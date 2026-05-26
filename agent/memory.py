import sqlite3
import json
import os
from datetime import datetime

import chromadb
from chromadb.config import Settings

MAX_MEMORY_SNIPPET_CHARS = 900
SQLITE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

class AgentMemory:
    def __init__(self, db_path="data"):
        os.makedirs(db_path, exist_ok=True)
        
        # 1. SQLite for exact relational chat history
        self.sqlite_path = os.path.join(db_path, "chat_history.db")
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_sqlite()

        # 2. ChromaDB for vector memory (semantic search of past conversations)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(db_path, "vector_store"))
        self.collection = self.chroma_client.get_or_create_collection(name="conversation_memory")

        # Curated long-term notes the agent chooses to keep, kept apart from
        # raw conversation memory so deliberate facts aren't drowned out.
        self.notes_collection = self.chroma_client.get_or_create_collection(name="agent_notes")

    def _init_sqlite(self):
        cursor = self.conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Check if messages table exists
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            cursor = self.conn.execute('''
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            ''')
            self.conn.commit()
        else:
            cursor = self.conn.execute("PRAGMA table_info(messages)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "conversation_id" not in columns:
                # Need to migrate
                cursor = self.conn.execute('ALTER TABLE messages RENAME TO messages_old')
                cursor = self.conn.execute('''
                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER,
                        role TEXT,
                        content TEXT,
                        image_path TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                ''')
                # Create a default conversation
                cursor = self.conn.execute('INSERT INTO conversations (name) VALUES ("Default Chat")')
                default_id = cursor.lastrowid
                
                # Copy old data
                cursor = self.conn.execute('INSERT INTO messages (id, conversation_id, role, content, timestamp) SELECT id, ?, role, content, timestamp FROM messages_old', (default_id,))
                cursor = self.conn.execute('DROP TABLE messages_old')
            else:
                if "image_path" not in columns:
                    cursor = self.conn.execute('ALTER TABLE messages ADD COLUMN image_path TEXT')
            self.conn.commit()

        # Ensure foreign keys are enabled
        cursor = self.conn.execute('PRAGMA foreign_keys = ON')

    def get_conversations(self):
        cursor = self.conn.execute('SELECT id, name, created_at, updated_at FROM conversations ORDER BY updated_at DESC')
        return [{"id": row[0], "name": row[1], "created_at": row[2], "updated_at": row[3]} for row in cursor.fetchall()]

    def create_conversation(self, name="New Chat"):
        cursor = self.conn.execute('INSERT INTO conversations (name) VALUES (?)', (name,))
        self.conn.commit()
        return cursor.lastrowid

    def rename_conversation(self, conversation_id, new_name):
        cursor = self.conn.execute('UPDATE conversations SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (new_name, conversation_id))
        self.conn.commit()

    def delete_conversation(self, conversation_id):
        cursor = self.conn.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        # Also delete messages since we might not have ON DELETE CASCADE if migrated weirdly or just to be safe
        cursor = self.conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        self.conn.commit()

    def fork_conversation(self, conversation_id, new_name=None):
        cursor = self.conn.execute('SELECT name FROM conversations WHERE id = ?', (conversation_id,))
        row = cursor.fetchone()
        if not row:
            return None
        orig_name = row[0]
        fork_name = new_name if new_name else f"{orig_name} (Fork)"
        
        new_id = self.create_conversation(fork_name)
        
        cursor = self.conn.execute('SELECT role, content, image_path, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conversation_id,))
        messages = cursor.fetchall()
        for msg in messages:
            cursor = self.conn.execute('INSERT INTO messages (conversation_id, role, content, image_path, timestamp) VALUES (?, ?, ?, ?, ?)', (new_id, msg[0], msg[1], msg[2], msg[3]))
        self.conn.commit()
        return new_id

    def add_message_to_sqlite(self, conversation_id, role, content, image_path=None):
        cursor = self.conn.execute('INSERT INTO messages (conversation_id, role, content, image_path) VALUES (?, ?, ?, ?)', (conversation_id, role, content, image_path))
        cursor = self.conn.execute('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (conversation_id,))
        self.conn.commit()
        return cursor.lastrowid

    def get_message_metadata(self, message_id):
        cursor = self.conn.execute(
            'SELECT conversation_id, role, timestamp FROM messages WHERE id = ?',
            (message_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {}
        return {
            "conversation_id": row[0],
            "role": row[1],
            "created_at": row[2],
        }

    def get_all_messages(self, conversation_id):
        cursor = self.conn.execute('SELECT role, content, image_path FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conversation_id,))
        return [{"role": row[0], "content": row[1], "image_path": row[2]} for row in cursor.fetchall()]

    def get_last_message(self, conversation_id):
        cursor = self.conn.execute(
            '''
            SELECT id, role, content, image_path
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 1
            ''',
            (conversation_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "image_path": row[3],
        }

    def clear_history(self, conversation_id):
        cursor = self.conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        self.conn.commit()

    def _current_timestamp(self):
        return datetime.utcnow().strftime(SQLITE_TIMESTAMP_FORMAT)

    def _prepare_vector_metadata(self, metadata):
        metadata = dict(metadata or {})
        message_id = metadata.get("message_id", metadata.get("id"))
        if message_id is not None:
            message_info = self.get_message_metadata(message_id)
            metadata.setdefault("id", message_id)
            metadata.setdefault("message_id", message_id)
            metadata.setdefault("conversation_id", message_info.get("conversation_id"))
            metadata.setdefault("role", message_info.get("role"))
            metadata.setdefault("created_at", message_info.get("created_at"))

        metadata.setdefault("created_at", self._current_timestamp())
        return {key: value for key, value in metadata.items() if value is not None}

    def add_to_vector_memory(self, text, metadata, embedding):
        """Store an embedding and its text in ChromaDB"""
        metadata = self._prepare_vector_metadata(metadata)
        doc_id = f"mem_{metadata.get('id', 'unknown')}_{hash(text)}"
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def _parse_timestamp(self, timestamp):
        if not timestamp:
            return None
        timestamp = str(timestamp).strip()
        for parser in (
            lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
            lambda value: datetime.strptime(value, SQLITE_TIMESTAMP_FORMAT),
        ):
            try:
                parsed = parser(timestamp)
                return parsed.replace(tzinfo=None)
            except ValueError:
                continue
        return None

    def _relative_age(self, timestamp):
        parsed = self._parse_timestamp(timestamp)
        if not parsed:
            return "age unknown"

        days = (datetime.utcnow() - parsed).days
        if days < 0:
            return "in the future"
        if days == 0:
            return "today"
        if days == 1:
            return "1 day ago"
        if days < 31:
            return f"{days} days ago"
        if days < 365:
            months = max(1, days // 30)
            return "1 month ago" if months == 1 else f"{months} months ago"

        years = max(1, days // 365)
        return "1 year ago" if years == 1 else f"{years} years ago"

    def _message_timestamp(self, metadata):
        metadata = metadata or {}
        timestamp = metadata.get("created_at") or metadata.get("timestamp")
        if timestamp:
            return timestamp

        message_id = metadata.get("message_id", metadata.get("id"))
        if message_id is None:
            return None
        return self.get_message_metadata(message_id).get("created_at")

    def _format_memory_line(self, role, text, timestamp=None):
        recorded = "Recorded at an unknown time"
        if timestamp:
            recorded = f"Recorded {str(timestamp)[:10]} ({self._relative_age(timestamp)})"
        return f"{recorded}: {role.capitalize()} said: {self._trim_memory_snippet(text)}"

    def _trim_memory_snippet(self, text):
        text = " ".join(str(text or "").split())
        if len(text) <= MAX_MEMORY_SNIPPET_CHARS:
            return text
        return text[:MAX_MEMORY_SNIPPET_CHARS].rstrip() + "..."

    def get_recent_memories(self, limit=6):
        cursor = self.conn.execute(
            '''
            SELECT role, content, timestamp
            FROM messages
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            ''',
            (limit,)
        )
        return [
            self._format_memory_line(role, content, timestamp)
            for role, content, timestamp in cursor.fetchall()
        ]

    def search_vector_memory(self, query_embedding, n_results=5):
        """Retrieve most relevant past conversation snippets based on vector similarity"""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_memories = []
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            for i in range(len(docs)):
                role = metas[i].get("role", "unknown").capitalize()
                timestamp = self._message_timestamp(metas[i])
                formatted_memories.append(self._format_memory_line(role, docs[i], timestamp))
            return formatted_memories
        return []

    def get_total_memories(self):
        """Return the total number of items stored in the vector database."""
        return self.collection.count()

    def add_note(self, text, embedding):
        """Store a durable note the agent chose to remember."""
        text = " ".join(str(text or "").split())
        if not text or embedding is None:
            return
        created_at = self._current_timestamp()
        doc_id = f"note_{hash(text)}_{created_at}"
        self.notes_collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"created_at": created_at, "memory_type": "note"}],
            ids=[doc_id],
        )

    def get_total_notes(self):
        """Return how many durable notes the agent has saved."""
        return self.notes_collection.count()

    def search_notes(self, query_embedding, n_results=3):
        """Retrieve the durable notes most relevant to the current query."""
        if self.notes_collection.count() == 0:
            return []

        results = self.notes_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        notes = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for i in range(len(docs)):
                timestamp = (metas[i] or {}).get("created_at")
                recorded = "saved at an unknown time"
                if timestamp:
                    recorded = f"saved {str(timestamp)[:10]} ({self._relative_age(timestamp)})"
                notes.append(f"- {self._trim_memory_snippet(docs[i])} ({recorded})")
        return notes

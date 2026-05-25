import sqlite3
import json
import os
import chromadb
from chromadb.config import Settings

MAX_MEMORY_SNIPPET_CHARS = 900

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

    def _init_sqlite(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Check if messages table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not self.cursor.fetchone():
            self.cursor.execute('''
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
            self.cursor.execute("PRAGMA table_info(messages)")
            columns = [info[1] for info in self.cursor.fetchall()]
            
            if "conversation_id" not in columns:
                # Need to migrate
                self.cursor.execute('ALTER TABLE messages RENAME TO messages_old')
                self.cursor.execute('''
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
                self.cursor.execute('INSERT INTO conversations (name) VALUES ("Default Chat")')
                default_id = self.cursor.lastrowid
                
                # Copy old data
                self.cursor.execute('INSERT INTO messages (id, conversation_id, role, content, timestamp) SELECT id, ?, role, content, timestamp FROM messages_old', (default_id,))
                self.cursor.execute('DROP TABLE messages_old')
            else:
                if "image_path" not in columns:
                    self.cursor.execute('ALTER TABLE messages ADD COLUMN image_path TEXT')
            self.conn.commit()

        # Ensure foreign keys are enabled
        self.cursor.execute('PRAGMA foreign_keys = ON')

    def get_conversations(self):
        self.cursor.execute('SELECT id, name, created_at, updated_at FROM conversations ORDER BY updated_at DESC')
        return [{"id": row[0], "name": row[1], "created_at": row[2], "updated_at": row[3]} for row in self.cursor.fetchall()]

    def create_conversation(self, name="New Chat"):
        self.cursor.execute('INSERT INTO conversations (name) VALUES (?)', (name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def rename_conversation(self, conversation_id, new_name):
        self.cursor.execute('UPDATE conversations SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (new_name, conversation_id))
        self.conn.commit()

    def delete_conversation(self, conversation_id):
        self.cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        # Also delete messages since we might not have ON DELETE CASCADE if migrated weirdly or just to be safe
        self.cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        self.conn.commit()

    def fork_conversation(self, conversation_id, new_name=None):
        self.cursor.execute('SELECT name FROM conversations WHERE id = ?', (conversation_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        orig_name = row[0]
        fork_name = new_name if new_name else f"{orig_name} (Fork)"
        
        new_id = self.create_conversation(fork_name)
        
        self.cursor.execute('SELECT role, content, image_path, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conversation_id,))
        messages = self.cursor.fetchall()
        for msg in messages:
            self.cursor.execute('INSERT INTO messages (conversation_id, role, content, image_path, timestamp) VALUES (?, ?, ?, ?, ?)', (new_id, msg[0], msg[1], msg[2], msg[3]))
        self.conn.commit()
        return new_id

    def add_message_to_sqlite(self, conversation_id, role, content, image_path=None):
        self.cursor.execute('INSERT INTO messages (conversation_id, role, content, image_path) VALUES (?, ?, ?, ?)', (conversation_id, role, content, image_path))
        self.cursor.execute('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (conversation_id,))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_all_messages(self, conversation_id):
        self.cursor.execute('SELECT role, content, image_path FROM messages WHERE conversation_id = ? ORDER BY id ASC', (conversation_id,))
        return [{"role": row[0], "content": row[1], "image_path": row[2]} for row in self.cursor.fetchall()]

    def clear_history(self, conversation_id):
        self.cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        self.conn.commit()

    def add_to_vector_memory(self, text, metadata, embedding):
        """Store an embedding and its text in ChromaDB"""
        doc_id = f"mem_{metadata.get('id', 'unknown')}_{hash(text)}"
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def _trim_memory_snippet(self, text):
        text = " ".join(str(text or "").split())
        if len(text) <= MAX_MEMORY_SNIPPET_CHARS:
            return text
        return text[:MAX_MEMORY_SNIPPET_CHARS].rstrip() + "..."

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
                text = self._trim_memory_snippet(docs[i])
                formatted_memories.append(f"{role} said: {text}")
            return formatted_memories
        return []

    def get_total_memories(self):
        """Return the total number of items stored in the vector database."""
        return self.collection.count()

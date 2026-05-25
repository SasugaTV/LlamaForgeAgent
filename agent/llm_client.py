import openai

class LlamaForgeClient:
    def __init__(
        self,
        base_url="http://localhost:1234/v1",
        chat_model="wd-thinking",
        embed_model="wd-embed",
        embedding_max_tokens=300,
        summary_max_tokens=900,
    ):
        # We point the standard OpenAI client at your local LlamaForge / LM Studio server
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.embedding_max_tokens = embedding_max_tokens
        self.summary_max_tokens = summary_max_tokens

    def count_tokens(self, text) -> int:
        """Returns a rough heuristic for tokens (1 token ~= 4 chars) to avoid heavy C++ bindings locking up the UI thread."""
        if isinstance(text, list):
            # Sum up text parts
            total_chars = 0
            for part in text:
                if part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    # Arbitrary token cost for image
                    total_chars += 4000
            return total_chars // 4
        return len(str(text)) // 4

    def _find_embedding_split(self, text, max_chars):
        window = text[:max_chars + 1]
        min_split = max(80, max_chars // 2)
        for delimiter in ("\n\n", "\n", ". ", "? ", "! ", " "):
            idx = window.rfind(delimiter)
            if idx >= min_split:
                return idx + len(delimiter)
        return max_chars

    def _chunk_text_for_embedding(self, text):
        max_chars = max(200, self.embedding_max_tokens * 4)
        remaining = str(text or "").strip()
        chunks = []

        while remaining:
            if self.count_tokens(remaining) <= self.embedding_max_tokens:
                chunks.append(remaining)
                break

            split_at = self._find_embedding_split(remaining, max_chars)
            chunk = remaining[:split_at].strip()
            if not chunk:
                chunk = remaining[:max_chars].strip()
                split_at = max_chars
            chunks.append(chunk)
            remaining = remaining[split_at:].strip()

        return chunks

    def _average_embeddings(self, embeddings):
        embeddings = [emb for emb in embeddings if emb]
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return embeddings[0]

        dims = min(len(emb) for emb in embeddings)
        return [
            sum(emb[i] for emb in embeddings) / len(embeddings)
            for i in range(dims)
        ]

    def _trim_text_to_token_budget(self, text, max_tokens):
        if self.count_tokens(text) <= max_tokens:
            return text

        max_chars = max(200, max_tokens * 4)
        split_at = self._find_embedding_split(text, max_chars)
        return text[:split_at].rstrip() + "\n[Summary truncated to fit context budget.]"

    def _get_embedding_chunk(self, text):
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_text = str(e).lower()
            if ("too large" in error_text or "batch size" in error_text) and len(text) > 200:
                split_at = self._find_embedding_split(text, max(100, len(text) // 2))
                left = text[:split_at].strip()
                right = text[split_at:].strip()
                if left and right:
                    print("Embedding chunk was still too large; retrying with smaller chunks.")
                    return self._average_embeddings([
                        self._get_embedding_chunk(left),
                        self._get_embedding_chunk(right),
                    ])
            print(f"Error getting embedding: {e}")
            return None

    def get_embedding(self, text: str):
        """Fetches one averaged vector embedding, chunking large inputs first."""
        chunks = self._chunk_text_for_embedding(text)
        if not chunks:
            return None

        if len(chunks) > 1:
            print(f"Embedding input split into {len(chunks)} chunks to stay under the local batch size.")

        embeddings = [self._get_embedding_chunk(chunk) for chunk in chunks]
        return self._average_embeddings(embeddings)

    def get_chat_response(self, messages, stream=True):
        """Gets a streaming or full response from the local chat model."""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"Error getting chat response: {e}")
            return None

    def summarize_messages(self, messages):
        """Asks the model to compress/summarize older conversation context."""
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Summarize the following conversation "
                    "so key facts and context are retained for future reference. "
                    f"Keep the summary under {self.summary_max_tokens} tokens."
                )
            },
        ]
        
        conversation_text = ""
        for msg in messages:
            content = msg['content']
            if isinstance(content, list):
                text_parts = [p['text'] for p in content if p.get('type') == 'text']
                content_str = " ".join(text_parts) + " [Image attached]"
            else:
                content_str = str(content)
            conversation_text += f"{msg['role'].capitalize()}: {content_str}\n"
            
        summary_prompt.append({"role": "user", "content": f"Summarize this conversation:\n\n{conversation_text}"})
        
        response = self.get_chat_response(summary_prompt, stream=False)
        if response and response.choices:
            return self._trim_text_to_token_budget(
                response.choices[0].message.content,
                self.summary_max_tokens
            )
        return "Failed to summarize."

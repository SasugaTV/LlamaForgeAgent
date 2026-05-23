import openai

class LlamaForgeClient:
    def __init__(self, base_url="http://localhost:1234/v1", chat_model="wd-thinking", embed_model="wd-embed"):
        # We point the standard OpenAI client at your local LlamaForge / LM Studio server
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self.chat_model = chat_model
        self.embed_model = embed_model

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

    def get_embedding(self, text: str):
        """Fetches the vector embedding for a piece of text from the local server."""
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

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
            {"role": "system", "content": "You are a helpful assistant. Please summarize the following conversation concisely so that key facts and context are retained for future reference."},
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
            return response.choices[0].message.content
        return "Failed to summarize."

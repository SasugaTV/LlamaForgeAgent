from llm_client import LlamaForgeClient

client = LlamaForgeClient()
print("Testing embedding...")
emb = client.get_embedding("Hello")
print("Embedding result type:", type(emb))

print("Testing chat...")
response = client.get_chat_response([{"role": "user", "content": "Hello"}], stream=False)
if response:
    print("Chat successful!")
else:
    print("Chat failed.")

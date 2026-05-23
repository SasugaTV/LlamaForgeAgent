from llm_client import LlamaForgeClient
import traceback

client = LlamaForgeClient()

print("Testing first message (System, User)...")
try:
    resp1 = client.get_chat_response([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ], stream=False)
    print("Response 1:", "Success" if resp1 else "Failed")
except Exception as e:
    print("Error 1:", e)

print("\nTesting second message with System in the middle (System, User, Assistant, System, User)...")
try:
    resp2 = client.get_chat_response([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "system", "content": "Relevant past memories: \nUser said hello."},
        {"role": "user", "content": "What is my name?"}
    ], stream=False)
    print("Response 2:", "Success" if resp2 else "Failed")
except Exception as e:
    print("Error 2:", e)

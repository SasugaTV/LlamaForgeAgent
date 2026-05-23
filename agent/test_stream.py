from llm_client import LlamaForgeClient
import traceback

client = LlamaForgeClient()
try:
    print("Sending stream request...")
    response = client.get_chat_response([
        {"role": "system", "content": "You are a bot"}, 
        {"role": "user", "content": "Hello"}
    ], stream=True)
    if response:
        print("Got response stream object")
        for chunk in response:
            delta = chunk.choices[0].delta
            print(f"Content: {getattr(delta, 'content', None)}, Reasoning: {getattr(delta, 'reasoning_content', None)}")
    else:
        print("Response is None")
except Exception as e:
    traceback.print_exc()

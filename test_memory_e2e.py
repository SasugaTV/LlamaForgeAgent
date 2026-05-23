import time
import os
import sys

# Ensure we are testing the agent folder
sys.path.append(os.path.join(os.path.dirname(__file__), "agent"))
from main import AgentApp

class HeadlessTester(AgentApp):
    def __init__(self):
        super().__init__()
        self.test_output = ""
        
    def append_to_display(self, text, tag=None):
        """Override the UI display to just capture text for the test."""
        # We don't want Tkinter errors if we run headless
        pass
        
    def capture_output(self, text):
        self.test_output += text

    def run_test(self):
        print("\n--- Starting Memory Smoke Test ---")
        
        # 1. Clean slate
        print("1. Creating fresh conversation...")
        self.new_conversation()
        
        # 2. Plant the memory
        fact = "My favorite fruit is a neon purple banana."
        print(f"2. User says: '{fact}'")
        
        # We need to manually simulate what process_message does because of the threading
        user_emb = self.llm.get_embedding(fact)
        user_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "user", fact, None)
        self.memory.add_to_vector_memory(fact, {"id": user_msg_id, "role": "user"}, user_emb)
        print("   -> Fact stored in Vector DB.")

        # Let the agent respond just to complete the turn
        self.working_context.append({"role": "user", "content": fact})
        resp1 = self.llm.get_chat_response(self._sanitize_messages(self.working_context), stream=False)
        ai_resp1 = resp1.choices[0].message.content if resp1 else "OK"
        ai_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "assistant", ai_resp1, None)
        ai_emb = self.llm.get_embedding(ai_resp1)
        self.memory.add_to_vector_memory(ai_resp1, {"id": ai_msg_id, "role": "assistant"}, ai_emb)
        print(f"   -> Agent replied: {ai_resp1[:30]}...")

        # 3. Create a NEW conversation to wipe the working context window
        print("\n3. Creating brand new conversation (wiping context window)...")
        self.new_conversation()
        
        # 4. Ask the question
        question = "What is my favorite fruit?"
        print(f"4. User asks: '{question}'")
        
        # Simulate process_message for the question
        user_emb2 = self.llm.get_embedding(question)
        
        # SEARCH FIRST (This is what we just fixed!)
        relevant_past = self.memory.search_vector_memory(user_emb2, n_results=5)
        print(f"   -> Vector Search found {len(relevant_past)} relevant memories:")
        for m in relevant_past:
            print(f"      - {m}")
            
        # Store question
        user_msg_id2 = self.memory.add_message_to_sqlite(self.current_conversation_id, "user", question, None)
        self.memory.add_to_vector_memory(question, {"id": user_msg_id2, "role": "user"}, user_emb2)
        
        # Inject context
        self.working_context.append({"role": "user", "content": question})
        query_context = list(self.working_context)
        
        if relevant_past and len(query_context) > 0 and query_context[0]["role"] == "system":
            memory_string = "\n".join(relevant_past)
            sys_msg = dict(query_context[0])
            sys_msg["content"] += f"\n\n[Relevant past memories for this query:\n{memory_string}]"
            query_context[0] = sys_msg
            
        # Get final answer
        resp2 = self.llm.get_chat_response(self._sanitize_messages(query_context), stream=False)
        final_answer = resp2.choices[0].message.content if resp2 else "Failed"
        print(f"\n5. Agent Final Answer: {final_answer}")
        
        if "neon purple banana" in final_answer.lower():
            print("\n✅ SMOKE TEST PASSED! The agent successfully retrieved and used the cross-conversation memory.")
        else:
            print("\n❌ SMOKE TEST FAILED! The agent did not use the injected memory.")

if __name__ == "__main__":
    tester = HeadlessTester()
    tester.run_test()
    # We must explicitly exit to avoid the Tkinter mainloop hanging the script
    os._exit(0)
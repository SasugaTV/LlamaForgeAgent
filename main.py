import customtkinter as ctk
import threading
import os
import shutil
import base64
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

from memory import AgentMemory
from llm_client import LlamaForgeClient

class AgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Settings ---
        self.max_context_window = 8192
        self.compression_threshold = self.max_context_window // 2
        self.system_prompt = self._load_system_prompt()
        self.current_conversation_id = None
        self.working_context = []
        self.attached_image_path = None
        self.inline_images = [] # Prevent garbage collection

        # --- Initialization ---
        self.title("LlamaForge Agent")
        self.geometry("1000x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.memory = AgentMemory(db_path="agent_data")
        self.llm = LlamaForgeClient()

        self._build_gui()
        self.load_conversations_list()
        
        # Load the latest conversation or create one
        convos = self.memory.get_conversations()
        if convos:
            self.load_conversation(convos[0]["id"])
        else:
            self.new_conversation()

    def _load_system_prompt(self):
        base_prompt = "You are a helpful, smart local AI assistant. You have access to vector memory to recall past events.\n"
        if os.path.exists("SOUL.md"):
            with open("SOUL.md", "r", encoding="utf-8") as f:
                base_prompt += f"\n--- AI PERSONA (SOUL) ---\n{f.read()}\n"
        if os.path.exists("USER.md"):
            with open("USER.md", "r", encoding="utf-8") as f:
                base_prompt += f"\n--- USER CONTEXT (USER) ---\n{f.read()}\n"
        return base_prompt

    def _build_gui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Conversations", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.new_chat_btn = ctk.CTkButton(self.sidebar_frame, text="+ New Chat", command=self.new_conversation)
        self.new_chat_btn.grid(row=1, column=0, padx=20, pady=10)

        self.conv_list_frame = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent")
        self.conv_list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Action Bar (Rename, Fork, Delete)
        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.title_label = ctk.CTkLabel(self.action_frame, text="Select a conversation", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(side="left", padx=10)

        self.del_btn = ctk.CTkButton(self.action_frame, text="Delete", width=60, fg_color="red", hover_color="darkred", command=self.delete_current_conversation)
        self.del_btn.pack(side="right", padx=5)
        
        self.fork_btn = ctk.CTkButton(self.action_frame, text="Fork", width=60, command=self.fork_current_conversation)
        self.fork_btn.pack(side="right", padx=5)

        self.rename_btn = ctk.CTkButton(self.action_frame, text="Rename", width=60, command=self.rename_current_conversation)
        self.rename_btn.pack(side="right", padx=5)

        # Chat display
        self.chat_display = ctk.CTkTextbox(self.main_frame, state="disabled", wrap="word", font=("Segoe UI", 14))
        self.chat_display.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.chat_display.tag_config("reasoning", foreground="#6E6E6E")

        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.attach_btn = ctk.CTkButton(self.input_frame, text="📎 Image", width=60, command=self.attach_image)
        self.attach_btn.grid(row=0, column=0, padx=(0, 10))

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...", font=("Segoe UI", 14))
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.send_message)
        self.send_button.grid(row=0, column=2)

        self.image_preview_label = ctk.CTkLabel(self.input_frame, text="", text_color="green")
        self.image_preview_label.grid(row=1, column=1, sticky="w", pady=(5,0))

    def load_conversations_list(self):
        for widget in self.conv_list_frame.winfo_children():
            widget.destroy()
            
        convos = self.memory.get_conversations()
        for conv in convos:
            btn = ctk.CTkButton(
                self.conv_list_frame, 
                text=conv["name"], 
                anchor="w", 
                fg_color="transparent", 
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                command=lambda c_id=conv["id"]: self.load_conversation(c_id)
            )
            btn.pack(fill="x", pady=2)
            
            if self.current_conversation_id == conv["id"]:
                btn.configure(fg_color=("gray75", "gray25"))

    def new_conversation(self):
        new_id = self.memory.create_conversation("New Chat")
        self.load_conversation(new_id)
        self.load_conversations_list()

    def load_conversation(self, conv_id):
        self.current_conversation_id = conv_id
        self.working_context = [{"role": "system", "content": self.system_prompt}]
        self.inline_images.clear()
        
        # Update title
        convos = self.memory.get_conversations()
        name = next((c["name"] for c in convos if c["id"] == conv_id), "Chat")
        self.title_label.configure(text=name)
        
        # Refresh sidebar highlights
        self.load_conversations_list()
        
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

        self.append_to_display("System: Agent initialized. Connected to local LlamaForge.\n" + "-"*50 + "\n\n")

        history = self.memory.get_all_messages(conv_id)
        
        # Rebuild working context and display
        for msg in history:
            role = msg["role"]
            text_content = msg["content"]
            img_path = msg.get("image_path")
            
            if img_path and os.path.exists(img_path):
                self.append_to_display(f"{role.capitalize()}: ")
                self.insert_image_to_display(img_path)
                self.append_to_display(f"\n{text_content}\n\n")
                
                with open(img_path, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode('utf-8')
                
                content_arr = [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
                self.working_context.append({"role": role, "content": content_arr})
            else:
                self.append_to_display(f"{role.capitalize()}: {text_content}\n\n")
                self.working_context.append({"role": role, "content": text_content})
                
    def rename_current_conversation(self):
        if not self.current_conversation_id: return
        dialog = ctk.CTkInputDialog(text="Enter new name:", title="Rename Conversation")
        new_name = dialog.get_input()
        if new_name:
            self.memory.rename_conversation(self.current_conversation_id, new_name)
            self.title_label.configure(text=new_name)
            self.load_conversations_list()

    def fork_current_conversation(self):
        if not self.current_conversation_id: return
        new_id = self.memory.fork_conversation(self.current_conversation_id)
        if new_id:
            self.load_conversation(new_id)

    def delete_current_conversation(self):
        if not self.current_conversation_id: return
        if messagebox.askyesno("Delete", "Are you sure you want to delete this conversation?"):
            self.memory.delete_conversation(self.current_conversation_id)
            self.current_conversation_id = None
            convos = self.memory.get_conversations()
            if convos:
                self.load_conversation(convos[0]["id"])
            else:
                self.new_conversation()

    def attach_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
        )
        if filepath:
            os.makedirs("artifacts", exist_ok=True)
            filename = os.path.basename(filepath)
            dest = os.path.join("artifacts", filename)
            # Avoid overwriting if same name
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join("artifacts", f"{base}_{counter}{ext}")
                counter += 1
                
            shutil.copy(filepath, dest)
            self.attached_image_path = dest
            self.image_preview_label.configure(text=f"Attached: {os.path.basename(dest)}")

    def insert_image_to_display(self, img_path):
        self.chat_display.configure(state="normal")
        try:
            pil_image = Image.open(img_path)
            # Resize image to fit nicely
            max_width = 300
            if pil_image.width > max_width:
                ratio = max_width / pil_image.width
                pil_image = pil_image.resize((max_width, int(pil_image.height * ratio)))
            
            photo = ImageTk.PhotoImage(pil_image)
            self.inline_images.append(photo) # keep reference
            
            textbox = self.chat_display._textbox
            textbox.image_create("end", image=photo)
        except Exception as e:
            self.chat_display.insert("end", f"[Image Error: {e}]")
        self.chat_display.configure(state="disabled")

    def append_to_display(self, text, tag=None):
        self.chat_display.configure(state="normal")
        if tag:
            self.chat_display.insert("end", text, tag)
        else:
            self.chat_display.insert("end", text)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _sanitize_messages(self, msgs):
        out = []
        for m in msgs:
            if not m.get("content"):
                continue
            if out and out[-1]["role"] == m["role"]:
                # If we need to merge content
                if isinstance(out[-1]["content"], list) or isinstance(m["content"], list):
                    # Combine lists/strings into a new list
                    c1 = out[-1]["content"] if isinstance(out[-1]["content"], list) else [{"type": "text", "text": out[-1]["content"]}]
                    c2 = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
                    out[-1] = {"role": m["role"], "content": c1 + [{"type": "text", "text": "\n\n"}] + c2}
                else:
                    out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
            else:
                out.append(dict(m))
        first_user = next((i for i, m in enumerate(out) if m["role"] == "user"), None)
        if first_user is None:
            return out
        return [m for i, m in enumerate(out) if m["role"] == "system" or i >= first_user]

    def manage_context(self):
        current_tokens = sum(self.llm.count_tokens(m["content"]) for m in self.working_context)
        if current_tokens > self.compression_threshold:
            print(f"Context window exceeded threshold ({current_tokens} > {self.compression_threshold}). Compressing...")
            if len(self.working_context) > 5:
                messages_to_compress = self.working_context[1:-4]
                recent_messages = self.working_context[-4:]
                summary = self.llm.summarize_messages(messages_to_compress)
                self.working_context = [
                    self.working_context[0],
                    {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
                ] + recent_messages
                print("Context compressed successfully.")

    def process_message(self, user_text, img_path):
        try:
            self.after(0, lambda: self.append_to_display("You: "))
            
            if img_path:
                self.after(0, lambda p=img_path: self.insert_image_to_display(p))
                self.after(0, lambda: self.append_to_display(f"\n{user_text}\n\n"))
                
                with open(img_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode('utf-8')
                
                content_for_llm = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            else:
                self.after(0, lambda: self.append_to_display(f"{user_text}\n\n"))
                content_for_llm = user_text
            
            user_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "user", user_text, img_path)
            self.working_context.append({"role": "user", "content": content_for_llm})
            
            self.after(0, self.load_conversations_list) # Update sidebar timestamps

            # Vector embed the text part
            user_emb = self.llm.get_embedding(user_text)
            if user_emb:
                self.memory.add_to_vector_memory(user_text, {"id": user_msg_id, "role": "user"}, user_emb)
                relevant_past = self.memory.search_vector_memory(user_emb, n_results=2)
                query_context = list(self.working_context)
                if relevant_past:
                    memory_string = "\n".join(relevant_past)
                    query_context.insert(-1, {"role": "system", "content": f"Relevant past memories:\n{memory_string}"})
            else:
                query_context = self.working_context

            self.manage_context()

            def start_agent_msg():
                self.append_to_display("Agent: ")
                
            self.after(0, start_agent_msg)
            
            query_context = self._sanitize_messages(query_context)
            response_stream = self.llm.get_chat_response(query_context, stream=True)
            
            full_response = ""
            in_reasoning = False
            if response_stream:
                for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning is None:
                        extra = getattr(delta, "model_extra", None) or {}
                        reasoning = extra.get("reasoning_content")
                    content = getattr(delta, "content", None)

                    if reasoning:
                        if not in_reasoning:
                            in_reasoning = True
                            self.after(0, lambda: self.append_to_display("[thinking] ", "reasoning"))
                        self.after(0, lambda t=reasoning: self.append_to_display(t, "reasoning"))

                    if content:
                        if in_reasoning:
                            in_reasoning = False
                            self.after(0, lambda: self.append_to_display("\n\n"))
                        full_response += content
                        self.after(0, lambda text=content: self.append_to_display(text))
            else:
                self.after(0, lambda: self.append_to_display("[Error: response_stream is None. Connection failed.]"))
                
            self.after(0, lambda: self.append_to_display("\n\n"))

            if full_response:
                ai_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "assistant", full_response, None)
                self.working_context.append({"role": "assistant", "content": full_response})
                ai_emb = self.llm.get_embedding(full_response)
                if ai_emb:
                    self.memory.add_to_vector_memory(full_response, {"id": ai_msg_id, "role": "assistant"}, ai_emb)

        except Exception as e:
            error_msg = f"\n[CRITICAL THREAD ERROR]: {str(e)}\n\n"
            self.after(0, lambda: self.append_to_display(error_msg))
            print(f"Exception in process_message: {e}")
            
        finally:
            def re_enable_input():
                self.send_button.configure(state="normal")
                self.entry.configure(state="normal")
                self.attach_btn.configure(state="normal")
                self.entry.focus()
                
            self.after(0, re_enable_input)

    def send_message(self):
        text = self.entry.get().strip()
        img = self.attached_image_path
        
        # At least one must be present
        if not text and not img:
            return

        self.entry.delete(0, "end")
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.attach_btn.configure(state="disabled")
        
        self.attached_image_path = None
        self.image_preview_label.configure(text="")

        threading.Thread(target=self.process_message, args=(text, img), daemon=True).start()

if __name__ == "__main__":
    app = AgentApp()
    app.mainloop()
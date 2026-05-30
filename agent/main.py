import customtkinter as ctk
import threading
import os
import json
import shutil
import subprocess
import base64
import tkinter as tk
import re
from datetime import datetime
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

from memory import AgentMemory
from llm_client import LlamaForgeClient
from agent_commands import parse_commands

class AgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Paths ---
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # --- Settings ---
        self.max_context_window = 8192
        self.compression_threshold = self.max_context_window // 2
        self.assistant_response_max_tokens = 1200
        self.assistant_visible_token_limit = 360
        self.assistant_reasoning_token_limit = 900
        self.system_prompt = self._load_system_prompt()
        self.current_conversation_id = None
        self.working_context = []
        self.attached_image_path = None
        self.inline_images = [] # Prevent garbage collection
        self.chat_font_size = 14
        self.agent_busy = False
        self.message_queue = []
        self.chat_follow_output = True
        self.chat_autoscroll_after_id = None
        self.llamaforge_status = {
            "loading": False,
            "tokens_up_active": False,
            "inferencing": False,
            "tokens_down_active": False,
            "tokens_up": 0,
            "tokens_down": 0,
        }
        
        # Reasoning UI state
        self.reasoning_states = {}
        self.reasoning_buttons = {}
        self.reasoning_counter = 0
        self.global_show_thinking = ctk.BooleanVar(value=True)

        # Text-to-speech state
        self.auto_tts_enabled = ctk.BooleanVar(value=False)
        self.tts_payloads = {}
        self.tts_buttons = {}
        self.tts_counter = 0
        self.tts_process = None
        self.tts_current_id = None

        # --- Initialization ---
        self.title("LlamaForge Agent")
        self.geometry("1000x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.memory = AgentMemory(db_path=os.path.join(self.data_dir, "agent_data"))
        self.llm = LlamaForgeClient()

        self._build_gui()
        self.load_conversations_list()
        
        # Load the latest conversation or create one
        convos = self.memory.get_conversations()
        if convos:
            self.load_conversation(convos[0]["id"])
        else:
            self.new_conversation()

        self.update_memory_count_display()
        self._refresh_location_label()

    def update_memory_count_display(self):
        count = self.memory.get_total_memories()
        notes = self.memory.get_total_notes()
        text = f"Memories: {count}"
        if notes:
            text += f"  |  Notes: {notes}"
        self.memory_count_label.configure(text=text)

    def _refresh_location_label(self):
        if not hasattr(self, "location_label"):
            return
        place = (self._get_location().get("location") or "").strip()
        self.location_label.configure(text=f"📍 {place}" if place else "📍 location unknown")

    def _load_system_prompt(self):
        base_prompt = "You are a helpful, smart local AI assistant. You have access to vector memory to recall past events.\n"
        soul_path = os.path.join(self.data_dir, "SOUL.md")
        user_path = os.path.join(self.data_dir, "USER.md")
        
        if os.path.exists(soul_path):
            with open(soul_path, "r", encoding="utf-8") as f:
                base_prompt += f"\n--- AI PERSONA (SOUL) ---\n{f.read()}\n"
        if os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                base_prompt += f"\n--- USER CONTEXT (USER) ---\n{f.read()}\n"
        base_prompt += self._self_management_instructions()
        return base_prompt

    def _self_management_instructions(self):
        return (
            "\n--- INSTRUCTIONS ---\n"
            "Focus purely on having a natural, helpful conversation with the user. "
            "Prefer compact, direct replies with only the detail the user needs right now. "
            "Your context is automatically managed by a background system, so you do not need to use "
            "any special bracket commands or manage your own memory. Just be conversational.\n"
        )

    def _current_time_context(self):
        now = datetime.now().astimezone()
        return (
            f"Current local date/time: {now.strftime('%Y-%m-%d %H:%M:%S %z')}. "
            "Treat recalled memories as dated observations; time-sensitive memories may be stale, resolved, or superseded."
        )

    def _notepad_path(self):
        return os.path.join(self.data_dir, "NOTEPAD.md")

    def _location_path(self):
        return os.path.join(self.data_dir, "location.json")

    def _load_notepad_context(self):
        notepad_path = self._notepad_path()
        if not os.path.exists(notepad_path):
            return ""
        with open(notepad_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _get_location(self):
        try:
            with open(self._location_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _humanize_minutes(self, minutes):
        if minutes < 0:
            return "just now"
        if minutes < 1:
            return "moments ago"
        if minutes < 60:
            return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"
        hours = minutes // 60
        if hours < 24:
            return "about an hour ago" if hours == 1 else f"about {hours} hours ago"
        days = hours // 24
        return "yesterday" if days == 1 else f"{days} days ago"

    def _location_context(self):
        data = self._get_location()
        place = (data.get("location") or "").strip()
        if not place:
            return ""
        suffix = ""
        set_at = data.get("set_at")
        if set_at:
            try:
                stated = datetime.fromisoformat(set_at)
                now = datetime.now(stated.tzinfo) if stated.tzinfo else datetime.now()
                minutes = int((now - stated).total_seconds() // 60)
                suffix = f" (stated {self._humanize_minutes(minutes)})"
            except ValueError:
                pass
        return (
            f"The user's most recently stated location is: {place}{suffix}. "
            "This may be stale - if their messages suggest they've moved, update it. "
            "Let the setting shape relevant, organic suggestions when it genuinely fits."
        )

    def _format_ui_timestamp(self, timestamp_str):
        if not timestamp_str:
            return datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            # Assuming SQLITE_TIMESTAMP_FORMAT is "%Y-%m-%d %H:%M:%S"
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return str(timestamp_str)[:16]

    def _set_location(self, place):
        place = " ".join(str(place or "").split())
        if not place:
            return
        payload = {
            "location": place,
            "set_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with open(self._location_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.after(0, self._refresh_location_label)

    def _append_note(self, text):
        text = " ".join(str(text or "").split())
        if not text:
            return
        line = f"- {datetime.now().strftime('%Y-%m-%d')}: {text}"
        existing = self._load_notepad_context()
        body = f"{existing}\n{line}" if existing else line
        with open(self._notepad_path(), "w", encoding="utf-8") as f:
            f.write(body.strip() + "\n")

    def _complete_note(self, text):
        needle = " ".join(str(text or "").split()).lower()
        if not needle or not os.path.exists(self._notepad_path()):
            return
        with open(self._notepad_path(), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        kept = [ln for ln in lines if needle not in ln.lower()]
        remaining = "\n".join(kept).strip()
        with open(self._notepad_path(), "w", encoding="utf-8") as f:
            f.write(remaining + "\n" if remaining else "")

    def _remember_note(self, text):
        text = " ".join(str(text or "").split())
        if not text:
            return
        embedding = self.llm.get_embedding(text)
        if embedding:
            self.memory.add_note(text, embedding)
            self.after(0, self.update_memory_count_display)

    def _apply_agent_commands(self, commands):
        handlers = {
            "LOCATION": self._set_location,
            "NOTE": self._append_note,
            "NOTE_DONE": self._complete_note,
            "REMEMBER": self._remember_note,
        }
        for keyword, value in commands:
            handler = handlers.get(keyword)
            if handler and value:
                try:
                    handler(value)
                except Exception as e:
                    print(f"Failed to apply agent command {keyword}: {e}")

    def _is_first_user_message(self):
        return not any(message.get("role") == "user" for message in self.working_context)

    def _user_message_count(self):
        return sum(1 for message in self.working_context if message.get("role") == "user")

    def _build_first_turn_memory_context(self):
        recent_memories = self.memory.get_recent_memories(limit=6)
        if recent_memories:
            return "Recent memories:\n" + "\n".join(recent_memories)
        return ""

    def cancel_generation(self):
        self.cancel_inference_flag = True

    def _append_query_system_context(self, query_context, title, content):
        if not content or not query_context or query_context[0].get("role") != "system":
            return query_context

        sys_msg = dict(query_context[0])
        sys_msg["content"] += f"\n\n[{title}:\n{content}]"
        query_context[0] = sys_msg
        return query_context

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_frame.grid_remove()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.grid()
            self.sidebar_visible = True

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

        self.memory_count_label = ctk.CTkLabel(self.sidebar_frame, text="Memories: 0", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray50")
        self.memory_count_label.grid(row=3, column=0, pady=(0, 2))

        self.location_label = ctk.CTkLabel(self.sidebar_frame, text="📍 location unknown", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray60")
        self.location_label.grid(row=4, column=0, pady=(0, 10))

        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Action Bar (Rename, Fork, Delete)
        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.sidebar_visible = True
        self.toggle_sidebar_btn = ctk.CTkButton(self.action_frame, text="☰", width=30, command=self.toggle_sidebar)
        self.toggle_sidebar_btn.pack(side="left", padx=(0, 10))

        self.title_label = ctk.CTkLabel(self.action_frame, text="Select a conversation", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(side="left", padx=10)

        self.tts_stop_btn = ctk.CTkButton(
            self.action_frame,
            text="Stop TTS",
            width=70,
            fg_color="#8B0000",
            hover_color="#5E0000",
            command=self.stop_tts
        )
        self.tts_stop_btn.pack(side="right", padx=5)

        self.auto_tts_switch = ctk.CTkSwitch(
            self.action_frame,
            text="Auto TTS",
            variable=self.auto_tts_enabled
        )
        self.auto_tts_switch.pack(side="right", padx=10)

        self.cancel_inference_flag = False
        self.cancel_btn = ctk.CTkButton(self.action_frame, text="Stop Generate", width=80, fg_color="#d48c00", hover_color="#a86e00", command=self.cancel_generation)
        self.cancel_btn.pack(side="right", padx=5)

        self.del_btn = ctk.CTkButton(self.action_frame, text="Delete", width=60, fg_color="red", hover_color="darkred", command=self.delete_current_conversation)
        self.del_btn.pack(side="right", padx=5)
        
        self.fork_btn = ctk.CTkButton(self.action_frame, text="Fork", width=60, command=self.fork_current_conversation)
        self.fork_btn.pack(side="right", padx=5)

        self.rename_btn = ctk.CTkButton(self.action_frame, text="Rename", width=60, command=self.rename_current_conversation)
        self.rename_btn.pack(side="right", padx=5)

        self.thinking_switch = ctk.CTkSwitch(self.action_frame, text="Show Thinking", variable=self.global_show_thinking, command=self.toggle_all_thinking)
        self.thinking_switch.pack(side="right", padx=10)

        self.zoom_out_btn = ctk.CTkButton(self.action_frame, text="A-", width=40, command=self.decrease_font_size)
        self.zoom_out_btn.pack(side="right", padx=5)

        self.zoom_in_btn = ctk.CTkButton(self.action_frame, text="A+", width=40, command=self.increase_font_size)
        self.zoom_in_btn.pack(side="right", padx=5)

        # Chat display
        self.chat_display = ctk.CTkTextbox(self.main_frame, state="disabled", wrap="word", font=("Segoe UI", self.chat_font_size))
        self.chat_display.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.chat_display.tag_config("reasoning", foreground="#6E6E6E")
        self.chat_display.tag_config("user_text", foreground="#5Dadec")
        self.chat_display.tag_config("agent_text", foreground="#48C774")
        self._bind_chat_scroll_events()

        # LlamaForge activity strip
        self.llamaforge_status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.llamaforge_status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            self.llamaforge_status_frame.grid_columnconfigure(col, weight=1)

        status_font = ctk.CTkFont(size=12, weight="bold")
        self.llamaforge_status_labels = {
            "loading": ctk.CTkLabel(self.llamaforge_status_frame, text="Loading", font=status_font),
            "tokens_up": ctk.CTkLabel(self.llamaforge_status_frame, text="Tokens up: 0", font=status_font),
            "inferencing": ctk.CTkLabel(self.llamaforge_status_frame, text="Inferencing", font=status_font),
            "tokens_down": ctk.CTkLabel(self.llamaforge_status_frame, text="Tokens down: 0", font=status_font),
        }
        for col, key in enumerate(("loading", "tokens_up", "inferencing", "tokens_down")):
            self.llamaforge_status_labels[key].grid(row=0, column=col, sticky="ew", padx=5)
        self._refresh_llamaforge_status()

        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=3, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...", font=("Segoe UI", self.chat_font_size))
        self.entry.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.button_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.button_frame.grid(row=1, column=0, sticky="ew")

        self.attach_btn = ctk.CTkButton(self.button_frame, text="📎 Image", width=60, command=self.attach_image)
        self.attach_btn.pack(side="left", padx=(0, 10))

        self.send_button = ctk.CTkButton(self.button_frame, text="Send", width=80, command=self.send_message)
        self.send_button.pack(side="left", padx=(0, 10))

        self.repost_button = ctk.CTkButton(
            self.button_frame,
            text="Repost Last",
            width=90,
            state="disabled",
            command=self.repost_last_message
        )
        self.repost_button.pack(side="left", padx=(0, 10))

        self.scroll_bottom_btn = ctk.CTkButton(self.button_frame, text="↓ Bottom", width=60, command=self.scroll_to_bottom)
        self.scroll_bottom_btn.pack(side="left", padx=(0, 10))

        self.status_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.status_frame.grid(row=2, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(1, weight=1)

        self.image_preview_label = ctk.CTkLabel(self.status_frame, text="", text_color="green")
        self.image_preview_label.grid(row=0, column=0, sticky="w", pady=(5,0))

        self.queue_status_label = ctk.CTkLabel(self.status_frame, text="", text_color="gray50")
        self.queue_status_label.grid(row=0, column=1, sticky="e", pady=(5,0))

    def delete_message_by_id(self, msg_id):
        if messagebox.askyesno("Delete Message", "Delete this message from history?"):
            self.memory.delete_message(msg_id)
            if self.current_conversation_id:
                self.load_conversation(self.current_conversation_id)

    def fork_from_message_by_id(self, msg_id):
        if messagebox.askyesno("Fork Conversation", "Create a new conversation branching from this message?"):
            new_id = self.memory.fork_conversation(self.current_conversation_id, up_to_message_id=msg_id)
            if new_id:
                self.load_conversation(new_id)

    def _ui_insert_message_controls(self, msg_id):
        if msg_id is None:
            return
            
        self.chat_display.configure(state="normal")
        
        fork_btn = ctk.CTkButton(
            self.chat_display,
            text="🔀",
            width=20,
            height=20,
            fg_color="transparent",
            text_color="#8AB4F8",
            hover_color="#333333",
            font=ctk.CTkFont(size=max(10, self.chat_font_size - 2)),
            command=lambda mid=msg_id: self.fork_from_message_by_id(mid)
        )
        self.chat_display._textbox.window_create("end", window=fork_btn)
        self.chat_display.insert("end", " ")
        
        del_btn = ctk.CTkButton(
            self.chat_display,
            text="🗑️",
            width=20,
            height=20,
            fg_color="transparent",
            text_color="#F28B82",
            hover_color="#333333",
            font=ctk.CTkFont(size=max(10, self.chat_font_size - 2)),
            command=lambda mid=msg_id: self.delete_message_by_id(mid)
        )
        self.chat_display._textbox.window_create("end", window=del_btn)
        self.chat_display.insert("end", " ")
        
        self.chat_display.configure(state="disabled")

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
        self.message_queue.clear()
        self._update_queue_status()
        self.stop_tts()
        self.chat_follow_output = True
        self._cancel_chat_autoscroll()
        self.current_conversation_id = conv_id
        self.working_context = [{"role": "system", "content": self.system_prompt}]
        self.inline_images.clear()
        self.tts_payloads.clear()
        self.tts_buttons.clear()
        
        self.reasoning_states.clear()
        self.reasoning_buttons.clear()
        self.reasoning_counter = 0
        
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
        
        # --- TWO STREAMS: Load the agent's compressed context if it exists ---
        agent_context_loaded = False
        context_file = os.path.join(self.data_dir, f"agent_context_{conv_id}.json")
        if os.path.exists(context_file):
            try:
                with open(context_file, "r", encoding="utf-8") as f:
                    saved_context = json.load(f)
                    if saved_context:
                        self.working_context = saved_context
                        # Refresh the system prompt in case the user edited SOUL.md / USER.md
                        if self.working_context[0].get("role") == "system":
                            self.working_context[0]["content"] = self.system_prompt
                        agent_context_loaded = True
            except Exception as e:
                print(f"Failed to load agent context from json: {e}")

        # Rebuild display (and fallback context if needed)
        for msg in history:
            role = msg["role"]
            text_content = msg["content"]
            img_path = msg.get("image_path")
            msg_id = msg.get("id")
            timestamp_str = msg.get("timestamp")
            display_time = self._format_ui_timestamp(timestamp_str)
            
            tag = "user_text" if role == "user" else "agent_text"
            role_label = f"[{display_time}] You" if role == "user" else f"[{display_time}] Agent"
            
            if img_path and os.path.exists(img_path):
                self.append_to_display(f"{role_label}: ", tag)
                self._ui_insert_message_controls(msg_id)
                self.insert_image_to_display(img_path)
                self.append_to_display(f"\n{text_content}\n\n")
                
                if not agent_context_loaded:
                    with open(img_path, "rb") as img_file:
                        base64_img = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    content_arr = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                    self.working_context.append({"role": role, "content": content_arr})
            else:
                self.append_to_display(f"{role_label}: ", tag)
                self._ui_insert_message_controls(msg_id)
                
                if role == "assistant":
                    reasoning_text, text_content = self._split_reasoning_text(text_content)
                    if reasoning_text:
                        reasoning_speech_id = self._create_tts_payload(reasoning_text)
                        
                        self.reasoning_counter += 1
                        b_id = f"reason_{self.reasoning_counter}"
                        self._ui_start_reasoning_block(b_id, reasoning_speech_id)
                        self._ui_append_reasoning(b_id, reasoning_text)
                        self._ui_end_reasoning_block(b_id)

                    if text_content.strip():
                        final_speech_id = self._create_tts_payload(text_content)
                        self._ui_insert_tts_button(final_speech_id)
                        self.append_to_display(" ")

                self.append_to_display(f"{text_content}\n\n")
                if not agent_context_loaded:
                    self.working_context.append({"role": role, "content": msg["content"]})
        self._update_repost_button_state()
                
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
            artifacts_dir = os.path.join(self.data_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            filename = os.path.basename(filepath)
            dest = os.path.join(artifacts_dir, filename)
            # Avoid overwriting if same name
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(artifacts_dir, f"{base}_{counter}{ext}")
                counter += 1
                
            shutil.copy(filepath, dest)
            self.attached_image_path = dest
            self.image_preview_label.configure(text=f"Attached: {os.path.basename(dest)}")

    def insert_image_to_display(self, img_path):
        should_follow = self.chat_follow_output
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
        if should_follow:
            self.chat_follow_output = True
            self._schedule_chat_autoscroll()

    def increase_font_size(self):
        if self.chat_font_size < 40:
            self.chat_font_size += 2
            self._update_fonts()

    def decrease_font_size(self):
        if self.chat_font_size > 8:
            self.chat_font_size -= 2
            self._update_fonts()

    def _update_fonts(self):
        new_font = ctk.CTkFont(family="Segoe UI", size=self.chat_font_size)
        self.chat_display.configure(font=new_font)
        self.entry.configure(font=new_font)
        reasoning_font = ctk.CTkFont(size=self.chat_font_size, slant="italic")
        speaker_font = ctk.CTkFont(size=max(12, self.chat_font_size))
        for btn in list(self.reasoning_buttons.values()):
            if self._widget_exists(btn):
                btn.configure(font=reasoning_font)
        for btn in list(self.tts_buttons.values()):
            if self._widget_exists(btn):
                btn.configure(font=speaker_font)

    def _on_mousewheel_zoom(self, event):
        if event.delta > 0:
            self.increase_font_size()
        else:
            self.decrease_font_size()

    def scroll_to_bottom(self):
        self.chat_follow_output = True
        self._schedule_chat_autoscroll()

    def append_to_display(self, text, tag=None):
        should_follow = self.chat_follow_output
            
        self.chat_display.configure(state="normal")
        if tag:
            self.chat_display.insert("end", text, tag)
        else:
            self.chat_display.insert("end", text)
        self.chat_display.configure(state="disabled")
        
        if should_follow:
            self.chat_follow_output = True
            self._schedule_chat_autoscroll()

    def _bind_chat_scroll_events(self):
        textbox = self.chat_display._textbox
        textbox.bind("<MouseWheel>", self._on_chat_user_scroll, add="+")
        textbox.bind("<Button-4>", self._on_chat_user_scroll, add="+")
        textbox.bind("<Button-5>", self._on_chat_user_scroll, add="+")
        textbox.bind("<Prior>", self._on_chat_user_scroll, add="+")
        textbox.bind("<Next>", self._on_chat_user_scroll, add="+")
        textbox.bind("<Home>", self._on_chat_user_scroll, add="+")
        textbox.bind("<End>", self._on_chat_user_scroll, add="+")

        scrollbar = getattr(self.chat_display, "_scrollbar", None)
        if scrollbar:
            scrollbar.bind("<ButtonPress-1>", self._on_chat_user_scroll, add="+")
            scrollbar.bind("<B1-Motion>", self._on_chat_scrollbar_drag, add="+")
            scrollbar.bind("<ButtonRelease-1>", self._on_chat_scrollbar_drag, add="+")

    def _on_chat_user_scroll(self, event=None):
        if self._event_scrolls_up(event):
            self.chat_follow_output = False
            self._cancel_chat_autoscroll()
            return
        self.after_idle(self._sync_chat_follow_state)

    def _on_chat_scrollbar_drag(self, event=None):
        self.chat_follow_output = False
        self._cancel_chat_autoscroll()
        self.after_idle(self._sync_chat_follow_state)

    def _event_scrolls_up(self, event):
        if event is None:
            return False
        if getattr(event, "num", None) == 4:
            return True
        if getattr(event, "keysym", None) in {"Prior", "Home"}:
            return True
        return getattr(event, "delta", 0) > 0

    def _is_chat_near_bottom(self):
        textbox = self.chat_display._textbox
        try:
            textbox.update_idletasks()
            yview = textbox.yview()
        except tk.TclError:
            return True
        if len(yview) != 2:
            return True
        return yview[1] >= 0.999

    def _sync_chat_follow_state(self):
        self.chat_follow_output = self._is_chat_near_bottom()

    def _cancel_chat_autoscroll(self):
        if not self.chat_autoscroll_after_id:
            return
        try:
            self.after_cancel(self.chat_autoscroll_after_id)
        except tk.TclError:
            pass
        self.chat_autoscroll_after_id = None

    def _schedule_chat_autoscroll(self):
        self._cancel_chat_autoscroll()
        self.chat_autoscroll_after_id = self.after(16, self._autoscroll_chat_to_bottom)

    def _autoscroll_chat_to_bottom(self):
        self.chat_autoscroll_after_id = None
        if not self.chat_follow_output:
            return
        self.chat_display.see("end")
        try:
            self.chat_display._textbox.yview_moveto(1.0)
        except tk.TclError:
            pass

    def _widget_exists(self, widget):
        try:
            return widget.winfo_exists()
        except tk.TclError:
            return False

    def _split_reasoning_text(self, text):
        reasoning_match = re.match(r"^\s*<think>\n?(.*?)\n?</think>\s*(.*)$", text, flags=re.DOTALL)
        if not reasoning_match:
            return "", text
        return reasoning_match.group(1).strip(), reasoning_match.group(2).strip()

    def _create_tts_payload(self, text):
        self.tts_counter += 1
        speech_id = f"tts_{self.tts_counter}"
        self._set_tts_payload(speech_id, text)
        return speech_id

    def _set_tts_payload(self, speech_id, text):
        self.tts_payloads[speech_id] = (text or "").strip()

    def _ui_insert_tts_button(self, speech_id):
        btn = ctk.CTkButton(
            self.chat_display,
            text="🔊",
            width=28,
            height=22,
            fg_color="transparent",
            text_color="#8AB4F8",
            hover_color="#333333",
            font=ctk.CTkFont(size=max(12, self.chat_font_size)),
            command=lambda sid=speech_id: self.toggle_tts(sid)
        )
        self.tts_buttons[speech_id] = btn

        self.chat_display.configure(state="normal")
        self.chat_display._textbox.window_create("end", window=btn)
        self.chat_display.configure(state="disabled")
        self._refresh_tts_buttons()

    def _get_powershell_exe(self):
        exe = shutil.which("powershell")
        if exe:
            return exe

        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        exe = os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        if os.path.exists(exe):
            return exe
        return "powershell.exe"

    def _start_tts_process(self, text):
        script = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
Add-Type -AssemblyName System.Speech
$text = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($text)) { exit 0 }
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SetOutputToDefaultAudioDevice()
    $synth.Speak($text)
}
finally {
    $synth.Dispose()
}
"""
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [
                self._get_powershell_exe(),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
        try:
            proc.stdin.write(text)
            proc.stdin.close()
        except OSError:
            proc.kill()
            raise
        return proc

    def _refresh_tts_buttons(self):
        for speech_id, btn in list(self.tts_buttons.items()):
            if not self._widget_exists(btn):
                self.tts_buttons.pop(speech_id, None)
                continue
            btn.configure(text="■" if speech_id == self.tts_current_id else "🔊")

    def toggle_tts(self, speech_id):
        if speech_id == self.tts_current_id:
            self.stop_tts()
            return

        text = self.tts_payloads.get(speech_id, "").strip()
        if not text:
            return

        self.stop_tts()
        try:
            self.tts_process = self._start_tts_process(text)
        except Exception as exc:
            messagebox.showerror("Text to Speech", f"Could not start Windows text to speech:\n{exc}")
            return

        self.tts_current_id = speech_id
        self._refresh_tts_buttons()
        self.after(200, lambda sid=speech_id, proc=self.tts_process: self._poll_tts_process(sid, proc))

    def stop_tts(self):
        proc = self.tts_process
        self.tts_process = None
        self.tts_current_id = None

        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

        self._refresh_tts_buttons()

    def _poll_tts_process(self, speech_id, proc):
        if proc is not self.tts_process:
            return
        if proc.poll() is None:
            self.after(200, lambda sid=speech_id, p=proc: self._poll_tts_process(sid, p))
            return

        self.tts_process = None
        if self.tts_current_id == speech_id:
            self.tts_current_id = None
        self._refresh_tts_buttons()

    def _finalize_final_tts(self, speech_id, text):
        self._set_tts_payload(speech_id, text)
        if self.auto_tts_enabled.get() and self.tts_payloads[speech_id]:
            self.toggle_tts(speech_id)

    def _format_token_count(self, count):
        return f"{max(0, int(count)):,}"

    def _count_single_message_tokens(self, message):
        return (
            self.llm.count_tokens(message.get("role", ""))
            + self.llm.count_tokens(message.get("content", ""))
            + 4
        )

    def _count_message_tokens(self, messages):
        return sum(self._count_single_message_tokens(message) for message in messages)

    def _context_metrics(self, messages=None):
        messages = self.working_context if messages is None else messages
        by_role = {}
        image_count = 0
        for message in messages:
            role = message.get("role", "unknown")
            by_role[role] = by_role.get(role, 0) + 1
            content = message.get("content", "")
            if isinstance(content, list):
                image_count += sum(1 for part in content if part.get("type") == "image_url")
        return {
            "tokens": self._count_message_tokens(messages),
            "messages": len(messages),
            "by_role": by_role,
            "images": image_count,
        }

    def _format_context_metrics(self, metrics):
        role_parts = [
            f"{role}={metrics['by_role'][role]}"
            for role in sorted(metrics["by_role"])
        ]
        role_text = ", ".join(role_parts) if role_parts else "no messages"
        image_text = f", images={metrics['images']}" if metrics["images"] else ""
        return (
            f"{self._format_token_count(metrics['tokens'])} est. tokens, "
            f"{metrics['messages']} messages ({role_text}{image_text})"
        )

    def _format_context_window_pct(self, tokens):
        if self.max_context_window <= 0:
            return "unknown"
        return f"{(tokens / self.max_context_window) * 100:.1f}%"

    def _format_token_delta(self, before_tokens, after_tokens):
        saved_tokens = before_tokens - after_tokens
        pct = (saved_tokens / before_tokens) * 100 if before_tokens else 0
        amount = self._format_token_count(abs(saved_tokens))
        if saved_tokens >= 0:
            return f"{amount} est. tokens saved ({abs(pct):.1f}%)"
        return f"grew by {amount} est. tokens ({abs(pct):.1f}%)"

    def _compression_summary_budget(self, chunk_tokens):
        return max(40, min(self.llm.summary_max_tokens, int(chunk_tokens * 0.45)))

    def _limit_visible_reply_to_budget(self, text):
        if self.llm.count_tokens(text) <= self.assistant_visible_token_limit:
            return text, False

        max_chars = max(120, int(self.assistant_visible_token_limit * 2.5))
        candidate = str(text)[:max_chars].rstrip()
        min_boundary = max(80, int(max_chars * 0.55))

        sentence_matches = list(re.finditer(r"[.!?](?:\s|$)", candidate))
        if sentence_matches and sentence_matches[-1].end() >= min_boundary:
            return candidate[:sentence_matches[-1].end()].rstrip(), True

        paragraph_boundary = candidate.rfind("\n\n")
        if paragraph_boundary >= min_boundary:
            return candidate[:paragraph_boundary].rstrip(), True

        word_boundary = candidate.rfind(" ")
        if word_boundary >= min_boundary:
            return candidate[:word_boundary].rstrip(), True

        return candidate, True

    def _close_response_stream(self, response_stream):
        close_stream = getattr(response_stream, "close", None)
        if callable(close_stream):
            close_stream()

    def _set_llamaforge_status(
        self,
        loading=None,
        tokens_up_active=None,
        inferencing=None,
        tokens_down_active=None,
        tokens_up=None,
        tokens_down=None,
    ):
        if loading is not None:
            self.llamaforge_status["loading"] = loading
        if tokens_up_active is not None:
            self.llamaforge_status["tokens_up_active"] = tokens_up_active
        if inferencing is not None:
            self.llamaforge_status["inferencing"] = inferencing
        if tokens_down_active is not None:
            self.llamaforge_status["tokens_down_active"] = tokens_down_active
        if tokens_up is not None:
            self.llamaforge_status["tokens_up"] = tokens_up
        if tokens_down is not None:
            self.llamaforge_status["tokens_down"] = tokens_down
        self._refresh_llamaforge_status()

    def _refresh_llamaforge_status(self):
        if not hasattr(self, "llamaforge_status_labels"):
            return

        status = self.llamaforge_status
        inactive = "gray45"
        self.llamaforge_status_labels["loading"].configure(
            text="Loading",
            text_color="#F0B429" if status["loading"] else inactive,
        )
        self.llamaforge_status_labels["tokens_up"].configure(
            text=f"Tokens up: {self._format_token_count(status['tokens_up'])}",
            text_color="#5Dadec" if status["tokens_up_active"] else inactive,
        )
        self.llamaforge_status_labels["inferencing"].configure(
            text="Inferencing",
            text_color="#48C774" if status["inferencing"] else inactive,
        )
        self.llamaforge_status_labels["tokens_down"].configure(
            text=f"Tokens down: {self._format_token_count(status['tokens_down'])}",
            text_color="#F7B267" if status["tokens_down_active"] else inactive,
        )

    def _start_llamaforge_request_status(self):
        self._set_llamaforge_status(
            loading=False,
            tokens_up_active=True,
            inferencing=False,
            tokens_down_active=False,
            tokens_up=0,
            tokens_down=0,
        )

    def _finish_llamaforge_request_status(self):
        self._set_llamaforge_status(
            loading=False,
            tokens_up_active=False,
            inferencing=False,
            tokens_down_active=False,
        )

    def toggle_all_thinking(self):
        state = self.global_show_thinking.get()
        for b_id in self.reasoning_states:
            self.reasoning_states[b_id] = state
            self.chat_display.tag_config(b_id, elide=not state)
            if b_id in self.reasoning_buttons and self.reasoning_buttons[b_id].winfo_exists():
                self.reasoning_buttons[b_id].configure(text="▼ Thinking" if state else "▶ Thinking")

    def toggle_single_reasoning(self, b_id):
        state = not self.reasoning_states.get(b_id, True)
        self.reasoning_states[b_id] = state
        self.chat_display.tag_config(b_id, elide=not state)
        if b_id in self.reasoning_buttons and self.reasoning_buttons[b_id].winfo_exists():
            self.reasoning_buttons[b_id].configure(text="▼ Thinking" if state else "▶ Thinking")

    def _ui_start_reasoning_block(self, b_id, speech_id=None):
        state = self.global_show_thinking.get()
        self.reasoning_states[b_id] = state
        
        btn = ctk.CTkButton(self.chat_display, text="▼ Thinking" if state else "▶ Thinking", 
                            width=80, height=20, fg_color="transparent", 
                            text_color="#6E6E6E", hover_color="#333333", anchor="w",
                            font=ctk.CTkFont(size=self.chat_font_size, slant="italic"), 
                            command=lambda id=b_id: self.toggle_single_reasoning(id))
        self.reasoning_buttons[b_id] = btn
        
        self.chat_display.configure(state="normal")
        self.chat_display._textbox.window_create("end", window=btn)
        if speech_id:
            self.chat_display.insert("end", " ")
            speech_btn = ctk.CTkButton(
                self.chat_display,
                text="🔊",
                width=28,
                height=20,
                fg_color="transparent",
                text_color="#8AB4F8",
                hover_color="#333333",
                font=ctk.CTkFont(size=max(12, self.chat_font_size)),
                command=lambda sid=speech_id: self.toggle_tts(sid)
            )
            self.tts_buttons[speech_id] = speech_btn
            self.chat_display._textbox.window_create("end", window=speech_btn)
        self.chat_display.insert("end", "\n")
        self.chat_display.tag_config(b_id, elide=not state)
        self.chat_display.configure(state="disabled")
        self._refresh_tts_buttons()

    def _ui_append_reasoning(self, b_id, text):
        self.append_to_display(text, tag=(b_id, "reasoning"))

    def _ui_end_reasoning_block(self, b_id):
        self.append_to_display("\n\n", tag=(b_id, "reasoning"))

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

    def manage_context(self, phase="context check", report_if_under_threshold=False):
        initial_metrics = self._context_metrics()
        initial_tokens = initial_metrics["tokens"]
        label = f"[Context compression: {phase}]"

        if initial_tokens <= self.compression_threshold:
            if report_if_under_threshold:
                print(
                    f"{label} no compression needed: "
                    f"{self._format_context_metrics(initial_metrics)}; "
                    f"window={self._format_context_window_pct(initial_tokens)} of "
                    f"{self._format_token_count(self.max_context_window)}, "
                    f"threshold={self._format_token_count(self.compression_threshold)}."
                )
            return

        print(
            f"{label} starting: {self._format_context_metrics(initial_metrics)}; "
            f"window={self._format_context_window_pct(initial_tokens)} of "
            f"{self._format_token_count(self.max_context_window)}, "
            f"threshold={self._format_token_count(self.compression_threshold)}."
        )

        step = 0
        while self._count_message_tokens(self.working_context) > self.compression_threshold:
            if len(self.working_context) <= 5:
                current_metrics = self._context_metrics()
                print(
                    f"{label} halted: {self._format_context_metrics(current_metrics)} "
                    "is over threshold, but there are too few messages to safely compress."
                )
                break

            step += 1
            system_prompt = self.working_context[0]
            recent_messages = self.working_context[-4:]
            middle_messages = self.working_context[1:-4]
            
            if not middle_messages:
                print(f"{label} halted: no older middle messages are available to compress.")
                break
                
            # Take a "bite" from the start of the middle messages (up to 5 at a time)
            chunk_size = min(5, len(middle_messages))
            chunk_to_compress = middle_messages[:chunk_size]
            remaining_middle = middle_messages[chunk_size:]
            before_step_metrics = self._context_metrics()
            chunk_metrics = self._context_metrics(chunk_to_compress)

            print(
                f"{label} step {step}: compressing {chunk_size} oldest middle messages "
                f"({self._format_context_metrics(chunk_metrics)}); "
                f"keeping {len(recent_messages)} recent messages unchanged."
            )
            
            summary_budget = self._compression_summary_budget(chunk_metrics["tokens"])
            print(
                f"{label} step {step}: requesting summary budget of "
                f"{self._format_token_count(summary_budget)} est. tokens."
            )
            summary = self.llm.summarize_messages(
                chunk_to_compress,
                max_tokens=summary_budget,
            )
            
            if summary == "Failed to summarize.":
                print(f"{label} step {step} failed: summarization failed; halting compression.")
                break
                
            new_summary_msg = {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
            self.working_context = [system_prompt, new_summary_msg] + remaining_middle + recent_messages
            after_step_metrics = self._context_metrics()
            if after_step_metrics["tokens"] >= before_step_metrics["tokens"]:
                tighter_budget = max(40, min(summary_budget, int(chunk_metrics["tokens"] * 0.25)))
                print(
                    f"{label} step {step}: model summary did not reduce context "
                    f"({self._format_token_count(before_step_metrics['tokens'])} -> "
                    f"{self._format_token_count(after_step_metrics['tokens'])}); "
                    f"trimming it to {self._format_token_count(tighter_budget)} est. tokens."
                )
                summary = self.llm._trim_text_to_token_budget(summary, tighter_budget)
                new_summary_msg = {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
                self.working_context = [system_prompt, new_summary_msg] + remaining_middle + recent_messages
                after_step_metrics = self._context_metrics()

                if after_step_metrics["tokens"] >= before_step_metrics["tokens"]:
                    self.working_context = [system_prompt] + middle_messages + recent_messages
                    print(
                        f"{label} step {step} halted: summary would make context larger, "
                        "so the original context was kept."
                    )
                    break

            print(
                f"{label} step {step} done: "
                f"{self._format_token_count(before_step_metrics['tokens'])} -> "
                f"{self._format_token_count(after_step_metrics['tokens'])} est. tokens; "
                f"{self._format_token_delta(before_step_metrics['tokens'], after_step_metrics['tokens'])}; "
                f"messages {before_step_metrics['messages']} -> {after_step_metrics['messages']}."
            )
            
            # --- DREAM STATE: Save the compressed summary to long-term memory ---
            print(f"{label} step {step}: consolidating summary into vector memory...")
            try:
                emb = self.llm.get_embedding(summary)
                if emb:
                    # Provide a dummy metadata dict that matches memory expectations
                    self.memory.add_to_vector_memory(
                        summary,
                        {"role": "system", "memory_type": "dream_summary", "id": -1},
                        emb
                    )
                    self.after(0, self.update_memory_count_display)
            except Exception as e:
                print(f"Failed to embed dream state summary: {e}")

        final_metrics = self._context_metrics()
        print(
            f"{label} complete: "
            f"{self._format_token_count(initial_tokens)} -> "
            f"{self._format_token_count(final_metrics['tokens'])} est. tokens; "
            f"{self._format_token_delta(initial_tokens, final_metrics['tokens'])}; "
            f"window={self._format_context_window_pct(final_metrics['tokens'])} of "
            f"{self._format_token_count(self.max_context_window)}, "
            f"messages {initial_metrics['messages']} -> {final_metrics['messages']}."
        )

    def _detect_loop(self, text):
        if not text or len(text) < 60:
            return False
        # Only check the last 1000 characters to keep it fast
        tail = text[-1000:]
        length = len(tail)
        # We look for a repeating pattern of length 15 to 200.
        # If it repeats 4 times consecutively at the end, it's an infinite loop.
        for pat_len in range(15, 200):
            if length >= pat_len * 4:
                pattern = tail[-pat_len:]
                if tail.endswith(pattern * 4):
                    return True
        return False

    def process_message(self, user_text, img_path, repost_existing=False):
        try:
            is_first_user_message = (
                self._user_message_count() <= 1
                if repost_existing else self._is_first_user_message()
            )
            first_turn_memory_context = (
                self._build_first_turn_memory_context() if is_first_user_message else ""
            )

            self.after(0, self._start_llamaforge_request_status)

            user_msg_id = None
            display_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if not repost_existing:
                user_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "user", user_text, img_path)
                
                self.after(0, lambda dt=display_time: self.append_to_display(f"[{dt}] You: ", "user_text"))
                self.after(0, lambda uid=user_msg_id: self._ui_insert_message_controls(uid))
                
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
                
                self.working_context.append({"role": "user", "content": content_for_llm})
                
                self.after(0, self.load_conversations_list) # Update sidebar timestamps

            # Vector embed the text part
            user_emb = self.llm.get_embedding(user_text)
            relevant_past = []
            relevant_notes = []
            if user_emb:
                # Search for past memories BEFORE adding the current message to avoid echoing the prompt
                relevant_past = self.memory.search_vector_memory(user_emb, n_results=5)
                relevant_notes = self.memory.search_notes(user_emb, n_results=3)
                
                # Now add the current message to the memory if it's long enough
                if not repost_existing and len(user_text.split()) > 3:
                    self.memory.add_to_vector_memory(
                        user_text,
                        {"id": user_msg_id, "role": "user", "memory_type": "message"},
                        user_emb
                    )
                    self.after(0, self.update_memory_count_display)

            self.manage_context(phase="pre-response safety check")
            
            max_attempts = 1
            attempt = 0
            
            ai_msg_id = None
            
            while attempt < max_attempts:
                attempt += 1
                query_context = list(self.working_context)

                query_context = self._append_query_system_context(
                    query_context,
                    "Current date and time",
                    self._current_time_context()
                )
                query_context = self._append_query_system_context(
                    query_context,
                    "User's current location",
                    self._location_context()
                )
                query_context = self._append_query_system_context(
                    query_context,
                    "Your notepad (always visible to you)",
                    self._load_notepad_context()
                )
                query_context = self._append_query_system_context(
                    query_context,
                    "Session startup memory",
                    first_turn_memory_context
                )
                query_context = self._append_query_system_context(
                    query_context,
                    "Relevant past memories for this query",
                    "\n".join(relevant_past)
                )
                query_context = self._append_query_system_context(
                    query_context,
                    "Relevant notes you saved earlier",
                    "\n".join(relevant_notes)
                )

                if attempt > 1:
                    query_context.append({
                        "role": "system",
                        "content": "SYSTEM WARNING: Your previous attempt got stuck in hidden reasoning or repetition. Do not plan, announce intentions, or restate the task. Give the direct answer now in a compact reply."
                    })
                    print("[Loop Detected! Restarting Generation...]")
                    
                if ai_msg_id is None:
                    ai_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "assistant", "", None)
                
                def start_agent_msg(a_msg_id, is_recovery=False):
                    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
                    self.append_to_display(f"[{dt}] Agent: ", "agent_text")
                    self._ui_insert_message_controls(a_msg_id)
                    if is_recovery:
                        self.append_to_display("[Recovering from loop...]\n")
                
                self.after(0, lambda a_id=ai_msg_id, recovering=attempt > 1: start_agent_msg(a_id, recovering))
                
                query_context = self._sanitize_messages(query_context)
                prompt_tokens = self._count_message_tokens(query_context)
                self.after(
                    0,
                    lambda count=prompt_tokens: self._set_llamaforge_status(
                        loading=True,
                        tokens_up_active=True,
                        inferencing=False,
                        tokens_down_active=False,
                        tokens_up=count,
                        tokens_down=0,
                    )
                )
                print(
                    "[Response budget] "
                    f"completion cap={self._format_token_count(self.assistant_response_max_tokens)} tokens; "
                    f"visible cap={self._format_token_count(self.assistant_visible_token_limit)} tokens; "
                    f"silent reasoning cap="
                    f"{self._format_token_count(self.assistant_reasoning_token_limit) if self.assistant_reasoning_token_limit else 'disabled'}."
                )
                response_stream = self.llm.get_chat_response(
                    query_context,
                    stream=True,
                    max_tokens=self.assistant_response_max_tokens,
                )
                self.after(0, lambda: self._set_llamaforge_status(tokens_up_active=False))
                
                full_response = ""
                clean_response = ""
                full_reasoning = ""
                in_reasoning = False
                current_block_id = None
                reasoning_speech_id = None
                final_speech_id = self._create_tts_payload("")
                final_button_inserted = False
                stream_started = False
                loop_detected = False
                response_limited = False
                reasoning_budget_exceeded = False
                
                if response_stream:
                    self.cancel_inference_flag = False
                    for chunk in response_stream:
                        if self.cancel_inference_flag:
                            self.after(0, lambda: self.append_to_display("\n[Inference Cancelled]\n"))
                            break
                        
                        if not stream_started:
                            stream_started = True
                            self.after(
                                0,
                                lambda: self._set_llamaforge_status(
                                    loading=False,
                                    inferencing=True,
                                    tokens_down_active=True,
                                )
                            )

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
                                self.reasoning_counter += 1
                                current_block_id = f"reason_{self.reasoning_counter}"
                                reasoning_speech_id = self._create_tts_payload("")
                                self.after(
                                    0,
                                    lambda b=current_block_id, sid=reasoning_speech_id: self._ui_start_reasoning_block(b, sid)
                                )
                                
                            full_reasoning += reasoning
                            if reasoning_speech_id:
                                self._set_tts_payload(reasoning_speech_id, full_reasoning)
                            self.after(0, lambda b=current_block_id, t=reasoning: self._ui_append_reasoning(b, t))

                            reasoning_tokens = self.llm.count_tokens(full_reasoning)
                            if (
                                self.assistant_reasoning_token_limit
                                and not clean_response
                                and reasoning_tokens > self.assistant_reasoning_token_limit
                            ):
                                reasoning_budget_exceeded = True
                                loop_detected = True
                                print(
                                    "[Response budget] Silent reasoning exceeded "
                                    f"{self._format_token_count(self.assistant_reasoning_token_limit)} tokens; stopping."
                                )
                                self._close_response_stream(response_stream)
                                break

                        if content:
                            if in_reasoning:
                                in_reasoning = False
                                self.after(0, lambda b=current_block_id: self._ui_end_reasoning_block(b))
                                
                            if not final_button_inserted:
                                final_button_inserted = True
                                self.after(0, lambda sid=final_speech_id: self._ui_insert_tts_button(sid))
                                self.after(0, lambda: self.append_to_display(" "))

                            full_response += content
                            previous_clean_response = clean_response
                            limited_response, response_limited = self._limit_visible_reply_to_budget(
                                clean_response + content
                            )
                            if len(limited_response) < len(previous_clean_response):
                                limited_response = previous_clean_response

                            visible_content = limited_response[len(previous_clean_response):]
                            clean_response = limited_response
                            self._set_tts_payload(final_speech_id, clean_response)
                            if visible_content:
                                self.after(0, lambda text=visible_content: self.append_to_display(text))
                            if response_limited:
                                print(
                                    "[Response budget] Visible reply capped at "
                                    f"{self._format_token_count(self.assistant_visible_token_limit)} tokens."
                                )
                                self._close_response_stream(response_stream)
                                break

                        if reasoning or content:
                            output_tokens = self.llm.count_tokens(full_reasoning) + self.llm.count_tokens(full_response)
                            self.after(
                                0,
                                lambda count=output_tokens: self._set_llamaforge_status(
                                    inferencing=True,
                                    tokens_down_active=True,
                                    tokens_down=count,
                                )
                            )
                        
                        if self._detect_loop(full_reasoning) or self._detect_loop(full_response):
                            loop_detected = True
                            print("[Loop Detected! Stopping current generation.]")
                            self._close_response_stream(response_stream)
                            break
                else:
                    self.after(0, lambda: self.append_to_display("[Error: response_stream is None. Connection failed.]"))

                if loop_detected:
                    if in_reasoning:
                        self.after(0, lambda b=current_block_id: self._ui_end_reasoning_block(b))
                    if attempt < max_attempts:
                        continue
                    else:
                        full_reasoning = ""
                        if reasoning_budget_exceeded:
                            print("[Response budget] Hidden reasoning overrun; stopped generation.")
                            if not clean_response:
                                clean_response = (
                                    "I got stuck in hidden thinking, so I stopped this response before it tied up the chat."
                                )
                                self.after(0, lambda text=clean_response: self.append_to_display(text))
                        else:
                            print("[Loop Detected!] Stopped current generation.")
                            if not clean_response:
                                clean_response = (
                                    "I got caught in a repetition loop, so I stopped that response."
                                )
                                self.after(0, lambda text=clean_response: self.append_to_display(text))

                break

            if full_reasoning and not clean_response:
                print("[Response budget] Model produced hidden reasoning but no visible answer.")
                full_reasoning = ""
                clean_response = (
                    "I started thinking through that without landing the answer. "
                    "Ask me again and I will answer directly."
                )
                self.after(0, lambda text=clean_response: self.append_to_display(text))

            self.after(0, self._finish_llamaforge_request_status)
            self.after(0, lambda: self.append_to_display("\n\n"))
            if full_reasoning and reasoning_speech_id:
                self.after(0, lambda sid=reasoning_speech_id, text=full_reasoning: self._set_tts_payload(sid, text))
            if clean_response:
                self.after(0, lambda sid=final_speech_id, text=clean_response: self._finalize_final_tts(sid, text))

            if clean_response or full_reasoning:
                db_content = clean_response
                if full_reasoning:
                    db_content = f"<think>\n{full_reasoning}\n</think>\n{clean_response}"

                if ai_msg_id is not None:
                    self.memory.update_message(ai_msg_id, db_content)
                else:
                    ai_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "assistant", db_content, None)
                    
                self.working_context.append({"role": "assistant", "content": db_content})
                ai_emb = self.llm.get_embedding(db_content)
                if ai_emb:
                    self.memory.add_to_vector_memory(
                        db_content,
                        {"id": ai_msg_id, "role": "assistant", "memory_type": "message"},
                        ai_emb
                    )
                    self.after(0, self.update_memory_count_display)

            # Compress the context AFTER the agent responds to avoid delaying the UI
            self.manage_context(phase="post-response", report_if_under_threshold=True)
            
            # Save the agent's (potentially newly compressed) state to disk
            self._save_agent_context()

            # Trigger background memory agent
            self._trigger_background_memory_agent()

        except Exception as e:
            error_msg = f"\n[CRITICAL THREAD ERROR]: {str(e)}\n\n"
            self.after(0, lambda: self.append_to_display(error_msg))
            print(f"Exception in process_message: {e}")
            
        finally:
            self.after(0, self._finish_llamaforge_request_status)
            self.after(0, self._finish_message_processing)

    def _save_agent_context(self):
        if not self.current_conversation_id:
            return
        path = os.path.join(self.data_dir, f"agent_context_{self.current_conversation_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.working_context, f)
        except Exception as e:
            print(f"Failed to save agent context: {e}")

    def _trigger_background_memory_agent(self):
        # We need the last user message and the agent's final clean response
        recent = self.working_context[-2:] if len(self.working_context) >= 2 else []
        if not recent or recent[-1]["role"] != "assistant":
            return
        
        # Make a copy of the content to safely pass to the thread
        conversation_slice = list(recent)
        
        threading.Thread(
            target=self._background_memory_task,
            args=(conversation_slice,),
            daemon=True
        ).start()

    def _background_memory_task(self, conversation_slice):
        try:
            # Build a localized context for the memory agent
            sys_prompt = (
                "You are an invisible background memory manager. Your sole job is to read the latest exchange "
                "and update the user's notepad and location. "
                "You MUST use EXACTLY DOUBLE BRACKETS for your commands. Example: [[NOTE_DONE: task]]\n"
                "Available Commands:\n"
                "[[LOCATION: place]] - Record where the user currently is.\n"
                "[[NOTE: text]] - Add a short-lived reminder to your notepad.\n"
                "[[NOTE_DONE: text]] - Remove a handled reminder from your notepad (match the text closely).\n"
                "[[REMEMBER: text]] - Save a durable fact to long-term memory.\n"
                "Do not write conversational text. Just output the necessary commands, or nothing if no updates are needed."
            )
            
            # Reconstruct the current notes so the agent knows what to delete
            current_notes = self._load_notepad_context()
            if current_notes:
                sys_prompt += f"\n\nCURRENT NOTEPAD:\n{current_notes}"
                
            query_context = [{"role": "system", "content": sys_prompt}]
            query_context.extend(conversation_slice)

            # Let's count tokens strictly to ensure we don't blow up
            prompt_tokens = self._count_message_tokens(query_context)
            # Run inference synchronously in this background thread
            response_stream = self.llm.get_chat_response(query_context, stream=True, max_tokens=200)
            
            if not response_stream:
                return
                
            full_response = ""
            full_reasoning = ""
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(getattr(delta, "model_extra", {}), "reasoning_content", "") or ""
                content = getattr(delta, "content", None) or ""
                full_reasoning += reasoning
                full_response += content
                
            combined_text = full_response + "\n" + full_reasoning
            print(f"\n--- Background Agent Output ---\n{combined_text.strip()}\n-------------------------------")
            
            # The background agent writes raw commands. Parse them!
            _, agent_cmds = parse_commands(combined_text)
            if agent_cmds:
                print(f"Executing Background Commands: {agent_cmds}")
                self._apply_agent_commands(agent_cmds)
                
        except Exception as e:
            print(f"Background memory agent failed: {e}")

    def _update_queue_status(self):
        queued_count = len(self.message_queue)
        if self.agent_busy:
            self.send_button.configure(text=f"Queue ({queued_count})" if queued_count else "Queue")
            self.queue_status_label.configure(
                text=f"{queued_count} queued" if queued_count else "Agent busy"
            )
        else:
            self.send_button.configure(text="Send")
            self.queue_status_label.configure(text="")
        self._update_repost_button_state()

    def _get_repost_candidate(self):
        if not self.current_conversation_id:
            return None
        latest_message = self.memory.get_last_message(self.current_conversation_id)
        if latest_message and latest_message.get("role") == "user":
            return latest_message
        return None

    def _update_repost_button_state(self):
        if not hasattr(self, "repost_button"):
            return
        state = (
            "normal"
            if not self.agent_busy and self._get_repost_candidate()
            else "disabled"
        )
        self.repost_button.configure(state=state)

    def _start_message_processing(self, text, img, repost_existing=False):
        self.agent_busy = True
        self.chat_follow_output = self._is_chat_near_bottom()
        self._update_queue_status()
        threading.Thread(
            target=self.process_message,
            args=(text, img, repost_existing),
            daemon=True
        ).start()

    def _finish_message_processing(self):
        if self.message_queue:
            text, img = self.message_queue.pop(0)
            self._update_queue_status()
            self._start_message_processing(text, img)
            return

        self.agent_busy = False
        self._update_queue_status()
        self.entry.focus()

    def repost_last_message(self):
        if self.agent_busy:
            return

        candidate = self._get_repost_candidate()
        if not candidate:
            self._update_repost_button_state()
            return

        text = (candidate.get("content") or "").strip()
        img = candidate.get("image_path")
        if img and not os.path.exists(img):
            messagebox.showwarning(
                "Repost Last",
                "The last message had an image attachment, but the image file is missing."
            )
            return

        if not text and not img:
            return

        self._start_message_processing(text, img, repost_existing=True)

    def send_message(self):
        text = self.entry.get().strip()
        img = self.attached_image_path
        
        # At least one must be present
        if not text and not img:
            return

        self.entry.delete(0, "end")
        self.attached_image_path = None
        self.image_preview_label.configure(text="")

        if self.agent_busy:
            self.message_queue.append((text, img))
            self._update_queue_status()
            self.entry.focus()
            return

        self._start_message_processing(text, img)

if __name__ == "__main__":
    app = AgentApp()
    app.mainloop()

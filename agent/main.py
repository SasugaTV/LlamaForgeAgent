import customtkinter as ctk
import threading
import os
import shutil
import subprocess
import base64
import tkinter as tk
import re
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

from memory import AgentMemory
from llm_client import LlamaForgeClient

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

    def update_memory_count_display(self):
        count = self.memory.get_total_memories()
        self.memory_count_label.configure(text=f"Memories: {count}")

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

        self.memory_count_label = ctk.CTkLabel(self.sidebar_frame, text="Memories: 0", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray50")
        self.memory_count_label.grid(row=3, column=0, pady=(0, 10))

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

        self.thinking_switch = ctk.CTkSwitch(self.action_frame, text="Show Thinking", variable=self.global_show_thinking, command=self.toggle_all_thinking)
        self.thinking_switch.pack(side="right", padx=10)

        self.zoom_out_btn = ctk.CTkButton(self.action_frame, text="A-", width=40, command=self.decrease_font_size)
        self.zoom_out_btn.pack(side="right", padx=5)

        self.zoom_in_btn = ctk.CTkButton(self.action_frame, text="A+", width=40, command=self.increase_font_size)
        self.zoom_in_btn.pack(side="right", padx=5)

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
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.attach_btn = ctk.CTkButton(self.input_frame, text="📎 Image", width=60, command=self.attach_image)
        self.attach_btn.grid(row=0, column=0, padx=(0, 10))

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...", font=("Segoe UI", self.chat_font_size))
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.send_message)
        self.send_button.grid(row=0, column=2)

        self.scroll_bottom_btn = ctk.CTkButton(self.input_frame, text="↓ Bottom", width=60, command=self.scroll_to_bottom)
        self.scroll_bottom_btn.grid(row=0, column=3, padx=(10, 0))

        self.image_preview_label = ctk.CTkLabel(self.input_frame, text="", text_color="green")
        self.image_preview_label.grid(row=1, column=1, sticky="w", pady=(5,0))

        self.queue_status_label = ctk.CTkLabel(self.input_frame, text="", text_color="gray50")
        self.queue_status_label.grid(row=1, column=2, columnspan=2, sticky="e", pady=(5,0))

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
        
        # Rebuild working context and display
        for msg in history:
            role = msg["role"]
            text_content = msg["content"]
            img_path = msg.get("image_path")
            
            tag = "user_text" if role == "user" else "agent_text"
            role_label = "You" if role == "user" else "Agent"
            
            if img_path and os.path.exists(img_path):
                self.append_to_display(f"{role_label}: ", tag)
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
                self.append_to_display(f"{role_label}: ", tag)
                
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
                self.working_context.append({"role": role, "content": msg["content"]})
                
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

    def _count_message_tokens(self, messages):
        total = 0
        for message in messages:
            total += self.llm.count_tokens(message.get("role", ""))
            total += self.llm.count_tokens(message.get("content", ""))
            total += 4
        return total

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
            self.after(0, self._start_llamaforge_request_status)
            self.after(0, lambda: self.append_to_display("You: ", "user_text"))
            
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
            query_context = list(self.working_context)
            if user_emb:
                # Search for past memories BEFORE adding the current message to avoid echoing the prompt
                relevant_past = self.memory.search_vector_memory(user_emb, n_results=5)
                
                # Now add the current message to the memory if it's long enough
                if len(user_text.split()) > 3:
                    self.memory.add_to_vector_memory(user_text, {"id": user_msg_id, "role": "user"}, user_emb)
                    self.after(0, self.update_memory_count_display)
                
                if relevant_past and len(query_context) > 0 and query_context[0]["role"] == "system":
                    memory_string = "\n".join(relevant_past)
                    # Safely append to the initial system prompt without mutating the persistent working_context
                    sys_msg = dict(query_context[0])
                    sys_msg["content"] += f"\n\n[Relevant past memories for this query:\n{memory_string}]"
                    query_context[0] = sys_msg

            self.manage_context()

            def start_agent_msg():
                self.append_to_display("Agent: ", "agent_text")
                
            self.after(0, start_agent_msg)
            
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
            response_stream = self.llm.get_chat_response(query_context, stream=True)
            self.after(0, lambda: self._set_llamaforge_status(tokens_up_active=False))
            
            full_response = ""
            full_reasoning = ""
            in_reasoning = False
            current_block_id = None
            reasoning_speech_id = None
            final_speech_id = self._create_tts_payload("")
            final_button_inserted = False
            stream_started = False
            
            if response_stream:
                for chunk in response_stream:
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

                    if content:
                        if in_reasoning:
                            in_reasoning = False
                            self.after(0, lambda b=current_block_id: self._ui_end_reasoning_block(b))
                            
                        if not final_button_inserted:
                            final_button_inserted = True
                            self.after(0, lambda sid=final_speech_id: self._ui_insert_tts_button(sid))
                            self.after(0, lambda: self.append_to_display(" "))

                        full_response += content
                        self._set_tts_payload(final_speech_id, full_response)
                        self.after(0, lambda text=content: self.append_to_display(text))

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
            else:
                self.after(0, lambda: self.append_to_display("[Error: response_stream is None. Connection failed.]"))

            self.after(0, self._finish_llamaforge_request_status)
            self.after(0, lambda: self.append_to_display("\n\n"))
            if full_reasoning and reasoning_speech_id:
                self.after(0, lambda sid=reasoning_speech_id, text=full_reasoning: self._set_tts_payload(sid, text))
            if full_response:
                self.after(0, lambda sid=final_speech_id, text=full_response: self._finalize_final_tts(sid, text))

            if full_response or full_reasoning:
                db_content = full_response
                if full_reasoning:
                    db_content = f"<think>\n{full_reasoning}\n</think>\n{full_response}"
                    
                ai_msg_id = self.memory.add_message_to_sqlite(self.current_conversation_id, "assistant", db_content, None)
                self.working_context.append({"role": "assistant", "content": db_content})
                ai_emb = self.llm.get_embedding(db_content)
                if ai_emb:
                    self.memory.add_to_vector_memory(db_content, {"id": ai_msg_id, "role": "assistant"}, ai_emb)
                    self.after(0, self.update_memory_count_display)

        except Exception as e:
            error_msg = f"\n[CRITICAL THREAD ERROR]: {str(e)}\n\n"
            self.after(0, lambda: self.append_to_display(error_msg))
            print(f"Exception in process_message: {e}")
            
        finally:
            self.after(0, self._finish_llamaforge_request_status)
            self.after(0, self._finish_message_processing)

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

    def _start_message_processing(self, text, img):
        self.agent_busy = True
        self.chat_follow_output = self._is_chat_near_bottom()
        self._update_queue_status()
        threading.Thread(target=self.process_message, args=(text, img), daemon=True).start()

    def _finish_message_processing(self):
        if self.message_queue:
            text, img = self.message_queue.pop(0)
            self._update_queue_status()
            self._start_message_processing(text, img)
            return

        self.agent_busy = False
        self._update_queue_status()
        self.entry.focus()

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

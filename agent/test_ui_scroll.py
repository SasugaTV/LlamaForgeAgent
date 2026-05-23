import customtkinter as ctk

app = ctk.CTk()
app.geometry("400x300")

txt = ctk.CTkTextbox(app)
txt.pack(fill="both", expand=True)

for i in range(100):
    txt.insert("end", f"Line {i}\n")
txt.see("end")

def stream_text():
    for i in range(50):
        def append_chunk(chunk=str(i)):
            # Without update_idletasks() first to see if it breaks
            yview = txt._textbox.yview()
            is_at_bottom = yview[1] >= 0.99
            print(f"[{chunk}] yview={yview[1]:.4f} is_at_bottom={is_at_bottom}")
            txt.insert("end", f" Chunk {chunk}\n")
            if is_at_bottom:
                txt.see("end")
        
        app.after(0, append_chunk)

app.after(500, stream_text)
app.after(1000, app.destroy)
app.mainloop()

import customtkinter as ctk

app = ctk.CTk()
app.geometry("400x300")

txt = ctk.CTkTextbox(app)
txt.pack(fill="both", expand=True)

for i in range(1000):
    txt.insert("end", f"Line {i}\n")
txt.see("end")

def check_yview():
    txt._textbox.update_idletasks()
    yview = txt._textbox.yview()
    bbox = txt._textbox.bbox("end-1c")
    print(f"At bottom: yview={yview[1]:.6f}, bbox={bbox is not None}")
    
    # Scroll up 5 lines
    txt._textbox.yview_scroll(-5, "units")
    txt._textbox.update_idletasks()
    yview2 = txt._textbox.yview()
    bbox2 = txt._textbox.bbox("end-1c")
    print(f"Scrolled up 5 lines: yview={yview2[1]:.6f}, bbox={bbox2 is not None}")
    
    app.destroy()

app.after(500, check_yview)
app.mainloop()
import customtkinter as ctk

app = ctk.CTk()
app.geometry("400x300")

txt = ctk.CTkTextbox(app)
txt.pack(fill="both", expand=True)

for i in range(100):
    txt.insert("end", f"Line {i}\n")
txt.see("end")

def test_bbox():
    app.update_idletasks()
    print("At bottom, bbox('end-1c'):", txt._textbox.bbox("end-1c"))
    print("At bottom, bbox('end-2c'):", txt._textbox.bbox("end-2c"))
    
    txt._textbox.yview_moveto(0.0)
    app.update_idletasks()
    print("At top, bbox('end-1c'):", txt._textbox.bbox("end-1c"))

app.after(500, test_bbox)
app.after(1000, app.destroy)
app.mainloop()

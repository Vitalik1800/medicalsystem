import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from model import predict_with_cam

def run_app():
    root = tk.Tk()
    root.title("Medical CV System")
    root.geometry("500x500")

    root.configure(bg="#f0f0f0")

    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path).convert("RGB")

            result, confidence, cam_image = predict_with_cam(img)

            cam_image = Image.fromarray(cam_image)

            img_tk = ImageTk.PhotoImage(cam_image)
            panel.config(image=img_tk)
            panel.image = img_tk

            result_label.config(
                text=f"Діагноз: {result}\nЙмовірність: {confidence*100:.1f}%"
            )

            if result == "Норма":
                result_label.config(fg="green")
            else:
                result_label.config(fg="red")

    title = tk.Label(root, text="Аналіз медичних зображень", font=("Arial", 16), bg="#f0f0f0")
    title.pack(pady=10)

    btn = tk.Button(root, text="Завантажити зображення", command=open_file, bg="#4CAF50", fg="white")
    btn.pack(pady=10)

    panel = tk.Label(root, bg="#f0f0f0")
    panel.pack()

    result_label = tk.Label(root, text="Результат: -", bg="#f0f0f0")
    result_label.pack(pady=10)

    root.mainloop()
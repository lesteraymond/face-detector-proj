import tkinter as tk
import cv2
import capture_dataset
import train_model
import os
from PIL import ImageTk, Image
from tkinter import messagebox
from random import randint


class MainFrame:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.cap = capture_dataset.Capture()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.root = tk.Tk()
        self.root.title("FACE DETECTOR")
        self.root.geometry("800x500")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<KeyPress>", self.key_press_hander)

        self.video_frame = tk.Label(self.root, height=350)
        self.video_frame.pack(padx=10, pady=10)

        self.show_rec_value = tk.IntVar()

        self.cb_show_rec = tk.Checkbutton(
            self.root,
            text="SHOW RECTANGLE",
            font=("Arial", 10),
            variable=self.show_rec_value,
        )
        self.cb_show_rec.pack(pady=5)

        self.capture_button = tk.Button(
            self.root,
            font=("Arial", 10),
            text="CAPTURE",
            command=self.capture_button_click,
        )
        self.capture_button.place(x=100, y=430, width=100)

        self.train_button = tk.Button(
            self.root, font=("Arial", 10), text="TRAIN", command=self.train_button_click
        )
        self.train_button.place(x=350, y=430, width=100)

        self.detect_button = tk.Button(self.root, font=("Arial", 10), text="DETECT")
        self.detect_button.place(x=600, y=430, width=100)

        #######
        self.capture_button_shortcut = tk.Label(
            self.root,
            font=("Arial", 10),
            text="Key: <c>",
        )
        self.capture_button_shortcut.place(x=100, y=460, width=100)

        self.train_button_shortcut = tk.Label(
            self.root,
            font=("Arial", 10),
            text="Key: <t>",
        )
        self.train_button_shortcut.place(x=350, y=460, width=100)

        self.detect_button_shortcut = tk.Label(
            self.root,
            font=("Arial", 10),
            text="Key: <d>",
        )
        self.detect_button_shortcut.place(x=600, y=460, width=100)

        # self.train_button = tk.Button(self.root, font=("Arial", 10), text="TRAIN")
        # self.train_button.place(x=350, y=430, width=100)

        # self.detect_button = tk.Button(self.root, font=("Arial", 10), text="DETECT")
        # self.detect_button.place(x=600, y=430, width=100)

        self.update_video_frame()

    def update_video_frame(self):
        r, self.f = self.video_capture.read()
        self.f = cv2.flip(self.f, 1)

        gray = cv2.cvtColor(self.f, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        rec_pos_x = 0
        rec_pos_y = 0
        if len(self.faces) > 0:
            status_text = "Face Detected"
            self.no_face_detected = False
            for x, y, w, h in self.faces:
                rec_pos_x = x
                rec_pos_y = y
                if self.show_rec_value.get() == 1:
                    cv2.rectangle(self.f, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # print(self.show_rec_value)
        else:
            status_text = "No Face Detected"
            self.no_face_detected = True

        if self.show_rec_value.get() == 1:
            cv2.putText(
                self.f,
                status_text,
                (rec_pos_x - 8, rec_pos_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        self.rgb = cv2.cvtColor(self.f, cv2.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(Image.fromarray(self.rgb))
        self.video_frame.img = self.img
        self.video_frame.config(image=self.img)
        self.root.after(10, self.update_video_frame)

    def capture_button_click(self):
        if self.no_face_detected:
            error_messages = [
                "Walang face na nakita. Paki try ulit.",
                "Hindi ka na detect ng camera. Ayusin mo lang pwesto mo.",
                "No face found. Check mo lighting or distance.",
                "hindi ka makita. Try mo lumapit konti.",
                "Camera cannot detect any face. Please try again.",
                "No face detected. Baka natakpan yung camera.",
                "Wala pa ring face. Ayusin mo lang angle mo.",
                "Face not detected. Try mo i-adjust yung position mo.",
                "Hindi nag register ang face mo. Try mo ulit.",
                "No face detected. Make sure nasa frame ka.",
            ]
            messagebox.showerror(
                title="ERROR",
                message=error_messages[randint(0, len(error_messages))],
            )
        else:
            # self.flush_image()
            self.show_image_preview_window()

    def train_button_click(self):
        files = os.listdir("dataset/")

        if len(files) == 0:
            messagebox.showerror("ERROR", "Walang laman ung dataset folder huhuhu")
            return

        if len(files) < 10:
            error_messages = [
                "Kulang pictures. Kailangan 10 o more.",
                "Medyo konti pa images mo, dagdagan mo muna.",
                "Dataset too small. Collect at least 10 pics.",
                "Hindi pedeng magtrain, kulang images.",
                "Add more photos muna bago magtrain.",
                "Kulang pa yung pictures. Try mo magcapture ng madami.",
                "Training failed. Need 10 or more images.",
                "Konti pa yung images. Dagdagan mo bago magtrain.",
                "Cannot proceed. Kulang pictures mo.",
                "Add more pictures. Minimum 10 required para magtrain.",
            ]
            messagebox.showerror(
                "ERROR", error_messages[randint(0, len(error_messages))]
            )
        else:
            train = train_model.Train()
            train.save_model()

    def flush_image(self):
        r, f = self.video_capture.read()
        f = cv2.flip(f, 1)

        self.cap.set_frame(f)
        self.cap.set_faces(self.faces)
        self.cap.write()
        self.preview_window.destroy()

    def show_image_preview_window(self):
        r, f = self.video_capture.read()
        f = cv2.flip(f, 1)

        x, y, w, h = self.faces[0]
        face = f[y : y + h, x : x + w]

        resize = cv2.resize(face, (0, 0), fx=1.5, fy=1.5)
        rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.preview_window = tk.Toplevel()
        self.preview_window.title("Preview")
        # self.preview_window.geometry("500x500")
        # self.preview_window.protocol("<KeyPress>", self.key_press_hander_preview_window)

        self.preview_image_container = tk.Label(self.preview_window)
        self.preview_image_container.img = img
        self.preview_image_container.config(image=img)
        self.preview_image_container.pack(pady=0)

        self.save_button = tk.Button(
            self.preview_window,
            text="Save",
            width=20,
            font=("Arial", 10),
            command=self.flush_image,
        )
        self.save_button.pack(side="left", padx=10, pady=0)

        self.capture_again_button = tk.Button(
            self.preview_window,
            text="Capture Again",
            width=20,
            font=("Arial", 10),
            command=self.preview_window.destroy,
        )
        self.capture_again_button.pack(side="right", padx=10, pady=0)

        # print(self.preview_window.winfo_height())

    def on_close(self):
        ask = messagebox.askyesno(title="Quit?", message="Do you want to quit?")
        if ask:
            self.root.destroy()

    def key_press_hander(self, event):
        char = event.char
        keysym = event.keysym
        keycode = event.keycode

        print(f"Keysym: {keysym} - Keycode: {keycode}")
        print(self.show_rec_value.get())
        # print(int(ord("q")))

        if keysym == "q":
            self.on_close()
        elif keysym == "c":
            self.capture_button_click()

    # def key_press_hander_preview_window(self, event):
    #     char = event.char
    #     keysym = event.keysym
    #     keycode = event.keycode

    #     print(f"Keysym: {keysym} - Keycode: {keycode}")
    #     print(self.show_rec_value.get())
    #     print("HANNI PHAM")

    #     if keysym == "Escape":
    #         self.preview_window.destroy()

    def show(self):
        self.root.mainloop()
        self.video_capture.release()


frame = MainFrame()
frame.show()

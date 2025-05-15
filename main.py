import cv2
import pygame
import time
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load sounds
sound_left = pygame.mixer.Sound("sounds/left.wav")
sound_center = pygame.mixer.Sound("sounds/center.wav")
sound_right = pygame.mixer.Sound("sounds/right.wav")

# Load YOLO model
model = YOLO("yolov8n.pt")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Directional Audio")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Video display label
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Status label
        self.status_label = tk.Label(root, text="Status: Stopped", font=("Arial", 14))
        self.status_label.pack(pady=5)

        # Start/Stop buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.cap = None
        self.running = False
        self.last_play_time = 0
        self.cooldown = 1.0  # seconds

    def start_detection(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        self.running = True
        self.status_label.config(text="Status: Detecting...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.detect_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.status_label.config(text="Status: Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        # Clear the video display
        self.video_label.config(image='')

    def detect_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]

            results = model(frame)[0]

            now = time.time()
            played_sound = False

            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_center = (x1 + x2) // 2

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if now - self.last_play_time > self.cooldown:
                    if obj_center < frame_width // 3:
                        sound_left.play()
                        played_sound = True
                    elif obj_center > 2 * frame_width // 3:
                        sound_right.play()
                        played_sound = True
                    else:
                        sound_center.play()
                        played_sound = True

            if played_sound:
                self.last_play_time = now

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update video label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.stop_detection()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

# See-Hear 🔊👁️

A real-time video object detection system that **speaks what it sees**. Powered by YOLOv8, this project uses your webcam to detect objects and plays an audio description for each detected item.

## How It Works
- Captures video using OpenCV
- Runs YOLOv8 detection on each frame
- Plays audio labels for detected objects using Pygame

## Requirements
- Python 3.10+
- OpenCV
- Pygame
- PIL (Pillow)
- Ultralytics YOLOv8

## Run It
```bash
python3 main.py

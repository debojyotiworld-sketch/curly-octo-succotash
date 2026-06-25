# Real-Time Human Detection and Tracking 🎯

A real-time dual-camera web application that utilizes advanced computer vision to detect and track human subjects. Built with **Flask**, **OpenCV**, and **PyTorch** using the state-of-the-art **YOLOv5** object detection model. 

## ✨ Features
* **Real-Time Detection:** Live streaming and processing of video feeds with sub-second latency.
* **YOLOv5 Integration:** Leverages the `yolov5s` model for high-speed, accurate human detection.
* **Dual-Camera Support:** Simultaneously captures, processes, and streams video from two independent webcams.
* **Smart Hardware Routing:** Automatically detects and utilizes a CUDA-enabled GPU if available, otherwise falls back to CPU.
* **Interactive Dashboard:** A responsive, dark-themed web interface displaying live feeds with dynamic, animated status borders.

## 🛠️ Tech Stack
* **Backend Framework:** Python, Flask
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning/AI:** PyTorch, Ultralytics YOLOv5
* **Frontend:** HTML5, CSS3

## 📁 Repository Structure
```text
.
├── detection.py           # Main application script (Flask + YOLOv5 logic)
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation

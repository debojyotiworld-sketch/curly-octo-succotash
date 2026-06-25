from flask import Flask, Response, render_template
import cv2
import torch
from PIL import Image

# Load the YOLOv5 model for object detection.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Set the device to GPU if available, otherwise use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize the Flask application.
app = Flask(__name__)

def detect_real_time(cam_index):
    """
    Capture video from a specified camera and perform real-time object detection.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Camera index {cam_index} could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        results = model(image)

        for detection in results.pred[0]:
            if detection[-1] == 0:  # Check if the detected object is a person
                box = detection[:4].cpu().numpy().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, 'Human', (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video1')
def video_feed1():
    return Response(detect_real_time(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video_feed2():
    return Response(detect_real_time(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Route for the main page. Load the HTML from templates folder.
    """
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='192.168.0.102', port=5000, debug=True)

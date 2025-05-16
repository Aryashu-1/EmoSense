from flask import Flask, render_template, Response, jsonify
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from resnet_emotion import ResNetEmotion
from collections import Counter

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emotion_counter = Counter()

# Load Model
model = ResNetEmotion(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))
model.eval()

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Image Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            face_img = Image.fromarray(roi)
            input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                emotion = EMOTIONS[pred.item()]
                emotion_counter[emotion] += 1

            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_stats')
def emotion_stats():
    return jsonify(dict(emotion_counter.most_common()))

if __name__ == '__main__':
    app.run(debug=True)

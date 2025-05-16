import torch
import cv2
import numpy as np
from torchvision import transforms
from es.resnet_emotion import ResNetEmotion
from PIL import Image
import pyttsx3

# ---------------------- CONFIG ---------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_QUOTES = {
    "Angry": "Take a deep breath. Calm brings clarity.",
    "Disgust": "Not everything needs a reaction. Choose peace.",
    "Fear": "Courage isn't the absence of fear, it's action in spite of it.",
    "Happy": "Smile big, it suits you!",
    "Neutral": "Every moment holds potential. Stay aware.",
    "Sad": "It's okay to feel down. Brighter days are ahead.",
    "Surprise": "Unexpected moments can be the most beautiful."
}

# ---------------------- MODEL SETUP ---------------------- #
model = ResNetEmotion(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))
model.eval()

# ---------------------- TEXT TO SPEECH ---------------------- #
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# ---------------------- IMAGE TRANSFORM ---------------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------------- FACE DETECTION ---------------------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

last_emotion = None
stable_counter = 0
required_stable_frames = 3
last_spoken_emotion = None
quote_to_display = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Only process first detected face
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = Image.fromarray(face)
        face = transform(face).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(face)
            _, pred = torch.max(output, 1)
            emotion = EMOTIONS[pred.item()]

        if emotion == last_emotion:
            stable_counter += 1
        else:
            stable_counter = 1  # Reset counter
            last_emotion = emotion

        # Only act if stable for required frames and not recently spoken
        if stable_counter >= required_stable_frames and emotion != last_spoken_emotion:
            quote_to_display = EMOTION_QUOTES[emotion]
            print(f"{emotion}: {quote_to_display}")
            engine.say(quote_to_display)
            engine.runAndWait()
            last_spoken_emotion = emotion

        # Draw face box and overlay text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, quote_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

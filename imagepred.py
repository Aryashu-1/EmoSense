import torch
import cv2
import numpy as np
from torchvision import transforms
from es.resnet_emotion import ResNetEmotion
from PIL import Image

# Load Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
model = ResNetEmotion(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))
model.eval()

# Labels (Modify based on dataset)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# OpenCV Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the image

image_path = "./images/happy.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Process each face
for (x, y, w, h) in faces:
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (48, 48))
    
    # Convert NumPy array to PIL Image
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0).to(DEVICE)

    # Predict emotion
    with torch.no_grad():
        output = model(face)
        _, pred = torch.max(output, 1)
        emotion = EMOTIONS[pred.item()]

    # Draw rectangle and label on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Emotion Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
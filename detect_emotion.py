import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import time
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("model/emotion_model.h5")

# Emotion labels (same order as dataset folders)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

# To store mood history with timestamps
emotion_history = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # (48,48,1)
        roi_gray = np.expand_dims(roi_gray, axis=0)   # (1,48,48,1)

        prediction = model.predict(roi_gray, verbose=0)
        emotion_label = class_names[np.argmax(prediction)]

        # Store emotion every 2 seconds with timestamp
        if time.time() - start_time >= 2:
            emotion_history.append({"time": time.strftime("%H:%M:%S"), "emotion": emotion_label})
            start_time = time.time()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---- Mood Mapping ----
# Convert to DataFrame
df = pd.DataFrame(emotion_history)

# Save to CSV
df.to_csv("mood_history.csv", index=False)
print("✅ Mood history saved to mood_history.csv")

# Plot mood distribution
counts = Counter(df["emotion"])
plt.bar(counts.keys(), counts.values(), color="skyblue")
plt.title("Mood Mapping - Emotion Distribution")
plt.xlabel("Emotions")
plt.ylabel("Frequency")
plt.show()

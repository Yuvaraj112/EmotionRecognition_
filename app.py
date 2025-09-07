from flask import Flask, render_template, Response, jsonify
import cv2
import time
import numpy as np
import tensorflow as tf
from collections import Counter
import atexit
import pandas as pd
from datetime import date

app = Flask(__name__)

# -----------------------------
# Load Emotion Model (robust)
# -----------------------------
model = None
model_load_error = None
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
try:
    # Prefer compile=False to avoid optimizer/loss deserialization issues
    model = tf.keras.models.load_model("model/emotion_model.h5", compile=False)
except Exception as e:
    # Fallback attempts for legacy/edge cases
    model_load_error = str(e)
    try:
        # Some legacy models load via tf.compat.v1 path
        model = tf.compat.v1.keras.models.load_model("model/emotion_model.h5", compile=False)
        model_load_error = None
    except Exception as e2:
        model_load_error = f"Primary and fallback load failed: {e2}"

# -----------------------------
# Global Variables
# -----------------------------
emotion_history = []
start_time = time.time()

# -----------------------------
# Initialize Camera (robust)
# -----------------------------
def initialize_camera() -> cv2.VideoCapture:
    """Try multiple indices and backends to find a working camera."""
    candidate_settings = [
        (0, cv2.CAP_DSHOW),
        (0, 0),
        (1, cv2.CAP_DSHOW),
        (1, 0),
    ]
    for index, backend in candidate_settings:
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            print(f"✅ Camera opened at index {index} backend {backend}")
            return cam
        cam.release()
    raise RuntimeError("❌ Cannot access the camera. Check if it's connected or in use.")

cap = initialize_camera()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Frame Generator
# -----------------------------
def gen_frames():
    global emotion_history, start_time
    print("🎥 Starting video stream...")
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame from camera")
            continue  # Skip if frame not read
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"📹 Processed {frame_count} frames")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            if model is not None:
                try:
                    prediction = model.predict(roi_gray, verbose=0)
                    emotion_label = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)
                    
                    # Debug: print prediction details
                    print(f"Prediction: {prediction[0]}")
                    print(f"Emotion: {emotion_label}, Confidence: {confidence:.3f}")
                    
                    # Only show emotion if confidence is above threshold
                    if confidence < 0.1:
                        emotion_label = "uncertain"
                except Exception as e:
                    print(f"Prediction error: {e}")
                    emotion_label = "error"
            else:
                emotion_label = "model_unavailable"

            # Save every 2 seconds
            if time.time() - start_time >= 2:
                emotion_history.append(emotion_label)
                start_time = time.time()

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"❌ Frame encoding error: {e}")
            continue

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')  # Your HTML dashboard

@app.route('/video_feed')
def video_feed():
    print("🎥 Video feed requested")
    resp = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # Prevent caching/stalling of the MJPEG stream
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/mood_data')
def mood_data():
    counts = Counter(emotion_history)
    return jsonify(counts)

@app.route('/health')
def health():
    return jsonify({
        "model_loaded": model is not None,
        "error": model_load_error
    })

# -----------------------------
# Save Mood Logs on Exit
# -----------------------------
def save_mood_logs():
    if emotion_history:
        df = pd.DataFrame({"emotion": emotion_history})
        filename = f"mood_log_{date.today()}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Mood log saved as {filename}")

atexit.register(save_mood_logs)

# -----------------------------
# Start Flask App
# -----------------------------
if __name__ == "__main__":
    # Use threaded=True for multiple clients; disable reloader to avoid double-opening the camera
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

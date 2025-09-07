# 🎭 AI-Powered Emotion Recognition System

A real-time emotion detection application that uses computer vision and machine learning to analyze facial expressions from webcam input and display emotion statistics through an interactive web dashboard.

## ✨ Features

- **Real-time Emotion Detection**: Live webcam feed with emotion labels overlaid on detected faces
- **Interactive Dashboard**: Beautiful web interface with emotion distribution charts
- **7 Emotion Categories**: Detects angry, disgust, fear, happy, neutral, sad, and surprise
- **Live Statistics**: Real-time bar chart showing emotion frequency
- **Timeline Visualization**: Multi-line chart tracking emotions over time
- **Mood History**: Automatic logging of detected emotions to CSV files
- **Responsive Design**: Modern, mobile-friendly interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/camera access
- Windows 10/11 (tested on Windows)

### Installation

1. **Clone or download the project**
   ```bash
   cd D:\Emotion_recognition
   ```

2. **Create a virtual environment (recommended)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   pip install flask tensorflow==2.15.0 keras==2.15.0 ml-dtypes~=0.2.0 opencv-python pandas
   ```

4. **Run the application**
   ```powershell
   python app.py
   ```

5. **Open in browser**
   - Navigate to: `http://127.0.0.1:5000`
   - Allow camera access when prompted

## 📁 Project Structure

```
Emotion_recognition/
├── app.py                 # Main Flask application
├── detect_emotion.py      # Emotion detection utilities
├── train_model.py         # Model training script
├── model/
│   └── emotion_model.h5   # Pre-trained emotion detection model
├── templates/
│   └── index.html         # Web dashboard template
├── mood_history.csv       # Emotion data logging
└── README.md             # This file
```

## 🎯 How It Works

1. **Camera Capture**: OpenCV captures frames from your webcam
2. **Face Detection**: Haar Cascade classifier detects faces in each frame
3. **Emotion Analysis**: Pre-trained CNN model analyzes facial features
4. **Real-time Display**: Emotion labels are overlaid on detected faces
5. **Data Visualization**: Charts update in real-time with emotion statistics
6. **Data Logging**: Emotions are saved to CSV for historical analysis

## 🛠️ Technical Details

### Model Architecture
- **Input**: 48x48 grayscale face images
- **Architecture**: Convolutional Neural Network (CNN)
- **Output**: 7 emotion classes with confidence scores
- **Framework**: TensorFlow/Keras

### Emotion Classes
- 😠 **Angry** - Red
- 🤢 **Disgust** - Purple  
- 😨 **Fear** - Orange
- 😊 **Happy** - Green
- 😐 **Neutral** - Gray
- 😢 **Sad** - Blue
- 😲 **Surprise** - Dark Orange

### API Endpoints
- `GET /` - Main dashboard
- `GET /video_feed` - MJPEG video stream
- `GET /mood_data` - Emotion statistics (JSON)
- `GET /health` - Model status check

## 🎨 Dashboard Features

### Live Webcam Feed
- Real-time video stream with emotion detection
- Green bounding boxes around detected faces
- Emotion labels displayed above each face
- Automatic face tracking and analysis

### Emotion Distribution Chart
- Colorful bar chart showing emotion frequency
- Real-time updates every 2 seconds
- Smooth animations and modern styling
- Responsive design for all screen sizes

### Mood Timeline
- Multi-line chart tracking all emotions over time
- Rolling 20-point window for performance
- Color-coded lines for each emotion
- Interactive legend and tooltips

## 🔧 Troubleshooting

### Camera Issues
- **No video feed**: Close other apps using the camera (Teams, Zoom, Camera app)
- **Permission denied**: Allow camera access in browser settings
- **Black screen**: Try refreshing the page or restarting the app

### Model Issues
- **"model_unavailable"**: Check if `model/emotion_model.h5` exists
- **Low accuracy**: Ensure good lighting and clear face visibility
- **Only one emotion**: Model may need retraining with more diverse data

### Performance Issues
- **Slow detection**: Close unnecessary applications
- **High CPU usage**: Reduce video quality in browser settings
- **Memory issues**: Restart the application periodically

## 📊 Data Export

The application automatically saves emotion data to CSV files:
- **File format**: `mood_log_YYYY-MM-DD.csv`
- **Columns**: emotion, timestamp
- **Location**: Project root directory
- **Frequency**: Every 2 seconds when faces are detected

## 🚀 Advanced Usage

### Custom Model Training
```python
# Use train_model.py to train your own model
python train_model.py
```

### API Integration
```python
# Get emotion data programmatically
import requests
response = requests.get('http://127.0.0.1:5000/mood_data')
emotions = response.json()
```

### Health Monitoring
```python
# Check model status
response = requests.get('http://127.0.0.1:5000/health')
status = response.json()
print(f"Model loaded: {status['model_loaded']}")
```

## 🛡️ Privacy & Security

- **Local Processing**: All analysis happens on your device
- **No Data Upload**: No facial data is sent to external servers
- **Camera Access**: Only used for real-time emotion detection
- **Data Storage**: Emotion logs stored locally in CSV format

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your camera is working with other applications
4. Check the terminal output for error messages

---

**Built with ❤️ using Flask, TensorFlow, OpenCV, and Chart.js**
# AuraBot - Smart Vision System

## Project Description
AuraBot is an intelligent computer vision system designed for the Fablab Project 1C.I. It combines object detection using YOLO with text-to-speech capabilities and optional Arduino integration for hardware feedback.

## Features
- **Object Detection**: Real-time object detection using YOLOv8
- **Text-to-Speech**: Audio feedback for detected objects
- **Arduino Integration**: Hardware connectivity for enhanced interaction (new_feature branch)
- **Face Recognition**: Face ID system with registration and cloud gallery (abdel branch)

## Installation

### Required Packages
```bash
pip install pyserial ultralytics opencv-python pyttsx3
```

### Docker Setup
```bash
docker-compose up --build
```

## Usage

### Main Object Detection
```bash
python Automatic_detector.py
```

### Face ID System
```bash
cd "face id"
python main.py
```

## Project Structure
- `Automatic_detector.py` - Main object detection script with Arduino support
- `yolov8n.pt` - YOLOv8 nano model weights
- `face id/` - Face recognition module
  - `main.py` - Face ID main application
  - `face_engine.py` - Face recognition engine
  - `audio_manager.py` - Audio management
  - `cloud_gallery.py` - Cloud storage integration
  - `registration_window.py` - User registration UI
  - `config.py` - Configuration settings
  - `encodings_cache.yml` - Face encodings database

## Controls
- Press **Q** to quit the object detection application

## Branches Merged
- **master** - Base implementation
- **new_feature** - Arduino serial communication integration
- **abdel** - Face recognition system

## License
Fablab Project 1C.I

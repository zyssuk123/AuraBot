import cv2
import numpy as np
import pyttsx3
import time
import threading
import os
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ─── Configuration from environment ──────────────────────────────────────────
ARDUINO_ENABLED = os.getenv('ARDUINO_ENABLED', 'true').lower() == 'true'
CAMERA_ENABLED = os.getenv('CAMERA_ENABLED', 'true').lower() == 'true'
GUI_ENABLED = os.getenv('GUI_ENABLED', 'true').lower() == 'true'

# ─── Auto-detect Arduino port ───────────────────────────────────────────────
arduino = None
SERIAL_PORT = None

if ARDUINO_ENABLED:
    def find_arduino_port():
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
                    return port.device
            if ports:
                return ports[0].device
        except Exception as e:
            print(f"⚠️ Error listing serial ports: {e}")
        return None

    SERIAL_PORT = find_arduino_port()
    
    if SERIAL_PORT:
        try:
            BAUD_RATE = 9600
            arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"✅ Connected to Arduino on {SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️ Could not connect to Arduino: {e}")
            arduino = None
    else:
        print("⚠️ Arduino not found. Running without hardware.")
else:
    print("ℹ️ Arduino disabled via environment variable.")

# ─── YOLO + Camera setup ─────────────────────────────────────────────────────
print("📦 Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("✅ YOLO model loaded!")

cap = None
if CAMERA_ENABLED:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Camera not available. Running in demo mode.")
        cap = None
    else:
        print("✅ Camera initialized!")
else:
    print("ℹ️ Camera disabled via environment variable.")

is_speaking = False
current_label = "Scanning..."
distance = 0

# ─── Demo frame generator ────────────────────────────────────────────────────
def create_demo_frame(frame_count):
    """Create a demo frame when camera is not available"""
    width, height = 640, 480
    demo = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background color (dark blue)
    demo[:] = (25, 25, 50)
    
    # Title
    cv2.putText(demo, "AURABOT - DEMO MODE", (120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Status
    cv2.putText(demo, "📷 Camera: Not Available", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(demo, "🔌 Arduino: " + ("Connected" if arduino else "Not Connected"), (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arduino else (0, 0, 255), 2)
    cv2.putText(demo, "🖥️  GUI: Enabled", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(demo, "Press Q to quit", (220, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(demo, "To enable camera:", (50, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "1. Edit .env file", (50, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "2. Set CAMERA_ENABLED=true", (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "3. Restart container", (50, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Frame counter
    cv2.putText(demo, f"Frame: {frame_count}", (520, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return demo

# ─── Text-to-speech ──────────────────────────────────────────────────────────
def speak(message):
    global is_speaking
    is_speaking = True
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(message)
        engine.runAndWait()
        del engine
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
    is_speaking = False

# ─── Serial communication thread ─────────────────────────────────────────────
def serial_thread():
    global current_label, distance
    while True:
        try:
            if arduino and arduino.is_open:
                arduino.write((current_label + '\n').encode('utf-8'))
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    if line.isdigit():
                        distance = int(line)
        except Exception as e:
            print(f"Serial error: {e}")
        time.sleep(0.3)

# Start serial thread only if Arduino is connected
if arduino:
    t_serial = threading.Thread(target=serial_thread, daemon=True)
    t_serial.start()

# ─── Main loop ───────────────────────────────────────────────────────────────
print("🚀 AuraBot started! Press Q to quit.")
print("=" * 50)

frame_count = 0
while True:
    ret = False
    frame = None
    
    if cap:
        ret, frame = cap.read()
    
    if not ret or frame is None:
        # Demo mode - create a test frame with instructions
        frame = None
        if frame_count % 30 == 0:  # Log every ~1 second
            print(f"⏳ Waiting for camera... (frame {frame_count})")
        
        # Create a demo frame for GUI display
        if GUI_ENABLED:
            demo_frame = create_demo_frame(frame_count)
            cv2.imshow("Blind Glasses - DEMO MODE", demo_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        time.sleep(0.033)  # ~30 FPS
        continue

    results = model(frame, verbose=False)[0]
    annotated = results.plot()

    detected_objects = []
    if results.boxes:
        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            if confidence > 0.5:
                detected_objects.append(label)

    unique_objects = list(set(detected_objects))

    if unique_objects:
        if len(unique_objects) == 1:
            message = unique_objects[0]
        else:
            message = ", ".join(unique_objects)

        if distance == 999:
            spoken = f"{message}, out of range"
        else:
            spoken = f"{message}, {distance} centimeters"

        current_label = message

        cv2.putText(annotated, f"{message} | {distance}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not is_speaking:
            print(f"🔊 Speaking: {spoken}")
            t = threading.Thread(target=speak, args=(spoken,))
            t.daemon = True
            t.start()
    else:
        current_label = "Scanning..."
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    dist_text = "Out of range" if distance == 999 else f"Distance: {distance} cm"
    cv2.putText(annotated, dist_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if GUI_ENABLED:
        cv2.imshow("Blind Glasses", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # In non-GUI mode, just log detections
        if unique_objects and frame_count % 10 == 0:
            print(f"👁️ Detected: {', '.join(unique_objects)}")

    frame_count += 1

# ─── Cleanup ─────────────────────────────────────────────────────────────────
print("\n🛑 Shutting down...")
if cap:
    cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
print("✅ Cleanup complete. Goodbye!")

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
CAMERA_ENABLED  = os.getenv('CAMERA_ENABLED',  'true').lower() == 'true'
GUI_ENABLED     = os.getenv('GUI_ENABLED',     'true').lower() == 'true'

# ─── Auto-detect Arduino port ─────────────────────────────────────────────────
arduino     = None
SERIAL_PORT = None

if ARDUINO_ENABLED:
    def find_arduino_port():
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if ('Arduino' in port.description or
                        'CH340'      in port.description or
                        'USB Serial' in port.description):
                    return port.device
            if ports:
                return ports[0].device
        except Exception as e:
            print(f"⚠️ Error listing serial ports: {e}")
        return None

    SERIAL_PORT = find_arduino_port()
    if SERIAL_PORT:
        try:
            BAUD_RATE = 115200          # must match Arduino Serial.begin(115200)
            arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)               # wait for Arduino reset after DTR toggle
            print(f"✅ Connected to Arduino on {SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️ Could not connect to Arduino: {e}")
            arduino = None
    else:
        print("⚠️ Arduino not found. Running without hardware.")
else:
    print("ℹ️ Arduino disabled via environment variable.")

# ─── YOLO + Camera setup ──────────────────────────────────────────────────────
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

# ─── Shared state ─────────────────────────────────────────────────────────────
is_speaking   = False
current_label = "Scanning..."

# FIX: protect `distance` with a lock so the serial thread and main loop
# never read/write it at the same time (avoids torn reads on 32-bit values).
distance      = 0
distance_lock = threading.Lock()

# ─── Demo frame generator ─────────────────────────────────────────────────────
def create_demo_frame(frame_count):
    """Return a placeholder frame shown when the camera is unavailable."""
    width, height = 640, 480
    demo = np.zeros((height, width, 3), dtype=np.uint8)

    demo[:] = (25, 25, 50)   # dark-blue background

    cv2.putText(demo, "AURABOT - DEMO MODE", (120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(demo, "Camera: Not Available", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(demo, "Arduino: " + ("Connected" if arduino else "Not Connected"),
                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if arduino else (0, 0, 255), 2)
    cv2.putText(demo, "GUI: Enabled", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(demo, "Press Q to quit",           (220, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(demo, "To enable camera:",          (50, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "1. Edit .env file",          (50, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "2. Set CAMERA_ENABLED=true", (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(demo, "3. Restart container",       (50, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(demo, f"Frame: {frame_count}", (520, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return demo

# ─── Text-to-speech ───────────────────────────────────────────────────────────
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

# ─── Serial communication thread ──────────────────────────────────────────────
def serial_thread():
    """
    Runs in the background:
      • Sends the latest detected object label to the Arduino (for LCD display).
      • Drains every line the Arduino has buffered and keeps only the last valid
        distance integer — prevents the read buffer from growing without bound
        when the main loop is slower than the Arduino's 50 ms send rate.
    """
    global current_label, distance
    while True:
        try:
            if arduino and arduino.is_open:
                # ── Send label to Arduino ──────────────────────────────────
                arduino.write((current_label + '\n').encode('utf-8'))

                # ── Read ALL buffered distance lines ──────────────────────
                # FIX: drain the full buffer (not just one line) so stale
                # readings don't accumulate and delay the displayed distance.
                latest = None
                while arduino.in_waiting > 0:
                    raw = arduino.readline().decode('utf-8').strip()
                    # Accept plain integers (positive only — distance is never negative)
                    if raw.isdigit():
                        latest = int(raw)

                if latest is not None:
                    # FIX: use a lock so the main loop never reads a half-written value
                    with distance_lock:
                        distance = latest

        except Exception as e:
            print(f"Serial error: {e}")

        time.sleep(0.3)

# Start serial thread only if Arduino is connected
if arduino:
    t_serial = threading.Thread(target=serial_thread, daemon=True)
    t_serial.start()

# ─── Main loop ────────────────────────────────────────────────────────────────
print("🚀 AuraBot started! Press Q to quit.")
print("=" * 50)

frame_count = 0
while True:
    ret   = False
    frame = None

    if cap:
        ret, frame = cap.read()

    # ── Demo / no-camera path ─────────────────────────────────────────────────
    if not ret or frame is None:
        if frame_count % 30 == 0:
            print(f"⏳ Waiting for camera... (frame {frame_count})")

        if GUI_ENABLED:
            demo_frame = create_demo_frame(frame_count)
            cv2.imshow("AuraBot - DEMO MODE", demo_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        time.sleep(0.033)   # ~30 FPS
        continue

    # ── YOLO inference ────────────────────────────────────────────────────────
    results  = model(frame, verbose=False)[0]
    annotated = results.plot()

    detected_objects = []
    if results.boxes:
        for box in results.boxes:
            lbl        = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            if confidence > 0.5:
                detected_objects.append(lbl)

    unique_objects = list(set(detected_objects))

    # FIX: take a single consistent snapshot of `distance` for this frame
    # so every display/speech call in the loop uses the exact same value.
    with distance_lock:
        current_distance = distance

    # ── Build announcement + overlay ─────────────────────────────────────────
    if unique_objects:
        message = unique_objects[0] if len(unique_objects) == 1 \
                  else ", ".join(unique_objects)

        spoken = (f"{message}, out of range"
                  if current_distance >= 295
                  else f"{message}, {current_distance} centimeters")

        current_label = message   # will be sent to Arduino LCD by serial_thread

        cv2.putText(annotated, f"{message} | {current_distance} cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not is_speaking:
            print(f"🔊 Speaking: {spoken}")
            t = threading.Thread(target=speak, args=(spoken,), daemon=True)
            t.start()
    else:
        current_label = "Scanning..."
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ── Distance overlay ─────────────────────────────────────────────────────
    dist_text = ("Out of range"
                 if current_distance >= 295
                 else f"Distance: {current_distance} cm")
    cv2.putText(annotated, dist_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ── Display / headless logging ────────────────────────────────────────────
    if GUI_ENABLED:
        cv2.imshow("AuraBot", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if unique_objects and frame_count % 10 == 0:
            print(f"👁️ Detected: {', '.join(unique_objects)} | {dist_text}")

    frame_count += 1

# ─── Cleanup ──────────────────────────────────────────────────────────────────
print("\n🛑 Shutting down...")
if cap:
    cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
print("✅ Cleanup complete. Goodbye!")
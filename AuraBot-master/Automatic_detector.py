import cv2
import numpy as np
import pyttsx3
import time
import threading
import os
import re
import serial
import serial.tools.list_ports
from dotenv import load_dotenv
from ultralytics import YOLO
from face_id.face_engine import FaceEngine

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
ARDUINO_ENABLED = os.getenv('ARDUINO_ENABLED', 'true').lower() == 'true'
CAMERA_ENABLED  = os.getenv('CAMERA_ENABLED',  'true').lower() == 'true'
GUI_ENABLED     = os.getenv('GUI_ENABLED',      'true').lower() == 'true'
FACEID_ENABLED  = os.getenv('FACEID_ENABLED',   'true').lower() == 'true'
CAMERA_INDEX    = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_MIRROR   = os.getenv('CAMERA_MIRROR',    'true').lower() == 'true'
ARDUINO_SEND_LABELS = os.getenv('ARDUINO_SEND_LABELS', 'true').lower() == 'true'
BAUD_RATE       = int(os.getenv('BAUD_RATE', '9600'))

# ─── Distance thresholds (must match Arduino sketch) ─────────────────────────
OOR_THRESHOLD  = 295   # anything >= this is "out of range"
SPEAK_MIN_CM   = 5     # ignore sensor noise below this
SPEAK_MAX_CM   = 80    # only speak when object is closer than this
SPEAK_COOLDOWN = 4     # seconds before repeating the same label

# ─── Shared state ─────────────────────────────────────────────────────────────
distance      = OOR_THRESHOLD  # updated by serial thread
current_label = "Scanning..."  # sent to Arduino
is_speaking   = False
_lock         = threading.Lock()

# ─── Auto-detect Arduino port ─────────────────────────────────────────────────
arduino = None
arduino_port = None

def find_arduino_port():
    try:
        ports = serial.tools.list_ports.comports()
        if ports:
            print("🔎 Serial ports:")
            for port in ports:
                print(f"   {port.device}: {port.description}")
        keywords = ('Arduino', 'CH340', 'CH341', 'USB Serial', 'ttyUSB', 'ttyACM')
        for port in ports:
            if any(kw in port.description for kw in keywords):
                return port.device
        non_bluetooth_ports = [
            port for port in ports
            if "bluetooth" not in port.description.lower()
        ]
        if len(non_bluetooth_ports) == 1:
            return non_bluetooth_ports[0].device
        if len(ports) == 1:
            return ports[0].device
    except Exception as e:
        print(f"⚠️ Error listing serial ports: {e}")
    return None

def connect_arduino(port):
    global arduino, arduino_port
    if port:
        try:
            arduino = serial.Serial(port, BAUD_RATE, timeout=0.05, write_timeout=0.2)
            arduino_port = port
            time.sleep(2)                    # wait for Arduino reset
            arduino.reset_input_buffer()
            print(f"✅ Arduino connected on {port} @ {BAUD_RATE} baud")
            return True
        except Exception as e:
            print(f"⚠️ Could not open {port}: {e}")
            arduino = None
            return False
    return False

def close_arduino():
    global arduino
    if arduino:
        try:
            arduino.close()
        except Exception:
            pass
    arduino = None

def reconnect_arduino():
    global distance, _last_sent_label
    port = os.getenv('ARDUINO_PORT') or arduino_port or find_arduino_port()
    close_arduino()
    with _lock:
        distance = OOR_THRESHOLD
    _last_sent_label = ""
    time.sleep(2)
    if port:
        connect_arduino(port)

def parse_distance(raw):
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        return int(float(match.group(0)))
    except ValueError:
        return None

if ARDUINO_ENABLED:
    port = os.getenv('ARDUINO_PORT') or find_arduino_port()
    if port:
        connect_arduino(port)
    else:
        print("⚠️ Arduino not found — running without hardware.")
else:
    print("ℹ️ Arduino disabled via environment variable.")

# ─── YOLO ─────────────────────────────────────────────────────────────────────
print("📦 Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("✅ YOLO model loaded!")

# ─── Face engine ──────────────────────────────────────────────────────────────
face_engine = None
if FACEID_ENABLED:
    print("👁️ Loading face recognition engine...")
    try:
        gallery_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "face_id", "galerie"))
        os.makedirs(gallery_path, exist_ok=True)
        face_engine = FaceEngine()
        face_engine.train_on_gallery(gallery_path)
        print("✅ Face recognition initialized!")
    except Exception as e:
        print(f"⚠️ Face recognition failed: {e}")
        face_engine = None

# ─── Camera ───────────────────────────────────────────────────────────────────
cap = None
if CAMERA_ENABLED:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"⚠️ Camera {CAMERA_INDEX} not available — demo mode.")
        cap = None
    else:
        print(f"✅ Camera {CAMERA_INDEX} initialized!")
else:
    print("ℹ️ Camera disabled via environment variable.")

# ─── Demo frame ───────────────────────────────────────────────────────────────
def create_demo_frame(frame_count):
    w, h = 640, 480
    demo = np.zeros((h, w, 3), dtype=np.uint8)
    demo[:] = (25, 25, 50)
    cv2.putText(demo, "AURABOT - DEMO MODE",   (120, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(demo, "Camera : Not Available", (50, 120),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),   2)
    cv2.putText(demo, "Arduino: " + ("Connected" if arduino else "Not Connected"),
                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if arduino else (0, 0, 255), 2)
    with _lock:
        d = distance
    cv2.putText(demo, f"Distance: {d} cm",      (50, 200),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(demo, "Press Q to quit",         (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(demo, f"Frame: {frame_count}",   (520, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
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
    finally:
        is_speaking = False

# ─── Serial thread ────────────────────────────────────────────────────────────
#
#   Protocol:
#     Python  → Arduino : "<label>\n"    (sent only when label changes)
#     Arduino → Python  : "<int_cm>\n"   (distance as plain integer)
#
_last_sent_label = ""

def serial_thread():
    global distance, _last_sent_label
    while True:
        try:
            if arduino and arduino.is_open:

                # Send label to Arduino only when it changed
                with _lock:
                    label_now = current_label
                if ARDUINO_SEND_LABELS and label_now != _last_sent_label:
                    arduino.write((label_now[:16] + '\n').encode('utf-8'))
                    _last_sent_label = label_now

                # Read distance from Arduino
                raw = arduino.readline().decode('utf-8', errors='ignore').strip()
                if raw:
                    val = parse_distance(raw)
                    if val is not None:
                        with _lock:
                            distance = val
            else:
                reconnect_arduino()

        except serial.SerialException as e:
            print(f"⚠️ Serial error: {e} — reconnecting")
            reconnect_arduino()
        except Exception as e:
            print(f"⚠️ Serial thread error: {e} — reconnecting")
            reconnect_arduino()

        time.sleep(0.05)  # 20 Hz — fast enough for smooth response

if arduino:
    t = threading.Thread(target=serial_thread, daemon=True)
    t.start()
    print("🔌 Serial thread started.")

# ─── Main loop ────────────────────────────────────────────────────────────────
print("🚀 AuraBot started! Press Q to quit.")
print("=" * 50)

frame_count   = 0
last_spoken   = ""
last_speak_ts = 0.0

while True:
    ret   = False
    frame = None

    if cap:
        ret, frame = cap.read()
        if ret and frame is not None and CAMERA_MIRROR:
            frame = cv2.flip(frame, 1)

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if not ret or frame is None:
        if frame_count % 30 == 0:
            print(f"⏳ Waiting for camera… (frame {frame_count})")
        if GUI_ENABLED:
            cv2.imshow("AuraBot - DEMO MODE", create_demo_frame(frame_count))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1
        time.sleep(0.033)
        continue

    # ── YOLO detection ────────────────────────────────────────────────────────
    results   = model(frame, verbose=False)[0]
    annotated = results.plot()

    detected_objects = []
    if results.boxes:
        for box in results.boxes:
            lbl        = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            if confidence > 0.5:
                detected_objects.append(lbl)

    unique_objects = list(set(detected_objects))

    # ── Face recognition ──────────────────────────────────────────────────────
    face_labels = []
    if face_engine:
        try:
            faces = face_engine.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_bgr = frame[y:y+h, x:x+w]
                name, conf = face_engine.predict(face_bgr)
                is_known   = name not in ("Inconnu", "Accès Refusé")
                color      = (30, 144, 255) if is_known else (0, 0, 255)
                text       = name if is_known else "Unknown"
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if is_known:
                    face_labels.append(name)
        except Exception as e:
            print(f"⚠️ Face recognition error: {e}")

    # ── Thread-safe distance snapshot ─────────────────────────────────────────
    with _lock:
        dist_now = distance

    out_of_range = dist_now >= OOR_THRESHOLD

    # ── Build label and spoken message ────────────────────────────────────────
    spoken = None

    if unique_objects or face_labels:

        if unique_objects:
            objects_str    = ", ".join(unique_objects)
            object_message = (f"{objects_str}, out of range"
                              if out_of_range
                              else f"{objects_str}, {dist_now} centimeters")
        else:
            object_message = None

        face_message = ("Personne détectée: " + ", ".join(set(face_labels))
                        if face_labels else None)

        if object_message and face_message:
            spoken = f"{object_message}. {face_message}"
            with _lock:
                current_label = f"{objects_str} | {', '.join(set(face_labels))}"
            cv2.putText(annotated, face_message, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif object_message:
            spoken = object_message
            with _lock:
                current_label = objects_str
        else:
            spoken = face_message
            with _lock:
                current_label = face_message

        cv2.putText(annotated, spoken[:60] if spoken else "", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── Speech gate ───────────────────────────────────────────────────────
        #   Speaks only when:
        #     1. Not already speaking
        #     2. Object is within 5–80 cm (matches red + yellow LED zones)
        #     3. Label changed OR cooldown elapsed
        now         = time.time()
        in_range    = SPEAK_MIN_CM <= dist_now < SPEAK_MAX_CM
        label_new   = spoken != last_spoken
        cooldown_ok = (now - last_speak_ts) >= SPEAK_COOLDOWN

        if spoken and not is_speaking and in_range and (label_new or cooldown_ok):
            print(f"🔊 Speaking: '{spoken}'  (distance={dist_now} cm)")
            last_spoken   = spoken
            last_speak_ts = now
            th = threading.Thread(target=speak, args=(spoken,), daemon=True)
            th.start()

    else:
        with _lock:
            current_label = "Scanning..."
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ── Distance overlay ──────────────────────────────────────────────────────
    dist_text = "Out of range" if out_of_range else f"Distance: {dist_now} cm"
    cv2.putText(annotated, dist_text, (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ── Arduino status indicator ──────────────────────────────────────────────
    hw_color = (0, 255, 0) if arduino else (0, 0, 255)
    hw_text  = "Arduino: OK" if arduino else "Arduino: disconnected"
    cv2.putText(annotated, hw_text, (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hw_color, 1)

    # ── Display ───────────────────────────────────────────────────────────────
    if GUI_ENABLED:
        cv2.imshow("AuraBot", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if unique_objects and frame_count % 10 == 0:
            print(f"👁️ Detected: {', '.join(unique_objects)}  |  {dist_text}")

    frame_count += 1

# ─── Cleanup ──────────────────────────────────────────────────────────────────
print("\n🛑 Shutting down…")
if cap:
    cap.release()
if arduino and arduino.is_open:
    arduino.close()
cv2.destroyAllWindows()
print("✅ Cleanup complete. Goodbye!")

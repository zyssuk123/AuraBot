import cv2
import pyttsx3
import time
import threading
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ─── Auto-detect Arduino port ───────────────────────────────────────────────
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
            return port.device
    # fallback: return first available port
    if ports:
        return ports[0].device
    return None

SERIAL_PORT = find_arduino_port()
BAUD_RATE = 9600

if SERIAL_PORT is None:
    print("❌ Arduino not found! Check USB connection.")
    exit()

print(f"✅ Connected to Arduino on {SERIAL_PORT}")
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # wait for Arduino to reset

# ─── YOLO + Camera setup ─────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

is_speaking = False
current_label = "Scanning..."
distance = 0

# ─── Text-to-speech ──────────────────────────────────────────────────────────
def speak(message):
    global is_speaking
    is_speaking = True
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(message)
    engine.runAndWait()
    del engine
    is_speaking = False

# ─── Serial communication thread ─────────────────────────────────────────────
def serial_thread():
    global current_label, distance
    while True:
        try:
            # Send current label to Arduino
            arduino.write((current_label + '\n').encode('utf-8'))

            # Read distance from Arduino
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if line.isdigit():
                    distance = int(line)
        except Exception as e:
            print(f"Serial error: {e}")
        time.sleep(0.3)

# Start serial thread
t_serial = threading.Thread(target=serial_thread, daemon=True)
t_serial.start()

# ─── Main loop ───────────────────────────────────────────────────────────────
print("Camera started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        # Add distance info to spoken message
        if distance == 999:
            spoken = f"{message}, out of range"
        else:
            spoken = f"{message}, {distance} centimeters"

        current_label = message  # send to Arduino LCD

        cv2.putText(annotated, f"{message} | {distance}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not is_speaking:
            print(f"Speaking: {spoken}")
            t = threading.Thread(target=speak, args=(spoken,))
            t.daemon = True
            t.start()
    else:
        current_label = "Scanning..."
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show distance on frame
    dist_text = "Out of range" if distance == 999 else f"Distance: {distance} cm"
    cv2.putText(annotated, dist_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Blind Glasses", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
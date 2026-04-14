import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

is_speaking = False

def speak(message):
    global is_speaking
    is_speaking = True
    engine = pyttsx3.init()  # create fresh engine each time
    engine.setProperty('rate', 150)
    engine.say(message)
    engine.runAndWait()
    del engine
    is_speaking = False

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
            message = f"{unique_objects[0]}"
        else:
            message = ", ".join(unique_objects)

        cv2.putText(annotated, message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Speak if not currently speaking
        if not is_speaking:
            print(f"Speaking: {message}")
            t = threading.Thread(target=speak, args=(message,))
            t.daemon = True
            t.start()
    else:
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Blind Glasses", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
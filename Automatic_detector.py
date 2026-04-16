import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# ── Distance estimation config ──────────────────────────────────────────────
# Approximate real-world heights (metres) for common COCO classes.
# Add / adjust entries as needed.
KNOWN_HEIGHTS = {
    "person":        1.70,
    "bicycle":       1.00,
    "car":           1.50,
    "motorcycle":    1.10,
    "bus":           3.00,
    "truck":         2.50,
    "cat":           0.25,
    "dog":           0.50,
    "chair":         0.90,
    "dining table":  0.75,
    "bottle":        0.25,
    "cup":           0.12,
    "laptop":        0.25,
    "tv":            0.60,
    "cell phone":    0.15,
    "book":          0.22,
    "clock":         0.30,
    "backpack":      0.50,
    "umbrella":      0.90,
    "handbag":       0.30,
    "suitcase":      0.65,
    "sports ball":   0.22,
    "keyboard":      0.04,
    "mouse":         0.04,
    "remote":        0.18,
    "scissors":      0.18,
    "toothbrush":    0.18,
    "vase":          0.30,
    "bowl":          0.12,
    "banana":        0.18,
    "apple":         0.08,
    "orange":        0.08,
    "sandwich":      0.10,
    "pizza":         0.30,
    "cake":          0.15,
    "couch":         0.85,
    "bed":           0.60,
    "toilet":        0.40,
    "sink":          0.25,
    "refrigerator":  1.80,
    "microwave":     0.30,
    "oven":          0.90,
}
DEFAULT_HEIGHT = 0.50          # fallback for unlisted objects (metres)

# Focal length (pixels).  Calibrate once with a known object at a known
# distance, then set FOCAL_LENGTH = (pixel_height × distance) / real_height.
# Default below works reasonably for a standard 720p / 1080p webcam.
FOCAL_LENGTH = 600             # pixels  – tune this for your camera

WARNING_DISTANCE = 1.0         # metres – alert threshold


def estimate_distance(label: str, pixel_height: float) -> float | None:
    """Return estimated distance in metres, or None if pixel_height is 0."""
    if pixel_height <= 0:
        return None
    real_h = KNOWN_HEIGHTS.get(label.lower(), DEFAULT_HEIGHT)
    return (real_h * FOCAL_LENGTH) / pixel_height


# ── TTS ─────────────────────────────────────────────────────────────────────
is_speaking = False


def speak(message: str):
    global is_speaking
    is_speaking = True
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(message)
    engine.runAndWait()
    del engine
    is_speaking = False


# ── Main loop ────────────────────────────────────────────────────────────────
print("Camera started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    annotated = results.plot()

    detected_objects   = []   # (label, distance_or_None)
    warning_objects    = []   # labels closer than WARNING_DISTANCE

    if results.boxes:
        for box in results.boxes:
            label      = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if confidence <= 0.5:
                continue

            # Bounding box coords  [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pixel_height    = y2 - y1

            distance = estimate_distance(label, pixel_height)

            detected_objects.append((label, distance))

            if distance is not None and distance < WARNING_DISTANCE:
                warning_objects.append((label, distance))

    # ── On-screen display ────────────────────────────────────────────────
    y_offset = 40
    if detected_objects:
        # De-duplicate labels, keep closest distance per label
        closest: dict[str, float | None] = {}
        for lbl, dist in detected_objects:
            if lbl not in closest or (dist is not None and (closest[lbl] is None or dist < closest[lbl])):
                closest[lbl] = dist

        for lbl, dist in closest.items():
            if dist is not None:
                dist_text = f"{dist:.2f} m"
                color     = (0, 0, 255) if dist < WARNING_DISTANCE else (0, 255, 0)
            else:
                dist_text = "? m"
                color     = (0, 255, 0)

            line = f"{lbl}: {dist_text}"
            cv2.putText(annotated, line, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            y_offset += 35

        # ── Warning banner ───────────────────────────────────────────────
        if warning_objects:
            # Closest threatening object
            warning_objects.sort(key=lambda t: t[1])
            warn_lbl, warn_dist = warning_objects[0]
            banner = f"WARNING: {warn_lbl} {warn_dist:.2f} m – TOO CLOSE!"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (0, 0, 200), -1)
            cv2.putText(annotated, banner, (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        # ── Voice ────────────────────────────────────────────────────────
        if not is_speaking:
            if warning_objects:
                warn_lbl, warn_dist = warning_objects[0]
                speech = (f"Warning! {warn_lbl} is {warn_dist:.1f} metres away. "
                          "Too close!")
            else:
                parts  = [f"{lbl}" for lbl, _ in closest.items()]
                speech = ", ".join(parts)

            print(f"Speaking: {speech}")
            t = threading.Thread(target=speak, args=(speech,))
            t.daemon = True
            t.start()

    else:
        cv2.putText(annotated, "Nothing detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Blind Glasses", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
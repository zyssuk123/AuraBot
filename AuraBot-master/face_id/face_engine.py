import os
import pickle
import shutil

import cv2
import numpy as np

try:
    from face_id.cloud_gallery import BLOB_TOKEN, download_gallery_to_tempdir
except ImportError:
    from cloud_gallery import BLOB_TOKEN, download_gallery_to_tempdir


class FaceEngine:
    """Face detection and recognition for Blindy.

    Uses OpenCV LBPH when opencv-contrib is installed. If it is not installed,
    falls back to a local template recognizer so the app still identifies saved
    faces instead of always returning "unknown".
    """

    FALLBACK_ACCEPT_DISTANCE = 0.28
    FALLBACK_STRONG_ACCEPT_DISTANCE = 0.12
    FALLBACK_MIN_MARGIN = 0.05
    FALLBACK_TOP_K = 7
    FALLBACK_CACHE_VERSION = 2

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self.recognizer = None
        self.use_fallback = False
        self.is_trained = False
        self.known_names = {}
        self.fallback_samples = []

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=10, grid_y=10
            )
        except Exception as e:
            print(f"[Engine] cv2.face unavailable: {e}")
            print("[Engine] Fallback recognizer active. Install opencv-contrib-python for LBPH.")
            self.use_fallback = True

        self.cache_file = os.path.join(self.base_dir, "encodings_cache.yml")
        self.names_file = os.path.join(self.base_dir, "names_cache.pkl")
        self.fallback_cache_file = os.path.join(self.base_dir, "fallback_faces.pkl")
        self.load_cache()

    def load_cache(self):
        if self.use_fallback:
            if not os.path.exists(self.fallback_cache_file):
                return
            try:
                with open(self.fallback_cache_file, "rb") as f:
                    data = pickle.load(f)
                if data.get("version") != self.FALLBACK_CACHE_VERSION:
                    print("[Engine] Ignoring old fallback cache; retraining required.")
                    return
                self.known_names = data.get("known_names", {})
                self.fallback_samples = data.get("samples", [])
                self.is_trained = bool(self.fallback_samples)
                if self.is_trained:
                    print(
                        f"[Engine] Fallback cache loaded: "
                        f"{len(self.known_names)} people ready."
                    )
            except Exception as e:
                print(f"[Engine] Fallback cache read error: {e}")
            return

        if self.recognizer is None:
            return
        if os.path.exists(self.cache_file) and os.path.exists(self.names_file):
            try:
                self.recognizer.read(self.cache_file)
                with open(self.names_file, "rb") as f:
                    self.known_names = pickle.load(f)
                self.is_trained = bool(self.known_names)
                if self.is_trained:
                    print(
                        f"[Engine] OpenCV cache loaded: "
                        f"{len(self.known_names)} people ready."
                    )
            except Exception as e:
                print(f"[Engine] Cache read error: {e}")

    def apply_clahe(self, gray_image):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def detect_faces(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.apply_clahe(gray)
        return self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )

    def _read_image_gray(self, path):
        # cv2.imread can fail on non-ASCII paths on Windows, so read bytes first.
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            image = None
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image

    def _prepare_face(self, image, cascade=None):
        if image is None:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = self.apply_clahe(gray)

        detector = cascade or self.cascade
        try:
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
        except Exception:
            faces = []

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            gray = gray[y : y + h, x : x + w]

        prepared = cv2.resize(gray, (160, 160))
        return cv2.equalizeHist(prepared)

    def _fallback_descriptor(self, prepared_face):
        face = prepared_face.astype(np.uint8)
        center = face[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        neighbors = (
            face[:-2, :-2],
            face[:-2, 1:-1],
            face[:-2, 2:],
            face[1:-1, 2:],
            face[2:, 2:],
            face[2:, 1:-1],
            face[2:, :-2],
            face[1:-1, :-2],
        )
        for bit, neighbor in enumerate(neighbors):
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        histograms = []
        grid_y, grid_x = 8, 8
        cell_h = lbp.shape[0] // grid_y
        cell_w = lbp.shape[1] // grid_x
        for gy in range(grid_y):
            for gx in range(grid_x):
                cell = lbp[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
                hist = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
                hist /= float(hist.sum()) or 1.0
                histograms.append(hist)
        return np.concatenate(histograms)

    @staticmethod
    def _fallback_distance(a, b):
        return float(0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-8)) / 64.0)

    def _fallback_class_scores(self, descriptor):
        distances_by_label = {}
        for label, sample in self.fallback_samples:
            distance = self._fallback_distance(descriptor, sample)
            distances_by_label.setdefault(label, []).append(distance)

        scores = []
        for label, distances in distances_by_label.items():
            nearest = sorted(distances)[: self.FALLBACK_TOP_K]
            score = float(np.median(nearest))
            scores.append((score, label))
        return sorted(scores)

    def train_on_gallery(self, gallery_path):
        local_gallery_path = os.path.abspath(gallery_path)
        gallery_roots = [local_gallery_path]
        tmp_dir = None

        use_cloud_training = os.getenv("USE_CLOUD_GALLERY_FOR_TRAINING", "false").lower() == "true"
        if BLOB_TOKEN and use_cloud_training:
            try:
                tmp_dir = download_gallery_to_tempdir()
                gallery_roots = [tmp_dir, local_gallery_path]
                print("[Engine] Training from cloud + local gallery...")
            except Exception as e:
                print(f"[Engine] Cloud unavailable, using local gallery: {e}")

        if not os.path.exists(local_gallery_path):
            os.makedirs(local_gallery_path)
            return 0, 0

        faces = []
        labels = []
        known_names = {}
        name_to_id = {}
        current_id = 0
        local_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )

        for gallery_root in gallery_roots:
            if not gallery_root or not os.path.exists(gallery_root):
                continue
            for person_dir_name in os.listdir(gallery_root):
                person_dir = os.path.join(gallery_root, person_dir_name)
                if not os.path.isdir(person_dir):
                    continue

                name = person_dir_name.replace("_", " ")
                if name not in name_to_id:
                    name_to_id[name] = current_id
                    known_names[current_id] = name
                    current_id += 1
                person_id = name_to_id[name]

                for filename in os.listdir(person_dir):
                    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    image_path = os.path.join(person_dir, filename)
                    image = self._read_image_gray(image_path)
                    prepared = self._prepare_face(image, local_cascade)
                    if prepared is None:
                        continue

                    if self.use_fallback:
                        faces.append(self._fallback_descriptor(prepared))
                    else:
                        faces.append(cv2.resize(prepared, (200, 200)))
                    labels.append(person_id)

        if faces and self.use_fallback:
            self.known_names = known_names
            self.fallback_samples = [
                (int(label), descriptor) for label, descriptor in zip(labels, faces)
            ]
            self.is_trained = True
            with open(self.fallback_cache_file, "wb") as f:
                pickle.dump(
                    {
                        "version": self.FALLBACK_CACHE_VERSION,
                        "known_names": self.known_names,
                        "samples": self.fallback_samples,
                    },
                    f,
                )
            print(
                f"[Engine] Fallback model trained: "
                f"{len(faces)} photos, {len(name_to_id)} people."
            )
            result = len(faces), len(name_to_id)
        elif faces and self.recognizer is not None:
            self.recognizer.train(faces, np.array(labels, dtype=np.int32))
            self.known_names = known_names
            self.is_trained = True
            self.recognizer.write(self.cache_file)
            with open(self.names_file, "wb") as f:
                pickle.dump(self.known_names, f)
            print(
                f"[Engine] LBPH model trained: "
                f"{len(faces)} photos, {len(name_to_id)} people."
            )
            result = len(faces), len(name_to_id)
        else:
            self.is_trained = False
            result = 0, 0

        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    def predict(self, face_bgr):
        if not self.is_trained:
            return "Inconnu", 100

        prepared = self._prepare_face(face_bgr)
        if prepared is None:
            return "Inconnu", 100

        if self.use_fallback:
            descriptor = self._fallback_descriptor(prepared)
            scores = self._fallback_class_scores(descriptor)
            if not scores:
                return "Inconnu", 100

            best_distance, best_label = scores[0]
            second_distance = scores[1][0] if len(scores) > 1 else float("inf")
            clear_margin = second_distance - best_distance >= self.FALLBACK_MIN_MARGIN
            strong_match = best_distance <= self.FALLBACK_STRONG_ACCEPT_DISTANCE

            if best_distance <= self.FALLBACK_ACCEPT_DISTANCE and (clear_margin or strong_match):
                return self.known_names.get(best_label, "Inconnu"), best_distance

            if best_distance <= self.FALLBACK_ACCEPT_DISTANCE:
                print(
                    "[Engine] Ambiguous face: "
                    f"{self.known_names.get(best_label, best_label)}={best_distance:.3f}, "
                    f"second={second_distance:.3f}"
                )
            return "Accès Refusé", best_distance

        if self.recognizer is None:
            return "Inconnu", 100

        label, confidence = self.recognizer.predict(cv2.resize(prepared, (200, 200)))
        if confidence < 95:
            return self.known_names.get(label, "Inconnu"), confidence
        return "Accès Refusé", confidence

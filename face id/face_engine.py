import cv2
import os
import shutil
import numpy as np
import pickle

from cloud_gallery import download_gallery_to_tempdir, BLOB_TOKEN

class FaceEngine:
    """Reconnaissance LBPH (Pure Python via OpenCV) pour éviter les dépendances complexes (sans C++)."""
    
    def __init__(self):
        # Charge le détecteur de visage (alt2 est plus précis)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        # Algorithme LBPH (Très rapide, pas besoin de dlib)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=10, grid_y=10)
        self.is_trained = False
        self.known_names = {}
        
        # 1. Fichiers de cache local
        self.cache_file = "encodings_cache.yml"
        self.names_file = "names_cache.pkl"
        self.load_cache()

    def load_cache(self):
        """Récupère les modèles sauvegardés au lieu d'entraîner depuis zero."""
        if os.path.exists(self.cache_file) and os.path.exists(self.names_file):
            try:
                self.recognizer.read(self.cache_file)
                with open(self.names_file, "rb") as f:
                    self.known_names = pickle.load(f)
                if len(self.known_names) > 0:
                    self.is_trained = True
                    print(f"[Engine] Cache OpenCV chargé : {len(self.known_names)} personnes prêtes.")
            except Exception as e:
                print(f"[Engine] Erreur de lecture du cache : {e}")

    def apply_clahe(self, gray_image):
        """Mode nuit : améliore la détection en zone sombre, sans C++ ni IA."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def detect_faces(self, frame_bgr):
        """Détecte les visages: passage en gris + Mode Nuit."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.apply_clahe(gray)
        # Retourne des boites (x, y, w, h)
        return self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

    def train_on_gallery(self, gallery_path):
        """Télécharge du cloud et apprend."""
        tmp_dir = None
        if BLOB_TOKEN:
            try:
                tmp_dir = download_gallery_to_tempdir()
                gallery_path = tmp_dir
                print("[Engine] Entraînement Cloud (LBPH)...")
            except Exception as e:
                print(f"[Engine] Erreur Cloud, fallback : {e}")

        faces, labels = [], []
        temp_names, current_id = {}, 0
        name_to_id = {}

        if not os.path.exists(gallery_path):
            os.makedirs(gallery_path)
            return 0, 0

        for item in os.listdir(gallery_path):
            path_item = os.path.join(gallery_path, item)
            if os.path.isdir(path_item):
                name = item.replace("_", " ")
                if name not in name_to_id:
                    name_to_id[name] = current_id
                    temp_names[current_id] = name
                    current_id += 1
                
                person_id = name_to_id[name]
                for filename in os.listdir(path_item):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_gray = cv2.imread(os.path.join(path_item, filename), cv2.IMREAD_GRAYSCALE)
                        if img_gray is None: continue
                        
                        img_gray = self.apply_clahe(img_gray)
                        detected = self.cascade.detectMultiScale(img_gray, 1.1, 5)
                        if len(detected) > 0:
                            (x,y,w,h) = detected[0]
                            roi = cv2.resize(img_gray[y:y+h, x:x+w], (200, 200))
                            faces.append(roi)
                            labels.append(person_id)
                        else:
                            faces.append(cv2.resize(img_gray, (200, 200)))
                            labels.append(person_id)

        # Mise à jour
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels, dtype=np.int32))
            self.is_trained = True
            self.known_names = temp_names
            
            # Sauvegarder dans le cache local super rapide
            self.recognizer.write(self.cache_file)
            with open(self.names_file, "wb") as f:
                pickle.dump(self.known_names, f)
            print("[Engine] ✅ Modèle LBPH mis en cache.")
            result = len(faces), len(name_to_id)
        else:
            self.is_trained = False
            result = 0, 0

        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    def predict(self, face_bgr):
        """Identification."""
        if not self.is_trained:
            return "Inconnu", 100
            
        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray_face = self.apply_clahe(gray_face)
        resized = cv2.resize(gray_face, (200, 200))
        
        id_label, confidence = self.recognizer.predict(resized)
        if confidence < 80: # Seuil réglé empiriquement
            return self.known_names.get(id_label, "Inconnu"), confidence
            
        return "Accès Refusé", confidence

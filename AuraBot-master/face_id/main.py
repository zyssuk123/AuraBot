import cv2
import os
import re
import threading
import time
import customtkinter as ctk
from dotenv import load_dotenv
from PIL import Image
try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Modules locaux (Blindy Core)
try:
    from face_id.face_engine import FaceEngine
    from face_id.audio_manager import AudioManager
    from face_id.registration_window import FaceRegistrationWindow
except ImportError:
    from face_engine import FaceEngine
    from audio_manager import AudioManager
    from registration_window import FaceRegistrationWindow

load_dotenv()

class BlindyApp(ctk.CTk):
    OOR_THRESHOLD = int(os.getenv("OOR_THRESHOLD", "295"))
    SPEAK_MIN_CM = int(os.getenv("SPEAK_MIN_CM", "5"))
    SPEAK_MAX_CM = int(os.getenv("SPEAK_MAX_CM", "80"))
    SPEAK_COOLDOWN = float(os.getenv("SPEAK_COOLDOWN", "4"))
    OBJECT_CONFIDENCE = float(os.getenv("OBJECT_CONFIDENCE", "0.5"))
    PROXIMITY_WARNING_CM = int(os.getenv("PROXIMITY_WARNING_CM", "80"))
    PROXIMITY_DANGER_CM = int(os.getenv("PROXIMITY_DANGER_CM", "30"))
    PROXIMITY_COOLDOWN = float(os.getenv("PROXIMITY_COOLDOWN", "3"))

    """L'intelligence artificielle Blindy : Interface Minimaliste et Contrôle Vocal."""
    def __init__(self):
        super().__init__()

        self.title("Blindy AI - L'œil intelligent")
        self.geometry("1100x800")
        self.configure(fg_color="#0D0D0D") # Fond ultra sombre pro

        self.gallery_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "galerie"))
        os.makedirs(self.gallery_path, exist_ok=True)
        self.faceid_enabled = os.getenv('FACEID_ENABLED', 'true').lower() == 'true'
        self.engine = FaceEngine() if self.faceid_enabled else None
        self.audio = AudioManager(callback_command=self.handle_voice_command)
        self.audio.callback_status = self.update_status_ui # Nouveau callback
        # Options imported from Automatic_detector.py
        self.camera_enabled = os.getenv('CAMERA_ENABLED', 'true').lower() == 'true'
        self.camera_mirror = os.getenv('CAMERA_MIRROR', 'true').lower() == 'true'
        self.object_detection_enabled = os.getenv('OBJECT_DETECTION_ENABLED', 'true').lower() == 'true'
        self.yolo_model = None
        if self.object_detection_enabled and YOLO is not None:
            try:
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'))
                self.yolo_model = YOLO(model_path)
                print('✅ YOLO loaded inside GUI app')
            except Exception as e:
                print(f'⚠️ Could not load YOLO model in GUI: {e}')
        
        # État de l'IA
        self.arduino_enabled = os.getenv('ARDUINO_ENABLED', 'true').lower() == 'true'
        self.arduino_send_labels = os.getenv('ARDUINO_SEND_LABELS', 'true').lower() == 'true'
        self.arduino = None
        self._arduino_port = None
        self._arduino_baud_rate = int(os.getenv('BAUD_RATE', '9600'))
        self._arduino_lock = threading.Lock()
        self._arduino_distance = None
        self._arduino_label = "Scanning..."
        self._arduino_last_sent_label = ""
        self._arduino_last_raw = ""
        self._arduino_last_rx_ts = 0.0
        self._closing = False
        self._init_arduino()

        self.capture = None
        self.is_camera_running = False
        self.is_listening = False
        self.is_registering = False
        self._last_object_announcement = ""
        self._last_object_announcement_ts = 0.0
        self._last_proximity_zone = ""
        self._last_proximity_ts = 0.0
        
        self.setup_ui()
        self.train_ia_thread()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Raccourci clavier 'I' pour l'inscription (Backup si la voix ne capte pas)
        self.bind("<i>", lambda e: self.open_registration_vocal())
        self.bind("<I>", lambda e: self.open_registration_vocal())

        # Démarrage automatique des systèmes
        self.after(500, self.start_systems)

    def find_arduino_port(self):
        if serial is None:
            return None
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

    def _init_arduino(self):
        if not self.arduino_enabled or serial is None:
            print("ℹ️ Arduino disabled or pyserial unavailable.")
            return

        port = os.getenv("ARDUINO_PORT") or self.find_arduino_port()
        if not port:
            print("⚠️ Arduino not found - running without hardware.")
            return

        self._connect_arduino(port, start_worker=True)

    def _connect_arduino(self, port, start_worker=False):
        try:
            self.arduino = serial.Serial(port, self._arduino_baud_rate, timeout=0.05, write_timeout=0.2)
            self._arduino_port = port
            time.sleep(2)
            self.arduino.reset_input_buffer()
            self._arduino_last_sent_label = ""
            if start_worker:
                threading.Thread(target=self._serial_worker, daemon=True).start()
            print(f"✅ Arduino connected on {port} @ {self._arduino_baud_rate} baud")
            return True
        except Exception as e:
            print(f"⚠️ Could not open Arduino port {port}: {e}")
            self.arduino = None
            return False

    def _close_arduino(self):
        if self.arduino:
            try:
                self.arduino.close()
            except Exception:
                pass
        self.arduino = None

    def _reconnect_arduino(self):
        port = os.getenv("ARDUINO_PORT") or self._arduino_port or self.find_arduino_port()
        self._close_arduino()
        with self._arduino_lock:
            self._arduino_distance = None
            self._arduino_last_raw = ""
            self._arduino_last_rx_ts = 0.0
        if not port:
            time.sleep(2)
            return
        time.sleep(2)
        self._connect_arduino(port, start_worker=False)

    def _serial_worker(self):
        while not self._closing:
            try:
                if self.arduino and self.arduino.is_open:
                    with self._arduino_lock:
                        label_now = self._arduino_label

                    if self.arduino_send_labels and label_now != self._arduino_last_sent_label:
                        try:
                            self.arduino.write((label_now[:16] + '\n').encode('utf-8'))
                            self._arduino_last_sent_label = label_now
                        except Exception as e:
                            print(f"⚠️ Serial write error: {e}")

                    raw = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if raw:
                        value = self._parse_distance(raw)
                        with self._arduino_lock:
                            self._arduino_last_raw = raw
                            self._arduino_last_rx_ts = time.time()
                            if value is not None:
                                self._arduino_distance = value
                else:
                    self._reconnect_arduino()
            except Exception as e:
                print(f"⚠️ Serial worker error: {e} - reconnecting")
                self._reconnect_arduino()
            time.sleep(0.05)

    @staticmethod
    def _parse_distance(raw):
        """Accept both '42' and verbose Arduino lines like 'Distance: 42 cm'."""
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        try:
            return int(float(match.group(0)))
        except ValueError:
            return None

    def _set_arduino_label(self, label):
        with self._arduino_lock:
            self._arduino_label = label or "Scanning..."

    def _get_distance(self):
        with self._arduino_lock:
            return self._arduino_distance

    def _get_last_serial_raw(self):
        with self._arduino_lock:
            return self._arduino_last_raw

    def _get_serial_status(self):
        with self._arduino_lock:
            return self._arduino_port, self._arduino_last_raw, self._arduino_last_rx_ts

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Header futuriste
        self.lbl_blindy = ctk.CTkLabel(self, text="BLINDY AI", font=ctk.CTkFont(size=30, weight="bold", family="Orbitron"), text_color="#1E90FF")
        self.lbl_blindy.grid(row=0, column=0, pady=(20, 5))
        
        self.lbl_status = ctk.CTkLabel(self, text="Dites 'Blindy' + votre question", font=ctk.CTkFont(size=16), text_color="#2ECC71")
        self.lbl_status.grid(row=1, column=0, pady=(0, 20))

        self.lbl_distance = ctk.CTkLabel(self, text="Distance: --", font=ctk.CTkFont(size=15, weight="bold"), text_color="#F1C40F")
        self.lbl_distance.grid(row=2, column=0, pady=(0, 5))

        # Affichage Vidéo Central
        self.video_frame = ctk.CTkFrame(self, fg_color="black", border_width=2, border_color="#1E90FF")
        self.video_frame.grid(row=3, column=0, padx=50, pady=20, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_display = ctk.CTkLabel(self.video_frame, text="CHARGEMENT...", font=ctk.CTkFont(size=20), text_color="#333333")
        self.video_display.grid(row=0, column=0, sticky="nsew")

    def update_status_ui(self, msg, color="#1E90FF"):
        """Met à jour le texte sous le titre en temps réel."""
        self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=color))

    def update_distance_ui(self, text, color="#F1C40F"):
        """Shows the Arduino distance outside the video frame."""
        self.after(0, lambda: self.lbl_distance.configure(text=text, text_color=color))

    def _handle_proximity_warning(self, dist_now, has_distance, object_spoken=False):
        if not has_distance or object_spoken:
            return
        if dist_now < self.SPEAK_MIN_CM or dist_now >= self.PROXIMITY_WARNING_CM:
            self._last_proximity_zone = ""
            return

        zone = "danger" if dist_now <= self.PROXIMITY_DANGER_CM else "near"
        now = time.time()
        cooldown_ok = now - self._last_proximity_ts >= self.PROXIMITY_COOLDOWN
        zone_changed = zone != self._last_proximity_zone
        tts_busy = getattr(self.audio, "_tts_busy", False)

        if tts_busy or not (zone_changed or cooldown_ok):
            return

        self._last_proximity_zone = zone
        self._last_proximity_ts = now
        if zone == "danger":
            self.audio.parler(f"Attention, obstacle tres proche, {dist_now} centimetres")
        else:
            self.audio.parler(f"Obstacle a {dist_now} centimetres")

    def start_systems(self):
        """Lance les fils d'écoute vocale et la caméra."""
        if not self.is_listening:
            self.is_listening = True
            threading.Thread(target=self.audio.ecouter_commande, daemon=True).start()
            self.lbl_status.configure(text="🎤 Micro actif - Dites 'Blindy' pour parler", text_color="#2ECC71")
            self.audio.parler(self.audio._text("ready"))

        if not self.is_camera_running:
            self.start_camera()

    def handle_voice_command(self, cmd):
        """Gère les ordres reçus par la voix."""
        if cmd == "REGISTRATION":
            self.after(0, self.open_registration_vocal)
        elif cmd == "DELETE_PERSON":
            self.after(0, self.open_deletion_vocal)

    def open_deletion_vocal(self):
        """Procédure vocale pour supprimer un profil sans écran."""
        if self.is_registering: return
        self.is_registering = True
        self.audio.is_paused = True
        
        def _delete_success(name):
            try:
                from face_id.cloud_gallery import delete_person
            except ImportError:
                from cloud_gallery import delete_person
            import os
            deleted = delete_person(name)
            if deleted > 0:
                self.audio.parler(self.audio._text("delete_done", name=name))
                # Nettoyage du cache pour forcer un nettoyage complet
                if os.path.exists("encodings_cache.yml"): os.remove("encodings_cache.yml")
                if os.path.exists("names_cache.pkl"): os.remove("names_cache.pkl")
                # Relancer l'entraînement
                self.train_ia_thread()
            else:
                self.audio.parler(self.audio._text("delete_missing", name=name))
            self.is_registering = False
            self.audio.is_paused = False

        def _delete_error(msg):
            self.audio.parler(self.audio._text("delete_cancel"))
            self.is_registering = False
            self.audio.is_paused = False

        self.audio.parler(self.audio._text("delete_ask"))
        import threading, time
        threading.Thread(
            target=lambda: (time.sleep(2.5), self.audio.ecouter_nom_inscription(_delete_success, _delete_error)),
            daemon=True
        ).start()

    def open_registration_vocal(self):
        """Ouvre la création Face ID par commande vocale."""
        if self.engine is None:
            self.audio.parler("Face ID est desactive.")
            return
        if self.is_registering: return
        self.is_registering = True
        self.audio.is_paused = True # Met en pause l'écoute principale
        self.stop_camera() # Libère la cam pour la fenêtre pop-up
        
        # Fenêtre d'inscription (Face ID Style)
        pop = FaceRegistrationWindow(self, self.gallery_path, self.engine, self.audio, self.on_registration_finished)
        pop.grab_set()

    def on_registration_finished(self):
        """Retour à la reconnaissance continue après inscription."""
        self.is_registering = False
        self.audio.is_paused = False # Reprend l'écoute principale
        self.train_ia_thread()
        self.start_camera()
        self.audio.parler(self.audio._text("registration_done"))

    def start_camera(self):
        if not self.camera_enabled:
            self.update_status_ui("Camera disabled via environment variable", color="#F39C12")
            self.is_camera_running = False
            return
        camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            self.update_status_ui(f"⚠️ Caméra {camera_index} indisponible", color="#FF5555")
            self.is_camera_running = False
            return
        self.is_camera_running = True
        self.update_video()

    def stop_camera(self):
        self.is_camera_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def train_ia_thread(self):
        if self.engine is not None:
            threading.Thread(target=self.engine.train_on_gallery, args=(self.gallery_path,), daemon=True).start()

    def update_video(self):
        if not self.is_camera_running or self.capture is None: return
        ret, frame = self.capture.read()
        if ret:
            if self.camera_mirror:
                frame = cv2.flip(frame, 1) # Effet miroir
            faces = self.engine.detect_faces(frame) if self.engine is not None else []

            # Optional object detection using YOLO
            detected_objects = []
            if self.yolo_model is not None:
                try:
                    yres = self.yolo_model(frame, verbose=False)[0]
                    if yres.boxes:
                        for box in yres.boxes:
                            label = yres.names[int(box.cls[0])]
                            confidence = float(box.conf[0])
                            if confidence > self.OBJECT_CONFIDENCE:
                                detected_objects.append(label)
                                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                                cv2.putText(
                                    frame,
                                    f"{label} {confidence:.2f}",
                                    (x1, max(y1 - 8, 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 180, 255),
                                    2,
                                )
                except Exception as e:
                    print(f"⚠️ YOLO inference error: {e}")

            noms_presents = set()
            for (x,y,w,h) in faces:
                name, score = self.engine.predict(frame[y:y+h, x:x+w])
                is_known = name not in ["Inconnu", "Accès Refusé", "AccÃ¨s RefusÃ©"]
                color = (30, 144, 255) if is_known else (255, 69, 0)
                
                # Rectangle Design (Coins seulement pour l'élégance)
                l = 30
                cv2.line(frame, (x, y), (x + l, y), color, 4)
                cv2.line(frame, (x, y), (x, y + l), color, 4)
                cv2.line(frame, (x + w, y), (x + w - l, y), color, 4)
                cv2.line(frame, (x + w, y), (x + w, y + l), color, 4)
                cv2.line(frame, (x, y + h), (x + l, y + h), color, 4)
                cv2.line(frame, (x, y + h), (x, y + h - l), color, 4)
                cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, 4)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, 4)
                
                if is_known:
                    noms_presents.add(name)
                    cv2.putText(frame, f"IDENTIFIE: {name}", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, "CIBLE INCONNUE", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if noms_presents:
                self.audio.annoncer_noms(list(noms_presents), cooldown_seconds=45)

            dist_now = self._get_distance()
            has_distance = dist_now is not None
            out_of_range = not has_distance or dist_now >= self.OOR_THRESHOLD

            # Announce objects using the same distance/cooldown rules as Automatic_detector.py
            unique_objects = sorted(set(detected_objects))
            object_spoken = False
            if unique_objects:
                obj_msg = ", ".join(unique_objects)
                spoken = (
                    f"{obj_msg}, out of range"
                    if out_of_range
                    else f"{obj_msg}, {dist_now} centimeters"
                )
                label_parts = [obj_msg]
                if noms_presents:
                    label_parts.append(", ".join(sorted(noms_presents)))
                self._set_arduino_label(" | ".join(label_parts))

                now = time.time()
                in_range = has_distance and self.SPEAK_MIN_CM <= dist_now < self.SPEAK_MAX_CM
                label_changed = spoken != self._last_object_announcement
                cooldown_ok = now - self._last_object_announcement_ts >= self.SPEAK_COOLDOWN
                tts_busy = getattr(self.audio, "_tts_busy", False)
                if in_range and not tts_busy and (label_changed or cooldown_ok):
                    self._last_object_announcement = spoken
                    self._last_object_announcement_ts = now
                    self.audio.parler(spoken)
                    object_spoken = True
            elif noms_presents:
                self._set_arduino_label(", ".join(sorted(noms_presents)))
            else:
                self._set_arduino_label("Scanning...")

            distance_color = "#E74C3C"
            if not has_distance:
                port, last_raw, last_rx_ts = self._get_serial_status()
                if last_raw:
                    age = time.time() - last_rx_ts
                    dist_text = f"Waiting distance: {last_raw[:20]} ({age:.0f}s)"
                else:
                    dist_text = f"No distance RX on {port or 'serial'}"
                distance_label = "Distance: no Arduino data"
                distance_color = "#E74C3C"
            else:
                dist_text = "Out of range" if out_of_range else f"Distance: {dist_now} cm"
                distance_label = dist_text
                if dist_now <= self.PROXIMITY_DANGER_CM:
                    distance_color = "#E74C3C"
                elif dist_now < self.PROXIMITY_WARNING_CM:
                    distance_color = "#F1C40F"
                else:
                    distance_color = "#2ECC71"

            self.update_distance_ui(distance_label, distance_color)
            self._handle_proximity_warning(dist_now, has_distance, object_spoken=object_spoken)

            hw_connected = self.arduino is not None and self.arduino.is_open
            hw_text = "Arduino: OK" if hw_connected else "Arduino: disconnected"
            hw_color = (0, 255, 0) if hw_connected else (0, 0, 255)
            cv2.putText(frame, dist_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, hw_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hw_color, 2)

            # Affichage UI
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(img, img, size=(800, 600))
            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img

        self.after(20, self.update_video)

    def on_closing(self):
        self._closing = True
        self.stop_camera()
        self._close_arduino()
        self.destroy()

if __name__ == "__main__":
    app = BlindyApp()
    app.mainloop()

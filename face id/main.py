import cv2
import os
import threading
import time
import customtkinter as ctk
from PIL import Image

# Modules locaux (Blindy Core)
from face_engine import FaceEngine
from audio_manager import AudioManager
from registration_window import FaceRegistrationWindow

class BlindyApp(ctk.CTk):
    """L'intelligence artificielle Blindy : Interface Minimaliste et Contrôle Vocal."""
    def __init__(self):
        super().__init__()

        self.title("Blindy AI - L'œil intelligent")
        self.geometry("1100x800")
        self.configure(fg_color="#0D0D0D") # Fond ultra sombre pro

        self.gallery_path = "galerie"
        self.engine = FaceEngine()
        self.audio = AudioManager(callback_command=self.handle_voice_command)
        self.audio.callback_status = self.update_status_ui # Nouveau callback
        
        # État de l'IA
        self.is_camera_running = False
        self.is_listening = False
        self.is_registering = False
        
        self.setup_ui()
        self.train_ia_thread()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Raccourci clavier 'I' pour l'inscription (Backup si la voix ne capte pas)
        self.bind("<i>", lambda e: self.open_registration_vocal())
        self.bind("<I>", lambda e: self.open_registration_vocal())

        # Démarrage automatique des systèmes
        self.after(500, self.start_systems)

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header futuriste
        self.lbl_blindy = ctk.CTkLabel(self, text="BLINDY AI", font=ctk.CTkFont(size=30, weight="bold", family="Orbitron"), text_color="#1E90FF")
        self.lbl_blindy.grid(row=0, column=0, pady=(20, 5))
        
        self.lbl_status = ctk.CTkLabel(self, text="Dites 'Blindy' + votre question", font=ctk.CTkFont(size=16), text_color="#2ECC71")
        self.lbl_status.grid(row=1, column=0, pady=(0, 20))

        # Affichage Vidéo Central
        self.video_frame = ctk.CTkFrame(self, fg_color="black", border_width=2, border_color="#1E90FF")
        self.video_frame.grid(row=2, column=0, padx=50, pady=20, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_display = ctk.CTkLabel(self.video_frame, text="CHARGEMENT...", font=ctk.CTkFont(size=20), text_color="#333333")
        self.video_display.grid(row=0, column=0, sticky="nsew")

    def update_status_ui(self, msg, color="#1E90FF"):
        """Met à jour le texte sous le titre en temps réel."""
        self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=color))

    def start_systems(self):
        """Lance les fils d'écoute vocale et la caméra."""
        if not self.is_listening:
            self.is_listening = True
            threading.Thread(target=self.audio.ecouter_commande, daemon=True).start()
            self.lbl_status.configure(text="🎤 Micro actif - Dites 'Blindy' pour parler", text_color="#2ECC71")

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
            from cloud_gallery import delete_person
            import os
            deleted = delete_person(name)
            if deleted > 0:
                self.audio.parler(f"Très bien. Le profil de {name} a été définitivement supprimé.")
                # Nettoyage du cache pour forcer un nettoyage complet
                if os.path.exists("encodings_cache.yml"): os.remove("encodings_cache.yml")
                if os.path.exists("names_cache.pkl"): os.remove("names_cache.pkl")
                # Relancer l'entraînement
                self.train_ia_thread()
            else:
                self.audio.parler(f"Je n'ai pas trouvé le profil de {name}.")
            self.is_registering = False
            self.audio.is_paused = False

        def _delete_error(msg):
            self.audio.parler("Annulation de la suppression.")
            self.is_registering = False
            self.audio.is_paused = False

        self.audio.parler("Quel profil dois-je effacer ?")
        import threading, time
        threading.Thread(
            target=lambda: (time.sleep(2.5), self.audio.ecouter_nom_inscription(_delete_success, _delete_error)),
            daemon=True
        ).start()

    def open_registration_vocal(self):
        """Ouvre la création Face ID par commande vocale."""
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
        self.audio.parler("Profil enregistré. Je reprends ma surveillance.")

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.is_camera_running = True
        self.update_video()

    def stop_camera(self):
        self.is_camera_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def train_ia_thread(self):
        threading.Thread(target=self.engine.train_on_gallery, args=(self.gallery_path,), daemon=True).start()

    def update_video(self):
        if not self.is_camera_running or self.capture is None: return
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1) # Effet miroir
            faces = self.engine.detect_faces(frame)

            noms_presents = set()
            for (x,y,w,h) in faces:
                name, score = self.engine.predict(frame[y:y+h, x:x+w])
                color = (30, 144, 255) if name not in ["Inconnu", "Accès Refusé"] else (255, 69, 0)
                
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
                
                if color == (30, 144, 255):
                    noms_presents.add(name)
                    cv2.putText(frame, f"IDENTIFIE: {name}", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, "CIBLE INCONNUE", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if noms_presents: self.audio.annoncer_noms(list(noms_presents))

            # Affichage UI
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(img, img, size=(800, 600))
            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img

        self.after(20, self.update_video)

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = BlindyApp()
    app.mainloop()

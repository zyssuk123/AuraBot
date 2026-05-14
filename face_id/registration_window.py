import cv2
import os
import io
import time
import threading
import customtkinter as ctk
from PIL import Image
from cloud_gallery import upload_photo

class FaceRegistrationWindow(ctk.CTkToplevel):
    """Fenêtre de scanner Blindy (Style Face ID futuriste)."""
    def __init__(self, master, gallery_path, engine, audio_mgr, on_complete):
        super().__init__(master)
        self.title("Blindy - Enregistrement Biométrique")
        self.geometry("800x700")
        self.configure(fg_color="#0D0D0D")
        
        self.gallery_path = gallery_path
        self.engine = engine
        self.audio = audio_mgr
        self.on_complete = on_complete
        
        self.capture = None
        self.is_scanning = False
        self.is_camera_running = False
        self.photos_taken = 0
        self.max_photos = 30 # Plus de photos pour plus de précision
        self.last_capture_time = 0
        
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Raccourci Touche 'I' pour relancer si échec
        self.bind("<Key-i>", lambda e: self.ask_name_vocal())
        self.bind("<Key-I>", lambda e: self.ask_name_vocal())
        
        self.after(500, self.start_camera)
        
        # Lancement de la demande de nom par la voix immédiatement
        self.after(1500, self.ask_name_vocal)

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        
        self.lbl_title = ctk.CTkLabel(self, text="INITIALISATION FACE ID", font=ctk.CTkFont(size=24, weight="bold", family="Orbitron"), text_color="#1E90FF")
        self.lbl_title.grid(row=0, column=0, pady=20)
        
        self.lbl_info = ctk.CTkLabel(self, text="Dites votre nom après le signal sonore...", font=ctk.CTkFont(size=14), text_color="#555555")
        self.lbl_info.grid(row=1, column=0, pady=5)

        self.entry_name = ctk.CTkEntry(self, width=300, placeholder_text="Nom du profil", fg_color="black", text_color="#1E90FF", border_color="#1E90FF")
        self.entry_name.grid(row=2, column=0, pady=10)

        self.video_display = ctk.CTkLabel(self, text="L'ŒIL S'OUVRE...", width=640, height=480, fg_color="black")
        self.video_display.grid(row=3, column=0, pady=20)
        
        self.progress = ctk.CTkProgressBar(self, width=640, progress_color="#1E90FF")
        self.progress.set(0)
        self.progress.grid(row=4, column=0, pady=10)
        
        self.btn_retry = ctk.CTkButton(self, text="REESSAYER (VOIX)", command=self.ask_name_vocal, fg_color="transparent", border_width=1, border_color="#1E90FF", text_color="#1E90FF")
        self.btn_retry.grid(row=5, column=0, pady=5)

    def ask_name_vocal(self):
        self.lbl_info.configure(text="Préparation de l'écoute...", text_color="#1E90FF")
        self.audio.parler("Dites votre nom après le bip.")
        # On attend plus longtemps (3.5s) que l'IA finisse de parler totalement
        self.after(3500, self.listen_name)

    def listen_name(self):
        self.lbl_info.configure(text="🎙️ PARLEZ MAINTENANT !", text_color="#FF4500") # Orange vif pour l'action
        self.entry_name.configure(border_color="#FF4500")
        
        def success(txt):
            self.entry_name.delete(0, 'end')
            self.entry_name.insert(0, txt.title())
            self.entry_name.configure(border_color="#2ECC71") # Vert si OK
            self.lbl_info.configure(text=f"✅ Nom capturé: {txt}. Scan du visage...", text_color="#2ECC71")
            self.start_scan_vocal()
            
        def error(msg):
            self.entry_name.configure(border_color="red")
            self.lbl_info.configure(text=f"❌ {msg}. Réessai automatique...", text_color="red")
            self.audio.parler("Désolée, je n'ai pas entendu. Veuillez recommencer.")
            # Relancer automatiquement après 4 secondes pour laisser le temps au message de finir
            self.after(4000, self.ask_name_vocal)

        threading.Thread(target=self.audio.ecouter_nom_inscription, args=(success, error), daemon=True).start()

    def start_scan_vocal(self):
        self.photos_taken = 0
        self.is_scanning = True
        self.audio.parler("C'est parti. Regardez bien dans le cercle bleu et bougez la tête lentement.")

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.is_camera_running = True
        self.update_camera()

    def update_camera(self):
        if not self.is_camera_running or self.capture is None: return
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            faces = self.engine.detect_faces(frame)
            
            # Design Futuriste (Guide Ovale)
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            color_guide = (255, 144, 30) # Bleu Blindy
            cv2.ellipse(frame, (cx, cy), (130, 180), 0, 0, 360, color_guide, 2, cv2.LINE_AA)
            
            if self.is_scanning and len(faces) > 0:
                big_f = max(faces, key=lambda r: r[2]*r[3])
                x,y,w,h = big_f
                t = time.time()
                if t - self.last_capture_time > 0.2:
                    name = self.entry_name.get().strip().replace(" ", "_").capitalize()
                    
                    # Encode la ROI en JPEG (mémoire, pas de fichier local)
                    roi = frame[max(0,y-30):y+h+30, max(0,x-30):x+w+30]
                    _, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    image_bytes = buf.tobytes()
                    
                    self.photos_taken += 1
                    self.last_capture_time = t
                    idx = self.photos_taken
                    
                    # Upload vers Vercel Blob en arrière-plan (non-bloquant)
                    def _upload(n=name, i=idx, b=image_bytes):
                        try:
                            upload_photo(n, i, b)
                        except Exception as e:
                            print(f"[Blob] Erreur upload photo {i}: {e}")
                    threading.Thread(target=_upload, daemon=True).start()
                    
                    self.progress.set(self.photos_taken / self.max_photos)
                    self.lbl_info.configure(
                        text=f"☁️ Upload {self.photos_taken}/{self.max_photos}...",
                        text_color="#1E90FF"
                    )
                    
                    if self.photos_taken >= self.max_photos:
                        self.is_scanning = False
                        self.lbl_info.configure(text="✅ PROFIL CLOUD ENREGISTRÉ", text_color="#2ECC71")
                        self.after(2500, self.finish)
                # Indicateur Scan
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(img, img, size=(640, 480))
            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img

        if self.is_camera_running:
            self.after(20, self.update_camera)

    def finish(self):
        self.on_closing()
        if self.on_complete: self.on_complete()

    def on_closing(self):
        self.is_camera_running = False
        if self.capture: self.capture.release()
        self.destroy()

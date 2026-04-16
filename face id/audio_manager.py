import os
import threading
import time
import queue
import numpy as np
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
from dotenv import load_dotenv
import datetime
import geocoder
import requests

# Chargement auto
load_dotenv()


class AudioManager:
    """Gestionnaire Blindy : IA Hybride (Gemini Cloud + Local Context).
    
    AMÉLIORATIONS v2.0 :
    - VAD (Voice Activity Detection) : détecte quand tu FINIS de parler
    - Seuil adaptatif : calibrage automatique selon ton micro
    - pyttsx3 Singleton : init une seule fois (x5 plus rapide)
    - TTS Queue : file d'attente vocale, zéro blocage
    - Gestion des conflits micro : pause propre entre écoute et inscription
    """

    # ── Constantes VAD ────────────────────────────────────────────────────────
    SAMPLE_RATE      = 16000   # Hz
    CHUNK_DURATION   = 0.3     # secondes par chunk d'analyse
    CHUNK_SAMPLES    = int(SAMPLE_RATE * CHUNK_DURATION)  # 4800 samples
    SILENCE_TIMEOUT  = 1.5     # secondes de silence avant de valider la parole
    MIN_SPEECH_DURATION = 0.4  # durée min (s) pour qu'un mot soit valide
    MAX_SPEECH_DURATION = 10.0 # durée max (s) pour éviter un enregistrement infini
    ENERGY_MULTIPLIER   = 2.5  # seuil = moyenne_bruit * ENERGY_MULTIPLIER

    def __init__(self, callback_command=None):
        print("[Blindy] Initialisation du système audio v2.0...")
        self.callback_command = callback_command
        self.callback_status  = None
        self.listening_continuous = False
        self.is_paused  = False
        self._mic_busy  = threading.Lock()   # protège l'accès physique au micro

        # ── IA / Config ───────────────────────────────────────────────────────
        self.nom_ia      = "Abdel AI"
        self.version     = "v2.0-VAD"
        self.ma_location = "Recherche..."
        self.api_key     = os.getenv("GEMINI_API_KEY")
        self.arduino     = None
        self.derniere_vue    = {}
        self.salutations_faites = {}

        # ── TTS : singleton + queue ───────────────────────────────────────────
        self._tts_engine = None
        self._tts_queue  = queue.Queue()
        self._tts_busy   = False
        self._init_tts_engine()
        threading.Thread(target=self._tts_worker, daemon=True).start()

        # ── Calibrage du bruit de fond ────────────────────────────────────────
        self.energy_threshold = 500  # valeur par défaut, sera calibrée
        threading.Thread(target=self._calibrer_bruit, daemon=True).start()

        # ── GPS ───────────────────────────────────────────────────────────────
        threading.Thread(target=self.initialiser_gps, daemon=True).start()

        print(f"[Blindy] {self.nom_ia} {self.version} prête.")

    # =========================================================================
    # TTS — Synthèse Vocale (Singleton + Queue)
    # =========================================================================

    def _init_tts_engine(self):
        """Crée le moteur pyttsx3 une seule fois."""
        try:
            self._tts_engine = pyttsx3.init()
            voices = self._tts_engine.getProperty('voices')
            for v in voices:
                if "FRA" in v.name.upper() or "FR" in v.name.upper():
                    self._tts_engine.setProperty('voice', v.id)
                    break
            self._tts_engine.setProperty('rate', 175)
            print("[TTS] Moteur pyttsx3 initialisé (singleton).")
        except Exception as e:
            print(f"[TTS] Erreur init pyttsx3 : {e}")
            self._tts_engine = None

    def _tts_worker(self):
        """Consommateur de la file TTS — lit les phrases dans l'ordre."""
        while True:
            texte = self._tts_queue.get()
            self._tts_busy = True
            try:
                if self._tts_engine:
                    self._tts_engine.say(texte)
                    self._tts_engine.runAndWait()
                else:
                    self._fallback_tts(texte)
            except Exception as e:
                print(f"[TTS] Erreur lecture : {e}")
                # Réinitialise le moteur si planté
                self._init_tts_engine()
                try:
                    self._fallback_tts(texte)
                except Exception:
                    pass
            finally:
                self._tts_busy = False
                self._tts_queue.task_done()

    def _fallback_tts(self, texte):
        """Fallback PowerShell si pyttsx3 est KO."""
        import subprocess
        t_safe = texte.replace("'", " ").replace('"', ' ')
        cmd = (
            f'powershell -Command "Add-Type -AssemblyName System.Speech; '
            f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Speak(\'{t_safe}\')"'
        )
        subprocess.run(cmd, shell=True, check=False, timeout=15)

    def parler(self, texte):
        """Envoie une phrase dans la file TTS (non-bloquant)."""
        if not texte:
            return
        print(f"📢 {self.nom_ia} : {texte}")
        self._tts_queue.put(texte)

    # =========================================================================
    # Calibrage du seuil de bruit
    # =========================================================================

    def _calibrer_bruit(self):
        """Mesure le bruit ambiant pendant 1s pour définir le seuil VAD."""
        try:
            print("[VAD] Calibrage du bruit de fond...")
            with self._mic_busy:
                samples = sd.rec(
                    self.SAMPLE_RATE,          # 1 seconde
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype='int16'
                )
                sd.wait()
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            self.energy_threshold = max(300, rms * self.ENERGY_MULTIPLIER)
            print(f"[VAD] Seuil calibré : {self.energy_threshold:.0f} (bruit RMS={rms:.0f})")
        except Exception as e:
            print(f"[VAD] Calibrage échoué, seuil par défaut : {e}")

    # =========================================================================
    # VAD — Enregistrement intelligent
    # =========================================================================

    def _calculer_energie(self, chunk: np.ndarray) -> float:
        """RMS du chunk audio."""
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def _enregistrer_avec_vad(self, max_silence=None, max_duration=None) -> bytes | None:
        """
        Enregistre jusqu'à ce que la voix s'arrête.
        
        Retourne les bytes audio ou None si rien capté.
        """
        if max_silence is None:
            max_silence = self.SILENCE_TIMEOUT
        if max_duration is None:
            max_duration = self.MAX_SPEECH_DURATION

        chunks_voix    = []
        silence_cumule = 0.0
        duree_parole   = 0.0
        en_parole      = False

        try:
            with self._mic_busy:
                stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype='int16',
                    blocksize=self.CHUNK_SAMPLES
                )
                with stream:
                    start_time = time.time()

                    while True:
                        elapsed = time.time() - start_time
                        if elapsed > max_duration + 2:
                            break  # sécurité absolue

                        chunk, _ = stream.read(self.CHUNK_SAMPLES)
                        energie = self._calculer_energie(chunk)

                        if energie > self.energy_threshold:
                            # On entend de la parole
                            en_parole       = True
                            silence_cumule  = 0.0
                            duree_parole   += self.CHUNK_DURATION
                            chunks_voix.append(chunk.copy())

                            if duree_parole >= max_duration:
                                break  # trop long, on coupe
                        else:
                            if en_parole:
                                # Silence après parole
                                silence_cumule += self.CHUNK_DURATION
                                chunks_voix.append(chunk.copy())  # garde la fin naturelle

                                if silence_cumule >= max_silence:
                                    break  # fin de phrase détectée ✅
                            # Sinon : silence avant parole → on attend sans rien enregistrer

        except Exception as e:
            print(f"[VAD] Erreur stream : {e}")
            return None

        if not en_parole or duree_parole < self.MIN_SPEECH_DURATION:
            return None  # bruit ou trop court

        audio_concat = np.concatenate(chunks_voix, axis=0)
        return audio_concat.tobytes()

    # =========================================================================
    # Écoute principale (boucle continue)
    # =========================================================================

    def ecouter_commande(self):
        """Écoute active avec VAD — ne capte que quand tu parles vraiment."""
        r = sr.Recognizer()
        self.listening_continuous = True

        print("[Blindy] Écoute VAD active.")

        while self.listening_continuous:
            if self.is_paused or self._tts_busy or self._mic_busy.locked():
                time.sleep(0.2)
                continue

            if self.callback_status:
                self.callback_status("🎤 EN ATTENTE", "#2ECC71")

            raw = self._enregistrer_avec_vad()
            if raw is None:
                continue  # silence ou bruit → on ignore

            if self.callback_status:
                self.callback_status("⌛ ANALYSE...", "#F39C12")

            try:
                audio_data = sr.AudioData(raw, self.SAMPLE_RATE, 2)
                texte = r.recognize_google(audio_data, language="fr-FR").lower()

                if not texte or len(texte) < 2:
                    continue

                print(f"👂 Capté : {texte}")

                # ── LOGIQUE DE COMMANDES ──────────────────────────────────────

                # 1. INSCRIPTION / FACE ID
                trig_face = [
                    "face id", "inscription", "enregistrer", "visage",
                    "heidi", "ajouter quelqu'un", "nouveau profil"
                ]
                if any(x in texte for x in trig_face):
                    self.callback_command("REGISTRATION")
                    continue

                # 1.bis SUPPRESSION BIOMÉTRIQUE (Pour les utilisateurs)
                trig_del = ["supprimer", "oublie", "effacer", "retirer"]
                if any(x in texte for x in trig_del) and any(y in texte for y in ["visage", "profil", "personne", "nom"]):
                    self.callback_command("DELETE_PERSON")
                    continue

                # 2. NAVIGATION
                if any(x in texte for x in ["aller", "direction", "itinéraire", "guide", "vers"]):
                    dest = texte.split("à")[-1].split("vers")[-1].strip()
                    if len(dest) > 2:
                        self.demander_navigation(dest)
                    continue

                # 3. QUESTIONS / IA
                triggers_ia = [
                    "blindy", "abdel", "météo", "actualité", "infos",
                    "qui est", "c'est quoi", "comment", "pourquoi", "quand"
                ]
                is_ia_call = any(x in texte for x in triggers_ia) or "?" in texte
                if not is_ia_call:
                    mots = texte.split()
                    if any(m in ["ai", "ia"] for m in mots):
                        is_ia_call = True

                if is_ia_call:
                    self.demander_ia(texte)

            except sr.UnknownValueError:
                pass  # parole inaudible → silencieux, pas d'erreur
            except sr.RequestError as e:
                print(f"[SR] Erreur réseau : {e}")
                if self.callback_status:
                    self.callback_status("❌ Pas de réseau", "#E74C3C")
                time.sleep(2)
            except Exception as e:
                print(f"[Blindy] Erreur inattendue : {e}")
                time.sleep(0.2)

    # =========================================================================
    # Écoute d'un nom (inscription)
    # =========================================================================

    def ecouter_nom_inscription(self, callback_success, callback_error):
        """Écoute un prénom pour l'inscription avec VAD (fin naturelle)."""
        # Attente que la boucle principale libère le micro
        for _ in range(20):
            if not self._mic_busy.locked():
                break
            time.sleep(0.15)

        raw = self._enregistrer_avec_vad(max_silence=1.0, max_duration=5.0)

        if raw is None:
            callback_error("Nom non détecté — veuillez réessayer")
            return

        r = sr.Recognizer()
        try:
            audio_data = sr.AudioData(raw, self.SAMPLE_RATE, 2)
            texte = r.recognize_google(audio_data, language="fr-FR").lower()
        except sr.UnknownValueError:
            callback_error("Je n'ai pas compris votre nom")
            return
        except sr.RequestError:
            callback_error("Erreur de connexion internet")
            return

        # Nettoyage du prénom
        for mot in ["mon nom est", "je m'appelle", "appelle-moi", "le petit signal", "signal", "bip"]:
            texte = texte.replace(mot, "")
        texte = texte.strip()

        if len(texte) < 2:
            callback_error("Nom vide ou invalide")
        else:
            callback_success(texte)

    # =========================================================================
    # IA Gemini
    # =========================================================================

    def demander_ia(self, question):
        """Requête Gemini en arrière-plan."""
        if not self.api_key:
            self.parler("Clé API manquante dans le fichier point env.")
            return

        def _ask():
            try:
                maintenant = datetime.datetime.now().strftime("%H:%M")
                instruction = (
                    f"Tu es {self.nom_ia}, assistant personnel IA. "
                    f"Lieu : {self.ma_location}. Heure : {maintenant}. "
                    f"Réponds très brièvement en français (15 mots max). "
                )
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/gemini-flash-latest:generateContent?key={self.api_key}"
                )
                data = {"contents": [{"parts": [{"text": instruction + question}]}]}
                response = requests.post(url, json=data, timeout=15)
                if response.status_code == 200:
                    rep = response.json()['candidates'][0]['content']['parts'][0]['text']
                    rep = rep.replace("*", "").strip()
                    self.parler(rep)
                else:
                    print(f"[Gemini] Erreur {response.status_code}: {response.text}")
                    self.parler("Connexion au cerveau interrompue.")
            except Exception as e:
                print(f"[Gemini] Erreur : {e}")
                self.parler("Désolé, erreur interne.")

        threading.Thread(target=_ask, daemon=True).start()

    def demander_navigation(self, destination):
        def _route():
            try:
                instr = f"Donne un itinéraire ultra-court de {self.ma_location} vers {destination}. 1 phrase."
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/gemini-flash-latest:generateContent?key={self.api_key}"
                )
                data = {"contents": [{"parts": [{"text": instr}]}]}
                response = requests.post(url, json=data, timeout=15)
                if response.status_code == 200:
                    txt = response.json()['candidates'][0]['content']['parts'][0]['text'].replace("*", "")
                    self.parler(txt)
            except Exception:
                self.parler("Erreur de guidage.")
        threading.Thread(target=_route, daemon=True).start()

    # =========================================================================
    # GPS & Salutations
    # =========================================================================

    def initialiser_gps(self):
        try:
            g = geocoder.ip('me')
            if g.city:
                self.ma_location = f"{g.city}, {g.country}"
        except Exception:
            self.ma_location = "Marrakech, Maroc"

    def annoncer_noms(self, noms):
        if not noms:
            return
        maintenant = time.time()
        a_saluer = [
            n for n in noms
            if maintenant - self.derniere_vue.get(n, 0) > 300
        ]
        for n in a_saluer:
            self.derniere_vue[n] = maintenant
        if a_saluer:
            self.parler(f"Bonjour {', '.join(a_saluer)}. Que puis-je faire pour vous ?")

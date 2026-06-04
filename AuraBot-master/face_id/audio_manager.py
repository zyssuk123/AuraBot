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
    LANGUAGE_OPTIONS = {
        "fr-FR": {
            "names": ("french", "francais", "français", "fra", "fr-"),
            "words": ("bonjour", "salut", "merci", "visage", "enregistrer", "supprimer", "profil", "itineraire", "itinéraire"),
            "reply": "français",
        },
        "en-US": {
            "names": ("english", "eng", "en-"),
            "words": ("hello", "thanks", "face", "register", "delete", "profile", "where", "what", "how"),
            "reply": "English",
        },
        "ar-MA": {
            "names": ("arabic", "arabe", "ara", "ar-", "ar_"),
            "words": (
                "سلام", "السلام", "مرحبا", "شكرا", "وجه", "وش", "تسجيل",
                "سجل", "حفظ", "احفظ", "حذف", "امسح", "ملف", "اسم",
                "فين", "أين", "اين", "شنو", "ماذا", "كيف", "طريق", "اتجاه",
                "اذهب", "الى", "إلى",
            ),
            "reply": "Arabic",
        },
    }
    LOCALIZED_MESSAGES = {
        "no_gemini": {
            "fr-FR": "Aucune clé Gemini valide n'est configurée. Ajoute une clé dans le fichier .env pour activer l'IA.",
            "en-US": "No valid Gemini key is configured. Add one in the .env file to activate AI.",
            "ar-MA": "مفتاح Gemini غير مضبوط. أضف المفتاح في ملف .env لتفعيل الذكاء الاصطناعي.",
        },
        "no_navigation": {
            "fr-FR": "Aucune clé Gemini valide n'est configurée. Le guidage vocal est désactivé.",
            "en-US": "No valid Gemini key is configured. Voice navigation is disabled.",
            "ar-MA": "مفتاح Gemini غير مضبوط. التوجيه الصوتي متوقف.",
        },
        "network": {
            "fr-FR": "Connexion au service Gemini interrompue. Vérifie ta connexion internet.",
            "en-US": "Gemini connection failed. Check your internet connection.",
            "ar-MA": "الاتصال بخدمة Gemini انقطع. تأكد من الإنترنت.",
        },
        "internal_error": {
            "fr-FR": "Erreur interne du service Gemini.",
            "en-US": "Internal Gemini service error.",
            "ar-MA": "حدث خطأ داخلي في خدمة Gemini.",
        },
        "navigation_error": {
            "fr-FR": "Erreur de guidage.",
            "en-US": "Navigation error.",
            "ar-MA": "حدث خطأ في التوجيه.",
        },
        "name_not_detected": {
            "fr-FR": "Nom non détecté — veuillez réessayer",
            "en-US": "Name not detected. Please try again.",
            "ar-MA": "لم أسمع الاسم. حاول مرة أخرى.",
        },
        "name_not_understood": {
            "fr-FR": "Je n'ai pas compris votre nom",
            "en-US": "I did not understand your name.",
            "ar-MA": "لم أفهم اسمك.",
        },
        "internet_error": {
            "fr-FR": "Erreur de connexion internet",
            "en-US": "Internet connection error.",
            "ar-MA": "خطأ في الاتصال بالإنترنت.",
        },
        "invalid_name": {
            "fr-FR": "Nom vide ou invalide",
            "en-US": "Empty or invalid name.",
            "ar-MA": "الاسم فارغ أو غير صالح.",
        },
        "greeting": {
            "fr-FR": "Bonjour {names}. Que puis-je faire pour vous ?",
            "en-US": "Hello {names}. What can I do for you?",
            "ar-MA": "مرحبا {names}. كيف يمكنني مساعدتك؟",
        },
        "ask_name": {
            "fr-FR": "Dites votre nom après le bip.",
            "en-US": "Say your name after the beep.",
            "ar-MA": "قل اسمك بعد الصافرة.",
        },
        "retry_name": {
            "fr-FR": "Désolée, je n'ai pas entendu. Veuillez recommencer.",
            "en-US": "Sorry, I did not hear you. Please try again.",
            "ar-MA": "آسف، لم أسمعك. حاول مرة أخرى.",
        },
        "scan_start": {
            "fr-FR": "C'est parti. Regardez bien dans le cercle bleu et bougez la tête lentement.",
            "en-US": "Let's start. Look inside the blue circle and move your head slowly.",
            "ar-MA": "لنبدأ. انظر داخل الدائرة الزرقاء وحرك رأسك ببطء.",
        },
        "ready": {
            "fr-FR": "Blindy est prêt.",
            "en-US": "Blindy is ready.",
            "ar-MA": "بليندي جاهز.",
        },
        "delete_ask": {
            "fr-FR": "Quel profil dois-je effacer ?",
            "en-US": "Which profile should I delete?",
            "ar-MA": "أي ملف تريد أن أحذف؟",
        },
        "delete_done": {
            "fr-FR": "Très bien. Le profil de {name} a été définitivement supprimé.",
            "en-US": "Done. The profile for {name} has been deleted.",
            "ar-MA": "تم. حذفت ملف {name} نهائيا.",
        },
        "delete_missing": {
            "fr-FR": "Je n'ai pas trouvé le profil de {name}.",
            "en-US": "I did not find the profile for {name}.",
            "ar-MA": "لم أجد ملف {name}.",
        },
        "delete_cancel": {
            "fr-FR": "Annulation de la suppression.",
            "en-US": "Deletion canceled.",
            "ar-MA": "تم إلغاء الحذف.",
        },
        "registration_done": {
            "fr-FR": "Profil enregistré. Je reprends ma surveillance.",
            "en-US": "Profile saved. I am resuming monitoring.",
            "ar-MA": "تم حفظ الملف. سأعود للمراقبة.",
        },
    }

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
        self.api_key     = self._sanitize_api_key(os.getenv("GEMINI_API_KEY"))
        if self._is_placeholder_gemini_key(self.api_key):
            self.api_key = ""
        self.arduino     = None
        self.derniere_vue    = {}
        self.salutations_faites = {}
        self._conversation_until = 0.0
        self.current_language = "fr-FR"

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

    @staticmethod
    def _sanitize_api_key(value):
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _is_placeholder_gemini_key(cls, value):
        key = cls._sanitize_api_key(value).lower()
        placeholders = {
            "",
            "your_gemini_api_key_here",
            "your_api_key",
            "replace_with_your_api_key",
            "changeme",
            "<your_api_key>",
        }
        return key in placeholders

    @classmethod
    def _is_valid_gemini_key(cls, value):
        key = cls._sanitize_api_key(value)
        return bool(key) and not cls._is_placeholder_gemini_key(key)

    @staticmethod
    def _get_gemini_error_message(status_code):
        messages = {
            401: "La clé Gemini est invalide ou expirée. Vérifie ta clé dans le fichier .env.",
            403: "L'accès à Gemini est refusé. Vérifie la configuration de ta clé.",
            429: "Le service Gemini est temporairement surchargé. Réessaie dans quelques instants.",
            500: "Le service Gemini est indisponible pour le moment. Réessaie plus tard.",
            502: "Le service Gemini a rencontré une erreur de passerelle. Réessaie plus tard.",
            503: "Le service Gemini est momentanément indisponible. Réessaie plus tard.",
            504: "Le service Gemini a pris trop de temps à répondre. Réessaie plus tard.",
        }
        return messages.get(status_code, "Le service Gemini est indisponible pour le moment. Réessaie plus tard.")

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
            item = self._tts_queue.get()
            if isinstance(item, tuple):
                texte, language = item
            else:
                texte, language = item, None
            self._tts_busy = True
            try:
                self._speak_text(texte, language)
            except Exception as e:
                print(f"[TTS] Erreur lecture : {e}")
            finally:
                self._tts_busy = False
                self._tts_queue.task_done()

    def _speak_text(self, texte, language=None):
        language = language or self._guess_language(texte)
        if os.name == "nt":
            try:
                self._sapi_tts(texte, language)
                return
            except Exception as e:
                print(f"[TTS] SAPI a echoue : {e}")

        try:
            self._pyttsx3_once(texte, language)
            return
        except Exception as e:
            print(f"[TTS] pyttsx3 a échoué : {e}")

        if os.name == "nt":
            self._sapi_tts(texte, language)

    def _pyttsx3_once(self, texte, language=None):
        """Speak with a fresh pyttsx3 engine. Slower, but reliable for AI replies."""
        engine = pyttsx3.init()
        try:
            voices = engine.getProperty('voices')
            voice_id = self._find_pyttsx3_voice(voices, language)
            if voice_id:
                engine.setProperty('voice', voice_id)
            engine.setProperty('rate', 175)
            engine.say(texte)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _sapi_tts(self, texte, language=None):
        """Use Windows SAPI directly. More reliable than pyttsx3 in worker threads."""
        import subprocess
        t_safe = texte.replace("'", "''")
        lang_prefix = (language or self.current_language or "fr-FR").split("-")[0].lower()
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$prefix = '{lang_prefix}'; "
            "$voice = $s.GetInstalledVoices() | "
            "Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name.ToLower().StartsWith($prefix) } | "
            "Select-Object -First 1; "
            "if ($voice) { $s.SelectVoice($voice.VoiceInfo.Name) }; "
            "$s.SetOutputToDefaultAudioDevice(); "
            f"$s.Speak('{t_safe}')"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(details or f"PowerShell exited with {result.returncode}")

    def parler(self, texte, language=None):
        """Envoie une phrase dans la file TTS (non-bloquant)."""
        if not texte:
            return
        language = language or self._guess_language(texte)
        self.current_language = language
        print(f"[Voix] {self.nom_ia} : {texte}")
        self._tts_queue.put((texte, language))

    def beep(self, frequency=1000, duration_ms=350):
        """Play a short cue before recording user speech."""
        try:
            if os.name == "nt":
                import winsound

                winsound.Beep(frequency, duration_ms)
            else:
                print("\a", end="", flush=True)
                time.sleep(duration_ms / 1000)
        except Exception as e:
            print(f"[Audio] Beep unavailable: {e}")

    def _find_pyttsx3_voice(self, voices, language=None):
        language = language or self.current_language or "fr-FR"
        options = self.LANGUAGE_OPTIONS.get(language, self.LANGUAGE_OPTIONS["fr-FR"])
        for voice in voices or []:
            haystack = " ".join(
                str(getattr(voice, attr, "")) for attr in ("id", "name", "languages")
            ).lower()
            if any(marker in haystack for marker in options["names"]):
                return voice.id
        return None

    def _guess_language(self, texte):
        if any("\u0600" <= ch <= "\u06ff" for ch in texte):
            return "ar-MA"
        lowered = texte.lower()
        scores = {}
        for lang, options in self.LANGUAGE_OPTIONS.items():
            scores[lang] = sum(1 for word in options["words"] if word in lowered)
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang
        return self.current_language or "fr-FR"

    def _text(self, key, language=None, **values):
        language = language or getattr(self, "current_language", "fr-FR")
        messages = self.LOCALIZED_MESSAGES.get(key, {})
        template = messages.get(language) or messages.get("fr-FR") or ""
        return template.format(**values)

    def _extract_destination(self, texte):
        lowered = texte.lower()
        markers = [
            "navigate to", "directions to", "go to", "route to", "vers", "à",
            "إلى", "الى", "نحو", "لـ", "ل ",
        ]
        for marker in markers:
            if marker in lowered:
                return lowered.split(marker, 1)[-1].strip()
        return lowered.strip()

    def _recognize_user_speech(self, recognizer, audio_data):
        best_text = None
        best_lang = self.current_language or "fr-FR"
        best_confidence = -1.0
        languages = [best_lang] + [lang for lang in self.LANGUAGE_OPTIONS if lang != best_lang]

        for language in languages:
            try:
                result = recognizer.recognize_google(audio_data, language=language, show_all=True)
            except sr.UnknownValueError:
                continue
            if not result:
                continue
            alternatives = result.get("alternative", []) if isinstance(result, dict) else []
            if not alternatives:
                continue

            text = alternatives[0].get("transcript", "").strip().lower()
            confidence = float(alternatives[0].get("confidence", 0.0))
            if confidence == 0.0 and text:
                confidence = 0.5
            if self._guess_language(text) == language:
                confidence += 0.2
            if confidence > best_confidence:
                best_text = text
                best_lang = language
                best_confidence = confidence

        if not best_text:
            best_text = recognizer.recognize_google(audio_data, language=best_lang).lower()

        self.current_language = self._guess_language(best_text) if best_text else best_lang
        return best_text, self.current_language

    def _activate_conversation(self, seconds=12.0):
        """Allow the next short user sentence to be treated as an AI question."""
        self._conversation_until = time.time() + seconds

    def _is_conversation_active(self):
        return time.time() < self._conversation_until

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
                texte, language = self._recognize_user_speech(r, audio_data)

                if not texte or len(texte) < 2:
                    continue

                print(f"[Ecoute] Capté : {texte}")

                # ── LOGIQUE DE COMMANDES ──────────────────────────────────────

                # 1. INSCRIPTION / FACE ID
                trig_face = [
                    "face id", "inscription", "enregistrer", "visage",
                    "heidi", "ajouter quelqu'un", "nouveau profil",
                    "register", "save face", "add face", "new profile",
                    "سجل وجه", "تسجيل وجه", "احفظ وجه", "حفظ وجه", "ملف جديد",
                    "شخص جديد", "اضف وجه", "أضف وجه",
                ]
                if any(x in texte for x in trig_face):
                    self.callback_command("REGISTRATION")
                    continue

                # 1.bis SUPPRESSION BIOMÉTRIQUE (Pour les utilisateurs)
                trig_del = [
                    "supprimer", "oublie", "effacer", "retirer", "delete", "forget", "remove",
                    "احذف", "حذف", "امسح", "انسى", "نسي", "شيل",
                ]
                target_words = [
                    "visage", "profil", "personne", "nom", "face", "profile", "person", "name",
                    "وجه", "ملف", "شخص", "اسم",
                ]
                if any(x in texte for x in trig_del) and any(y in texte for y in target_words):
                    self.callback_command("DELETE_PERSON")
                    continue

                # 2. NAVIGATION
                nav_triggers = [
                    "aller", "direction", "itinéraire", "guide", "vers",
                    "go to", "directions", "route", "navigate to",
                    "اذهب", "روح", "طريق", "اتجاه", "دلني", "إلى", "الى", "فين",
                ]
                if any(x in texte for x in nav_triggers):
                    dest = self._extract_destination(texte)
                    if len(dest) > 2:
                        self.demander_navigation(dest, language=language)
                    continue

                # 3. QUESTIONS / IA
                # If the sentence reached this point, it was not a local command.
                # Send it to Gemini so every captured question gets an answer.
                self._activate_conversation()
                self.demander_ia(texte, language=language)

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
            callback_error(self._text("name_not_detected"))
            return

        r = sr.Recognizer()
        try:
            audio_data = sr.AudioData(raw, self.SAMPLE_RATE, 2)
            texte, _ = self._recognize_user_speech(r, audio_data)
        except sr.UnknownValueError:
            callback_error(self._text("name_not_understood"))
            return
        except sr.RequestError:
            callback_error(self._text("internet_error"))
            return

        # Nettoyage du prénom
        for mot in [
            "mon nom est", "je m'appelle", "appelle-moi", "le petit signal", "signal", "bip",
            "my name is", "i am", "call me",
            "اسمي هو", "اسمي", "أنا اسمي", "انا اسمي", "سميتي", "ناديني", "الاسم", "الصافرة",
        ]:
            texte = texte.replace(mot, "")
        texte = texte.strip()

        if len(texte) < 2:
            callback_error(self._text("invalid_name"))
        else:
            callback_success(texte)

    # =========================================================================
    # IA Gemini
    # =========================================================================

    def demander_ia(self, question, language=None):
        """Requête Gemini en arrière-plan."""
        if not self._is_valid_gemini_key(self.api_key):
            if self.callback_status:
                self.callback_status("⚠️ IA désactivée", "#F39C12")
            self.parler(self._text("no_gemini", language), language=language)
            return

        print(f"[IA] Question envoyée : {question}")

        def _ask():
            try:
                maintenant = datetime.datetime.now().strftime("%H:%M")
                reply_language = self.LANGUAGE_OPTIONS.get(
                    language or self.current_language,
                    self.LANGUAGE_OPTIONS["fr-FR"],
                )["reply"]
                instruction = (
                    f"Tu es {self.nom_ia}, assistant personnel IA. "
                    f"Lieu : {self.ma_location}. Heure : {maintenant}. "
                    f"Réponds dans la même langue que l'utilisateur: {reply_language}. "
                    f"Si la langue est Arabic, réponds en arabe clair, pas en transcription latine. "
                    f"Reste très bref (15 mots max). "
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
                    self.parler(rep, language=language)
                else:
                    print(f"[Gemini] Erreur {response.status_code}: {response.text}")
                    self.parler(self._get_gemini_error_message(response.status_code))
            except requests.RequestException as e:
                print(f"[Gemini] Erreur réseau : {e}")
                self.parler(self._text("network", language), language=language)
            except Exception as e:
                print(f"[Gemini] Erreur : {e}")
                self.parler(self._text("internal_error", language), language=language)

        threading.Thread(target=_ask, daemon=True).start()

    def demander_navigation(self, destination, language=None):
        if not self._is_valid_gemini_key(self.api_key):
            self.parler(self._text("no_navigation", language), language=language)
            return

        def _route():
            try:
                reply_language = self.LANGUAGE_OPTIONS.get(
                    language or self.current_language,
                    self.LANGUAGE_OPTIONS["fr-FR"],
                )["reply"]
                instr = f"Donne un itinéraire ultra-court de {self.ma_location} vers {destination}. 1 phrase. Réponds en {reply_language}. Si Arabic, utilise l'écriture arabe."
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/gemini-flash-latest:generateContent?key={self.api_key}"
                )
                data = {"contents": [{"parts": [{"text": instr}]}]}
                response = requests.post(url, json=data, timeout=15)
                if response.status_code == 200:
                    txt = response.json()['candidates'][0]['content']['parts'][0]['text'].replace("*", "")
                    self.parler(txt, language=language)
                else:
                    print(f"[Gemini] Erreur navigation {response.status_code}: {response.text}")
                    self.parler(self._get_gemini_error_message(response.status_code))
            except requests.RequestException as e:
                print(f"[Gemini] Erreur réseau navigation : {e}")
                self.parler(self._text("network", language), language=language)
            except Exception:
                self.parler(self._text("navigation_error", language), language=language)
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

    def annoncer_noms(self, noms, cooldown_seconds=45):
        if not noms:
            return
        maintenant = time.time()
        a_saluer = [
            n for n in noms
            if maintenant - self.derniere_vue.get(n, 0) > cooldown_seconds
        ]
        for n in a_saluer:
            self.derniere_vue[n] = maintenant
        if a_saluer:
            self._activate_conversation()
            self.parler(self._text("greeting", names=", ".join(a_saluer)))

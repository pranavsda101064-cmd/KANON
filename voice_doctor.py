"""
KANON AI — Voice Doctor
Run: python voice_doctor.py

Requires:
  pip install speechrecognition gtts playsound==1.2.2 pyaudio requests
  Ollama running: https://ollama.com/download
  Backend running: python main.py
"""

import os
import sys
import time
import threading
import requests
from datetime import datetime

# ── Optional imports with friendly errors ──
try:
    import speech_recognition as sr
except ImportError:
    print("❌ Missing: pip install speechrecognition pyaudio")
    sys.exit(1)

try:
    from gtts import gTTS
except ImportError:
    print("❌ Missing: pip install gtts")
    sys.exit(1)

try:
    from playsound import playsound
except ImportError:
    print("❌ Missing: pip install playsound==1.2.2")
    sys.exit(1)

# ── Config ──────────────────────────────────
BACKEND_URL = "http://localhost:8001"
TEMP_AUDIO  = "temp_kanon_tts.mp3"

LANGUAGES = {
    "1": {"name": "English",  "code": "en", "speech": "en-IN"},
    "2": {"name": "Kannada",  "code": "kn", "speech": "kn-IN"},
    "3": {"name": "Tamil",    "code": "ta", "speech": "ta-IN"},
}

WAKE_WORDS = {
    "en": ["report", "done", "finished", "that's all", "i'm done", "generate report"],
    "kn": ["ವರದಿ", "ಮುಗಿದಿದೆ", "ಸಾಕು", "ಇಷ್ಟು ಸಾಕು"],
    "ta": ["அறிக்கை", "முடிந்தது", "போதும்", "அவ்வளவுதான்"],
}

GREETINGS = {
    "en": "Hello! I'm Dr. KANON AI. Could you please tell me your age and gender, then describe what brings you in today?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ಡಾ. KANON AI. ದಯವಿಟ್ಟು ನಿಮ್ಮ ವಯಸ್ಸು ಮತ್ತು ಲಿಂಗ ತಿಳಿಸಿ, ನಂತರ ಇಂದಿನ ಸಮಸ್ಯೆ ವಿವರಿಸಿ.",
    "ta": "வணக்கம்! நான் டாக்டர் KANON AI. தயவுசெய்து உங்கள் வயது மற்றும் பாலினம் கூறுங்கள், பின்னர் இன்றைய பிரச்சனை விளக்குங்கள்.",
}

WAKE_INSTRUCTIONS = {
    "en": "Say 'report' or 'done' when you want your medical report.",
    "kn": "ವರದಿ ಪಡೆಯಲು 'ವರದಿ' ಅಥವಾ 'ಮುಗಿದಿದೆ' ಎಂದು ಹೇಳಿ.",
    "ta": "அறிக்கை பெற 'அறிக்கை' அல்லது 'முடிந்தது' என்று சொல்லுங்கள்.",
}

REPORT_READY = {
    "en": "Your medical report is ready. Please check the screen.",
    "kn": "ನಿಮ್ಮ ವೈದ್ಯಕೀಯ ವರದಿ ಸಿದ್ಧವಾಗಿದೆ. ದಯವಿಟ್ಟು ಪರದೆ ನೋಡಿ.",
    "ta": "உங்கள் மருத்துவ அறிக்கை தயாராக உள்ளது. திரையில் பாருங்கள்.",
}


class VoiceDoctor:
    def __init__(self):
        self.recognizer  = sr.Recognizer()
        self.microphone  = sr.Microphone()
        self.lang_code   = "en"
        self.lang_name   = "English"
        self.speech_code = "en-IN"
        self.session_id  = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.speaking    = False

        print("🎤 Calibrating microphone... (stay quiet for 2 seconds)")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Microphone ready.\n")

    def choose_language(self):
        print("\n🌐 Select language / ಭಾಷೆ ಆಯ್ಕೆ / மொழி தேர்வு:")
        print("   1. English")
        print("   2. ಕನ್ನಡ (Kannada)")
        print("   3. தமிழ் (Tamil)")
        while True:
            choice = input("Enter 1, 2, or 3: ").strip()
            if choice in LANGUAGES:
                lang = LANGUAGES[choice]
                self.lang_code   = lang["code"]
                self.lang_name   = lang["name"]
                self.speech_code = lang["speech"]
                print(f"\n✅ Language: {self.lang_name}\n")
                break
            print("Invalid. Enter 1, 2, or 3.")

    def speak(self, text: str):
        """Text-to-speech in selected language (non-blocking)."""
        def _run():
            self.speaking = True
            try:
                tts = gTTS(text=text, lang=self.lang_code, slow=False)
                tts.save(TEMP_AUDIO)
                playsound(TEMP_AUDIO)
                if os.path.exists(TEMP_AUDIO):
                    os.remove(TEMP_AUDIO)
            except Exception as e:
                print(f"⚠️  TTS error: {e}")
            finally:
                self.speaking = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()  # wait for speech to finish before listening

    def listen(self) -> str | None:
        """Listen for speech and return transcribed text."""
        with self.microphone as source:
            print("🎙️  Listening... (speak now)")
            try:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
                print("   Processing...", end="", flush=True)
                text = self.recognizer.recognize_google(audio, language=self.speech_code)
                print(f" ✅\n👤 You: \"{text}\"")
                return text
            except sr.WaitTimeoutError:
                print("\n⏰ No speech detected.")
                return None
            except sr.UnknownValueError:
                print("\n❓ Couldn't understand. Please repeat.")
                return None
            except sr.RequestError as e:
                print(f"\n⚠️  Speech recognition error: {e}")
                return None

    def send_message(self, message: str) -> dict:
        """Send message to backend Ollama endpoint."""
        try:
            r = requests.post(
                f"{BACKEND_URL}/ollama/chat",
                data={
                    "session_id": self.session_id,
                    "message":    message,
                    "language":   self.lang_code,
                },
                timeout=120,
            )
            return r.json()
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": "Cannot connect to backend. Is main.py running?"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def save_report(self, report: str, health_score: int):
        """Save report to file and print it."""
        print("\n" + "="*60)
        print(report)
        print("="*60)

        filename = f"KANON_Report_{self.lang_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"KANON AI Medical Consultation Report\n")
            f.write(f"Language: {self.lang_name}\n")
            f.write(f"Health Score: {health_score}/100\n")
            f.write("="*50 + "\n")
            f.write(report)
        print(f"\n✅ Report saved: {filename}")
        return filename

    def run(self):
        # Check backend
        try:
            r = requests.get(f"{BACKEND_URL}/ollama/status", timeout=5)
            status = r.json()
            if not status.get("available"):
                print("\n⚠️  Ollama is not running!")
                print("   Download: https://ollama.com/download")
                print(f"   Then run: ollama pull {status.get('model', 'llama3.1:8b')}")
                print("   Falling back to Gemini if configured...\n")
        except Exception:
            print("\n⚠️  Cannot reach backend. Make sure main.py is running.\n")
            sys.exit(1)

        self.choose_language()

        print("\n" + "="*60)
        print(f"🩺  KANON AI VOICE DOCTOR — {self.lang_name}".center(60))
        print("="*60)
        print(f"\n🗣️  {WAKE_INSTRUCTIONS[self.lang_code]}")
        print("   Press Ctrl+C to quit\n")

        greeting = GREETINGS[self.lang_code]
        print(f"🤖 Dr. KANON: {greeting}")
        self.speak(greeting)

        while True:
            user_input = self.listen()
            if user_input is None:
                continue

            print("🤖 Dr. KANON: ", end="", flush=True)
            data = self.send_message(user_input)

            if not data.get("success"):
                err = data.get("message", "Unknown error")
                print(err)
                self.speak(err)
                continue

            reply = data.get("message", "")
            print(reply)
            self.speak(reply)

            if data.get("is_final"):
                # Report generated
                report       = data.get("diagnosis", "")
                health_score = data.get("health_score", 75)
                risk         = data.get("risk_level", "LOW")

                print(f"\n📊 Health Score: {health_score}/100  |  Risk: {risk}")
                self.save_report(report, health_score)
                self.speak(REPORT_READY[self.lang_code])
                break


if __name__ == "__main__":
    doctor = VoiceDoctor()
    try:
        doctor.run()
    except KeyboardInterrupt:
        bye = {
            "en": "\n👋 Consultation ended. Take care!",
            "kn": "\n👋 ಸಮಾಲೋಚನೆ ಮುಗಿದಿದೆ. ನಿಮ್ಮ ಆರೋಗ್ಯ ನೋಡಿಕೊಳ್ಳಿ!",
            "ta": "\n👋 ஆலோசனை முடிந்தது. உடல்நலம் கவனித்துக்கொள்ளுங்கள்!",
        }
        print(bye.get(doctor.lang_code, bye["en"]))

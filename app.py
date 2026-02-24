from flask import Flask, send_from_directory, jsonify, request
from flask_sock import Sock
import os, json, tempfile, base64
import numpy as np
import soundfile as sf
import cv2
import webbrowser
from threading import Timer
import traceback

# ===============================
# OPENAI SETUP
# ===============================
from openai import OpenAI

client = None
if os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
else:
    print("⚠ OPENAI_API_KEY not set. Transcription disabled.")

# ===============================
# AUTO OPEN BROWSER
# ===============================
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
sock = Sock(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ===============================
# STATIC PAGES
# ===============================
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "welcome.html")

@app.route("/publicRom.html")
def public_room():
    return send_from_directory(BASE_DIR, "publicRom.html")

@app.route("/proffesionalRom.html")
def professional_room():
    return send_from_directory(BASE_DIR, "proffesionalRom.html")

@app.route("/teacher_LO&RE.html")
def teacher_page():
    return send_from_directory(BASE_DIR, "teacher_LO&RE.html")

@app.route("/index.html")
def index_page():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/home1.html")
def home1_page():
    return send_from_directory(BASE_DIR, "home1.html")

@app.route("/home")
def home_page():
    return send_from_directory(BASE_DIR, "home.html")

@app.route("/models/<path:filename>")
def serve_models(filename):
    return send_from_directory(os.path.join(BASE_DIR, "models"), filename)

# ===============================
# AUTH — FIXED
# ===============================
@app.route("/auth_teacher", methods=["POST"])
def auth_teacher():
    try:
        if not request.is_json:
            return jsonify({"ok": False, "error": "Invalid request format"}), 400

        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"ok": False, "error": "Username and password required"}), 400

        if username == "admin" and password == "admin123":
            return jsonify({"ok": True}), 200

        return jsonify({"ok": False, "error": "Invalid username or password"}), 401

    except Exception as e:
        print("🔥 AUTH ERROR:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

# ===============================
# STUDENT AUTH (DEMO)
# ===============================
@app.route("/register_student", methods=["POST"])
def register_student():
    try:
        data = request.get_json()
        if not all([data.get("full_name"), data.get("email"), data.get("password")]):
            return jsonify({"ok": False, "error": "All fields required"}), 400

        import uuid
        student_id = str(uuid.uuid4())[:8].upper()
        return jsonify({"ok": True, "student_id": student_id})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

@app.route("/login", methods=["POST"])
def login_student():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if email and password:
            return jsonify({
                "ok": True,
                "student_id": "STU_" + email.split("@")[0].upper()
            })

        return jsonify({"ok": False, "error": "Invalid credentials"}), 401

    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

# ===============================
# FACE DETECTION
# ===============================
@app.route("/detect_face", methods=["POST"])
def detect_face():
    try:
        data = request.get_json()
        img_b64 = data["image"].split(",")[1]

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_b64), np.uint8),
            cv2.IMREAD_COLOR
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face.detectMultiScale(gray, 1.1, 5)

        return jsonify({"ok": True, "count": len(faces)})

    except Exception as e:
        print("FACE ERROR:", e)
        return jsonify({"ok": False, "error": "Detection failed"}), 500

# ===============================
# WHISPER LIVE TRANSCRIPTION (SINGLE & FIXED)
# ===============================
@sock.route("/ws/transcribe")
def ws_transcribe(ws):
    if client is None:
        print("⚠ Whisper disabled (no API key)")
        try:
            while ws.receive() is not None:
                pass
        except:
            pass
        return

    print("🎤 Transcription WebSocket connected")
    buffer = bytearray()

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            if not isinstance(data, (bytes, bytearray)):
                continue

            buffer.extend(data)

            if len(buffer) >= 16000 * 2 * 2:
                audio_np = np.frombuffer(buffer, dtype=np.int16)
                if audio_np.size == 0:
                    buffer.clear()
                    continue

                audio = audio_np.astype(np.float32) / 32768.0

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, 16000, subtype="PCM_16")

                with open(f.name, "rb") as af:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af
                    )

                if result.text.strip():
                    ws.send(json.dumps({"text": result.text}))

                buffer.clear()

    except Exception:
        print("🔥 TRANSCRIPTION ERROR")
        traceback.print_exc()

    finally:
        print("🔌 Transcription WebSocket closed")

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )

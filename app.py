from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import os

app = Flask(__name__)
CORS(app)

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.route("/")
def home():
    return "Server is running 🚀"

@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    if request.method == 'OPTIONS':
        return '', 200

    file = request.files["file"]
    filepath = "temp_audio.webm"
    file.save(filepath)

    segments, _ = model.transcribe(filepath, language="ar")
    text = " ".join([s.text for s in segments])

    os.remove(filepath)
    return jsonify({"text": text})

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
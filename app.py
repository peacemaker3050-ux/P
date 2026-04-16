from flask import Flask, request
from flask_cors import CORS
import whisper
import os

app = Flask(__name__)
CORS(app)
model = whisper.load_model("base")

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
    result = model.transcribe(filepath)
    os.remove(filepath)
    return {"text": result["text"]}

app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            file=("audio.webm", f),
            model="whisper-large-v3",
            language="ar",
            response_format="text"
        )

    os.remove(filepath)
    return jsonify({"text": result})

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
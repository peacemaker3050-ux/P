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

@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    transcript = data.get("text", "")

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"أنت مساعد أكاديمي. لخّص هذه المحاضرة الجامعية بشكل منظم، مع إبراز أهم النقاط والمفاهيم الرئيسية. اكتب الملخص باللغة العربية.\n\nنص المحاضرة:\n{transcript}"}],
        max_tokens=1024
    )
    summary = completion.choices[0].message.content
    return jsonify({"summary": summary})

@app.route("/analyze_schedule", methods=["POST", "OPTIONS"])
def analyze_schedule():
    if request.method == 'OPTIONS':
        return '', 200

    import json
    data = request.get_json()
    image_base64 = data.get("image", "")

    prompt = """أنت محلل جداول دراسية جامعية متخصص.
## قاعدة التصنيف:
- الخلايا ذات الخلفية الزرقاء الممتدة على عرض الجدول = "lecture"
- أي خلية في عمود سكشن معين (حتى لو مكتوب فيها معمل) = "section"
- رقم السكشن من رقم العمود: الأول="1"، الثاني="2"، الثالث="3"
أوقات بصيغة 12 ساعة مع ص/م — تجاهل الخلايا الفارغة — لا تضيف isLab
أرجع JSON فقط بدون أي نص أو backticks:
{"days":{"الأحد":[{"type":"lecture","subject":"","doctor":"","hall":"","from":"9:00 ص","to":"10:30 ص","period":1}]}}"""

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=4000
    )

    raw = completion.choices[0].message.content
    clean = raw.replace('```json', '').replace('```', '').strip()
    result = json.loads(clean)
    return jsonify(result)


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    messages = data.get("messages", [])
    system_prompt = data.get("system", "")

    # Build messages list for Groq (text-only, no image/pdf support in Groq)
    groq_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # If content is a list (multipart), extract text parts only
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)
        groq_messages.append({"role": role, "content": content})

    # Prepend system prompt as first user message if provided (Groq uses system role)
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(groq_messages)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=full_messages,
        max_tokens=1024
    )
    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
from flask import Flask, request, render_template, redirect, url_for
import os
import whisper
from groq import Groq
from gtts import gTTS
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load Whisper model once
model = whisper.load_model("base")

# Groq Clientimport os
groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = None
    summary = None
    audio_path = None

    if request.method == "POST":
        if "audio" not in request.files:
            return "No file part"
        
        file = request.files["audio"]
        if file.filename == "":
            return "No selected file"
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Transcribe
        result = model.transcribe(filepath)
        transcription = result["text"]

        # Summarize using Groq
        prompt = f"Summarize the following text in 1-2 lines:\n\n{transcription}"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768"
        )
        summary = response.choices[0].message.content.strip()

        # Convert to speech
        tts = gTTS(summary)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_path = os.path.join(STATIC_FOLDER, "output.mp3")
        tts.save(audio_path)

        return render_template("index.html",
                               transcription=transcription,
                               summary=summary,
                               audio_url=url_for('static', filename="output.mp3"))
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

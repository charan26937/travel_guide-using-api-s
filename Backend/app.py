from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
import tempfile
import requests
import base64
import os

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MURF_API_KEY = os.environ.get("MURF_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
if not MURF_API_KEY:
    raise RuntimeError("Missing MURF_API_KEY environment variable")

client = genai.Client(api_key=GEMINI_API_KEY)

PROMPTS = {
    "Summary": """
        You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.

Respond ONLY in {language}.
    """,
    "Detailed": """
        You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.

Respond ONLY in {language}.
    """
}


def generate_speech(text, voice_id, locale):
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO"
    }
    response = requests.post(url, headers=headers, json=data, timeout=120)
    if response.status_code == 200:
        with open(temp_audio.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print(f"Murf API error: {response.status_code} - {response.text}")
        raise RuntimeError(f"Murf voice generation failed: {response.status_code}")

    return temp_audio


def generate_description(place, answer_type, language):
    prompt = PROMPTS[answer_type].format(place=place, language=language)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    place = data.get("place")
    answer_type = data.get("answerType")
    language = data.get("language")
    voice_id = data.get("voiceId")
    locale = data.get("locale")

    if not all([place, answer_type, language, voice_id, locale]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        text_description = generate_description(place, answer_type, language)
        audio_path = generate_speech(text_description, voice_id, locale)
        with open(audio_path.name, "rb") as audio_file:
            audio_bytes = audio_file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return jsonify({
            "description": text_description,
            "audioBase64": encoded_audio
        })
    except Exception as exc:
        app.logger.exception("Audio guide generation failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

from flask import Flask, request, jsonify, send_file
import whisper
from transformers import pipeline
from TTS.api import TTS

# Initialize Flask app
app = Flask(__name__)

# Load Whisper for Speech-to-Text
whisper_model = whisper.load_model("base")

# Load LLaMA 2 or GPT-3.5 for text generation
generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# Load Coqui TTS for Text-to-Speech
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Convert speech to text using Whisper."""
    audio_file = request.files["audio"]
    result = whisper_model.transcribe(audio_file)
    return jsonify({"text": result["text"]})

@app.route("/generate", methods=["POST"])
def generate_response():
    """Generate a response using LLaMA 2."""
    user_input = request.json["text"]
    response = generator(user_input, max_length=100, num_return_sequences=1)
    return jsonify({"response": response[0]["generated_text"]})

@app.route("/synthesize", methods=["POST"])
def synthesize_speech():
    """Convert text to speech using Coqui TTS."""
    text = request.json["text"]
    tts.tts_to_file(text=text, file_path="output.wav")
    return send_file("output.wav", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

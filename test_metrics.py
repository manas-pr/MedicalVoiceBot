import time
import requests

# Test Speech-to-Text
def test_stt(audio_file):
    start_time = time.time()
    response = requests.post("http://127.0.0.1:5000/transcribe", files={"audio": audio_file})
    latency = time.time() - start_time
    return response.json()["text"], latency

# Test Generative AI
def test_generate(text):
    start_time = time.time()
    response = requests.post("http://127.0.0.1:5000/generate", json={"text": text})
    latency = time.time() - start_time
    return response.json()["response"], latency

# Test Text-to-Speech
def test_tts(text):
    start_time = time.time()
    response = requests.post("http://127.0.0.1:5000/synthesize", json={"text": text})
    latency = time.time() - start_time
    return len(response.content), latency

# Example usage
audio_file = open("test.wav", "rb")
text, stt_latency = test_stt(audio_file)
print(f"STT Latency: {stt_latency:.2f}s")

response, gen_latency = test_generate(text)
print(f"Generation Latency: {gen_latency:.2f}s")

audio_size, tts_latency = test_tts(response)
print(f"TTS Latency: {tts_latency:.2f}s")

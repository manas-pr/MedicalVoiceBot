import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:5000"

st.title("Medical VoiceBot App")

# Upload audio file
audio_file = st.file_uploader("Upload your voice query", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Transcribe audio to text
    response = requests.post(f"{API_URL}/transcribe", files={"audio": audio_file})
    if response.status_code == 200:
        user_text = response.json()["text"]
        st.write(f"**You said:** {user_text}")

        # Generate response using LLaMA 2
        response = requests.post(f"{API_URL}/generate", json={"text": user_text})
        if response.status_code == 200:
            bot_response = response.json()["response"]
            st.write(f"**VoiceBot says:** {bot_response}")

            # Convert response to speech
            response = requests.post(f"{API_URL}/synthesize", json={"text": bot_response})
            if response.status_code == 200:
                st.audio(response.content, format="audio/wav")

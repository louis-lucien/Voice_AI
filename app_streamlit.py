import streamlit as st
import json
import openai
import os
import subprocess
import langdetect
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import pdfplumber

# Chargement des variables d'environnement
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
model_whisper = whisper.load_model("base")

detected_language = "fr"
VOICE_IDS = {
    "fr": "b6nVfb3l2zshrLZTvqbs",
    "en": "P7x743VjyZEOihNNygQ9",
    "ar": "tavIIPLplRB883FzWU0V"
}

# Extraction et chargement FAQ
@st.cache_data
def load_faq():
    with pdfplumber.open("FAQs_ProBoutik.pdf") as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    sections = text.split("\n")
    faq_data, current_question, answer = [], None, ""
    for section in sections:
        section = section.strip()
        if section.endswith("?"):
            if current_question and answer.strip():
                faq_data.append({"question": current_question, "answer": answer.strip()})
            current_question, answer = section, ""
        else:
            answer += " " + section
    if current_question and answer.strip():
        faq_data.append({"question": current_question, "answer": answer.strip()})
    return faq_data

faq_data = load_faq()

# Langue

def detect_language(text):
    global detected_language
    try:
        detected_lang = langdetect.detect(text)
        detected_language = detected_lang if detected_lang in VOICE_IDS else "fr"
        return detected_language
    except:
        detected_language = "fr"
        return "fr"

# Whisper

def recognize_speech_with_whisper(duration=10):
    fs = 16000
    st.info("Parlez maintenant...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wav.write(temp_audio.name, fs, recording)
        result = model_whisper.transcribe(temp_audio.name)
        return result["text"]

# Assistant OpenAI

def ask_openai_with_faq_via_assistant(question):
    detect_language(question)
    faq_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data])
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        instructions=f"Tu es un assistant de support ProBoutik. Réponds dans la langue détectée ({detected_language}). Voici la FAQ:\n{faq_text}"
    )
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):
        if msg.role == "assistant":
            return msg.content[0].text.value
    return "Je n'ai pas pu générer de réponse."

# Synthèse vocale

def generate_voice_response(text):
    voice_id = VOICE_IDS.get(detected_language, VOICE_IDS["fr"])
    audio_stream = elevenlabs_client.generate(
        text=text,
        voice=voice_id,
        model="eleven_multilingual_v2"
    )
    audio_path = "response.mp3"
    with open(audio_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
    return audio_path

# Interface Streamlit

st.title("Assistant IA ProBoutik")
mode = st.radio("Choisissez le mode :", ["Texte", "Voix"])

if mode == "Texte":
    user_input = st.text_input("Posez votre question :")
    if user_input:
        response = ask_openai_with_faq_via_assistant(user_input)
        st.markdown(f"**Bot :** {response}")
        audio_file = generate_voice_response(response)
        st.audio(audio_file, format="audio/mp3")

elif mode == "Voix":
    if st.button("🎙️ Enregistrer et poser votre question"):
        user_input = recognize_speech_with_whisper()
        st.markdown(f"**Vous avez dit :** {user_input}")
        response = ask_openai_with_faq_via_assistant(user_input)
        st.markdown(f"**Bot :** {response}")
        audio_file = generate_voice_response(response)
        st.audio(audio_file, format="audio/mp3")

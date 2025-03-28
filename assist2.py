import json
import openai
import os
import subprocess
import langdetect
import pdfplumber
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile

print("Importation réussie !")

# Charger les variables d'environnement
load_dotenv()

# Configuration OpenAI et ElevenLabs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

if not ELEVENLABS_API_KEY:
    raise ValueError("La clé API ElevenLabs est manquante. Veuillez la définir dans vos variables d'environnement.")

# Initialisation des clients
client = openai.OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
model_whisper = whisper.load_model("base")

VOICE_IDS = {
    "fr": "b6nVfb3l2zshrLZTvqbs",
    "en": "P7x743VjyZEOihNNygQ9",
    "ar": "tavIIPLplRB883FzWU0V"
}

# Détection de la langue de la question utilisateur (stockée pour la réponse)
detected_language = "fr"

def detect_language(text):
    global detected_language
    try:
        detected_lang = langdetect.detect(text)
        detected_language = detected_lang if detected_lang in VOICE_IDS else "fr"
        return detected_language
    except:
        detected_language = "fr"
        return "fr"

def recognize_speech_with_whisper(duration=10):
    fs = 16000
    print("🎙️ Parlez maintenant...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wav.write(temp_audio.name, fs, recording)
        result = model_whisper.transcribe(temp_audio.name)
        print(f"✅ Reconnu : {result['text']}")
        return result["text"]

def extract_faq_from_pdf(pdf_path):
    faq_data = []
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    sections = text.split("\n")
    current_question = None
    answer = ""
    for section in sections:
        section = section.strip()
        if section.endswith("?"):
            if current_question and answer.strip():
                faq_data.append({"question": current_question, "answer": answer.strip()})
            current_question = section
            answer = ""
        else:
            answer += " " + section
    if current_question and answer.strip():
        faq_data.append({"question": current_question, "answer": answer.strip()})
    return faq_data

def save_faq_to_json(pdf_path, json_path="faq_proboutik.json"):
    faq_data = extract_faq_from_pdf(pdf_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, indent=4, ensure_ascii=False)
    print(f"FAQ sauvegardée dans {json_path}")

save_faq_to_json("FAQs_ProBoutik.pdf")

with open("faq_proboutik.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

print(f"Nombre d'entrées FAQ chargées: {len(faq_data)}")

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
        instructions=f"Tu es un assistant de support ProBoutik. Réponds dans la langue détectée ({detected_language}). Utilise le faqs comme base de connaissance. Si l'information n'est pas disponible, indique que vous n'avez pas cette information et propose de contacter le support.\n" + faq_text
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

def generate_voice_response(text):
    voice_id = VOICE_IDS.get(detected_language, VOICE_IDS["fr"])
    print(f"Langue détectée: {detected_language}")
    print(f"Voix utilisée: {voice_id}")
    audio_stream = elevenlabs_client.generate(
        text=text,
        voice=voice_id,
        model="eleven_multilingual_v2"
    )
    print("Lecture de la réponse vocale...")
    audio_path = "response.mp3"
    with open(audio_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
    subprocess.run(["start", audio_path], shell=True)

def chatbot():
    while True:
        mode = input("Mode ? (1 = texte, 2 = audio, quit = quitter) : ").strip().lower()
        if mode == "quit":
            break

        if mode == "1":
            user_input = input("Vous: ")
        elif mode == "2":
            user_input = recognize_speech_with_whisper()
        else:
            print("❌ Choix invalide.")
            continue

        response = ask_openai_with_faq_via_assistant(user_input)
        print("Bot:", response)
        generate_voice_response(response)

if __name__ == "__main__":
    chatbot()

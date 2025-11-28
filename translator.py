import sounddevice as sd
import numpy as np
import wave
import threading
import whisper
import ollama
import torch
import pyttsx3
import warnings

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ---------------- Config ----------------
FS = 16000
WHISPER_MODEL = "large-v2"
OLLAMA_MODEL = "llama3.2:1b"
# ----------------------------------------

# 1️⃣ Initialize pyttsx3
speak_engine = pyttsx3.init()
voices = speak_engine.getProperty("voices")
speak_engine.setProperty("voice", voices[1].id)  # Change 0 or 1 for male/female
speak_engine.setProperty("rate", 175)           # Words per minute


def say(text):
    speak_engine.say(text)
    speak_engine.runAndWait()


# 2️⃣ Manual start/stop recording
def record_audio(stop_event, frames):
    with sd.InputStream(samplerate=FS, channels=1, dtype='int16') as stream:
        while not stop_event.is_set():
            data, _ = stream.read(1024)
            frames.append(data)


def manual_start_stop_recording(filename):
    input("🎤 Press Enter to START recording...")
    frames = []
    stop_event = threading.Event()
    thread = threading.Thread(target=record_audio, args=(stop_event, frames))
    thread.start()
    input("🔴 Recording... Press Enter to STOP.")
    stop_event.set()
    thread.join()

    if frames:
        audio = np.concatenate(frames, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(audio.tobytes())
        print(f"✅ Saved recording: {filename}")
        return True
    else:
        print("⚠️ No audio recorded!")
        return False

# 3️⃣ Transcribe audio with Whisper (GPU if available)
def transcribe_audio(filename, model_name=WHISPER_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading Whisper model '{model_name}' on {device.upper()}...")
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(filename)
    text = result["text"].strip()
    print(f"📝 Your Question: {text}")
    return text

# 4️⃣ Ask Ollama in English
def ask_ollama_in_english(question, model_name=OLLAMA_MODEL):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": f"Answer the following question in English only:\n\n{question}"
            }]
        )
        answer = response["message"]["content"].strip()
        print(f"🤖 Ollama Answer (English): {answer}")
        return answer
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return ""


# 5️⃣ Continuous Q&A pipeline
def main():
    print("🎙️ Continuous Voice Q&A (say 'exit' to quit)")
    while True:
        wav_file = "question.wav"
        if not manual_start_stop_recording(wav_file):
            continue

        question_text = transcribe_audio(wav_file)
        if not question_text:
            continue

        if question_text.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        answer_text = ask_ollama_in_english(question_text)
        if answer_text:
            say(answer_text)  # Speak the answer


if __name__ == "__main__":
    main()
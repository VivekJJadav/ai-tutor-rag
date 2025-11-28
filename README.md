# AI Tutor - RAG & Voice Server

The AI engine behind the AI Tutor platform. This server handles Retrieval-Augmented Generation (RAG) for textbook Q&A and Voice-to-Text processing.

## 🧠 Features

-   **RAG Pipeline:** Indexes PDF textbooks using FAISS and retrieves relevant context for answering student questions.
-   **Voice Transcription:** Uses OpenAI's Whisper model to transcribe voice input with high accuracy.
-   **Test Generation:** Generates multiple-choice questions based on textbook content.
-   **API:** Provides endpoints for `/transcribe`, `/generate`, and `/generate_test`.

## 🛠️ Tech Stack

-   **Framework:** Flask
-   **AI Models:**
    -   **LLM:** Ollama (Llama 3.2 / Mistral)
    -   **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
    -   **Speech-to-Text:** OpenAI Whisper (`base` model)
-   **Vector DB:** FAISS
-   **PDF Processing:** pdfplumber

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VivekJJadav/ai-tutor-rag.git
    cd ai-tutor-rag
    ```

2.  **Setup Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Dependencies:**
    -   Install [Ollama](https://ollama.com/) and pull a model: `ollama pull llama3.2`
    -   Ensure you have `ffmpeg` installed for audio processing.

4.  **Run the Server:**
    ```bash
    python voice_api_server.py
    ```
    The server will start on `http://127.0.0.1:5002`.

## 🔗 Related Repositories

-   **Main Application:** [ai-tutor](https://github.com/VivekJJadav/ai-tutor) - The Frontend and Django Backend.

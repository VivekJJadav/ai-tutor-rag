import os
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, pdf_path, ollama_host="http://127.0.0.1:11434", model_name="llama3.2:1b"):
        self.pdf_path = pdf_path
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.embed_model = None
        self.index = None
        self.chunks = []
        self.is_initialized = False

    def extract_text_from_pdf(self):
        logger.info(f"📑 Extracting text from PDF: {self.pdf_path}")
        text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    def split_text(self, text, chunk_size=500, overlap=50):
        logger.info("✂️ Splitting text into chunks...")
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def create_embeddings(self, chunks):
        device = "cuda" if False else "cpu"  # SentenceTransformer CPU/GPU
        logger.info(f"🔎 Creating embeddings on {device.upper()}...")
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(chunks, show_progress_bar=True)
        return model, embeddings

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))
        return index

    def initialize(self):
        """Load PDF, create embeddings, and build index."""
        if self.is_initialized:
            logger.info("RAG Pipeline already initialized.")
            return

        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

            text = self.extract_text_from_pdf()
            self.chunks = self.split_text(text)
            self.embed_model, embeddings = self.create_embeddings(self.chunks)
            self.index = self.build_faiss_index(np.array(embeddings))
            self.is_initialized = True
            logger.info("✅ RAG Pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise

    def query_ollama(self, prompt):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            r = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "No response from model.")
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return f"Error querying Ollama: {str(e)}"

    def query(self, question, top_k=3):
        """Answer a question using the RAG pipeline."""
        if not self.is_initialized:
            return "System is not initialized."

        try:
            q_emb = self.embed_model.encode([question], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, top_k)
            retrieved_chunks = [self.chunks[i] for i in I[0]]
            context = "\n".join(retrieved_chunks)

            prompt = f"""You are an assistant that answers using ONLY the context below.
Format your answer as follows:
- Use LaTeX formatting for ALL mathematical formulas (enclosed in $...$).
- Use bullet points for any lists, theories, or theorems.
- Ensure the output is readable and well-structured.

Context:
{context}

Question: {question}
Answer:"""
            answer = self.query_ollama(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            return f"Error processing your question: {e}"

# ------------------------------
# Run standalone
# ------------------------------
if __name__ == "__main__":
    pdf_path = r"Merged_Science_Textbook.pdf"
    rag = RAGPipeline(pdf_path)
    rag.initialize()

    print("\n✅ PDF loaded. You can now ask questions (type 'exit' to quit).\n")
    while True:
        question = input("👉 Enter your question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting RAG system.")
            break
        answer = rag.query(question)
        print("\n📌 Answer:\n", answer, "\n")
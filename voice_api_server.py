#!/usr/bin/env python3
"""
Voice-to-text API server based on translator.py functionality.
Provides endpoints for audio transcription and Ollama responses.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import whisper
import ollama
import torch
import warnings
import logging
from ragsystem import RAGPipeline

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL = "base"
OLLAMA_MODEL = "llama3.2:1b"
PDF_PATH = "Merged_Science_Textbook.pdf"

# Global variables for models
whisper_model = None
rag_pipeline = None

def load_whisper_model():
    """Load Whisper model once at startup."""
    global whisper_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on {device.upper()}...")
        whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        logger.info("Whisper model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        return False

def load_rag_pipeline():
    """Load RAG pipeline once at startup."""
    global rag_pipeline
    try:
        logger.info(f"Initializing RAG pipeline with {PDF_PATH}...")
        rag_pipeline = RAGPipeline(PDF_PATH)
        rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading RAG pipeline: {str(e)}")
        return False

def transcribe_audio_file(audio_file_path):
    """Transcribe audio file using Whisper."""
    global whisper_model

    if whisper_model is None:
        return None, "Whisper model not loaded"

    try:
        result = whisper_model.transcribe(audio_file_path)
        text = result["text"].strip()
        logger.info(f"Transcribed text: {text}")
        return text, None
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None, str(e)

def ask_rag_question(question):
    """Ask question to RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        return None, "RAG pipeline not loaded"
    
    try:
        answer = rag_pipeline.query(question)
        logger.info(f"RAG answer: {answer}")
        return answer, None
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'whisper_loaded': whisper_model is not None,
        'rag_loaded': rag_pipeline is not None and rag_pipeline.is_initialized
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio file."""
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Transcribe audio
            transcribed_text, error = transcribe_audio_file(temp_file_path)

            if error:
                return jsonify({
                    'error': f'Transcription failed: {error}',
                    'success': False
                }), 500

            if not transcribed_text:
                return jsonify({
                    'error': 'No speech detected in audio',
                    'success': False
                }), 400

            return jsonify({
                'transcribed_text': transcribed_text,
                'success': True
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    """Complete voice chat pipeline: transcribe audio + get RAG response."""
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Step 1: Transcribe audio
            transcribed_text, transcribe_error = transcribe_audio_file(temp_file_path)

            if transcribe_error:
                return jsonify({
                    'error': f'Transcription failed: {transcribe_error}',
                    'success': False
                }), 500

            if not transcribed_text:
                return jsonify({
                    'error': 'No speech detected in audio',
                    'success': False
                }), 400

            # Step 2: Get RAG response
            rag_answer, rag_error = ask_rag_question(transcribed_text)

            if rag_error:
                return jsonify({
                    'error': f'RAG failed: {rag_error}',
                    'success': False
                }), 500

            return jsonify({
                'question': transcribed_text,
                'answer': rag_answer,
                'success': True
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error in voice_chat endpoint: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate response for text chat using RAG."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        logger.info(f"Received text chat request: {message[:50]}...")
        
        # Get RAG response
        rag_answer, rag_error = ask_rag_question(message)

        if rag_error:
            return jsonify({
                'error': f'RAG failed: {rag_error}',
                'success': False
            }), 500

        return jsonify({
            'response': rag_answer,
            'success': True
        })

    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/generate_test', methods=['POST'])
def generate_test():
    """Generate a multiple-choice test using RAG."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        chapter_id = data.get('chapter_id')
        topic = data.get('topic', f'Chapter {chapter_id}')
        
        logger.info(f"Generating test for: {topic}")
        
        # Ask RAG to generate a test
        prompt = f"Create a test with 5 multiple-choice questions about '{topic}'. Format the output as a JSON array of objects, where each object has 'question', 'options' (array of 4 strings), 'correct_answer' (string), and 'explanation' (string). Do not include any markdown formatting or other text, just the raw JSON."
        
        rag_answer, rag_error = ask_rag_question(prompt)

        if rag_error:
            return jsonify({'error': f'RAG failed: {rag_error}', 'success': False}), 500

        # Try to parse the JSON response
        import json
        import re
        
        try:
            # Clean up potential markdown code blocks
            cleaned_json = rag_answer.replace('```json', '').replace('```', '').strip()
            # Find the first [ and last ]
            start = cleaned_json.find('[')
            end = cleaned_json.rfind(']') + 1
            if start != -1 and end != -1:
                cleaned_json = cleaned_json[start:end]
                
            questions = json.loads(cleaned_json)
            
            return jsonify({
                'questions': questions,
                'success': True
            })
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from RAG response: {rag_answer}")
            # Fallback: Return raw text if JSON parsing fails
            return jsonify({
                'raw_text': rag_answer,
                'success': True,
                'warning': 'Failed to parse structured quiz'
            })

    except Exception as e:
        logger.error(f"Error in generate_test endpoint: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    logger.info("Starting Voice API server...")

    # Load Whisper model at startup
    whisper_loaded = load_whisper_model()
    
    # Load RAG pipeline at startup
    rag_loaded = load_rag_pipeline()

    if whisper_loaded and rag_loaded:
        logger.info("Voice API server ready! Starting Flask app...")
        app.run(host='127.0.0.1', port=5002, debug=False, threaded=True)
    else:
        logger.error("Failed to load models. Exiting.")
        if not whisper_loaded:
            logger.error("- Whisper model failed to load")
        if not rag_loaded:
            logger.error("- RAG pipeline failed to load")
        exit(1)
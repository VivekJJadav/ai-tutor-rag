#!/bin/bash
# Startup script for the voice API server

echo "Starting Voice-to-Text API Server..."
echo "This will load Whisper model initially..."
echo "Server will run on http://127.0.0.1:5002"
echo ""

cd "/Users/vicky/PycharmProjects/Study Buddy inference"
/Users/vicky/miniconda3/bin/python voice_api_server.py
#!/usr/bin/env python3
"""
Real-time transcription using fine-tuned Whisper model.
Records from microphone and transcribes Nuer speech.
"""

import whisper
import numpy as np
import sounddevice as sd
from pathlib import Path
import torch

# Configuration
MODELS_DIR = Path("/home/buayctrlz/nuer-whisper-asr/models")
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds

# Load model
# For now, use base model. After fine-tuning, point to your model
print("Loading Whisper model...")
model = whisper.load_model("small")

def record_audio(duration, sample_rate=16000):
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype=np.float32)
    sd.wait()
    return audio.flatten()

def transcribe_audio(audio, model):
    """Transcribe audio using Whisper."""
    # Save to temp file (Whisper works with files)
    temp_path = "/tmp/temp_recording.wav"
    import scipy.io.wavfile as wav
    wav.write(temp_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    # Transcribe
    result = model.transcribe(temp_path, language=None)
    return result["text"]

print("\n" + "="*50)
print("NUER REAL-TIME TRANSCRIPTION")
print("="*50)
print("Commands:")
print("  [r] Record and transcribe")
print("  [q] Quit")
print("="*50)

while True:
    cmd = input("\nEnter command [r/q]: ").strip().lower()
    
    if cmd == 'q':
        break
    
    if cmd == 'r':
        # Record
        audio = record_audio(CHUNK_DURATION)
        
        # Transcribe
        print("Transcribing...")
        text = transcribe_audio(audio, model)
        print(f"Transcription: {text}")

print("Goodbye!")

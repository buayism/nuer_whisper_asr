#!/usr/bin/env python3
"""
Fine-tune Whisper on Nuer language data.
Uses Whisper's built-in fine-tuning capabilities.
"""

import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import whisper
from whisper import Whisper

# Configuration
DATA_DIR = Path("/home/buayctrlz/nuer-whisper-asr/data")
MODELS_DIR = Path("/home/buayctrlz/nuer-whisper-asr/models")
TRAIN_DATA = DATA_DIR / "train.json"
TEST_DATA = DATA_DIR / "test.json"

# Model settings
BASE_MODEL = "small"  # Options: tiny, base, small, medium, large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-5

print(f"Device: {DEVICE}")
print(f"Base model: {BASE_MODEL}")

# Load dataset
def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_dataset(TRAIN_DATA)
test_data = load_dataset(TEST_DATA)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Load pre-trained model
print(f"Loading Whisper {BASE_MODEL} model...")
model = whisper.load_model(BASE_MODEL)
model = model.to(DEVICE)

# For fine-tuning, we need to prepare data in Whisper's expected format
class NuerDataset(torch.utils.data.Dataset):
    def __init__(self, data, base_dir=Path("/home/buayctrlz/nuer-speech-to-text")):
        self.data = data
        self.base_dir = base_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = self.base_dir / item["audio"]
        text = item["text"]
        
        # Load audio
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
        
        # Tokenize text
        tokens = whisper.tokenizer.Tokenizer(model.tokenizer).encode(text)
        
        return mel, tokens

print("\nNote: Full fine-tuning requires more setup.")
print("For now, let's test the model on a sample...")

# Test on first sample
sample = train_data[0]
audio_path = Path("/home/buayctrlz/nuer-speech-to-text") / sample["audio"]

print(f"\nTesting on: {sample['text']}")
print(f"Audio: {audio_path}")

if audio_path.exists():
    result = model.transcribe(str(audio_path), language=None)
    print(f"Transcription: {result['text']}")
else:
    print(f"Audio file not found: {audio_path}")

print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("1. Install dependencies: pip install openai-whisper torch torchaudio")
print("2. For proper fine-tuning, use HuggingFace transformers:")
print("   pip install transformers datasets")
print("3. Run: python scripts/finetune_whisper_hf.py")
print("="*50)

#!/usr/bin/env python3
"""
Convert Nuer dataset to Whisper format.
Whisper needs: audio_path, text (transcription)
"""

import json
import os
import random
from pathlib import Path

# Paths
DATA_DIR = Path("/home/buayctrlz/nuer-whisper-asr/data")
NUER_DATASET = DATA_DIR / "nuer_dataset.json"

# Load the dataset
with open(NUER_DATASET, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} utterances from dataset")

# Convert to Whisper format: list of {"audio": path, "text": transcription}
whisper_data = []
for entry in data:
    whisper_data.append({
        "audio": entry["audio"],
        "text": entry["text"],
        "id": entry["id"],
        "duration": entry.get("duration", 0)
    })

# Split: 80% train, 20% test (manual split without sklearn)
random.seed(42)
random.shuffle(whisper_data)
split_idx = int(len(whisper_data) * 0.8)
train_data = whisper_data[:split_idx]
test_data = whisper_data[split_idx:]

print(f"Train: {len(train_data)} utterances")
print(f"Test: {len(test_data)} utterances")

# Save splits
train_path = DATA_DIR / "train.json"
test_path = DATA_DIR / "test.json"
all_path = DATA_DIR / "all.json"

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

with open(all_path, 'w', encoding='utf-8') as f:
    json.dump(whisper_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved:")
print(f"  - {train_path}")
print(f"  - {test_path}")
print(f"  - {all_path}")

# Print sample
print(f"\nSample entry:")
print(json.dumps(train_data[0], indent=2, ensure_ascii=False))

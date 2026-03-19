#!/usr/bin/env python3
"""
Fine-tune Whisper on Nuer data using HuggingFace Transformers.
This is the recommended approach for fine-tuning Whisper.
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# Paths
DATA_DIR = Path("/home/buayctrlz/nuer-whisper-asr/data")
MODELS_DIR = Path("/home/buayctrlz/nuer-whisper-asr/models")
TRAIN_FILE = DATA_DIR / "train.json"
TEST_FILE = DATA_DIR / "test.json"

# Model configuration
MODEL_NAME = "openai/whisper-small"  # Options: tiny, base, small, medium
LANGUAGE = "nuer"  # Custom language code
TASK = "transcribe"

# Training configuration
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
MAX_STEPS = 500
SAVE_STEPS = 100
EVAL_STEPS = 100

print(f"Loading data from {DATA_DIR}")

# Load data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Convert to HuggingFace Dataset format
def prepare_dataset(data, base_dir=Path("/home/buayctrlz/nuer-speech-to-text")):
    """Prepare dataset for HuggingFace."""
    audio_paths = []
    texts = []
    
    for item in data:
        audio_path = str(base_dir / item["audio"])
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)
            texts.append(item["text"])
    
    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": texts
    })
    
    # Cast audio column to Audio type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset

print("Preparing datasets...")
train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)

# Load processor
print(f"Loading Whisper processor: {MODEL_NAME}")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# Preprocessing function
def prepare_dataset_batch(batch):
    """Process a batch of data."""
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label IDs
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch

print("Processing datasets...")
train_dataset = train_dataset.map(prepare_dataset_batch, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset_batch, remove_columns=test_dataset.column_names)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Load model
print(f"Loading model: {MODEL_NAME}")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODELS_DIR / "whisper-nuer"),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nStarting training...")
print("="*50)
trainer.train()

# Save model
print("\nSaving model...")
trainer.save_model(str(MODELS_DIR / "whisper-nuer-final"))
processor.save_pretrained(str(MODELS_DIR / "whisper-nuer-final"))

print("\nTraining complete!")
print(f"Model saved to: {MODELS_DIR / 'whisper-nuer-final'}")

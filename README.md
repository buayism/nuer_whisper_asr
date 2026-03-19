# Nuer Whisper ASR Project

Real-time speech recognition for Nuer language using OpenAI Whisper.

## Project Structure

```
nuer-whisper-asr/
├── data/
│   ├── nuer_dataset.json    # Original data from Kaldi project
│   ├── train.json           # Training split (80%)
│   ├── test.json            # Test split (20%)
│   └── all.json             # Full dataset
├── models/                  # Trained models saved here
├── scripts/
│   ├── convert_to_whisper.py    # Convert data to Whisper format
│   ├── finetune_whisper.py      # Basic fine-tuning (OpenAI whisper)
│   ├── finetune_whisper_hf.py   # Full fine-tuning (HuggingFace)
│   └── transcribe_live.py       # Real-time transcription
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Convert data (already done):
```bash
python scripts/convert_to_whisper.py
```

## Training

### Option 1: Quick Test (OpenAI Whisper)
```bash
python scripts/finetune_whisper.py
```

### Option 2: Full Fine-tuning (HuggingFace) - RECOMMENDED
```bash
python scripts/finetune_whisper_hf.py
```

## Real-time Transcription

```bash
python scripts/transcribe_live.py
```

## Notes

- Audio files are referenced from the original Kaldi project location
- Data was copied from: `/home/buayctrlz/nuer-speech-to-text/`
- 188 utterances total (150 train, 38 test)

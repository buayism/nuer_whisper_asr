cd /home/buayctrlz/nuer-whisper-asr
pip install -r requirements.txt --break-system-packages
python3 scripts/finetune_whisper_hf.py
python3 scripts/transcribe_live.py
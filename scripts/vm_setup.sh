#!/usr/bin/env bash
# One-time VM setup. Run: ssh rogii 'bash -s' < scripts/vm_setup.sh
set -euo pipefail
sudo mkdir -p /mnt/voice && sudo chown azureuser:azureuser /mnt/voice
mkdir -p /mnt/voice/hf /mnt/voice/repo /mnt/voice/data
sudo apt-get install -y -q ffmpeg libsndfile1 >/dev/null
python3 -m venv ~/voice-venv
~/voice-venv/bin/pip install -q --upgrade pip
~/voice-venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
~/voice-venv/bin/pip install -q "transformers>=4.46,<5" "peft>=0.12" "datasets>=2.19,<4" accelerate evaluate jiwer librosa soundfile huggingface_hub pytest fastapi httpx python-multipart
~/voice-venv/bin/python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
# NeMo fork for IndicConformer benchmark (separate venv; failure here is non-blocking)
python3 -m venv ~/nemo-venv || true
~/nemo-venv/bin/pip install -q --upgrade pip || true
~/nemo-venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || true
~/nemo-venv/bin/pip install -q "nemo_toolkit[asr] @ git+https://github.com/AI4Bharat/NeMo.git@nemo-v2" "pytorch_lightning<2.5" "lightning<2.5" jiwer "datasets>=2.19,<4" soundfile || echo "NEMO INSTALL FAILED (non-blocking)"
echo SETUP DONE

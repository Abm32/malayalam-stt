"""python scripts/transcribe.py file.wav [--base MODEL] [--adapter runs/medium_lora/best]"""
import argparse

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from scripts.normalize import normalize


def load(base, adapter, device):
    proc = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, device=0 if device == "cuda" else -1,
                    chunk_length_s=15, stride_length_s=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--base", default="vrclc/Whisper-medium-Malayalam")
    ap.add_argument("--adapter", default="runs/medium_lora/best")
    ap.add_argument("--raw", action="store_true", help="skip normalizer")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = load(a.base, a.adapter or None, device)
    y, _ = librosa.load(a.audio, sr=16000, mono=True)
    text = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "ml", "task": "transcribe"})["text"]
    print(text if a.raw else normalize(text))


if __name__ == "__main__":
    main()

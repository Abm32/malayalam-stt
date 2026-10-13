"""Evaluate Whisper checkpoints (optionally + LoRA adapter) on a saved split.
Usage: python scripts/eval.py --root /mnt/voice/data --split own_test \
         --models openai/whisper-large-v3 vrclc/Whisper-medium-Malayalam [--adapter runs/x/best]
"""
import argparse
import json
from pathlib import Path

import jiwer

from scripts.normalize import normalize


def score(refs, hyps):
    r = [normalize(x) or "<empty>" for x in refs]
    h = [normalize(x) for x in hyps]
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h)}


def transcribe_all(model_id, ds, adapter=None, batch_size=8, device="cuda"):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    hyps = []
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        feats = proc.feature_extractor([x["array"] for x in batch["audio"]], sampling_rate=16000,
                                       return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, language="ml", task="transcribe", num_beams=1, max_new_tokens=440)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    return hyps


def main():
    from datasets import load_from_disk
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/voice/data")
    ap.add_argument("--split", default="own_test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="results/eval.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ds = load_from_disk(Path(a.root) / a.split)
    if a.limit:
        ds = ds.select(range(a.limit))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in a.models:
        hyps = transcribe_all(m, ds, a.adapter)
        s = score(ds["text"], hyps)
        row = {"model": m, "adapter": a.adapter, "split": a.split, "n": len(ds), **s}
        rows.append(row)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        Path(a.out).with_suffix("").mkdir(exist_ok=True)
        (Path(a.out).with_suffix("") / (m.replace("/", "_") + ("_ft" if a.adapter else "") + ".hyp.txt")).write_text(
            "\n".join(f"{r}\t{h}" for r, h in zip(ds["text"], hyps)), encoding="utf-8")
        print(f"{m} {'+'+a.adapter if a.adapter else ''}: WER={s['wer']:.3f} CER={s['cer']:.3f}")
    print("\n| model | adapter | n | WER | CER |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['adapter'] or '-'} | {r['n']} | {r['wer']:.3f} | {r['cer']:.3f} |")


if __name__ == "__main__":
    main()

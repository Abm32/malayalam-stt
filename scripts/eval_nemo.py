"""Benchmark AI4Bharat IndicConformer (stock) on own_test. Runs in ~/nemo-venv.
Usage: ~/nemo-venv/bin/python scripts/eval_nemo.py --root /mnt/voice/data
Model may be gated: accept terms on HF and `huggingface-cli login` first.
"""
import argparse
import json
import tempfile
from pathlib import Path

import soundfile as sf

from scripts.eval import score

MODEL = "ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large"


def main():
    import nemo.collections.asr as nemo_asr
    from datasets import load_from_disk
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/voice/data")
    ap.add_argument("--split", default="own_test")
    ap.add_argument("--out", default="results/eval.jsonl")
    a = ap.parse_args()
    ds = load_from_disk(Path(a.root) / a.split)
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for i, x in enumerate(ds["audio"]):
        p = tmp / f"{i}.wav"
        sf.write(p, x["array"], 16000)
        paths.append(str(p))
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL)
    model.cur_decoder = "rnnt"
    hyps = model.transcribe(paths, batch_size=8, language_id="ml")
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    s = score(ds["text"], hyps)
    row = {"model": MODEL, "adapter": None, "split": a.split, "n": len(ds), **s}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row)


if __name__ == "__main__":
    main()

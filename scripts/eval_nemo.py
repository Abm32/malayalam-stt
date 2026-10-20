"""Benchmark stock NeMo Malayalam models (IndicConformer, Koyal) on own_test. Runs in ~/nemo-venv.
Usage: ~/nemo-venv/bin/python scripts/eval_nemo.py --root /mnt/voice/data
Models are gated: accept terms on HF and `huggingface-cli login` first.
"""
import argparse
import json
import tempfile
from pathlib import Path

import soundfile as sf

from scripts.eval import score

MODELS = ["ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large", "adalat-ai/koyal-ml-120m-1.0"]


def load(repo):
    """NeMo from_pretrained needs the .nemo named after the repo; these repos don't. Fetch + restore."""
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import HfApi, hf_hub_download
    name = next(f for f in HfApi().list_repo_files(repo) if f.endswith(".nemo"))
    path = hf_hub_download(repo, name)
    cls = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    cfg = cls.restore_from(path, return_config=True)
    # koyal ships ctc strategy "greedy_batch"; the AI4Bharat NeMo 1.23 fork only has "greedy"
    if cfg.get("aux_ctc") and cfg.aux_ctc.decoding.strategy == "greedy_batch":
        cfg.aux_ctc.decoding.strategy = "greedy"
    return cls.restore_from(path, override_config_path=cfg)


def main():
    from datasets import load_from_disk
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/voice/data")
    ap.add_argument("--models", nargs="+", default=MODELS)
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
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for repo in a.models:
        model = load(repo)
        if hasattr(model, "cur_decoder"):
            model.cur_decoder = "rnnt"
        kw = {"language_id": "ml"} if "indicconformer" in repo else {}  # koyal is monolingual
        hyps = model.transcribe(paths, batch_size=8, **kw)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        hyps = [h.text if hasattr(h, "text") else h for h in hyps]
        s = score(ds["text"], hyps)
        row = {"model": repo, "adapter": None, "split": a.split, "n": len(ds), **s}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        Path("results/eval").mkdir(exist_ok=True)
        Path(f"results/eval/{repo.replace('/', '_')}.hyp.txt").write_text("\n".join(hyps) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()

"""Transcribe 16 kHz mono WAVs with the winning model (koyal, CER 0.048 on own voice). Runs in ~/nemo-venv.
Usage: ~/nemo-venv/bin/python scripts/transcribe.py a.wav b.wav [--beam]
"""
import argparse

from scripts.eval_nemo import load

MODEL = "adalat-ai/koyal-ml-120m-1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--beam", action="store_true", help="TDT maes beam search (small gain over greedy)")
    a = ap.parse_args()
    model = load(MODEL)
    if a.beam:
        from omegaconf import open_dict
        cfg = model.cfg.decoding
        with open_dict(cfg):
            cfg.strategy, cfg.beam = "maes", {"beam_size": 4}
        model.change_decoding_strategy(cfg)
    hyps = model.transcribe(a.wavs, batch_size=8)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    for p, h in zip(a.wavs, hyps):
        print(p, "\t", h.text if hasattr(h, "text") else h)


if __name__ == "__main__":
    main()

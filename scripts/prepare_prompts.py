"""Build data/prompts.txt from IMaSC (+ Common Voice ml if accessible)."""
import argparse
import random
import re
from pathlib import Path

_ML = re.compile(r"^[ഀ-ൿ\s.,?!‌‍]+$")


def filter_sentences(lines):
    seen, out = set(), []
    for raw in lines:
        s = re.sub(r"\s+", " ", raw).strip()
        if not s or not _ML.match(s):
            continue
        n = len(s.split())
        if n < 5 or n > 20 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def load_sources(hf_token=None):
    """IMaSC text column only (parquet column pruning over HfFileSystem; skips ~4.7 GB of audio)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=hf_token)
    texts = []
    for path in sorted(fs.glob("datasets/thennal/IMaSC/data/*.parquet")):
        with fs.open(path) as f:
            texts += pq.read_table(f, columns=["text"]).column("text").to_pylist()
    try:
        from datasets import load_dataset
        cv = load_dataset("mozilla-foundation/common_voice_17_0", "ml",
                          split="train+validation+test", token=hf_token)
        texts += cv["sentence"]
    except Exception as e:  # gated or offline; IMaSC alone is enough
        print(f"common voice skipped: {type(e).__name__}")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/prompts.txt")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sents = filter_sentences(load_sources())
    random.Random(a.seed).shuffle(sents)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(sents[: a.n]) + "\n", encoding="utf-8")
    print(f"wrote {min(a.n, len(sents))} of {len(sents)} candidates to {a.out}")


if __name__ == "__main__":
    main()

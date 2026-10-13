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
        words = s.split()
        n = len(words)
        if n < 5 or n > 20 or s in seen or len(words[-1].rstrip(".,?!")) <= 2:
            continue
        seen.add(s)
        out.append(s)
    return out


def load_sources(hf_token=None):
    """Malayalam Wikipedia sentences (wikimedia/wikipedia 20231101.ml, text column only).
    Deliberately NOT IMaSC/Common Voice: public Malayalam ASR models train on those sentences,
    which would inflate baseline scores on the own-voice test set."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=hf_token)
    texts = []
    for path in sorted(fs.glob("datasets/wikimedia/wikipedia/20231101.ml/*.parquet")):
        with fs.open(path) as f:
            for article in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                texts += re.split(r"(?<=[.?!])\s+|\n+", article)
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

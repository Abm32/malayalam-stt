"""Build HF datasets: own-voice splits + public mix-in. Run on VM.
Usage: python scripts/prepare_data.py --manifest data/own/manifest.jsonl --root /mnt/voice/data
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from scripts.normalize import normalize

SR = 16000


def split_ids(ids, seed=42):
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_dev = n_test = max(1, n // 10)
    return {"test": sorted(ids[:n_test]),
            "dev": sorted(ids[n_test:n_test + n_dev]),
            "train": sorted(ids[n_test + n_dev:])}


def augment(y, rng):
    """Speed perturb 0.9-1.1 (pitch follows, like sox) + noise at 20-30 dB SNR."""
    import librosa
    f = rng.uniform(0.9, 1.1)
    y = librosa.resample(y, orig_sr=SR, target_sr=int(SR / f))
    snr = rng.uniform(20, 30)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rms / (10 ** (snr / 20)), len(y))
    return np.clip(y + noise, -1, 1).astype(np.float32)


def main():
    from datasets import Audio, Dataset, concatenate_datasets, load_dataset
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/own/manifest.jsonl")
    ap.add_argument("--root", default="/mnt/voice/data")
    ap.add_argument("--own_share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    splits = split_ids(list(by_id), a.seed)

    def own_ds(ids, aug=False):
        recs = []
        for i in ids:
            r = by_id[i]
            if aug:
                y, _ = sf.read(r["path"], dtype="float32")
                recs.append({"audio": {"array": augment(y, rng), "sampling_rate": SR},
                             "text": normalize(r["text"]), "source": "own_aug"})
            else:
                recs.append({"audio": r["path"], "text": normalize(r["text"]), "source": "own"})
        return Dataset.from_list(recs).cast_column("audio", Audio(sampling_rate=SR))

    for name in ("train", "dev", "test"):
        d = own_ds(splits[name])
        d.save_to_disk(root / f"own_{name}")
        print(name, len(d))

    own_train = concatenate_datasets([own_ds(splits["train"]), own_ds(splits["train"], aug=True)])
    n_public = int(len(own_train) * (1 - a.own_share) / a.own_share)

    pubs = []
    imasc = load_dataset("thennal/IMaSC", split="train")
    tcol = "text" if "text" in imasc.column_names else imasc.column_names[0]
    imasc = imasc.rename_column(tcol, "text").shuffle(seed=a.seed).select(range(min(n_public, len(imasc))))
    imasc = imasc.map(lambda x: {"text": normalize(x["text"]), "source": "imasc"},
                      remove_columns=[c for c in imasc.column_names if c not in ("audio", "text")])
    pubs.append(imasc.cast_column("audio", Audio(sampling_rate=SR)))
    try:
        cv = load_dataset("mozilla-foundation/common_voice_17_0", "ml", split="train+validation")
        cv = cv.shuffle(seed=a.seed).select(range(min(n_public // 2, len(cv))))
        cv = cv.map(lambda x: {"text": normalize(x["sentence"]), "source": "cv"},
                    remove_columns=[c for c in cv.column_names if c != "audio"])
        pubs.append(cv.cast_column("audio", Audio(sampling_rate=SR)))
    except Exception as e:
        print("common voice skipped:", type(e).__name__)
    public = concatenate_datasets(pubs).shuffle(seed=a.seed).select(range(min(n_public, sum(len(p) for p in pubs))))

    mix = concatenate_datasets([own_train, public]).shuffle(seed=a.seed)
    mix.save_to_disk(root / "mix_train")
    own_frac = sum(s.startswith("own") for s in mix["source"]) / len(mix)
    print(f"mix_train {len(mix)} own_share={own_frac:.2f}")
    assert 0.3 <= own_frac <= 0.5, "own share outside 30-50%"
    (root / "splits.json").write_text(json.dumps(splits))


if __name__ == "__main__":
    main()

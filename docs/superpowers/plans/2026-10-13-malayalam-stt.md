# Malayalam Own-Voice STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record ~1000 Malayalam sentences in Abhimanyu's voice, benchmark public Malayalam ASR models on a held-out slice, LoRA-fine-tune the best Whisper checkpoint, and produce a before/after WER/CER table.

**Architecture:** Three loosely coupled pieces. (1) A laptop-side recorder (FastAPI + one HTML page) that produces `data/own/manifest.jsonl` + 16 kHz WAVs. (2) Data scripts that build HF `datasets` on disk with deterministic splits and public mix-in. (3) Eval/train/transcribe scripts that run on the Azure VM `rogii-gpu` against those datasets. A single text normalizer is shared by every piece so WER/CER are comparable.

**Tech Stack:** Python 3.10+, FastAPI + uvicorn, ffmpeg, HF `transformers` + `peft` + `datasets<4` + `jiwer`, torch/CUDA on 4x T4, AI4Bharat NeMo fork (benchmark only).

**Spec:** `docs/superpowers/specs/2026-10-13-malayalam-stt-design.md`

## Global Constraints

- Audio everywhere: 16 kHz, mono, WAV (PCM 16-bit).
- Own-voice split 80/10/10 by sentence id, seed 42; test ids never in train.
- Own-voice share of training examples 30–50%.
- Every WER/CER computed after `scripts/normalize.py:normalize()` on both ref and hyp.
- CER is the primary metric; WER reported alongside.
- `data/` is gitignored. `results/` is committed.
- VM: `rogii-gpu`, resource group `ROGII-RG`, ssh host `rogii`, data root `/mnt/voice`, venv `/home/azureuser/voice-venv`. Deallocate when idle.
- Pin `datasets>=2.19,<4` (v4 moved audio decoding to torchcodec; avoid on VM).
- Fine-tune: LoRA via peft on `q_proj`,`v_proj`, fp16, one GPU per run, greedy decoding for eval.

## Review Focus

1. Recorder upload with an empty/0-byte blob (user hits stop instantly): server must reject with 400, not write a manifest line. Test added to Task 3.
2. Manifest resume after a crash mid-write: a truncated last line must not break `/next`. Test added to Task 3 (reader skips malformed lines).
3. Audio longer than 30 s in `transcribe.py`: Whisper truncates silently at 30 s. Chunked pipeline path pinned in Task 9.
4. Normalizer on text containing ZWJ/ZWNJ and old-style chillu sequences: must map to atomic chillu so `ന്‍` and `ൻ` do not count as errors. Test in Task 1.
5. `prepare_data.py` run twice must produce identical splits (determinism), otherwise baseline and fine-tuned evals are on different test sets. Test in Task 5.

---

### Task 1: Repo skeleton + text normalizer

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize(text: str) -> str`. Every later task imports this.

- [ ] **Step 1: Create requirements and gitignore**

`requirements.txt`:
```
fastapi
uvicorn
python-multipart
soundfile
librosa
jiwer
datasets>=2.19,<4
transformers>=4.44
peft>=0.12
accelerate
evaluate
huggingface_hub
pytest
```

`.gitignore`:
```
data/
__pycache__/
*.pyc
.pytest_cache/
runs/
```

- [ ] **Step 2: Write the failing test**

`tests/test_normalize.py`:
```python
from scripts.normalize import normalize


def test_old_chillu_sequence_maps_to_atomic():
    # ന + ് + ZWJ  ->  ൻ (U+0D7B)
    assert normalize("അവന്‍") == "അവൻ"


def test_strips_punctuation_and_zw_chars():
    assert normalize("ഇത്‌ ഒരു, പരീക്ഷണം.") == "ഇത് ഒരു പരീക്ഷണം"


def test_collapses_whitespace_and_lowercases_latin():
    assert normalize("  ഹലോ   World  ") == "ഹലോ world"


def test_idempotent():
    s = "അവന്‍ വന്നു!"
    assert normalize(normalize(s)) == normalize(s)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/abhimanyu/Desktop/digital_agency/voice && python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.normalize'`

- [ ] **Step 4: Write implementation**

`scripts/normalize.py`:
```python
"""Single text normalizer applied to every ref and hyp before WER/CER."""
import re
import unicodedata

# old-style chillu: consonant + virama (U+0D4D) + ZWJ (U+200D) -> atomic chillu
_CHILLU = {
    "ണ്‍": "ൺ",  # ണ്‍ -> ൺ
    "ന്‍": "ൻ",  # ന്‍ -> ൻ
    "ര്‍": "ർ",  # ര്‍ -> ർ
    "ല്‍": "ൽ",  # ല്‍ -> ൽ
    "ള്‍": "ൾ",  # ള്‍ -> ൾ
    "ക്‍": "ൿ",  # ക്‍ -> ൿ
}
_PUNCT = re.compile(r"[^\w\sഀ-ൿ]", re.UNICODE)
_ZW = re.compile("[​‌‍﻿]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for old, new in _CHILLU.items():
        text = text.replace(old, new)
    text = _ZW.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = text.replace("_", " ")
    text = _WS.sub(" ", text).strip()
    return text.lower()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore scripts/__init__.py scripts/normalize.py tests/test_normalize.py
git commit -m "feat(voice): add repo skeleton and Malayalam text normalizer"
```

---

### Task 2: Prompt sentence list

**Files:**
- Create: `scripts/prepare_prompts.py`
- Test: `tests/test_prepare_prompts.py`

**Interfaces:**
- Produces: `data/prompts.txt`, one sentence per line, ~1000 lines, line number = sentence id (1-based). `filter_sentences(lines: list[str]) -> list[str]` pure function.

- [ ] **Step 1: Write the failing test**

`tests/test_prepare_prompts.py`:
```python
from scripts.prepare_prompts import filter_sentences


def test_filter_keeps_only_malayalam_5_to_20_words_deduped():
    lines = [
        "ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു",  # 7 words, keep
        "ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു",  # duplicate
        "ചെറുത്",  # 1 word, drop
        "This is english text with many words here ok",  # latin, drop
        "അവൻ വന്നു " * 12,  # >20 words, drop
    ]
    out = filter_sentences(lines)
    assert out == ["ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare_prompts.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Write implementation**

`scripts/prepare_prompts.py`:
```python
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
    from datasets import load_dataset
    texts = []
    ds = load_dataset("thennal/IMaSC", split="train")
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    texts += ds[col]
    try:
        cv = load_dataset("mozilla-foundation/common_voice_17_0", "ml",
                          split="train+validation+test", token=hf_token)
        texts += cv["sentence"]
    except Exception as e:  # gated or offline; IMaSC alone is enough
        print(f"common voice skipped: {e}")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prepare_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Generate real prompts (laptop)**

Run: `pip install -r requirements.txt && python scripts/prepare_prompts.py`
Expected: `wrote 1000 of N candidates to data/prompts.txt`, N in the thousands. If IMaSC column is not `text`, inspect `ds.column_names` and fix the column pick.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_prompts.py tests/test_prepare_prompts.py
git commit -m "feat(voice): build Malayalam prompt list from IMaSC/Common Voice"
```

---

### Task 3: Recorder server + page

**Files:**
- Create: `record/server.py`
- Create: `record/index.html`
- Test: `tests/test_recorder.py`

**Interfaces:**
- Produces: `data/own/wav/{id}.wav`, `data/own/manifest.jsonl` lines `{"id": int, "path": "data/own/wav/{id}.wav", "text": str, "duration_s": float}`, `data/own/skipped.txt` one id per line.
- `read_manifest(path) -> list[dict]` skips malformed lines.
- `next_id(prompts: list[str], done: set[int], skipped: set[int]) -> int | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_recorder.py`:
```python
import json
from record.server import read_manifest, next_id


def test_read_manifest_skips_truncated_line(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"id": 1, "path": "x", "text": "a", "duration_s": 1.0}) + "\n"
                 + '{"id": 2, "pa')
    rows = read_manifest(p)
    assert [r["id"] for r in rows] == [1]


def test_next_id_skips_done_and_skipped():
    prompts = ["a", "b", "c", "d"]
    assert next_id(prompts, done={1}, skipped={2}) == 3
    assert next_id(prompts, done={1, 3, 4}, skipped={2}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Write the server**

`record/server.py`:
```python
"""Laptop-side prompted recorder. Run: uvicorn record.server:app --port 8765"""
import json
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "prompts.txt"
OWN = ROOT / "data" / "own"
WAV = OWN / "wav"
MANIFEST = OWN / "manifest.jsonl"
SKIPPED = OWN / "skipped.txt"

app = FastAPI()


def read_manifest(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def next_id(prompts, done, skipped):
    for i in range(1, len(prompts) + 1):
        if i not in done and i not in skipped:
            return i
    return None


def _prompts():
    return PROMPTS.read_text(encoding="utf-8").splitlines()


def _skipped():
    return set(int(x) for x in SKIPPED.read_text().split()) if SKIPPED.exists() else set()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/next")
def nxt():
    prompts = _prompts()
    done = {r["id"] for r in read_manifest(MANIFEST)}
    i = next_id(prompts, done, _skipped())
    return {"id": i, "text": prompts[i - 1] if i else None,
            "done": len(done), "total": len(prompts)}


@app.post("/skip")
def skip(id: int = Form(...)):
    OWN.mkdir(parents=True, exist_ok=True)
    with SKIPPED.open("a") as f:
        f.write(f"{id}\n")
    return {"ok": True}


@app.post("/upload")
async def upload(id: int = Form(...), audio: UploadFile = None):
    blob = await audio.read()
    if len(blob) < 1000:
        raise HTTPException(400, "empty recording")
    prompts = _prompts()
    WAV.mkdir(parents=True, exist_ok=True)
    out = WAV / f"{id}.wav"
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as t:
        t.write(blob)
        tmp = t.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out)], check=True)
    Path(tmp).unlink(missing_ok=True)
    info = sf.info(out)
    if info.duration < 0.5:
        out.unlink()
        raise HTTPException(400, "recording under 0.5s")
    row = {"id": id, "path": str(out.relative_to(ROOT)), "text": prompts[id - 1],
           "duration_s": round(info.duration, 3)}
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return JSONResponse(row)
```

- [ ] **Step 4: Write the page**

`record/index.html`:
```html
<!doctype html>
<html lang="ml"><head><meta charset="utf-8"><title>Record</title>
<style>
body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
#text{font-size:2.2rem;line-height:1.5;margin:24px 0;min-height:4em}
button{font-size:1.1rem;padding:10px 18px;margin-right:8px}
#rec.on{background:#c00;color:#fff}
#status{color:#666;margin-top:12px}
</style></head><body>
<div id="prog"></div>
<div id="text"></div>
<button id="rec">Record (Space)</button>
<button id="ok" disabled>Accept (Enter)</button>
<button id="redo" disabled>Re-record (R)</button>
<button id="skip">Skip (S)</button>
<audio id="play" controls style="display:block;margin-top:16px"></audio>
<div id="status"></div>
<script>
let cur=null, mr=null, chunks=[], blob=null, stream=null;
const $=id=>document.getElementById(id);
async function load(){
  const r=await (await fetch('/next')).json();
  cur=r; $('prog').textContent=`${r.done} / ${r.total}`;
  if(!r.id){$('text').textContent='All done.';return;}
  $('text').textContent=r.text; blob=null; $('play').src='';
  $('ok').disabled=true; $('redo').disabled=true; $('status').textContent='';
}
async function toggle(){
  if(mr&&mr.state==='recording'){mr.stop();return;}
  stream=stream||await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,noiseSuppression:false,echoCancellation:false}});
  chunks=[]; mr=new MediaRecorder(stream,{mimeType:'audio/webm;codecs=opus'});
  mr.ondataavailable=e=>chunks.push(e.data);
  mr.onstop=()=>{blob=new Blob(chunks,{type:'audio/webm'}); $('play').src=URL.createObjectURL(blob);
    $('ok').disabled=false; $('redo').disabled=false; $('rec').classList.remove('on'); $('rec').textContent='Record (Space)';};
  mr.start(); $('rec').classList.add('on'); $('rec').textContent='Stop (Space)';
}
async function accept(){
  if(!blob) return;
  const fd=new FormData(); fd.append('id',cur.id); fd.append('audio',blob,'a.webm');
  const r=await fetch('/upload',{method:'POST',body:fd});
  if(!r.ok){$('status').textContent='Rejected: '+(await r.text());return;}
  load();
}
async function skip(){const fd=new FormData(); fd.append('id',cur.id); await fetch('/skip',{method:'POST',body:fd}); load();}
$('rec').onclick=toggle; $('ok').onclick=accept; $('redo').onclick=toggle; $('skip').onclick=skip;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='BUTTON') e.preventDefault();
  if(e.code==='Space'){e.preventDefault();toggle();}
  else if(e.key==='Enter'){accept();}
  else if(e.key==='r'||e.key==='R'){toggle();}
  else if(e.key==='s'||e.key==='S'){skip();}
});
load();
</script></body></html>
```

- [ ] **Step 5: Run unit tests**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: 2 PASS

- [ ] **Step 6: Manual smoke test**

Run: `uvicorn record.server:app --port 8765` then open http://localhost:8765 in Chrome. Record 3 sentences, accept each. Then:
```bash
wc -l data/own/manifest.jsonl          # 3
python -c "import soundfile as sf;i=sf.info('data/own/wav/1.wav');print(i.samplerate,i.channels,i.subtype)"  # 16000 1 PCM_16
```
Restart server, reload page: progress shows `3 / 1000`, sentence 4 is displayed. Hit Space twice instantly (near-empty blob), Enter: status shows `Rejected: ... empty recording` and manifest still 3 lines.

- [ ] **Step 7: Commit**

```bash
git add record/server.py record/index.html tests/test_recorder.py
git commit -m "feat(voice): browser prompted recorder with resume and ffmpeg conversion"
```

---

### Task 4: VM helper + environment setup

**Files:**
- Create: `scripts/vm.sh`
- Create: `scripts/vm_setup.sh`

**Interfaces:**
- Produces: `scripts/vm.sh start|stop|sync|ssh`. VM venv at `/home/azureuser/voice-venv`, NeMo venv at `/home/azureuser/nemo-venv`, `HF_HOME=/mnt/voice/hf`, repo mirror at `/mnt/voice/repo`.

- [ ] **Step 1: Write vm.sh**

`scripts/vm.sh`:
```bash
#!/usr/bin/env bash
# Usage: scripts/vm.sh start|stop|sync|ssh [cmd]
set -euo pipefail
RG=ROGII-RG; VM=rogii-gpu; HOST=rogii; REMOTE=/mnt/voice/repo
HERE="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  start) az vm start -g $RG -n $VM -o none && echo started ;;
  stop)  az vm deallocate -g $RG -n $VM -o none && echo deallocated ;;
  sync)  rsync -az --delete --exclude .git --exclude __pycache__ --exclude 'data/public' \
           "$HERE/" "$HOST:$REMOTE/" && echo synced ;;
  ssh)   shift; ssh $HOST "cd $REMOTE && export HF_HOME=/mnt/voice/hf && source ~/voice-venv/bin/activate && $*" ;;
  *) echo "usage: $0 start|stop|sync|ssh [cmd]"; exit 1 ;;
esac
```

- [ ] **Step 2: Write vm_setup.sh (runs ON the VM)**

`scripts/vm_setup.sh`:
```bash
#!/usr/bin/env bash
# One-time VM setup. Run: ssh rogii 'bash -s' < scripts/vm_setup.sh
set -euo pipefail
sudo mkdir -p /mnt/voice && sudo chown azureuser:azureuser /mnt/voice
mkdir -p /mnt/voice/hf /mnt/voice/repo /mnt/voice/data
sudo apt-get install -y -q ffmpeg libsndfile1 >/dev/null
python3 -m venv ~/voice-venv
~/voice-venv/bin/pip install -q --upgrade pip
~/voice-venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
~/voice-venv/bin/pip install -q "transformers>=4.44" "peft>=0.12" "datasets>=2.19,<4" accelerate evaluate jiwer librosa soundfile huggingface_hub pytest
~/voice-venv/bin/python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
# NeMo fork for IndicConformer benchmark (separate venv; failure here is non-blocking)
python3 -m venv ~/nemo-venv || true
~/nemo-venv/bin/pip install -q --upgrade pip || true
~/nemo-venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || true
~/nemo-venv/bin/pip install -q "git+https://github.com/AI4Bharat/NeMo.git@nemo-v2#egg=nemo_toolkit[asr]" jiwer || echo "NEMO INSTALL FAILED (non-blocking)"
echo SETUP DONE
```

- [ ] **Step 3: Run it**

```bash
chmod +x scripts/vm.sh scripts/vm_setup.sh
az vm show -d -g ROGII-RG -n rogii-gpu --query powerState -o tsv   # expect "VM running"; else scripts/vm.sh start
ssh rogii 'bash -s' < scripts/vm_setup.sh
scripts/vm.sh sync
scripts/vm.sh ssh python -m pytest tests -q
```
Expected: `cuda True 4`, `SETUP DONE`, tests pass on VM.

- [ ] **Step 4: Commit**

```bash
git add scripts/vm.sh scripts/vm_setup.sh
git commit -m "chore(voice): VM start/stop/sync helper and one-time setup"
```

---

### Task 5: Dataset preparation

**Files:**
- Create: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `data/own/manifest.jsonl` (Task 3), `normalize` (Task 1).
- Produces: `split_ids(ids: list[int], seed=42) -> dict[str, list[int]]` with keys `train/dev/test`. On disk: `{data_root}/own_{train,dev,test}` and `{data_root}/mix_train` as `datasets.Dataset.save_to_disk` with columns `audio` (16 kHz), `text` (normalized), `source` (`own`/`own_aug`/`imasc`/`cv`).

- [ ] **Step 1: Write the failing test**

`tests/test_prepare_data.py`:
```python
from scripts.prepare_data import split_ids


def test_split_is_deterministic_and_disjoint():
    ids = list(range(1, 1001))
    a = split_ids(ids, seed=42)
    b = split_ids(ids, seed=42)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["train"]) & set(a["dev"])
    assert not set(a["dev"]) & set(a["test"])
    assert len(a["train"]) == 800 and len(a["dev"]) == 100 and len(a["test"]) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare_data.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Write implementation**

`scripts/prepare_data.py`:
```python
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
        print("common voice skipped:", e)
    public = concatenate_datasets(pubs).shuffle(seed=a.seed).select(range(min(n_public, sum(len(p) for p in pubs))))

    mix = concatenate_datasets([own_train, public]).shuffle(seed=a.seed)
    mix.save_to_disk(root / "mix_train")
    own_frac = sum(s.startswith("own") for s in mix["source"]) / len(mix)
    print(f"mix_train {len(mix)} own_share={own_frac:.2f}")
    assert 0.3 <= own_frac <= 0.5, "own share outside 30-50%"
    (root / "splits.json").write_text(json.dumps(splits))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prepare_data.py -v`
Expected: PASS

- [ ] **Step 5: Run on VM (after recordings exist)**

```bash
rsync -az data/own/ rogii:/mnt/voice/repo/data/own/
scripts/vm.sh sync
scripts/vm.sh ssh python scripts/prepare_data.py --root /mnt/voice/data
scripts/vm.sh ssh python scripts/prepare_data.py --root /mnt/voice/data_check
scripts/vm.sh ssh "diff <(python -c \"import json;print(json.load(open('/mnt/voice/data/splits.json'))['test'])\") <(python -c \"import json;print(json.load(open('/mnt/voice/data_check/splits.json'))['test'])\") && echo SPLITS IDENTICAL"
```
Expected: `mix_train N own_share=0.4x`, `SPLITS IDENTICAL`.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_data.py tests/test_prepare_data.py
git commit -m "feat(voice): deterministic own-voice splits with public mix-in and augmentation"
```

---

### Task 6: Whisper evaluation script

**Files:**
- Create: `scripts/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `{root}/own_test` (Task 5), `normalize`.
- Produces: `score(refs: list[str], hyps: list[str]) -> dict` with `wer`, `cer` (floats, after normalize). CLI appends rows to `results/eval.jsonl` `{model, adapter, split, n, wer, cer}` and prints a markdown table.

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
from scripts.eval import score


def test_score_normalizes_before_metrics():
    refs = ["അവൻ വന്നു"]
    hyps = ["അവന്‍ വന്നു."]  # old chillu + punctuation, same after normalize
    s = score(refs, hyps)
    assert s["wer"] == 0.0 and s["cer"] == 0.0


def test_score_counts_errors():
    s = score(["ഒന്ന് രണ്ട്"], ["ഒന്ന് മൂന്ന്"])
    assert s["wer"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Write implementation**

`scripts/eval.py`:
```python
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
            ids = model.generate(feats, language="ml", task="transcribe", num_beams=1, max_new_tokens=225)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval.py -v`
Expected: 2 PASS

- [ ] **Step 5: Find the adalat-ai Malayalam checkpoint id**

```bash
python -c "from huggingface_hub import list_models; [print(m.id) for m in list_models(author='adalat-ai', search='mal')]"
```
Expected: one or more ids containing `malayalam`/`ml`. Record the medium one for Task 8. If none, drop it from the baseline list and note in results.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval.py tests/test_eval.py
git commit -m "feat(voice): Whisper eval with normalized WER/CER and hyp dumps"
```

---

### Task 7: IndicConformer benchmark script (non-blocking)

**Files:**
- Create: `scripts/eval_nemo.py`

**Interfaces:**
- Consumes: `{root}/own_test`, `score` from Task 6 (import only jiwer-based `score`; no torch/transformers needed).
- Produces: a row in `results/eval.jsonl` with `model: ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large`.

- [ ] **Step 1: Write script**

`scripts/eval_nemo.py`:
```python
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
    ap.add_argument("--out", default="results/eval.jsonl")
    a = ap.parse_args()
    ds = load_from_disk(Path(a.root) / "own_test")
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
    row = {"model": MODEL, "adapter": None, "split": "own_test", "n": len(ds), **s}
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on VM (after Task 5 data exists)**

```bash
scripts/vm.sh sync
ssh rogii 'cd /mnt/voice/repo && HF_HOME=/mnt/voice/hf ~/nemo-venv/bin/python scripts/eval_nemo.py'
```
Expected: a printed row with wer/cer. If NeMo import or model download fails, record the error in `results/README.md` under "IndicConformer: not benchmarked (reason)" and continue. Do not spend more than one hour on this.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_nemo.py
git commit -m "feat(voice): IndicConformer stock benchmark script"
```

---

### Task 8: Baseline table

**Files:**
- Create: `results/README.md`

- [ ] **Step 1: Run Whisper baselines on VM**

```bash
scripts/vm.sh ssh python scripts/eval.py --root /mnt/voice/data --split own_test --models \
  openai/whisper-large-v3 vrclc/Whisper-medium-Malayalam sajilck/whisper-small-malayalam-v2 <ADALAT_ID_FROM_TASK6>
```
Expected: four lines `model: WER=… CER=…`, rows appended to `results/eval.jsonl`, hyp dumps under `results/eval/`.

- [ ] **Step 2: Reproducibility check**

Re-run the best model alone; WER/CER must match to 3 decimals (greedy decoding).

- [ ] **Step 3: Write results/README.md**

```markdown
# Own-voice Malayalam STT results

Test set: own_test, N=<n> utterances, <minutes> min. Metrics after scripts/normalize.py.

## Baselines (stock)

| model | WER | CER |
|---|---|---|
| openai/whisper-large-v3 | … | … |
| vrclc/Whisper-medium-Malayalam | … | … |
| sajilck/whisper-small-malayalam-v2 | … | … |
| <adalat id> | … | … |
| ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large | … | … |

Base chosen for fine-tune: <model> (lowest CER among Whisper).
```

- [ ] **Step 4: Commit**

```bash
rsync -az rogii:/mnt/voice/repo/results/ results/
git add results/
git commit -m "results(voice): stock baseline WER/CER on own-voice test set"
```

---

### Task 9: LoRA fine-tune

**Files:**
- Create: `scripts/train.py`

**Interfaces:**
- Consumes: `{root}/mix_train`, `{root}/own_dev` (Task 5).
- Produces: `runs/{name}/best/` LoRA adapter dir loadable by `peft.PeftModel.from_pretrained` (Task 6 `--adapter`, Task 10).

- [ ] **Step 1: Write script**

`scripts/train.py`:
```python
"""LoRA fine-tune a Whisper checkpoint on mix_train, early-stop on own_dev CER.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/train.py --base vrclc/Whisper-medium-Malayalam --name medium_lora
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from scripts.eval import score


@dataclass
class Collator:
    proc: WhisperProcessor

    def __call__(self, feats):
        inp = self.proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.proc.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        inp["labels"] = labels
        return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="/mnt/voice/data")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=32)
    a = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(a.base, language="malayalam", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base)
    model.generation_config.language = "ml"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()

    def prep(x):
        x["input_features"] = proc.feature_extractor(x["audio"]["array"], sampling_rate=16000).input_features[0]
        x["labels"] = proc.tokenizer(x["text"]).input_ids
        return x

    train = load_from_disk(Path(a.root) / "mix_train").map(prep, remove_columns=["audio", "text", "source"], num_proc=4)
    dev = load_from_disk(Path(a.root) / "own_dev").map(prep, remove_columns=["audio", "text", "source"])
    train = train.filter(lambda x: len(x["labels"]) < 448)

    def metrics(p):
        ids = p.predictions
        ids[ids == -100] = proc.tokenizer.pad_token_id
        labels = p.label_ids
        labels[labels == -100] = proc.tokenizer.pad_token_id
        s = score(proc.batch_decode(labels, skip_special_tokens=True), proc.batch_decode(ids, skip_special_tokens=True))
        return {"wer": s["wer"], "cer": s["cer"]}

    out = Path("runs") / a.name
    args = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=a.bs, per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=a.lr, warmup_steps=50, num_train_epochs=a.epochs,
        fp16=True, gradient_checkpointing=True, eval_strategy="steps", eval_steps=100, save_steps=100,
        logging_steps=25, predict_with_generate=True, generation_max_length=225,
        load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        save_total_limit=2, report_to="none", remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                             data_collator=Collator(proc), compute_metrics=metrics,
                             processing_class=proc.feature_extractor,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])
    trainer.train()
    model.save_pretrained(out / "best")
    proc.save_pretrained(out / "best")
    print("saved", out / "best", trainer.evaluate())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on VM (tiny)**

```bash
scripts/vm.sh sync
scripts/vm.sh ssh CUDA_VISIBLE_DEVICES=0 python scripts/train.py --base <BEST_WHISPER> --name smoke --epochs 0.05
```
Expected: `trainable params: …`, a few training steps, one eval with `eval_cer`, `saved runs/smoke/best`. If `processing_class` errors on the installed transformers, replace with `tokenizer=proc.feature_extractor`.

- [ ] **Step 3: Full run (background, log to file)**

```bash
scripts/vm.sh ssh "CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py --base <BEST_WHISPER> --name medium_lora > runs/medium_lora.log 2>&1 &"
```
Poll with `scripts/vm.sh ssh "tail -5 runs/medium_lora.log; grep eval_cer runs/medium_lora.log | tail -3"`. Expect dev CER to drop below the baseline CER within the first few hundred steps. Expected duration on one T4: 1–3 hours.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(voice): LoRA fine-tune script with dev-CER early stopping"
```

---

### Task 10: Post fine-tune eval + transcribe CLI

**Files:**
- Create: `scripts/transcribe.py`
- Modify: `results/README.md`

**Interfaces:**
- Consumes: `runs/medium_lora/best` (Task 9), `eval.py --adapter` (Task 6).

- [ ] **Step 1: Evaluate fine-tuned on own_test**

```bash
scripts/vm.sh ssh python scripts/eval.py --root /mnt/voice/data --split own_test --models <BEST_WHISPER> --adapter runs/medium_lora/best
```
Expected: a row with adapter set; CER lower than the same model's baseline row.

- [ ] **Step 2: Write transcribe.py**

`scripts/transcribe.py`:
```python
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
                    chunk_length_s=30, stride_length_s=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--base", default="vrclc/Whisper-medium-Malayalam")
    ap.add_argument("--adapter", default="runs/medium_lora/best")
    ap.add_argument("--raw", action="store_true", help="skip normalizer")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = load(a.base, a.adapter, device)
    y, _ = librosa.load(a.audio, sr=16000, mono=True)
    text = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "ml", "task": "transcribe"})["text"]
    print(text if a.raw else normalize(text))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test >30 s input on VM**

```bash
scripts/vm.sh ssh "ffmpeg -y -loglevel error -i data/own/wav/1.wav -i data/own/wav/2.wav -i data/own/wav/3.wav -i data/own/wav/4.wav -i data/own/wav/5.wav -i data/own/wav/6.wav -filter_complex 'concat=n=6:v=0:a=1' /tmp/long.wav && python scripts/transcribe.py /tmp/long.wav"
```
Expected: output contains words from all six sentences, not just the first ~30 s. (If six utterances are under 30 s total, concat more.)

- [ ] **Step 4: Update results/README.md**

Add section:
```markdown
## After fine-tune (LoRA on own voice + mix-in)

| model | WER | CER |
|---|---|---|
| <base> stock | … | … |
| <base> + runs/medium_lora/best | … | … |

Delta: CER … -> …, WER … -> ….
```

- [ ] **Step 5: Commit, then deallocate VM**

```bash
rsync -az rogii:/mnt/voice/repo/results/ results/
git add scripts/transcribe.py results/
git commit -m "feat(voice): transcribe CLI and fine-tuned results"
scripts/vm.sh stop
```

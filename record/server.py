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
    with MANIFEST.open("a+b") as f:
        f.seek(0, 2)
        if f.tell() and (f.seek(-1, 2), f.read(1))[1] != b"\n":
            f.write(b"\n")  # previous write was cut mid-line; do not merge into it
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return JSONResponse(row)

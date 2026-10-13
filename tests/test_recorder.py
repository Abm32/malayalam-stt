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


def _client(tmp_path, monkeypatch):
    import record.server as srv
    from fastapi.testclient import TestClient
    monkeypatch.setattr(srv, "ROOT", tmp_path)
    monkeypatch.setattr(srv, "PROMPTS", tmp_path / "prompts.txt")
    monkeypatch.setattr(srv, "OWN", tmp_path / "own")
    monkeypatch.setattr(srv, "WAV", tmp_path / "own" / "wav")
    monkeypatch.setattr(srv, "MANIFEST", tmp_path / "own" / "manifest.jsonl")
    monkeypatch.setattr(srv, "SKIPPED", tmp_path / "own" / "skipped.txt")
    (tmp_path / "prompts.txt").write_text("ഒന്ന് രണ്ട്\nമൂന്ന് നാല്\n", encoding="utf-8")
    return TestClient(srv.app)


def _webm(tmp_path, seconds):
    import subprocess
    p = tmp_path / "in.webm"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-ar", "48000", "-c:a", "libopus", str(p)], check=True)
    return p.read_bytes()


def test_upload_converts_to_16k_mono_and_appends_manifest(tmp_path, monkeypatch):
    import soundfile as sf
    c = _client(tmp_path, monkeypatch)
    r = c.post("/upload", data={"id": 1}, files={"audio": ("a.webm", _webm(tmp_path, 2), "audio/webm")})
    assert r.status_code == 200, r.text
    info = sf.info(tmp_path / "own" / "wav" / "1.wav")
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert read_manifest(tmp_path / "own" / "manifest.jsonl")[0]["text"] == "ഒന്ന് രണ്ട്"
    assert c.get("/next").json() == {"id": 2, "text": "മൂന്ന് നാല്", "done": 1, "total": 2}


def test_upload_rejects_empty_blob_without_manifest_line(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/upload", data={"id": 1}, files={"audio": ("a.webm", b"\x1a\x45", "audio/webm")})
    assert r.status_code == 400
    assert not (tmp_path / "own" / "manifest.jsonl").exists()

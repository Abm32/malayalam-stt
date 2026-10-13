# Malayalam STT fine-tuned on own voice — design

Date: 2026-10-13
Status: approved (conversational), pending written review

## Goal

A Malayalam speech-to-text model that transcribes Abhimanyu's own speech
more accurately than any public stock model. First deliverable is a
measured before/after table on a held-out set of his own recordings, not a
product. Use case decided afterwards.

Success criterion: fine-tuned model beats the best stock Malayalam model on
the same own-voice test set on both WER and CER. CER is the primary number
(Malayalam is agglutinative, word boundaries are noisy).

Out of scope for this iteration: TTS, streaming/real-time, mobile export,
IndicConformer fine-tuning (fallback only if Whisper plateaus).

## Decisions made

| Question | Decision |
|---|---|
| TTS or STT | STT first |
| Use case | Prove pipeline, choose use later |
| Training audio | Record fresh, prompted reading in browser |
| Base model | Whisper, best public Malayalam checkpoint chosen by benchmark |
| Also benchmark | IndicConformer ml stock (inference only, NeMo fork) |
| Compute | Existing Azure VM `rogii-gpu` (4x T4, eastus). Deallocate when idle. |

## Components

### 1. Recorder (`record/`)

Runs on the laptop. Mic is local.

- `record/server.py`: FastAPI. Routes: `GET /` page, `GET /next` returns
  next unrecorded sentence, `POST /upload` takes webm + sentence id,
  converts via `ffmpeg -ar 16000 -ac 1` to WAV at
  `data/own/wav/{id}.wav`, appends `{id, path, text, duration_s}` to
  `data/own/manifest.jsonl`. `POST /skip` marks a sentence skipped.
- `record/index.html`: shows one sentence large, buttons + keys: Space
  record/stop, Enter accept and next, R re-record, S skip. Playback before
  accept. Progress counter.
- Resume-safe: "next" = first sentence id not in manifest and not skipped.
- Sentence list: `data/prompts.txt`, one per line, built by
  `scripts/prepare_prompts.py` from Malayalam Wikipedia text (not IMaSC or
  Common Voice: public Malayalam ASR models train on those sentences, which
  would inflate baseline scores on the own-voice test set): dedupe, 5–20
  words, Malayalam script only, no sentence cut at an initial, shuffled.
  Target 1000. (Prompts 1–9 were recorded from IMaSC before this change.)
- Audio target: ~1000 utterances, 1.5–2 h, 16 kHz mono WAV.

### 2. Data (`scripts/prepare_data.py`)

- Own voice split by sentence id, deterministic seed: 80/10/10
  train/dev/test. Test sentences never in train.
- Public mix-in: IMaSC (HF `thennal/IMaSC`) + Common Voice Malayalam
  (HF `mozilla-foundation/common_voice_17_0`, config `ml`). Sampled per
  epoch so own-voice is 30–50% of training examples.
- Augmentation on own-voice train only: speed perturb 0.9–1.1, mild
  additive noise. None on dev/test.
- Text normalizer (`scripts/normalize.py`), applied to refs and hyps
  everywhere: Unicode NFC, strip punctuation and Latin digits
  normalisation, chillu normalisation (atomic chillu forms), collapse
  whitespace. One function, one test.
- Output: HF `datasets` on disk at `/mnt/voice/data/` on VM.

### 3. Baseline + fine-tune (`scripts/eval.py`, `scripts/train.py`)

On `rogii-gpu`, fresh venv `/home/azureuser/voice-venv`
(torch, transformers, peft, datasets, jiwer, librosa, soundfile,
accelerate). Separate venv for NeMo fork.

Baselines on own-voice test set, WER + CER after normalizer:
1. `openai/whisper-large-v3` (stock)
2. `vrclc/Whisper-medium-Malayalam`
3. adalat-ai Vividh medium high-LR
4. `sajilck/whisper-small-malayalam-v2`
5. `ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large` (NeMo fork, RNNT decoder)

Best Whisper by CER becomes base. Fine-tune:
- HF `Seq2SeqTrainer`, LoRA (peft) on q/v projections of encoder+decoder,
  fp16, gradient checkpointing, one T4 per run.
- Eval every N steps on own-voice dev, early stop on CER, keep best.
- Re-run `eval.py` on test with the adapter. Produce
  `results/baseline_vs_finetuned.md`.

### 4. Inference (`scripts/transcribe.py`)

`python scripts/transcribe.py file.wav` loads base + adapter, prints
normalised Malayalam text. Chunked for >30 s input.

## Repo layout

```
voice/
  record/        server.py, index.html
  scripts/       prepare_prompts.py, prepare_data.py, normalize.py,
                 eval.py, train.py, transcribe.py
  data/          gitignored (prompts.txt, own/, public/)
  results/       eval tables (committed)
  docs/superpowers/specs/
  requirements.txt
```

Sync to VM with rsync over existing `rogii` SSH host. Data lives on VM at
`/mnt/voice/`.

## Infra and cost

- `rogii-gpu` Standard_NC64as_T4_v3, ~$4/hr while running. Deallocate when
  idle (approved). Start with `az vm start -g ROGII-RG -n rogii-gpu`.
- No new resources needed.
- Common Voice via HF requires accepting dataset terms with an HF token.
  IMaSC and model checkpoints are open.

## Testing

- `normalize.py`: one `test_normalize.py` with a handful of ref/hyp pairs.
- Recorder: manual smoke (record 3 sentences, check WAV is 16 kHz mono,
  manifest has 3 lines, restart server resumes at 4th).
- `prepare_data.py`: assert no test sentence id appears in train.
- Eval: baseline table is itself the test; numbers must be reproducible
  across two runs (greedy decoding, fixed seed).

## Risks

- 1–2 h own voice may overfit; mix-in and LoRA (not full FT) mitigate.
- IndicConformer NeMo fork may not install cleanly on VM CUDA; it is
  benchmark-only, so failure there does not block the main path.
- Common Voice gating may delay mix-in; IMaSC alone is sufficient fallback.

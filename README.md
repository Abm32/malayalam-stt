# Malayalam own-voice STT benchmark

Which open Malayalam speech-to-text model is best on *my* voice, and does fine-tuning on ~2 h of my own recordings help?

**Answer:** `adalat-ai/koyal-ml-120m-1.0` (NeMo, 120M params) — CER 0.048 / WER 0.275 on 88 held-out own-voice clips. Fine-tuning Whisper on own voice made it worse. Full table and notes in [`results/README.md`](results/README.md).

| model | WER | CER |
|---|---|---|
| **adalat-ai/koyal-ml-120m-1.0** (NeMo) | **0.275** | **0.048** |
| ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large (NeMo) | 0.281 | 0.049 |
| adalat-ai/whisper-medium-ml-high-lr | 0.358 | 0.078 |
| adalat-ai/whisper-medium-ml-high-lr + own-voice LoRA | 0.397 | 0.081 |
| adalat-ai/whisper-medium-ml-rmft | 0.405 | 0.084 |
| vrclc/Whisper-medium-Malayalam | 0.575 | 0.128 |
| sajilck/whisper-small-malayalam-v2 | 0.582 | 0.149 |
| openai/whisper-large-v3 | 3.318 | 1.067 |

## Use the winner

```bash
pip install "nemo_toolkit[asr]" soundfile
huggingface-cli login   # koyal is gated: accept terms at https://huggingface.co/adalat-ai/koyal-ml-120m-1.0
```

```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("adalat-ai/koyal-ml-120m-1.0")
print(model.transcribe(["speech.wav"])[0].text)   # 16 kHz mono WAV in, punctuated Malayalam out
```

Or in this repo: `python scripts/transcribe.py a.wav b.wav [--beam]`.

## What is here

- `record/` — browser recorder that prompts ~1000 Malayalam sentences (first 9 IMaSC, rest Malayalam Wikipedia so the test set is not in public ASR training data). `python -m uvicorn record.server:app --port 8765`.
- `scripts/prepare_data.py` — splits own recordings (train/dev/test), mixes with IMaSC, saves HF datasets.
- `scripts/eval.py`, `scripts/eval_nemo.py` — WER/CER on `own_test` for Whisper and NeMo models, one normalizer (`scripts/normalize.py`) for refs and hyps.
- `scripts/train.py` — LoRA fine-tune of a Whisper checkpoint with early stopping on dev CER.
- `scripts/vm_setup.sh`, `scripts/vm.sh` — Azure T4 VM setup and helpers (two venvs: `voice-venv` for Whisper, `nemo-venv` with the AI4Bharat NeMo fork and `numpy<2`).
- `docs/superpowers/specs/` — design spec; `RESUME.md` — run log and decisions.

Recordings themselves are not in the repo.

## Why fine-tuning did not help

998 clips (137 min) at 40 % of a 4100-sample mix could not move a model already trained on far more Malayalam; dev CER was best at step 100 and rose monotonically after. The winner's residual errors are rare proper nouns and digit formatting ("100" vs "നൂറ്"), not speaker mismatch, so more own-voice data would not pay off either.

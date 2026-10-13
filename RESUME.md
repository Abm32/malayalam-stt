# Resume notes — Malayalam own-voice STT

Last updated: 2026-10-13. Recordings accepted so far: 384 / 1000 (laptop, `data/own/`).

## State
- All code committed on branch `testdoc` (voice/ paths only). Suite: `python -m pytest tests -q` → 18 pass.
- Spec: `docs/superpowers/specs/2026-10-13-malayalam-stt-design.md`. Plan: `docs/superpowers/plans/2026-10-13-malayalam-stt.md`.
- Execution ledger (rulings, review findings, deferred minors): `~/Desktop/.superpowers/sdd/2026-10-13-malayalam-stt/progress.md`.
- Pipeline fully dry-run on VM with proxy voice; code review done, 2 Critical + 4 Important fixed, 6 Minor deferred (listed in ledger).
- VM `rogii-gpu` is DEALLOCATED. Env ready: `~/voice-venv` (torch 2.5.1, transformers 4.57), `~/nemo-venv` (NeMo fork works). HF cache `/mnt/voice/hf` (IMaSC + sajilck small already cached). Repo mirror `/mnt/voice/repo`.

## Still needed from user
1. Finish recording: `cd ~/Desktop/digital_agency/voice; python -m uvicorn record.server:app --port 8765` → http://localhost:8765. Resumes automatically.
2. HF login (gated models): `huggingface-cli login`, then accept terms at
   https://huggingface.co/vrclc/Whisper-medium-Malayalam and
   https://huggingface.co/ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large
   Then copy token to VM: `scp ~/.cache/huggingface/token rogii:/mnt/voice/hf/token`

## Next steps (Tasks 8–10 of the plan), in order
```bash
scripts/vm.sh start
scripts/vm.sh sync
rsync -az data/own/ rogii:/mnt/voice/repo/data/own/
scripts/vm.sh ssh python scripts/prepare_data.py --root /mnt/voice/data
# baselines (Task 8)
scripts/vm.sh ssh "CUDA_VISIBLE_DEVICES=0 python scripts/eval.py --root /mnt/voice/data --models openai/whisper-large-v3 sajilck/whisper-small-malayalam-v2 adalat-ai/whisper-medium-ml-high-lr adalat-ai/whisper-medium-ml-rmft vrclc/Whisper-medium-Malayalam"
scripts/vm.sh ssh "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/voice/repo ~/nemo-venv/bin/python scripts/eval_nemo.py --root /mnt/voice/data"   # non-blocking
# pick best Whisper by CER -> BEST; fine-tune (Task 9)
scripts/vm.sh ssh "CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py --base BEST --name medium_lora > runs/medium_lora.log 2>&1 &"
# after training (Task 10)
scripts/vm.sh ssh "CUDA_VISIBLE_DEVICES=0 python scripts/eval.py --root /mnt/voice/data --models BEST --adapter runs/medium_lora/best"
rsync -az rogii:/mnt/voice/repo/results/ results/   # write results/README.md table, commit
scripts/vm.sh stop
```
Gotchas: run scripts with `scripts/vm.sh ssh` (sets PYTHONPATH + HF_HOME); never run two commands after one `cd ... &&` with `&` (second runs in $HOME). Candidate also worth benchmarking: `adalat-ai/koyal-ml-120m-1.0`.

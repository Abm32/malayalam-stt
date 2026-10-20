# Malayalam STT results (own_test, 88 recordings, 2026-10-19/20)

Scored by `scripts/eval.py` (Whisper) and `scripts/eval_nemo.py` (NeMo) on the VM (T4, GPU2); hypotheses in `results/eval/*.hyp.txt`, raw rows in `results/eval.jsonl`. Refs and hyps pass through `scripts/normalize.py` (punctuation stripped, digits kept).

| model | adapter | WER | CER |
|---|---|---|---|
| openai/whisper-large-v3 | – | 3.318 | 1.067 |
| sajilck/whisper-small-malayalam-v2 | – | 0.582 | 0.149 |
| vrclc/Whisper-medium-Malayalam | – | 0.575 | 0.128 |
| adalat-ai/whisper-medium-ml-rmft | – | 0.405 | 0.084 |
| adalat-ai/whisper-medium-ml-high-lr | – | 0.358 | 0.078 |
| adalat-ai/whisper-medium-ml-high-lr | runs/medium_lora/best (LoRA, mix_train 4100, 40 % own) | 0.397 | 0.081 |
| ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large (NeMo, rnnt) | – | 0.281 | 0.049 |
| **adalat-ai/koyal-ml-120m-1.0** (NeMo, TDT) | – | **0.275** | **0.048** |

- Best off-the-shelf: `adalat-ai/koyal-ml-120m-1.0` (CER 0.048), IndicConformer a hair behind (0.049). Both NeMo conformers beat every Whisper by ~40 % relative CER. Koyal is 120M params and emits punctuation + digits ("100-ലധികം" where the ref spells "നൂറിലധികം"), so its true CER is slightly better than shown.
- Best Whisper: `adalat-ai/whisper-medium-ml-high-lr` (CER 0.078).
- LoRA fine-tune (`scripts/train.py`, early stopping on dev CER) did not help: dev CER 0.0793 at step 100, then monotonically worse (0.0797, 0.0808, 0.0833, 0.0875 at 200–500) and stopped; on own_test the adapter is +0.003 CER worse than the base. 998 own clips are too few to move a model already trained on far more Malayalam; the own-data signal is dominated by the mixed public data.
- NeMo models run in `~/nemo-venv` (AI4Bharat NeMo fork 1.23, `numpy<2`); `eval_nemo.py` restores from the downloaded `.nemo` and downgrades koyal's CTC decoding strategy `greedy_batch` to `greedy` for that fork. large-v3 is unusable for Malayalam (CER > 1 from hallucinated repeats).

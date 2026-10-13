# SDD ledger — plan: docs/superpowers/plans/2026-10-13-malayalam-stt.md
Spec: docs/superpowers/specs/2026-10-13-malayalam-stt-design.md (read)
Setup: Ruling: no git worktree — repo root is ~/Desktop (huge multi-project dump, dirty tree); work in place on existing non-main branch `testdoc`, commits scoped to voice/ paths — cost if wrong: unrelated dirt in commits (mitigated by explicit `git add` paths)
Pre-flight (shared interfaces):
- T1 normalize() -> T2/T5/T6/T9/T10 consume `scripts.normalize.normalize(str)->str`: consistent
- T3 manifest.jsonl {id,path,text,duration_s} -> T5 reads same keys: consistent
- T5 saves own_{train,dev,test}, mix_train with cols audio/text/source -> T6 reads text+audio, T9 removes audio/text/source: consistent
- T6 score(refs,hyps)->{wer,cer} -> T7, T9 import: consistent
- T9 runs/{name}/best (peft save_pretrained) -> T6 --adapter / T10 PeftModel.from_pretrained: consistent
- T4 vm.sh ssh wraps cd+HF_HOME+venv -> T5..T10 commands rely on it: consistent
Task 1: complete (commits 6200e8d..4ed84cc, tests: python -m pytest tests -q → 4 passed in 0.25s)
Task 2: Ruling: load_sources reads IMaSC 'text' column via pyarrow+HfFileSystem instead of load_dataset — avoids 4.7 GB audio download on laptop for prompts — cost if wrong: none (same text). Common Voice skipped (EmptyDatasetError, no HF token). adalat-ai ids found: whisper-medium-ml-high-lr, whisper-medium-ml-rmft, koyal-ml-120m-1.0. vrclc medium + IndicConformer are gated: HF token needed on VM.
Task 2: complete (commits 4ed84cc..2926512, tests: python -m pytest tests -q → 5 passed in 0.26s)
Task 3: Ruling: manual mic smoke replaced by TestClient tests (synthetic opus webm via ffmpeg -> 16k mono PCM_16 + manifest; empty blob -> 400, no manifest line). Written after handler existed, so passed first run (characterization, not RED/GREEN) — cost if wrong: a test that cannot fail; mitigated: asserts real ffmpeg/soundfile output. Real mic smoke left to user at recording time.
Task 3: complete (commits 2926512..ef70a85, tests: python -m pytest tests -q → 9 passed, 1 warning in 1.20s)
Task 5: Ruling: Step 5 (build datasets on VM + SPLITS IDENTICAL check) deferred until user records; code+unit test complete. Task 4 Step 3 (VM setup verification) pending background install, see ~/setup.log on VM.
Task 5: complete (commits 244b898..665f7d7, tests: python -m pytest tests -q → 10 passed, 1 warning in 1.62s)
Task 6: complete (commits 665f7d7..5517e19, tests: python -m pytest tests -q → 12 passed, 1 warning in 1.14s)
Task 7: script committed (2f1b226); VM run pending env + data. Non-blocking per plan.
Task 9: script committed (577608c); smoke + full run pending env + data.
Task 10: transcribe.py committed (ce1fadd); eval/README pending fine-tune.
Ruling: before user records, run full pipeline dry-run on VM with proxy 'own voice' = 60 clips of one IMaSC speaker (scratchpad make_dryrun.py, outputs under /mnt/voice/dryrun, discarded) — validates prepare_data/eval/train/transcribe end to end — cost if wrong: ~30 min GPU time.
Task 4: complete (commits ef70a85..ce1fadd, tests: scripts/vm.sh ssh python -m pytest tests -q → 12 passed in 0.50s)
Task 5: Ruling: 'python scripts/x.py' cannot import scripts.normalize (sys.path[0]=scripts/); vm.sh ssh now exports PYTHONPATH=/mnt/voice/repo; direct ssh invocations must set PYTHONPATH too — cost if wrong: none.
Task 4: Ruling: transformers pinned >=4.46,<5 — 5.x needs torch>=2.6 (torch.accelerator), VM has torch 2.5.1+cu121 — cost if wrong: none, 4.57 has every API used.
Task 7: NeMo fork imports after pinning pytorch_lightning<2.5 (NeptuneLogger removed in 2.5).
Task 2: Ruling: prompts switched from IMaSC to Malayalam Wikipedia (wikimedia/wikipedia 20231101.ml) after dry-run showed sajilck v2 scores 0.000 WER on IMaSC sentences (trained on them). Prompts 1-9 (already recorded) kept as IMaSC; 10-1000 Wikipedia; backup at data/prompts_imasc_backup.txt. Spec updated. — cost if wrong: 9 test/train sentences overlap public training text, negligible.
Dry-run (proxy voice, /mnt/voice/dryrun): prepare_data ok (48/6/6, mix 239, own 0.40); eval ok; train smoke ok (LoRA 3.5M params, eval_cer logged, adapter saved runs/dry_smoke/best); eval --adapter ok; transcribe 39.8s ok (chunking covers >30s; tail clause dropped — minor, revisit with real data).
Task 9: Step 2 smoke complete on dry-run. Full run pending real data.
Task 10: Step 3 (>30s) verified on dry-run. Steps 1,4 pending real data.
Blocked on user: ~1000 recordings (9 done), HF login for gated vrclc medium + IndicConformer + Common Voice.
VM deallocated after dry-run (idle policy).
Final review: fresh reviewer (fable) on review-6200e8d10..116d969bf.diff. 2 Critical, 4 Important, 6 Minor.
Final: fixed C1 generation cap 225->448/440 + prompt token filter — test_filter_by_tokens_drops_long_sentences RED→GREEN
Final: fixed C2 transcribe chunk 30/5->15/3 — verified by CER on >=60s real-recording concat (see below)
Final: fixed I3 collator sot strip — test_collator_strips_startoftranscript_so_shift_right_does_not_double_it RED→GREEN
Final: fixed I4 manifest newline guard — test_upload_after_truncated_manifest_line_starts_a_fresh_line RED→GREEN
Final: fixed I5 per-id hash split — test_recording_more_later_never_moves_existing_ids RED→GREEN
Final: Ruling: split sizes now ~10%/10% (hash buckets) instead of exact 100/100 — stability across growing manifest outranks exact counts — cost if wrong: test set 80-120 instead of 100
Final: fixed I6 nta encodings — test_all_three_nta_encodings_collapse RED→GREEN
Suite 18/18.
Final: minor (deferred): server.py id not bounds-checked; missing audio field -> 500; ffmpeg temp leak on failure
Final: minor (deferred): index.html Enter during re-record uploads previous take
Final: minor (deferred): transcribe.py default --adapter must exist; --adapter "" undocumented
Final: minor (deferred): eval.py one --adapter merged into every --models entry
Final: minor (deferred): prepare_data own-share assert after save_to_disk
Final: minor (deferred): vm.sh bare `ssh` exits silently; train.py no CUDA_VISIBLE_DEVICES default
Final: C2 verified on 72.4 s concat of 10 real recordings, stock sajilck small: chunk 30/5 CER 0.514 (hyp 504 chars of 734 ref, tail lost) -> chunk 15/3 CER 0.322 (hyp 830 chars, full coverage). Residual error is model quality on user's voice (the thing fine-tuning targets).

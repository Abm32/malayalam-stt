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
        sot = self.proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == sot).all():
            labels = labels[:, 1:]  # shift_tokens_right re-adds decoder_start_token
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
    ap.add_argument("--eval_steps", type=int, default=100)
    a = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(a.base, language="malayalam", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base)
    model.generation_config.language = "ml"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for LoRA + gradient checkpointing
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()

    def prep(x):
        x["input_features"] = proc.feature_extractor(x["audio"]["array"], sampling_rate=16000).input_features[0]
        x["labels"] = proc.tokenizer(x["text"]).input_ids
        return x

    train = load_from_disk(Path(a.root) / "mix_train").map(prep, remove_columns=["audio", "text", "source"])
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
        fp16=True, gradient_checkpointing=True, eval_strategy="steps", eval_steps=a.eval_steps, save_steps=a.eval_steps,
        logging_steps=25, predict_with_generate=True, generation_max_length=448,
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

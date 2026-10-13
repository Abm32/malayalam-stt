import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
from transformers import WhisperProcessor  # noqa: E402

from scripts.train import Collator  # noqa: E402


def test_collator_strips_startoftranscript_so_shift_right_does_not_double_it():
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="malayalam", task="transcribe")
    sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    labels = proc.tokenizer("അവൻ വന്നു").input_ids
    assert labels[0] == sot  # tokenizer prepends it
    import numpy as np
    feats = [{"input_features": np.zeros((80, 3000), dtype="float32"), "labels": labels}]
    out = Collator(proc)(feats)
    assert out["labels"][0, 0].item() != sot
    assert out["labels"][0, 0].item() == labels[1]

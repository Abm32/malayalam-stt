from scripts.eval import score


def test_score_normalizes_before_metrics():
    refs = ["അവൻ വന്നു"]
    hyps = ["അവന്‍ വന്നു."]  # old chillu + punctuation, same after normalize
    s = score(refs, hyps)
    assert s["wer"] == 0.0 and s["cer"] == 0.0


def test_score_counts_errors():
    s = score(["ഒന്ന് രണ്ട്"], ["ഒന്ന് മൂന്ന്"])
    assert s["wer"] == 0.5

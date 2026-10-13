from scripts.prepare_prompts import filter_sentences


def test_filter_keeps_only_malayalam_5_to_20_words_deduped():
    lines = [
        "ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു",  # 7 words, keep
        "ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു",  # duplicate
        "ചെറുത്",  # 1 word, drop
        "This is english text with many words here ok",  # latin, drop
        "അവൻ വന്നു " * 12,  # >20 words, drop
    ]
    out = filter_sentences(lines)
    assert out == ["ഇത് ഒരു നല്ല ദിവസം ആണ് എന്ന് തോന്നുന്നു"]

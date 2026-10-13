from scripts.normalize import normalize


def test_old_chillu_sequence_maps_to_atomic():
    # ന + ് + ZWJ  ->  ൻ (U+0D7B)
    assert normalize("അവന്‍") == "അവൻ"


def test_strips_punctuation_and_zw_chars():
    assert normalize("ഇത്‌ ഒരു, പരീക്ഷണം.") == "ഇത് ഒരു പരീക്ഷണം"


def test_collapses_whitespace_and_lowercases_latin():
    assert normalize("  ഹലോ   World  ") == "ഹലോ world"


def test_idempotent():
    s = "അവന്‍ വന്നു!"
    assert normalize(normalize(s)) == normalize(s)


def test_all_three_nta_encodings_collapse():
    # ന്റ (virama), ന്‍റ (old ZWJ form), ൻ്റ (chillu+virama) are the same cluster
    assert normalize("എന്‍റെ") == "എന്റെ"
    assert normalize("എൻ്റെ") == "എന്റെ"

"""Single text normalizer applied to every ref and hyp before WER/CER."""
import re
import unicodedata

# old-style chillu: consonant + virama (U+0D4D) + ZWJ (U+200D) -> atomic chillu
_CHILLU = {
    "ണ്‍": "ൺ",  # ണ്‍ -> ൺ
    "ന്‍": "ൻ",  # ന്‍ -> ൻ
    "ര്‍": "ർ",  # ര്‍ -> ർ
    "ല്‍": "ൽ",  # ല്‍ -> ൽ
    "ള്‍": "ൾ",  # ള്‍ -> ൾ
    "ക്‍": "ൿ",  # ക്‍ -> ൿ
}
_PUNCT = re.compile(r"[^\w\sഀ-ൿ]", re.UNICODE)
_ZW = re.compile("[​‌‍﻿]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u0d28\u0d4d\u200d\u0d31", "\u0d28\u0d4d\u0d31")  # ന്‍റ (old ZWJ nta) -> ന്റ
    for old, new in _CHILLU.items():
        text = text.replace(old, new)
    text = text.replace("\u0d7b\u0d4d\u0d31", "\u0d28\u0d4d\u0d31")  # ൻ്റ (chillu nta) -> ന്റ
    text = _ZW.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = text.replace("_", " ")
    text = _WS.sub(" ", text).strip()
    return text.lower()

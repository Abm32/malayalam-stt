from scripts.prepare_data import split_ids


def test_split_is_deterministic_and_disjoint():
    ids = list(range(1, 1001))
    a = split_ids(ids, seed=42)
    b = split_ids(ids, seed=42)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["train"]) & set(a["dev"])
    assert not set(a["dev"]) & set(a["test"])
    assert 80 <= len(a["test"]) <= 120 and 80 <= len(a["dev"]) <= 120
    assert len(a["train"]) + len(a["dev"]) + len(a["test"]) == 1000


def test_recording_more_later_never_moves_existing_ids():
    a = split_ids(list(range(1, 601)), seed=42)
    b = split_ids(list(range(1, 1001)), seed=42)
    assert set(a["test"]) <= set(b["test"])
    assert set(a["dev"]) <= set(b["dev"])
    assert set(a["train"]) <= set(b["train"])

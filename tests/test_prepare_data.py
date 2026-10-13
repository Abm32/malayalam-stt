from scripts.prepare_data import split_ids


def test_split_is_deterministic_and_disjoint():
    ids = list(range(1, 1001))
    a = split_ids(ids, seed=42)
    b = split_ids(ids, seed=42)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["train"]) & set(a["dev"])
    assert not set(a["dev"]) & set(a["test"])
    assert len(a["train"]) == 800 and len(a["dev"]) == 100 and len(a["test"]) == 100

import importlib

import pytest


def test_splits_are_reproducible_disjoint_and_complementary():
    fn = importlib.import_module("experiments.hrf_estimation.splits").make_splits
    ids = tuple(f"sub-07_ses-nsd11_task-nsdcore_run-{i:02}" for i in range(1, 13))
    splits = fn(ids, seed=20261107, n_partitions=10)
    assert splits == fn(ids, seed=20261107, n_partitions=10)
    assert len(splits) == 20
    for first, second in zip(splits[::2], splits[1::2]):
        assert len(first.train) == len(first.test) == 6
        assert not set(first.train) & set(first.test)
        assert set(first.train + first.test) == set(ids)
        assert first.train == second.test and first.test == second.train
    assert len({frozenset(s.train) for s in splits}) == 20


@pytest.mark.parametrize("bad", ["session", "odd", "duplicate", "too_many"])
def test_invalid_split_requests_fail(bad):
    fn = importlib.import_module("experiments.hrf_estimation.splits").make_splits
    ids = [f"sub-07_ses-nsd11_task-nsdcore_run-{i:02}" for i in range(1, 5)]
    count = 1
    if bad == "session":
        ids[0] = ids[0].replace("nsd11", "nsd12")
    elif bad == "odd":
        ids.pop()
    elif bad == "duplicate":
        ids[0] = ids[1]
    else:
        count = 4  # only three unordered 2/2 partitions
    with pytest.raises(ValueError):
        fn(tuple(ids), 0, count)

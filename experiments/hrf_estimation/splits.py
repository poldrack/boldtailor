"""Stored, paired shuffle splits within one subject/session."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    test: tuple[str, ...]


def make_splits(run_ids, seed, n_partitions):
    ids = tuple(run_ids)
    sessions = {tuple(x.split("_")[:2]) for x in ids}
    if len(ids) < 4 or len(ids) % 2 or len(set(ids)) != len(ids) or len(sessions) != 1:
        raise ValueError(
            "splits require unique runs from one session and an even count >= 4"
        )
    # Each unordered partition occurs once by anchoring the first run in training.
    halves = [(ids[0], *tail) for tail in combinations(ids[1:], len(ids) // 2 - 1)]
    if not isinstance(n_partitions, int) or not 1 <= n_partitions <= len(halves):
        raise ValueError("requested more partitions than available, or invalid count")
    selected = np.random.default_rng(seed).choice(
        len(halves), n_partitions, replace=False
    )
    result = []
    for index in selected:
        train = halves[index]
        test = tuple(x for x in ids if x not in train)
        result.extend([Split(train, test), Split(test, train)])
    return tuple(result)

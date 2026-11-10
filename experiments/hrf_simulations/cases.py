"""Deterministic primary and sensitivity case grids; no RT resampling."""

from hashlib import sha256
from itertools import product
import json

import numpy as np


def simulation_cases(stage="pilot", replicates=None, sessions=None):
    if stage not in ("pilot", "full", "sensitivity"):
        raise ValueError("stage must be pilot, full, or sensitivity")
    replicates = (20 if stage == "pilot" else 100) if replicates is None else replicates
    sessions = (
        ([11] if stage in ("pilot", "sensitivity") else list(range(10, 20)))
        if sessions is None
        else sessions
    )
    if replicates < 1 or not sessions or any(s not in range(10, 20) for s in sessions):
        raise ValueError("positive replicates and sessions within 10–19 required")
    truths = (
        ("canonical", "offgrid")
        if stage != "full"
        else ("canonical", "fast", "slow", "empirical", "offgrid")
    )
    snrs = ("q50",) if stage != "full" else ("q25", "q50", "q75")
    mechanisms = [
        (m, c)
        for m in ("none", "amplitude", "duration", "additive_duration")
        for c in ("brief", "stimulus")
        if not (m == "duration" and c == "stimulus")
    ]
    variants = [dict(gamma=0.5, betas=[1.0, 1.0], match_rt_rms=False)]
    if stage == "sensitivity":
        variants += [
            dict(gamma=g, betas=[1.0, 1.0], match_rt_rms=False) for g in (0.25, 1.0)
        ]
        variants += [
            dict(gamma=0.5, betas=[1.0, 1.25], match_rt_rms=False),
            dict(gamma=0.5, betas=[1.0, 1.0], match_rt_rms=True),
        ]
    result = []
    for s, t, (m, c), noise, q, r, v in product(
        sessions,
        truths,
        mechanisms,
        ("white", "ar1", "spectral"),
        snrs,
        range(replicates),
        variants,
    ):
        seed = int(np.random.SeedSequence([20261110, s, r]).generate_state(1)[0])
        case = dict(
            session=s,
            truth=t,
            mechanism=m,
            convention=c,
            noise=noise,
            snr_level=q,
            repeat=r,
            seed=seed,
            **v,
        )
        case["id"] = sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]
        result.append(case)
    return result

"""Deterministic primary and sensitivity case grids; no RT resampling."""

from hashlib import sha256
from itertools import product
import json

import numpy as np


def simulation_cases(stage="pilot", replicates=None, sessions=None):
    if stage == "expanded":
        return expanded_cases(replicates, sessions)
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


def expanded_cases(replicates=None, sessions=None):
    """Balanced core and targeted controls, avoiding a giant factorial grid."""
    sessions = [10, 11, 19] if sessions is None else sessions
    replicates = 20 if replicates is None else replicates
    base = simulation_cases("full", replicates, sessions)
    result = []
    for source in base:
        c = dict(source, run_coefficient_sd=0.0, regressor_scaling="native")
        if c["snr_level"] == "q50" and c["noise"] != "white":
            result.append(dict(c, block="core"))
            if c["truth"] == "offgrid":
                result.extend(
                    dict(c, truth=t, block="core")
                    for t in ("offgrid_fast", "offgrid_slow")
                )
        if (
            c["session"] == sessions[0]
            and c["noise"] == "white"
            and c["snr_level"] == "q50"
        ):
            result.append(dict(c, block="white"))
        if c["truth"] not in ("canonical", "offgrid") or c["noise"] == "white":
            continue
        if c["snr_level"] != "q50":
            result.append(dict(c, block="snr"))
            continue
        if c["session"] != sessions[0]:
            continue
        result.extend(
            [
                dict(c, block="contrast", betas=[1.0, 1.25]),
                dict(c, block="run_variation", run_coefficient_sd=0.25),
                dict(c, block="scaling", regressor_scaling="unit_rms"),
            ]
        )
        if c["mechanism"] in ("amplitude", "additive_duration"):
            result.extend(dict(c, block="rt_strength", gamma=g) for g in (0.25, 1.0))
            result.append(dict(c, block="matched_rt", match_rt_rms=True))
    for c in result:
        c.pop("id")
        c["id"] = sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
    return result

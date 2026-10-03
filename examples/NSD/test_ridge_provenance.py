"""Global tuning identities describe the analysis, not the installed software."""

import numpy as np
import pandas as pd
import pytest

from boldtailor._software import software_environment
from boldtailor.data import from_arrays
from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.ridge_selection import score_ridge_candidates, select_ridge_penalty
from examples.NSD.ridge_provenance import tuning_provenance
from tests.oracles import glm_run_sources


def _problem():
    rng = np.random.default_rng(91)
    signals, events, predictors, times = [], [], [], []
    for run in range(3):
        t = 0.75 + 1.5 * np.arange(70)
        p = pd.DataFrame(
            dict(response_time=rng.uniform(0.3, 1.8, 7), trial_type=np.arange(7) % 2)
        )
        events.append(p.assign(onset=8 + 12.0 * np.arange(7) + run, duration=1.5))
        predictors.append(p)
        times.append(t)
        signals.append(rng.normal(100, 1, (len(t), 3)))
    sources = [glm_run_sources(run) for run in range(3)]
    data = from_arrays(signals, events, frame_times=times, sources=sources)
    return data, predictors


def _tuning(fractional):
    data, predictors = _problem()
    if fractional:
        scores = score_fraction_candidates(data, predictors, fractions=[0.5, 1.0])
        return tuning_provenance(scores, select_ridge_fractions(scores))
    scores = score_ridge_candidates(data, predictors, alphas=[0.0, 1.0])
    return tuning_provenance(scores, select_ridge_penalty(scores))


@pytest.mark.parametrize("fractional", [False, True])
def test_tuning_identity_does_not_depend_on_software(monkeypatch, fractional):
    first = _tuning(fractional)
    assert first.analysis_fingerprint is not None
    changed = {k: f"changed-{v}" for k, v in software_environment().items()}
    for module in ("_fit_lifecycle", "data", "prepared"):
        monkeypatch.setattr(
            f"boldtailor.{module}.software_environment", lambda: changed
        )
    second = _tuning(fractional)
    software = [a.get("software") for a in second.to_dict()["activities"]]
    assert changed in software
    assert first.analysis_fingerprint == second.analysis_fingerprint

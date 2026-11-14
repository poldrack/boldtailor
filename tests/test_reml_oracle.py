import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from boldtailor.ar1 import whiten
from boldtailor.reml import RemlGroup, fit_reml
from tests.reml_oracles import dense_fit, mixedlm_fit, simulate

UNEQUAL = (40, 73, 55)
SEED = 17


def _scans(n_groups, equal):
    if equal:
        return (60,) * n_groups
    return tuple(UNEQUAL[g % len(UNEQUAL)] for g in range(n_groups))


def _case_id(case):
    scans = "eq" if case["equal"] else "uneq"
    extra = "-dup" if case.get("duplicate") else ""
    extra += f"-ar{case['rho']}" if case.get("rho") else ""
    return f"g{case['groups']}-{scans}-z{case['n_random']}-r{case['ratio']}{extra}"


def _grid_cases():
    levels = itertools.product((1, 3, 12), (True, False), (1, 3, 6), (0.05, 1.0, 20.0))
    keys = ("groups", "equal", "n_random", "ratio")
    return [dict(zip(keys, values, strict=True)) for values in levels]


EXTRA_CASES = [
    dict(groups=3, equal=True, n_random=3, ratio=1.0, duplicate=True),
    dict(groups=3, equal=False, n_random=3, ratio=1.0, rho=0.5),
    dict(groups=12, equal=True, n_random=6, ratio=0.05, rho=0.5),
    dict(groups=1, equal=True, n_random=6, ratio=20.0, rho=0.5),
]
CASES = _grid_cases() + EXTRA_CASES

# Cases where MixedLM(reml=True, BFGS, gtol=1e-10) does not reach the REML
# optimum. Each was confirmed: boldtailor matches dense_fit within 1e-5, and
# the dense criterion scores MixedLM's estimate worse.
MIXEDLM_FAILURES = {
    "g1-eq-z1-r0.05": "REML optimum on boundary ratio=0; boldtailor matches "
    "dense_fit exactly (ratio 0, sigma2 rel. diff 2.7e-16); dense criterion is "
    "2.0e-3 worse at MixedLM's ratio=4.4e-4",
    "g1-uneq-z3-r0.05": "MixedLM stuck away from the optimum; boldtailor "
    "matches dense_fit (ratio rel. diff 1.9e-7); dense criterion is 0.11 worse "
    "at MixedLM's ratio=0.0173 (optimum 0.00669)",
}


def _mixedlm_param(case):
    reason = MIXEDLM_FAILURES.get(_case_id(case))
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(case, id=_case_id(case), marks=marks)


def _duplicate_column(random):
    return [np.column_stack([z, z[:, :1]]) for z in random]


def _case_data(case):
    signals, fixed, random = simulate(
        np.random.default_rng(SEED),
        scans=_scans(case["groups"], case["equal"]),
        n_random=case["n_random"],
        ratio=case["ratio"],
        rho=case.get("rho", 0.0),
    )
    if case.get("duplicate"):
        random = _duplicate_column(random)
    return signals, fixed, random


def _fit(signals, fixed, random, rho):
    groups = [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]
    return fit_reml(groups, random, rho=rho)


def _whitened(parts, rho):
    return [[whiten(a, rho) for a in part] for part in parts]


@pytest.mark.parametrize("case", [_mixedlm_param(c) for c in CASES])
def test_matches_mixedlm(case):
    rho = case.get("rho", 0.0)
    data = _case_data(case)
    fit = _fit(*data, rho)
    ratio, sigma2 = mixedlm_fit(*_whitened(data, rho))
    assert_allclose(fit.ratio, [ratio], rtol=2e-4)
    assert_allclose(fit.noise_variance, [sigma2], rtol=2e-4)


@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_matches_dense_fit(case):
    rho = case.get("rho", 0.0)
    data = _case_data(case)
    fit = _fit(*data, rho)
    ratio, sigma2 = dense_fit(*data, rho)
    assert_allclose(fit.ratio, ratio, rtol=1e-5)
    assert_allclose(fit.noise_variance, sigma2, rtol=1e-5)

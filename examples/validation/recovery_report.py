"""Run the recovery tests exactly as written and print the quantities behind them.

Usage: ``uv run python examples/validation/recovery_report.py``

The tests in ``tests/test_recovery.py`` only assert pass/fail. This report wraps
the selection functions they call, runs each test unchanged, and prints the
selected fractions, penalties, CV scores, recovery rates, and RMSE values so the
numbers can be compared with ``docs/validation/recovery-2026-10.md``.
"""

import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tests.test_recovery as recovery  # noqa: E402

WRAPPED = (
    "select_ridge_fractions",
    "select_ridge_penalty",
    "select_hrfs",
    "fit_single_trials",
    "score_fraction_candidates",
    "score_ridge_candidates",
)
TESTS = (
    recovery.test_fractional_cv_prefers_shrinkage_when_trials_overlap_and_noise_is_high,
    recovery.test_fractional_cv_selects_ols_when_noise_is_negligible,
    recovery.test_shared_alpha_cv_prefers_a_positive_penalty_under_high_noise,
    recovery.test_hrf_selection_recovers_the_generating_kernel_under_ar1_noise,
    recovery.test_hrf_selection_keeps_canonical_when_truth_is_canonical,
)
HRF_TRUE_IDS = np.array([0, 1, 2, 3, 2, 1, 0, 3])


def _recording(name, function, records):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        records.setdefault(name, []).append(result)
        return result

    return wrapped


def _install_recorders(records):
    originals = {name: getattr(recovery, name) for name in WRAPPED}
    for name, function in originals.items():
        setattr(recovery, name, _recording(name, function, records))
    return originals


def _run(test, records):
    records.clear()
    try:
        test()
    except AssertionError as error:
        status = f"FAIL ({error})"
    else:
        status = "PASS"
    print(f"\n## {test.__name__}: {status}")
    return status


def _profile(scores):
    medians = np.nanmedian(scores.cv_r2, axis=1)
    return {float(g): round(float(m), 4) for g, m in zip(scores.grid, medians)}


def _report_shrinkage(records):
    choice = records["select_ridge_fractions"][0]
    print("  selected fractions:", choice.ridge_fraction.tolist())
    print("  at_boundary:", choice.at_boundary.tolist())
    print(
        "  median CV R² per fraction:",
        _profile(records["score_fraction_candidates"][0]),
    )
    ols, ridge = records["fit_single_trials"]
    _, _, truth = recovery._dense_trial_problem(np.random.default_rng(11))
    ridge_rmse = recovery._centered_rmse(ridge, truth)
    ols_rmse = recovery._centered_rmse(ols, truth)
    print(f"  centered RMSE: ridge {ridge_rmse:.3f} vs OLS {ols_rmse:.3f}")


def _report_ols_control(records):
    choice = records["select_ridge_fractions"][0]
    print("  selected fractions:", choice.ridge_fraction.tolist())
    print(
        "  median CV R² per fraction:",
        _profile(records["score_fraction_candidates"][0]),
    )


def _report_alpha(records):
    choice = records["select_ridge_penalty"][0]
    print("  selected alpha:", choice.ridge_alpha, " at_boundary:", choice.at_boundary)
    print("  median CV R² per alpha:", _profile(records["score_ridge_candidates"][0]))


def _report_hrf_recovery(records):
    selection = records["select_hrfs"][0]
    recovered = float(np.mean(selection.hrf_indices == HRF_TRUE_IDS))
    print(
        "  selected ids:",
        selection.hrf_indices.tolist(),
        " truth:",
        HRF_TRUE_IDS.tolist(),
    )
    print(f"  recovered: {recovered:.3f}")
    delta = np.nanmedian(selection.delta_cv_r2[HRF_TRUE_IDS != 0])
    print("  median delta_cv_r2 (non-canonical truth):", round(float(delta), 4))
    print("  at_parameter_bound:", selection.at_parameter_bound.tolist())


def _report_canonical_control(records):
    selection = records["select_hrfs"][0]
    canonical = float(np.mean(selection.hrf_indices == 0))
    print("  selected ids:", selection.hrf_indices.tolist())
    print(f"  canonical fraction: {canonical:.3f}")


REPORTERS = (
    _report_shrinkage,
    _report_ols_control,
    _report_alpha,
    _report_hrf_recovery,
    _report_canonical_control,
)


def main():
    """Run every recovery test, print its quantities, and return the statuses."""
    records = {}
    originals = _install_recorders(records)
    statuses = {}
    try:
        for test, reporter in zip(TESTS, REPORTERS, strict=True):
            statuses[test.__name__] = _run(test, records)
            reporter(records)
    finally:
        for name, function in originals.items():
            setattr(recovery, name, function)
    return statuses


if __name__ == "__main__":
    main()

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from experiments.hrf_estimation.sensitivity import synthetic_templates


def test_case_grid_preserves_observed_rt_and_pairs_seeds():
    from experiments.hrf_simulations.cases import simulation_cases

    cases = simulation_cases("pilot", replicates=2)
    assert len(cases) == 2 * 2 * 7 * 3
    assert {c["session"] for c in cases} == {11}
    assert {c["mechanism"] for c in cases} == {
        "none",
        "amplitude",
        "duration",
        "additive_duration",
    }
    duration = [c for c in cases if c["mechanism"] == "duration"]
    assert all(c["convention"] == "brief" for c in duration)
    assert len({c["seed"] for c in cases if c["repeat"] == 0}) == 1
    assert len({c["id"] for c in cases}) == len(cases)


@pytest.fixture
def tiny_case():
    runs = tuple(
        replace(r, run_id=f"sub-test_ses-test_run-{i}")
        for i, r in enumerate(synthetic_templates())
    )
    library = HrfLibrary(
        (
            HrfCandidate(0, "spm", (6, 16, 1, 1, 6, 0, 32)),
            HrfCandidate(1, "double_gamma", (3.5, 12, 0.8, 1.5, 6, 0, 36)),
        )
    )
    profile = dict(
        feature_id=0,
        rho=0.3,
        variance=1.0,
        calibration_sessions=[10],
        snr_quantiles=np.array([0.2, 0.5, 1.0]),
        sources=[],
        runs=[
            dict(
                residual=np.sin(np.arange(len(r.frame_times))),
                nuisance=np.zeros(len(r.confounds.columns)),
            )
            for r in runs
        ],
    )
    case = dict(
        id="test",
        session=11,
        truth="canonical",
        mechanism="amplitude",
        convention="brief",
        noise="ar1",
        snr=0.5,
        seed=5,
        repeat=0,
        gamma=0.5,
        betas=[1.0, 1.0],
    )
    return runs, library, profile, case


def test_complete_case_resume_corruption_and_report(tmp_path, tiny_case):
    from experiments.hrf_simulations.runner import run_case
    from experiments.hrf_simulations.reporting import build_report

    runs, library, profile, case = tiny_case
    folder = tmp_path / "case-test"
    result = run_case(runs, library, profile, case, folder)
    assert len(result) == 36  # 4 models x (iid, estimated AR1, oracle AR1) x 3 fits
    assert set(result.split) == {"full", "half1", "half2"}
    assert set(result.estimator) == {"iid", "ar1", "ar1_oracle"}
    for name in (
        "candidate_id",
        "curve_rmse",
        "failed",
        "rho",
        "task_reconstruction_rmse",
    ):
        assert name in result
    stamp = (folder / "scores.tsv").stat().st_mtime_ns
    cached = run_case(runs, library, profile, case, folder)
    assert len(cached) == len(result)
    assert (folder / "scores.tsv").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="manifest"):
        run_case(runs, library, profile, {**case, "gamma": 0.9}, folder)
    build_report(tmp_path, tmp_path / "report")
    summary = pd.read_csv(tmp_path / "report/summary.tsv", sep="\t")
    assert summary.n.sum() == 36
    assert (tmp_path / "report/recovery.png").is_file()
    assert (tmp_path / "report/paired.tsv").is_file()
    stability = pd.read_csv(tmp_path / "report/stability.tsv", sep="\t")
    assert len(stability) == 12
    assert 'half_shape_correlation' in stability
    with (folder / "scores.tsv").open("a") as out:
        out.write("corruption")
    with pytest.raises(ValueError, match="corrupt"):
        run_case(runs, library, profile, case, folder)


def test_noiseless_case_and_zero_signal_metrics(tmp_path, tiny_case):
    from experiments.hrf_simulations.runner import run_case

    runs, library, profile, case = tiny_case
    result = run_case(
        runs,
        library,
        profile,
        {**case, "snr": float("inf")},
        tmp_path / "noiseless",
        models=("ConsDurRTMod",),
        splits=False,
    )
    assert result.loc[result.estimator == "iid", "candidate_id"].iloc[0] == 0


def test_cli_plan_does_not_require_nsd(capsys):
    from experiments.hrf_simulations.cli import main

    main(["plan", "--stage", "pilot", "--replicates", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["datasets"] == 42
    assert result["sessions"] == [11]


def test_failure_rate_includes_failed_estimates():
    from experiments.hrf_simulations.reporting import summarize

    rows = pd.DataFrame(
        dict(
            session=[11, 11],
            truth=["canonical"] * 2,
            mechanism=["amplitude"] * 2,
            convention=["brief"] * 2,
            noise=["ar1"] * 2,
            snr=[0.5] * 2,
            model=["ConsDurRTMod"] * 2,
            estimator=["iid"] * 2,
            split=["full"] * 2,
            failed=[False, True],
            curve_rmse=[0.1, np.nan],
            peak_absolute_error=[0.0, np.nan],
        )
    )
    out = summarize(rows)
    assert out.n.iloc[0] == 2
    assert out.failure_rate.iloc[0] == 0.5
    assert out.curve_rmse_mean.iloc[0] == 0.1

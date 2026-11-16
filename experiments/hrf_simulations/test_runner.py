import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from experiments.hrf_estimation.sensitivity import synthetic_templates


def test_case_grid_preserves_observed_rt_and_pairs_seeds():
    """Verify case grid preserves observed response-time and pairs seeds."""
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
    """Verify complete case resume corruption and report."""
    from experiments.hrf_simulations.runner import run_case
    from experiments.hrf_simulations.reporting import build_report

    runs, library, profile, case = tiny_case
    folder = tmp_path / "case-test"
    result = run_case(runs, library, profile, case, folder)
    assert (
        len(result) == 36
    )  # 3 models x (CV, iid REML, estimated/oracle AR1 REML) x 3 fits
    assert set(result.model) == {"ConsDurNoRT", "RTDur", "ConsDurRTMod"}
    assert set(result.selection_method) == {"cv", "reml"}
    assert set(result.split) == {"full", "half1", "half2"}
    assert set(result.estimator) == {"cv", "iid", "ar1", "ar1_oracle"}
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
    assert "half_shape_correlation" in stability
    with (folder / "scores.tsv").open("a") as out:
        out.write("corruption")
    with pytest.raises(ValueError, match="corrupt"):
        run_case(runs, library, profile, case, folder)


def test_noiseless_case_and_zero_signal_metrics(tmp_path, tiny_case):
    """Verify noiseless case and zero signal metrics."""
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
    # Noiseless data need not hit a search boundary when the fitted task model
    # differs from the generating model. The table must retain the actual
    # comparison's qualification, rather than infer it from the noise level.
    fits = json.loads((tmp_path / "noiseless/fits.json").read_text())
    assert [f["selection_method"] for f in fits] == ["cv", "reml", "reml", "reml"]
    assert result.loc[result.selection_method == "reml", "provisional"].tolist() == [
        f["provisional"][0] for f in fits if f["selection_method"] == "reml"
    ]


def test_cli_plan_does_not_require_nsd(capsys):
    """Verify CLI plan does not require NSD."""
    from experiments.hrf_simulations.cli import main

    main(["plan", "--stage", "pilot", "--replicates", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["datasets"] == 42
    assert result["sessions"] == [11]
    assert result["default_n_jobs"] == 15


def test_runner_defaults_to_fifteen_workers_with_override():
    """Verify runner defaults to fifteen workers with override."""
    from experiments.hrf_simulations.cli import _parser

    args = ["run", "--calibration", "calibration", "--output", "outputs"]
    assert _parser().parse_args(args).n_jobs == 15
    assert _parser().parse_args(args + ["--n-jobs", "2"]).n_jobs == 2


def test_failure_rate_includes_failed_estimates():
    """Verify failure rate includes failed estimates."""
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
            provisional=[True, False],
            curve_rmse=[0.1, np.nan],
            peak_absolute_error=[0.0, np.nan],
        )
    )
    out = summarize(rows)
    assert out.n.iloc[0] == 2
    assert out.failure_rate.iloc[0] == 0.5
    assert out.curve_rmse_mean.iloc[0] == 0.1
    assert out.n_qualified.iloc[0] == 2
    assert out.n_provisional.iloc[0] == 1
    assert out.provisional_rate.iloc[0] == 0.5
    legacy = summarize(rows.drop(columns="provisional"))
    assert legacy.n_qualified.iloc[0] == 0
    assert np.isnan(legacy.provisional_rate.iloc[0])


def test_paired_report_compares_reml_with_cv():
    """Paired errors compare both REML noise models against CV on the same case."""
    from experiments.hrf_simulations.reporting import paired_differences

    table = pd.DataFrame(
        dict(
            id=["one"] * 3,
            model=["ConsDurNoRT"] * 3,
            convention=["brief"] * 3,
            split=["full"] * 3,
            session=[11] * 3,
            truth=["canonical"] * 3,
            mechanism=["none"] * 3,
            noise=["white"] * 3,
            snr=[1.0] * 3,
            repeat=[0] * 3,
            estimator=["cv", "iid", "ar1"],
            curve_rmse=[0.4, 0.2, 0.3],
        )
    )
    paired = paired_differences(table)
    assert paired["iid_minus_cv_rmse"].iloc[0] == pytest.approx(-0.2)
    assert paired["ar1_minus_cv_rmse"].iloc[0] == pytest.approx(-0.1)

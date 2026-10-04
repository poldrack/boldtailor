"""boldtailor run: argument parsing, dry run, exit codes."""

import json
import logging
import shlex

import pytest

from boldtailor import cli
from boldtailor.model import Modulator
from boldtailor.workflow import report


def _argv(root, *extra):
    return [
        "run",
        "--bids-dir",
        str(root),
        "--subject",
        "sub-07",
        "--session",
        "ses-nsd10",
        "--task",
        "nsdcore",
        *extra,
    ]


def test_parser_builds_settings_with_defaults_and_overrides(four_runs):
    root, _ = four_runs
    args = cli.build_parser().parse_args(
        _argv(
            root,
            "--modulator",
            "response_time:indicator",
            "--modulator",
            "trial_type",
            "--ridge-mode",
            "cv",
            "--ridge-alphas",
            "0",
            "1",
            "--skip-stage",
            "reliability",
            "--n-jobs",
            "2",
            "--no-surface-maps",
            "--hrf-library",
            "sobol",
            "--hrf-n-samples",
            "8",
        )
    )
    settings = cli.settings_from_args(args)
    assert settings.modulators == (
        Modulator("response_time", missing="indicator"),
        Modulator("trial_type"),
    )
    assert settings.ridge_mode == "cv" and settings.ridge_alphas == (0.0, 1.0)
    assert settings.stages == frozenset({"glms", "betas", "summaries"})
    assert settings.n_jobs == 2 and settings.surface_maps is False
    assert (
        settings.output_dir == root / "derivatives" / "boldtailor_hrf-sobol8s0_ridge-cv"
    )


def test_dry_run_prints_the_plan_and_exits_zero(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--dry-run", "--hrf-library", "canonical")) == 0
    out = capsys.readouterr().out
    assert (
        "run-04" in out
        and "response_time" in out
        and "boldtailor_hrf-canonical_ridge-fractionalcv" in out
    )


def test_settings_errors_exit_one_with_one_line(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--ridge-mode", "lasso")) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "ridge_mode" in err[0]
    assert cli.main(_argv(root, "--surface-mesh", "left=a.gii")) == 1


def test_input_errors_exit_two_and_name_what_is_missing(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--task", "other", "--dry-run")) == 2
    assert "task-other" in capsys.readouterr().err
    assert cli.main(_argv(root, "--modulator", "stimulus_id", "--dry-run")) == 2
    assert "stimulus_id" in capsys.readouterr().err


def test_run_writes_the_derivative_and_a_second_run_exits_one(
    four_runs, tmp_path, capsys
):
    root, _ = four_runs
    argv = _argv(
        root,
        "--output-dir",
        str(tmp_path / "out"),
        "--hrf-library",
        "canonical",
        "--ridge-mode",
        "off",
        "--skip-stage",
        "reliability",
        "--skip-stage",
        "betas",
        "--skip-stage",
        "summaries",
        "--n-jobs",
        "1",
        "--block-size",
        "2",
        "--no-surface-maps",
    )
    assert cli.main(argv) == 0
    assert (tmp_path / "out" / "sub-07_ses-nsd10_task-nsdcore_report.html").exists()
    assert cli.main(argv) == 1
    assert "existing_results" in capsys.readouterr().err
    assert cli.main([*argv, "--existing-results", "overwrite"]) == 0


def test_skip_stage_cannot_remove_glms(four_runs):
    root, _ = four_runs
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(_argv(root, "--skip-stage", "glms"))


def test_help_exits_zero(capsys):
    assert cli.main(["run", "--help"]) == 0
    assert "--bids-dir" in capsys.readouterr().out


def test_usage_error_exits_one_with_one_stderr_line(four_runs, capsys):
    root, _ = four_runs
    argv = [
        "run",
        "--bids-dir",
        str(root),
        "--subject",
        "sub-07",
        "--session",
        "ses-nsd10",
    ]
    assert cli.main(argv) == 1
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_report_command_line_round_trips_through_the_parser(four_runs, tmp_path):
    root, _ = four_runs
    args = cli.build_parser().parse_args(
        _argv(
            root,
            *("--modulator", "response_time:indicator", "--modulator", "trial_type"),
            *("--output-dir", str(tmp_path / "o"), "--hrf-library", "sobol"),
            *("--hrf-n-samples", "8", "--hrf-seed", "3"),
            *("--ridge-mode", "fixed", "--ridge-alpha", "0.5"),
            *("--ridge-alphas", "0", "2.5", "--ridge-fractions", "0.5", "1"),
            *("--ridge-percentile", "80", "--encoding-mode", "absolute"),
            *("--skip-stage", "reliability", "--no-surface-maps"),
            *("--no-rt-in-hrf-selection", "--n-jobs", "2", "--block-size", "64"),
            *("--max-grayordinates", "100", "--existing-results", "overwrite"),
            *("--surface-mesh", "left=l.gii", "--surface-mesh", "right=r.gii"),
        )
    )
    settings = cli.settings_from_args(args)
    tokens = shlex.split(report.command_line(settings))
    assert tokens[:2] == ["boldtailor", "run"]
    again = cli.settings_from_args(cli.build_parser().parse_args(tokens[1:]))
    assert again == settings


def test_skipping_a_stage_also_skips_the_stages_that_require_it(four_runs):
    root, _ = four_runs
    args = cli.build_parser().parse_args(_argv(root, "--skip-stage", "betas"))
    assert cli.settings_from_args(args).stages == frozenset({"glms", "reliability"})


def test_settings_stay_strict_about_stage_requirements(four_runs):
    from boldtailor.workflow.settings import WorkflowSettings

    root, _ = four_runs
    with pytest.raises(ValueError, match="summaries requires stage betas"):
        WorkflowSettings(
            bids_dir=root,
            subject="sub-07",
            session="ses-nsd10",
            task="nsdcore",
            stages=frozenset({"glms", "summaries"}),
        )


def test_overwrite_into_the_fmriprep_directory_is_refused(four_runs, capsys):
    root, prep = four_runs
    bold = sorted(prep.rglob("*_bold.dtseries.nii"))
    argv = _argv(root, "--output-dir", str(prep), "--existing-results", "overwrite")
    assert cli.main([*argv, "--hrf-library", "canonical", "--ridge-mode", "off"]) == 1
    assert "output_dir" in capsys.readouterr().err
    assert bold and all(p.exists() for p in bold)


def test_dry_run_with_string_trial_type_is_task_only(four_runs, capsys):
    from tests.workflow.synthetic_bids import face_house_events, rewrite_events

    root, _ = four_runs
    rewrite_events(root, face_house_events)
    argv = _argv(root, "--dry-run", "--hrf-library", "canonical", "--ridge-mode", "off")
    assert cli.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["task_model"]["regressors"] == ["task"]
    assert "trial_type is not binary 0/1; not used as a modulator" in plan["notes"]


def test_no_modulators_fits_a_task_only_model(four_runs, capsys):
    root, _ = four_runs
    argv = _argv(root, "--dry-run", "--hrf-library", "canonical", "--ridge-mode", "off")
    assert cli.main([*argv, "--no-modulators"]) == 0
    assert json.loads(capsys.readouterr().out)["task_model"]["regressors"] == ["task"]
    args = cli.build_parser().parse_args(_argv(root, "--no-modulators"))
    settings = cli.settings_from_args(args)
    assert settings.modulators == ()
    tokens = shlex.split(report.command_line(settings))
    assert "--no-modulators" in tokens
    assert cli.settings_from_args(cli.build_parser().parse_args(tokens[1:])) == settings


def test_no_modulators_excludes_modulator(four_runs, capsys):
    root, _ = four_runs
    argv = _argv(root, "--no-modulators", "--modulator", "trial_type", "--dry-run")
    assert cli.main(argv) == 1
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def _quick(root, tmp_path, *extra):
    return _argv(
        root,
        *("--output-dir", str(tmp_path / "out"), "--hrf-library", "canonical"),
        *("--ridge-mode", "off", "--n-jobs", "1", "--block-size", "2"),
        "--no-surface-maps",
        *extra,
    )


def test_malformed_confounds_exit_two(four_runs, tmp_path, capsys):
    import pandas as pd

    root, prep = four_runs
    path = next(prep.rglob("*run-02*confounds_timeseries.tsv"))
    pd.read_csv(path, sep="\t").drop(columns="cosine00").to_csv(
        path, sep="\t", index=False
    )
    assert cli.main(_quick(root, tmp_path, "--dry-run")) == 2
    assert "cosine" in capsys.readouterr().err
    assert cli.main(_quick(root, tmp_path)) == 2
    assert "cosine" in capsys.readouterr().err


def test_errors_during_fitting_propagate_with_a_traceback(
    four_runs, tmp_path, monkeypatch
):
    from boldtailor.workflow import run as workflow_run

    def fail(*args, **kwargs):
        raise ValueError("numerical failure inside fitting")

    monkeypatch.setattr(workflow_run.analysis, "fit_glms", fail)
    with pytest.raises(ValueError, match="numerical failure inside fitting"):
        cli.main(_quick(root=four_runs[0], tmp_path=tmp_path))


def _boldtailor_info(caplog):
    return [
        r
        for r in caplog.records
        if r.name.startswith("boldtailor") and r.levelno == logging.INFO
    ]


def test_run_logs_progress_unless_quiet(four_runs, tmp_path, caplog):
    root, _ = four_runs
    assert cli.main(_quick(root, tmp_path, "--skip-stage", "betas")) == 0
    assert _boldtailor_info(caplog)
    caplog.clear()
    quiet = _quick(root, tmp_path, "--skip-stage", "betas", "--quiet")
    assert cli.main([*quiet, "--existing-results", "overwrite"]) == 0
    assert not _boldtailor_info(caplog)


def test_missing_surface_mesh_exits_two(four_runs, tmp_path, capsys):
    root, _ = four_runs
    argv = [a for a in _quick(root, tmp_path) if a != "--no-surface-maps"]
    mesh = ("--surface-mesh", f"left={tmp_path / 'l.gii'}")
    assert cli.main([*argv, *mesh, "--surface-mesh", f"right={tmp_path}/r.gii"]) == 2
    assert "l.gii" in capsys.readouterr().err

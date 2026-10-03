"""boldtailor run: argument parsing, dry run, exit codes."""

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

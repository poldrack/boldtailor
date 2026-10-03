"""Command line: `boldtailor run` executes the workflow on one BIDS subject and session."""

import argparse
import json
import sys
from pathlib import Path

from boldtailor.workflow.settings import (
    ENCODING_MODES,
    EXISTING_RESULTS,
    HRF_LIBRARIES,
    RIDGE_MODES,
    STAGE_REQUIRES,
    SUPPORTED_SPACES,
    WorkflowSettings,
    parse_modulator,
)

_SKIPPABLE = ("reliability", "betas", "summaries")
_RIDGE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_RIDGE_ALPHAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class _OneLineParser(argparse.ArgumentParser):
    """Usage errors are one stderr line and exit status 1."""

    def error(self, message):
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(1)


def _mesh(text):
    side, _, path = text.partition("=")
    if side not in ("left", "right") or not path:
        raise argparse.ArgumentTypeError("surface mesh must be left=PATH or right=PATH")
    return side, Path(path)


def _choice(group, flag, choices, default, **kwargs):
    # validated by WorkflowSettings, so a bad value is a one-line settings error
    group.add_argument(
        flag,
        default=default,
        metavar="{" + "|".join(choices) + "}",
        **kwargs,
    )


def _add_inputs(run):
    group = run.add_argument_group("inputs")
    group.add_argument("--bids-dir", type=Path, required=True)
    group.add_argument("--subject", required=True)
    group.add_argument("--session", required=True)
    group.add_argument("--task", required=True)
    group.add_argument("--fmriprep-dir", type=Path)
    group.add_argument("--output-dir", type=Path)
    _choice(group, "--space", SUPPORTED_SPACES, "fsLR-91k")
    group.add_argument(
        "--modulator",
        action="append",
        type=parse_modulator,
        metavar="COLUMN[:indicator]",
        help="task modulator; repeat to list all; default detects response_time and trial_type",
    )


def _add_hrf(run):
    group = run.add_argument_group("HRF selection")
    _choice(group, "--hrf-library", HRF_LIBRARIES, "default")
    group.add_argument("--hrf-n-samples", type=int, default=512)
    group.add_argument("--hrf-seed", type=int, default=0)
    group.add_argument("--no-rt-in-hrf-selection", action="store_true")


def _add_ridge(run):
    group = run.add_argument_group("beta series")
    _choice(group, "--ridge-mode", RIDGE_MODES, "fractional_cv")
    group.add_argument("--ridge-alpha", type=float, default=0.1)
    group.add_argument(
        "--ridge-fractions", type=float, nargs="+", default=_RIDGE_FRACTIONS
    )
    group.add_argument("--ridge-alphas", type=float, nargs="+", default=_RIDGE_ALPHAS)
    group.add_argument("--ridge-percentile", type=float, default=90.0)
    _choice(group, "--encoding-mode", ENCODING_MODES, "within_run")


def _add_stages(run):
    group = run.add_argument_group("stages and figures")
    group.add_argument("--skip-stage", action="append", choices=_SKIPPABLE, default=[])
    group.add_argument("--no-surface-maps", action="store_true")
    group.add_argument(
        "--surface-mesh", action="append", type=_mesh, metavar="left=PATH|right=PATH"
    )


def _add_execution(run):
    group = run.add_argument_group("execution")
    group.add_argument("--n-jobs", type=int, default=4)
    group.add_argument("--block-size", type=int, default=4096)
    group.add_argument("--max-grayordinates", type=int)
    _choice(group, "--existing-results", EXISTING_RESULTS, "error")
    group.add_argument(
        "--dry-run", action="store_true", help="print the resolved plan and exit"
    )


def build_parser():
    parser = _OneLineParser(
        prog="boldtailor", description="First-level fMRI modelling with tailored HRFs."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the full workflow on one subject and session")
    for add in (_add_inputs, _add_hrf, _add_ridge, _add_stages, _add_execution):
        add(run)
    return parser


def _stages(skipped):
    """Enabled stages; skipping a stage also skips every stage that requires it."""
    stages = {"glms", *_SKIPPABLE} - set(skipped)
    while dropped := {s for s in stages if STAGE_REQUIRES.get(s, s) not in stages}:
        stages -= dropped
    return frozenset(stages)


def settings_from_args(args):
    meshes = dict(args.surface_mesh) if args.surface_mesh else None
    return WorkflowSettings(
        bids_dir=args.bids_dir,
        subject=args.subject,
        session=args.session,
        task=args.task,
        fmriprep_dir=args.fmriprep_dir,
        output_dir=args.output_dir,
        space=args.space,
        modulators=tuple(args.modulator) if args.modulator else None,
        hrf_library=args.hrf_library,
        hrf_n_samples=args.hrf_n_samples,
        hrf_seed=args.hrf_seed,
        hrf_selection_rt=not args.no_rt_in_hrf_selection,
        ridge_mode=args.ridge_mode,
        ridge_alpha=args.ridge_alpha,
        ridge_fractions=tuple(args.ridge_fractions),
        ridge_alphas=tuple(args.ridge_alphas),
        ridge_percentile=args.ridge_percentile,
        encoding_mode=args.encoding_mode,
        stages=_stages(args.skip_stage),
        surface_maps=not args.no_surface_maps,
        surface_meshes=meshes,
        n_jobs=args.n_jobs,
        block_size=args.block_size,
        max_grayordinates=args.max_grayordinates,
        existing_results=args.existing_results,
    )


def _run(args):
    from boldtailor.workflow import run as workflow_run

    settings = settings_from_args(args)
    if args.dry_run:
        plan = dict(
            settings=settings.to_dict(), **workflow_run.describe_inputs(settings)
        )
        print(json.dumps(plan, indent=2))
        return 0
    result = workflow_run.run_workflow(settings)
    print(
        f"wrote {len(result.paths)} files under {settings.output_dir}; report: {result.report_path}"
    )
    return 0


def _execute(args):
    from boldtailor.workflow.inputs import InputError

    try:
        return _run(args)
    except (InputError, FileNotFoundError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code or 0
    return _execute(args)

"""Calibrate NSD noise, inspect workloads, run cases, and summarize recovery."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from boldtailor.hrf_library import HrfLibrary, default_hrf_library
from experiments.hrf_estimation.inputs import discover_runs, load_run
from experiments.hrf_simulations.calibration import calibrate, draw_profile
from experiments.hrf_simulations.cases import simulation_cases
from experiments.hrf_simulations.io import file_hash, verify, write_json
from experiments.hrf_simulations.runner import implementation_fingerprint, run_case

ROOT = Path(__file__).resolve().parent
DEFAULT_MASK = (
    ROOT.parent
    / "hrf_estimation/pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz"
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print case count without reading NSD")
    calibration = commands.add_parser(
        "calibrate", help="fit reference noise profiles; writes local private artifacts"
    )
    run = commands.add_parser(
        "run", help="execute/resume selected real-NSD simulation cases"
    )
    smoke = commands.add_parser(
        "smoke", help="synthetic end-to-end control, no NSD required"
    )
    report = commands.add_parser("report", help="summarize completed cases without NSD")
    for command in (plan, run):
        command.add_argument(
            "--stage", choices=["pilot", "full", "sensitivity"], default="pilot"
        )
        command.add_argument("--replicates", type=int)
        command.add_argument("--sessions", nargs="+", type=int)
    for command in (calibration, run):
        command.add_argument(
            "--bids-root", type=Path, default=Path("/Volumes/extdata1/NSD/BIDS")
        )
        command.add_argument("--subject", default="sub-07")
        command.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    calibration.add_argument(
        "--sessions", nargs="+", type=int, default=list(range(10, 20))
    )
    calibration.add_argument("--output", type=Path, required=True)
    run.add_argument("--calibration", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--case", help="case ID from plan --list")
    run.add_argument(
        "--limit", type=int, help="maximum number of cases; useful for timing a pilot"
    )
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument(
        "--snr", type=float, help="explicit fixed-SNR control; inf gives noiseless data"
    )
    run.add_argument("--zero-nuisance", action="store_true")
    run.add_argument(
        "--no-splits",
        action="store_true",
        help="full-session fit only, labeled in manifest",
    )
    plan.add_argument("--list", action="store_true", dest="list_cases")
    smoke.add_argument("--output", type=Path, required=True)
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def _inventory(args):
    sources = []
    for s in args.sessions:
        for spec in discover_runs(args.bids_root, args.subject, [s]):
            sources.append(
                dict(
                    session=s,
                    run_id=spec.run_id,
                    files={
                        str(path): file_hash(path)
                        for path in (
                            spec.bold_path,
                            spec.events_path,
                            spec.confounds_path,
                        )
                    },
                )
            )
    return dict(
        sources=sources,
        mask=str(args.mask),
        mask_hash=file_hash(args.mask),
        implementation=implementation_fingerprint(),
        python=platform.python_version(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        hpf_seconds=80.0,
        subject=args.subject,
    )


def _calibrate(args):
    if any(s not in range(10, 20) for s in args.sessions):
        raise ValueError("use sessions 10–19 only")
    vertices = np.load(args.mask)["feature_ids"]
    provenance = _inventory(args)

    def sessions():
        for s in args.sessions:
            specs = discover_runs(args.bids_root, args.subject, [s])
            yield s, tuple(load_run(spec, vertices, 80.0) for spec in specs)

    calibrate(sessions(), args.output, provenance)


def _run(args):
    if args.n_jobs < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("n_jobs and limit must be positive")
    verify(args.calibration)
    cases = simulation_cases(args.stage, args.replicates, args.sessions)
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise ValueError("case ID not found in configured grid")
    if args.limit:
        cases = cases[: args.limit]
    if args.snr is not None:
        for case in cases:
            case["snr"] = args.snr
            case["snr_level"] = "fixed_control"
    for case in cases:
        case["zero_nuisance"] = args.zero_nuisance
    args.output.mkdir(parents=True, exist_ok=True)
    # Unique invocation records do not overwrite a previous frozen grid.
    from hashlib import sha256

    grid_id = sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()[:12]
    write_json(args.output / f"grid-{grid_id}.json", cases)
    vertices = np.load(args.mask)["feature_ids"][:1]
    library = default_hrf_library()
    for s in sorted({c["session"] for c in cases}):
        runs = tuple(
            load_run(spec, vertices, 80.0)
            for spec in discover_runs(args.bids_root, args.subject, [s])
        )
        selected = [c for c in cases if c["session"] == s]
        profiles = {
            seed: draw_profile(args.calibration, s, runs, seed)
            for seed in sorted({c["seed"] for c in selected})
        }
        print(
            f"SESSION {s}: {len(selected)} datasets; n_jobs={args.n_jobs}", flush=True
        )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.n_jobs)(
                delayed(_execute)(
                    runs, library, profiles[c["seed"]], c, args.output, args.no_splits
                )
                for c in selected
            )


def _execute(runs, library, profile, case, output, no_splits):
    table = run_case(
        runs,
        library,
        profile,
        case,
        output / f"case-{case['id']}",
        splits=not no_splits,
    )
    print(f"COMPLETE {case['id']}: {len(table)} fits", flush=True)


def _smoke(args):
    from experiments.hrf_estimation.sensitivity import synthetic_templates
    from experiments.hrf_simulations.reporting import build_report

    full = default_hrf_library()
    library = HrfLibrary(
        tuple(replace(k, id=i) for i, k in enumerate(full.candidates[:3]))
    )
    runs = tuple(
        replace(r, run_id=f"sub-synthetic_ses-synthetic_run-{i}")
        for i, r in enumerate(synthetic_templates())
    )
    profile = dict(
        feature_id=0,
        rho=0.4,
        variance=1.0,
        calibration_sessions=[],
        sources=[],
        calibration_label="synthetic control, not empirically calibrated",
        runs=[
            dict(
                residual=np.sin(np.arange(len(r.frame_times))),
                nuisance=np.zeros(len(r.confounds.columns)),
            )
            for r in runs
        ],
    )
    case = dict(
        id="synthetic-smoke",
        session=11,
        truth="canonical",
        mechanism="amplitude",
        convention="brief",
        noise="ar1",
        snr=2.0,
        seed=42,
        repeat=0,
        gamma=0.5,
        betas=[1.0, 1.0],
    )
    table = run_case(runs, library, profile, case, args.output / "case-synthetic-smoke")
    build_report(args.output, args.output / "report")
    print(f"Synthetic smoke complete: {len(table)} fits", flush=True)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "plan":
        cases = simulation_cases(args.stage, args.replicates, args.sessions)
        payload = dict(
            datasets=len(cases),
            sessions=sorted({c["session"] for c in cases}),
            default_n_jobs=1,
            includes_oracle=True,
        )
        if args.list_cases:
            payload["cases"] = cases
        print(json.dumps(payload, indent=2))
    elif args.command == "calibrate":
        _calibrate(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "smoke":
        _smoke(args)
    else:
        from experiments.hrf_simulations.reporting import build_report

        build_report(args.input, args.output)


if __name__ == "__main__":
    main()

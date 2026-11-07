"""Reproducible four-run canonical/library CV smoke comparison."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.hrf_library import default_hrf_library
from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_estimation.inputs import discover_runs, load_run
from experiments.hrf_estimation.library_cv import fit_library_cv
from experiments.hrf_estimation.scoring import score_hrfs
from experiments.hrf_estimation.splits import make_splits
from experiments.hrf_estimation.provenance import (
    identities,
    cached_result,
    completion_record,
)


def _atomic_text(path, text):
    temporary = path.with_name(".partial-" + path.name)
    temporary.write_text(text)
    temporary.replace(path)


def _save_fit(estimate, destination):
    times = np.arange(360) * 0.1
    curves = np.full((len(estimate.kernels), len(times)), np.nan)
    parameters = np.full((len(estimate.kernels), 7), np.nan)
    ids = np.full(len(estimate.kernels), -1, dtype=int)
    for i, kernel in enumerate(estimate.kernels):
        if kernel is None:
            continue
        samples = kernel.kernel(0.1, 1)
        curves[i] = np.interp(times, np.arange(len(samples)) * 0.1, samples, right=0)
        ids[i] = kernel.id
        if kernel.kind != "glmsingle":
            parameters[i] = kernel.parameters
    temporary = destination.with_name(".partial-" + destination.name)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            feature_ids=estimate.feature_ids,
            curves=curves,
            times=times,
            parameters=parameters,
            candidate_ids=ids,
            status=estimate.status,
            training_run_ids=estimate.training_run_ids,
        )
    temporary.replace(destination)


def _save_score(score, brain_axis, path):
    values = np.full(len(brain_axis), np.nan, dtype=np.float32)
    values[score.index.to_numpy(dtype=int)] = score.r2_task_partial
    header = nib.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(["r2_task_partial"]), brain_axis)
    )
    temporary = path.with_name(".partial-" + path.name)
    nib.save(nib.Cifti2Image(values[None, :], header=header), temporary)
    temporary.replace(path)
    _atomic_text(
        path.with_suffix("").with_suffix(".json"),
        json.dumps(
            {
                "metric": "1 - pooled full-model SSE / pooled nuisance-model SSE",
                "amplitudes": "refitted independently per evaluation run",
                "unprocessed_vertices": "NaN",
            },
            indent=2,
        )
        + "\n",
    )


def _estimate(runs, library, method, settings):
    if method == "librarycv":
        return fit_library_cv(runs, library, settings)
    if method == "reml":
        from experiments.hrf_estimation.reml import fit_library_reml

        return fit_library_reml(runs, library, settings)
    if method in ("remlar1", "remlcomponents", "remlshared", "remlcombined"):
        from experiments.hrf_estimation.reml_extensions import fit_extended_reml

        return fit_extended_reml(runs, library, {**settings, "variant": method.removeprefix("reml")})
    if method in ("continuouslbfgsb", "continuouspowell"):
        from experiments.hrf_estimation.continuous import fit_continuous

        optimizer = "L-BFGS-B" if method == "continuouslbfgsb" else "Powell"
        return fit_continuous(
            runs, {**settings, "library": library, "optimizer": optimizer}
        )
    if method != "canonical":
        raise ValueError(f"unknown method: {method}")
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        (library.candidates[0],) * len(runs[0].feature_ids),
        ("ok",) * len(runs[0].feature_ids),
        tuple(r.run_id for r in runs),
        {},
        runs[0].brain_axis,
    )


def run_comparison(
    runs,
    library,
    brain_axis,
    destination,
    seed=20261107,
    methods=("canonical", "librarycv"),
    settings=None,
    n_partitions=1,
):
    for run in runs:
        if run.brain_axis is not None and not np.array_equal(
            run.brain_axis, brain_axis[run.feature_ids]
        ):
            raise ValueError("export axis does not match run anatomy")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    splits = make_splits(tuple(r.run_id for r in runs), seed, n_partitions)
    by_id = {r.run_id: r for r in runs}
    settings = dict(settings or {})
    manifest = dict(
        seed=seed,
        cutoff_seconds=80,
        metric="HRF-transfer R2 with refitted amplitudes",
        library_fingerprint=library.fingerprint,
        splits=[asdict(s) for s in splits],
        rt_mode=settings.get("rt_mode", "amplitude"),
        methods=list(methods),
        estimator_settings=settings,
        **identities(runs, brain_axis),
    )
    cached = cached_result(destination, manifest)
    if cached is not None:
        return cached
    (destination / "COMPLETE.json").unlink(missing_ok=True)
    _atomic_text(destination / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    evaluations = [("insample", runs, runs)] + [
        (
            f"transfer{i:02}",
            tuple(by_id[x] for x in s.train),
            tuple(by_id[x] for x in s.test),
        )
        for i, s in enumerate(splits, 1)
    ]
    tables = []
    for method in methods:
        for label, train, test in evaluations:
            tables.append(
                _evaluate(
                    method,
                    label,
                    train,
                    test,
                    library,
                    brain_axis,
                    destination,
                    settings,
                )
            )
    table = pd.concat(tables, ignore_index=True)
    _atomic_text(destination / "scores.tsv", table.to_csv(sep="\t", index=False))
    _atomic_text(
        destination / "COMPLETE.json",
        json.dumps(completion_record(destination, len(table)), indent=2) + "\n",
    )
    return table


def _evaluate(method, label, train, test, library, brain_axis, destination, settings):
    start = time.monotonic()
    estimate = _estimate(train, library, method, settings)
    score = score_hrfs(test, estimate, settings.get("rt_mode", "amplitude"))
    prefix = (
        "_".join(train[0].run_id.split("_")[:2])
        + f"_space-fsLR_den-32k_desc-{method}{label}"
    )
    _save_fit(estimate, destination / f"{prefix}_hrf.npz")
    _atomic_text(
        destination / f"{prefix}_diagnostics.json",
        json.dumps(estimate.diagnostics, default=lambda value: value.tolist(), indent=2)
        + "\n",
    )
    _save_score(
        score.summary, brain_axis, destination / f"{prefix}_stat-r2task.dscalar.nii"
    )
    _atomic_text(
        destination / f"{prefix}_runmetrics.tsv",
        score.per_run.to_csv(sep="\t", index=False),
    )
    seconds = time.monotonic() - start
    print(f"{method} {label}: {seconds:.1f}s", flush=True)
    return score.summary.reset_index().assign(
        method=method, evaluation=label, seconds=seconds
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-samples", type=int, default=512)
    parser.add_argument(
        "--methods", nargs="+", default=["canonical", "librarycv", "reml"]
    )
    parser.add_argument("--session", type=int, choices=range(10, 20), default=11)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--full-roi", action="store_true")
    args = parser.parse_args()
    specs = discover_runs(args.bids_root, "sub-07", (args.session,))[: args.runs]
    with np.load(args.mask) as mask:
        vertices = mask["feature_ids" if args.full_roi else "pilot_feature_ids"]
    runs = tuple(load_run(s, vertices, 80) for s in specs)
    brain = nib.load(specs[0].bold_path).header.get_axis(1)
    if any(not np.array_equal(r.brain_axis, brain[vertices]) for r in runs):
        raise ValueError("CIFTI brain axes differ across runs")
    run_comparison(
        runs,
        default_hrf_library(args.library_samples),
        brain,
        args.output,
        methods=tuple(args.methods),
        n_partitions=args.partitions,
    )


if __name__ == "__main__":
    main()

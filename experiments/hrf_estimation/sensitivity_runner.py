"""Persist a factorial noise/RT recovery study separately from empirical VTC fits."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.hrf_library import HrfCandidate, default_hrf_library
from experiments.hrf_estimation.full_vtc import estimator_settings
from experiments.hrf_estimation.pilot import _estimate, _save_fit, _atomic_text
from experiments.hrf_estimation.provenance import identities, cached_result, completion_record
from experiments.hrf_estimation.scoring import score_hrfs
from experiments.hrf_estimation.sensitivity import parametric_library, simulate_runs, synthetic_templates

METHODS = ("canonical", "librarycv", "reml", "parametriccv", "parametricreml",
           "continuouslbfgsb", "continuouspowell")


def simulation_cases():
    names = ("canonical", "fast", "offgrid", "empirical")
    return [dict(truth=name, generating_rt=generating, fitted_rt=fitted,
                 rho=rho, snr=snr, seed=20261107 + repeat)
            for name, generating, fitted, rho, snr, repeat in product(
                names, ("amplitude", "duration"), ("amplitude", "duration"),
                (0., .4, .7), (.5, 2.), range(3))]


def true_kernel(name):
    choices = dict(canonical=HrfCandidate(0, "spm", (6, 16, 1, 1, 6, 0, 32)),
                   fast=HrfCandidate(0, "double_gamma", (3.5, 12, .8, 1.5, 6, 0, 36)),
                   offgrid=HrfCandidate(0, "double_gamma", (4.1, 12, .8, 1.5, 6, 0, 36)),
                   empirical=HrfCandidate(0, "glmsingle", (10,)))
    return choices[name]


def recovery_metrics(estimate, truth):
    kernel = estimate.kernels[0]
    if kernel is None:
        return dict(curve_rmse=np.nan, curve_correlation=np.nan, peak_bias_seconds=np.nan)
    def curve(k):
        h = k.kernel(.1, 1)
        return np.interp(np.arange(360) * .1, np.arange(len(h)) * .1, h, right=0)
    expected, actual = curve(truth), curve(kernel)
    return dict(curve_rmse=float(np.sqrt(np.mean((actual - expected)**2))),
                curve_correlation=float(np.corrcoef(actual, expected)[0, 1]),
                peak_bias_seconds=float((np.argmax(actual) - np.argmax(expected)) * .1))


def _method_row(runs, library, truth, method, settings, destination):
    selected = parametric_library(library) if method.startswith("parametric") else library
    implementation = {"parametriccv": "librarycv", "parametricreml": "reml"}.get(method, method)
    start = time.monotonic()
    estimate = _estimate(runs[:2], selected, implementation, settings)
    score = score_hrfs(runs[2:], estimate, settings["rt_mode"]).summary.iloc[0].to_dict()
    _save_fit(estimate, destination / f"{method}_hrf.npz")
    _atomic_text(destination / f"{method}_diagnostics.json", json.dumps(
        estimate.diagnostics, default=lambda x: x.tolist(), indent=2))
    return dict(method=method, seconds=time.monotonic() - start,
                **score, **recovery_metrics(estimate, truth))


def evaluate_case(templates, library, case, destination, settings, methods=METHODS):
    truth = true_kernel(case["truth"])
    runs = simulate_runs(templates, truth, case["generating_rt"],
                         case["rho"], case["snr"], case["seed"])
    settings = {**settings, "rt_mode": case["fitted_rt"]}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    brain = nib.cifti2.BrainModelAxis.from_surface([0], 1, "CortexLeft")
    manifest = dict(case=case, settings=settings, methods=list(methods),
                    truth=dict(kind=truth.kind, parameters=truth.parameters),
                    library_fingerprint=library.fingerprint,
                    training_run_ids=[r.run_id for r in runs[:2]],
                    test_run_ids=[r.run_id for r in runs[2:]],
                    **identities(runs, brain))
    cached = cached_result(destination, manifest)
    if cached is not None:
        return cached
    (destination / "COMPLETE.json").unlink(missing_ok=True)
    _atomic_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows = [_method_row(runs, library, truth, method, settings, destination) for method in methods]
    table = pd.DataFrame(rows).assign(**case)
    _atomic_text(destination / "scores.tsv", table.to_csv(sep="\t", index=False))
    _atomic_text(destination / "COMPLETE.json", json.dumps(completion_record(destination, len(table)), indent=2))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, choices=range(288))
    args = parser.parse_args()
    templates = synthetic_templates()
    library = default_hrf_library()
    for index, case in enumerate(simulation_cases()):
        if args.case is not None and args.case != index:
            continue
        print(f"START case {index}: {json.dumps(case)}", flush=True)
        settings = estimator_settings(case["fitted_rt"], 1)
        settings["verbose"] = 0
        result = evaluate_case(templates, library, case, args.output / f"case-{index:03}", settings)
        print(f"DONE case {index}: {result.seconds.sum():.1f}s", flush=True)


if __name__ == "__main__":
    main()

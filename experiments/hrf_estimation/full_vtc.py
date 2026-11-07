"""Matched full-fusiform comparison; output directories are method-level restart units."""

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np

from boldtailor.hrf_library import default_hrf_library
from experiments.hrf_estimation.inputs import discover_runs, load_run
from experiments.hrf_estimation.pilot import run_comparison
from experiments.hrf_estimation.sensitivity import parametric_library


def estimator_settings(mode, workers):
    return dict(rt_mode=mode, n_jobs=workers, prepared=True, verbose=10,
                lambda_grid=[0, .001, .01, .1, 1], max_starts=4,
                maxiter=500, maxfun=5000)


def comparison_jobs():
    groups = [("fullbaseline", "full", ("canonical", "librarycv", "reml")),
              ("parametriccv", "parametric", ("librarycv",)),
              ("parametricreml", "parametric", ("reml",)),
              ("continuouslbfgsb", "full", ("continuouslbfgsb",)),
              ("continuouspowell", "full", ("continuouspowell",))]
    return [dict(session=session, rt_mode=mode, name=name, library=library, methods=methods)
            for mode in ("amplitude", "duration") for session in (11, 12)
            for name, library, methods in groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--job", type=int, choices=range(20))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    mask = Path(__file__).parent / "pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz"
    with np.load(mask) as data:
        vertices = data["feature_ids"]
    full = default_hrf_library()
    libraries = dict(full=full, parametric=parametric_library(full))
    jobs = comparison_jobs()
    for index, job in enumerate(jobs):
        if args.job is not None and args.job != index:
            continue
        print(f"START job {index}: {json.dumps(job)}", flush=True)
        specs = discover_runs(args.bids_root, "sub-07", (job["session"],))
        if len(specs) != 12:
            raise ValueError("matched full-VTC comparison requires exactly 12 runs")
        runs = tuple(load_run(s, vertices, 80) for s in specs)
        brain = nib.load(specs[0].bold_path).header.get_axis(1)
        destination = args.output / f"session{job['session']}-{job['rt_mode']}-{job['name']}"
        run_comparison(runs, libraries[job["library"]], brain, destination,
                       methods=job["methods"], settings=estimator_settings(job["rt_mode"], args.workers))
        print(f"DONE job {index}", flush=True)


if __name__ == "__main__":
    main()

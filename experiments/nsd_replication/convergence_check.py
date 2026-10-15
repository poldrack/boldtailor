"""R6 sensitivity to LinearSVC's iteration limit.

Refits chosen (version, threshold) cells of the nsdpreproc R6 table with a
larger ``max_iter`` and writes one row per fold as it finishes, next to the
accuracy the pipeline table recorded at the default limit.

    python -m experiments.nsd_replication.convergence_check \
        --config experiments/nsd_replication/configs/pilot.toml \
        --subject sub-07 --max-iter 10000 --cells b1:0.3 b2:0.3 b1:0.0 lss-assume:0.0
"""

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.ladder import LEVELS, LSS_LEVELS
from experiments.nsd_replication.metrics import _fold_inputs
from experiments.nsd_replication.run import (
    SubjectData,
    _configure_logging,
    _mask,
    _metrics_dir,
    _subject_betas,
    _subject_roi,
)
from experiments.nsd_replication.trials import repetition_array

log = logging.getLogger(__name__)


def _fold(data, k, labels, max_iter):
    start = time.monotonic()
    n_reps = data.shape[0]
    train = np.delete(data, k, axis=0).reshape(-1, data.shape[2])
    train, test, dual = _fold_inputs(train, data[k])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LinearSVC(random_state=0, dual=dual, max_iter=max_iter).fit(
            train, np.tile(labels, n_reps - 1)
        )
    n_iter = int(np.max(model.n_iter_))
    return dict(
        fold=k + 1,
        accuracy=float((model.predict(test) == labels).mean()),
        n_iter=n_iter,
        converged=n_iter < max_iter,
        seconds=round(time.monotonic() - start),
    )


def _job(version, threshold, cell, k, labels, max_iter):
    return version, threshold, cell.shape[2], _fold(cell, k, labels, max_iter)


def _cell(spec):
    version, threshold = spec.rsplit(":", 1)
    return version, float(threshold)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--cells", nargs="+", type=_cell, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    _configure_logging()
    logging.getLogger(__name__).setLevel(logging.INFO)

    config = load_config(args.config)
    metrics = _metrics_dir(config, "nsdpreproc", args.subject)
    out = config.output_dir / "checks" / "nsdpreproc" / args.subject
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"r6_max_iter_{args.max_iter}.tsv"

    columns = np.flatnonzero(_subject_roi(config, args.subject))
    data = SubjectData(
        lambda lv: _subject_betas(config, "nsdpreproc", args.subject, lv, columns),
        LEVELS + LSS_LEVELS,
        np.ones(len(columns), bool),
        args.subject,
    )
    baseline = pd.read_csv(metrics / "r6.tsv", sep="\t")
    labels = np.arange(len(data.images))
    jobs = []
    for version, threshold in args.cells:
        reps = repetition_array(data.betas[version], data.trials, data.images)
        mask = _mask(data.roi, data.composite, threshold)
        cell = np.nan_to_num(reps[:, :, mask])
        jobs += [(version, threshold, cell, k) for k in range(reps.shape[0])]

    rows = []
    fits = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(_job)(*job, labels, args.max_iter) for job in jobs
    )
    for version, threshold, n_features, fold in fits:
        before = baseline[
            (baseline["version"] == version)
            & np.isclose(baseline["threshold"], threshold)
        ]
        row = dict(
            version=version,
            threshold=threshold,
            n_features=n_features,
            max_iter=args.max_iter,
            **fold,
            table_accuracy=float(before["accuracy"].iloc[0]),
            table_n_unconverged=int(before["n_unconverged"].iloc[0]),
        )
        rows.append(row)
        log.info("%s", row)
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()

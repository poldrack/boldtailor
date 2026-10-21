"""Denoising diagnostic: noise pool x count-scoring rule, on one session.

For real and null-shifted events, every candidate PC count k is scored under
two noise pools and two scoring rules, each through the same pcstop choice:

Pools (one threshold: GLMsingle's mixture threshold of the ON-OFF R²):

- ``onoff``: ON-OFF R² below the threshold (GLMsingle; boldtailor's default);
- ``full``: in-sample R² of the full task model (ON plus the session's
  modulators, canonical HRF, coefficients shared across runs) below the
  threshold, so a feature is noise only if no part of the task explains it.

The scoring set is GLMsingle's for both: ON-OFF R² above the threshold. The
full model nests ON-OFF, so its R² is never lower and the ``full`` pool is
disjoint from the scoring set as well.

Scoring rules:

- ``timeseries``: boldtailor's ``select_component_count`` (median pooled
  held-out R² of time-series prediction; the held-out run is not denoised);
- ``consistency``: GLMsingle's rule. Single-trial betas (trial events only:
  the single-trial design has no modulator columns) are refitted on the
  scoring set with the first k pool PCs in every run's confounds, and each
  feature's leave-one-run-out ``calcbadness`` (squared error of each trial
  against the mean of its image's trials in the other runs) is summarised as
  ``-median`` over the scoring set.

HRFs are frozen: b2's selection (session task model, baseline confounds).

    python -m experiments.nsd_replication.denoising_check \
        --config experiments/nsd_replication/configs/dev.toml \
        --subject sub-01 --session ses-nsd11 --n-jobs 16
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from boldtailor._denoising_cv import choose_count, select_component_count
from boldtailor._denoising_pool import (
    CANONICAL,
    _onoff_terms,
    analysis_components,
)
from boldtailor._hrf_cv import pooled_amplitude, prepare_runs
from boldtailor.hrf_library import default_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.parallel import map_blocks
from boldtailor.single_trial import fit_selected_hrfs

from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.features import null_session
from experiments.nsd_replication.inputs import analysis_data, load_nsdpreproc
from experiments.nsd_replication.ladder import (
    _labels,
    _stack,
    select_session_denoising,
)
from experiments.nsd_replication.run import _configure_logging
from experiments.nsd_replication.trials import trial_table

log = logging.getLogger(__name__)
COUNTS = tuple(range(11))
PCSTOP = 1.05
POOLS = ("onoff", "full")
RULES = ("timeseries", "consistency")


def task_r2(data, library, task_model, run_labels):
    """In-sample R² of ``task_model`` with the canonical HRF, coefficients
    shared across runs, each run's confounds plus intercept as nuisance
    (``onoff_r2`` with the full task model in place of the ON regressor)."""
    runs = prepare_runs(data, library, task_model, labels=run_labels)
    terms = [_onoff_terms(*args) for args in zip(runs, data.signals, run_labels)]
    a = sum(x.T @ x for x, _ in terms)
    b = sum(x.T @ yr for x, yr in terms)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        raise ValueError("full task design is singular across runs")
    sse = sum(np.sum((yr - x @ amplitude[0]) ** 2, axis=0) for x, yr in terms)
    sst = sum(np.sum(yr**2, axis=0) for _, yr in terms)
    ratio = np.full(data.n_features, np.nan)
    np.divide(sse, sst, out=ratio, where=sst > 0)
    return 1 - ratio


def badness(betas, trials):
    """GLMsingle ``calcbadness`` for leave-one-run-out: per feature, the summed
    squared error of each trial against the mean of its image's trials in the
    other runs (trials whose image has no trial in another run are skipped)."""
    images = trials["image"].to_numpy()
    runs = trials["run"].to_numpy()
    out = np.zeros(betas.shape[1])
    for run in np.unique(runs):
        test, train = runs == run, runs != run
        means = pd.DataFrame(betas[train]).groupby(images[train]).mean()
        held = images[test]
        keep = np.isin(held, means.index)
        if keep.any():
            predicted = means.loc[held[keep]].to_numpy()
            out += np.sum((betas[test][keep] - predicted) ** 2, axis=0)
    return out


def _blocks(features, size=2048):
    return [features[i : i + size] for i in range(0, len(features), size)]


def _selection_block(indices, session, library):
    return select_hrfs(
        analysis_data(session, indices),
        library=library,
        task_model=session.task_model,
        run_labels=_labels(session),
    )


def _fit_block(indices, session, selection, confounds):
    result = fit_selected_hrfs(
        analysis_data(session, indices, confounds),
        hrf_selection=selection[tuple(indices)],
        run_labels=_labels(session),
    )
    return _stack(result)


def _confounds(session, components, k):
    if k == 0:
        return session.confounds
    names = [f"pc_{i:02d}" for i in range(k)]
    return tuple(
        pd.concat(
            [frame, pd.DataFrame(c.prefix(k), columns=names, index=frame.index)],
            axis=1,
        )
        for frame, c in zip(session.confounds, components, strict=True)
    )


def _available(components, k):
    return all(not c.unavailable_reason(k) for c in components)


def consistency_perf(session, scoring, components, trials, library, n_jobs):
    """Per count: -median over ``scoring`` of leave-one-run-out badness of
    single-trial betas refitted with the first k PCs in every run."""
    blocks = _blocks(np.flatnonzero(scoring))
    selection = dict(
        (tuple(i), s)
        for i, s in map_blocks(
            _selection_block, blocks, args=(session, library), n_jobs=n_jobs
        )
    )
    perf = []
    for k in COUNTS:
        if not _available(components, k):
            perf.append(np.nan)
            continue
        start = time.monotonic()
        confounds = _confounds(session, components, k)
        parts = dict(
            (tuple(i), b)
            for i, b in map_blocks(
                _fit_block, blocks, args=(session, selection, confounds), n_jobs=n_jobs
            )
        )
        betas = np.hstack([parts[tuple(i)] for i in blocks])
        if betas.shape[0] != len(trials):
            raise ValueError("single-trial betas must have one row per trial")
        perf.append(-float(np.median(badness(betas, trials))))
        log.info("consistency k=%d: %.6g (%.0fs)", k, perf[-1], time.monotonic() - start)
    return np.array(perf)


def check(session, n_jobs=1, library=None):
    """Long table: one row per pool, rule and count; ``attrs`` hold choices."""
    library = default_hrf_library() if library is None else library
    labels = _labels(session)
    denoising = select_session_denoising(session, session.task_model, library)
    data = analysis_data(session, np.arange(session.signals[0].shape[1]))
    threshold = denoising.noise_pool_threshold
    scoring = np.asarray(denoising.scoring_mask)
    full = task_r2(data, library, session.task_model, labels)
    pools = {
        "onoff": np.asarray(denoising.noise_pool),
        "full": np.isfinite(full) & (full < threshold),
    }
    if (pools["full"] & scoring).any():
        raise ValueError("the full-model pool overlaps the scoring set")
    trials = trial_table(session.events, session.labels, session.session, "73k_id")
    hrf_indices = denoising.initial_selection.hrf_indices
    rows, choices = [], {}
    for pool_name, pool in pools.items():
        components = analysis_components(data, pool)
        timeseries = select_component_count(
            data,
            hrf_indices=hrf_indices,
            components=components,
            scoring=scoring,
            task_model=session.task_model,
            library=library,
            counts=COUNTS,
            pcstop=PCSTOP,
            run_labels=labels,
        ).scores["perf"].to_numpy(dtype=float)
        perfs = {
            "timeseries": timeseries,
            "consistency": consistency_perf(
                session, scoring, components, trials, library, n_jobs
            ),
        }
        for rule, perf in perfs.items():
            choices[f"{pool_name}_{rule}"] = choose_count(COUNTS, perf, PCSTOP)
            for k, p in zip(COUNTS, perf):
                rows.append(
                    dict(pool=pool_name, rule=rule, count=k, perf=p, curve=p - perf[0])
                )
        log.info("pool %s (%d features): %s", pool_name, pool.sum(),
                 {r: choices[f"{pool_name}_{r}"] for r in RULES})
    table = pd.DataFrame(rows)
    table.attrs.update(
        choices,
        selected=int(denoising.n_components),
        pcstop_count=int(denoising.pcstop_count),
        threshold=float(threshold),
        n_scoring=int(scoring.sum()),
        n_pool_onoff=int(pools["onoff"].sum()),
        n_pool_full=int(pools["full"].sum()),
    )
    return table


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    _configure_logging()
    logging.getLogger(__name__).setLevel(logging.INFO)

    config = load_config(args.config)
    real = load_nsdpreproc(config, args.subject, args.session)
    # The features stage's null seed: (subject number, session index).
    seed = (int(args.subject.removeprefix("sub-")), config.sessions.index(args.session))
    out = config.output_dir / "checks" / "denoising" / args.subject
    out.mkdir(parents=True, exist_ok=True)
    for condition, session in (("real", real), ("null", null_session(real, seed))):
        start = time.monotonic()
        table = check(session, args.n_jobs)
        path = out / f"{args.session}_{condition}.tsv"
        table.assign(condition=condition, **table.attrs).to_csv(
            path, sep="\t", index=False
        )
        log.info("%s %s in %.0fs: %s; wrote %s", args.session, condition,
                 time.monotonic() - start, table.attrs, path)


if __name__ == "__main__":
    main()

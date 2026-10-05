"""Resumable fit and metrics stages of the NSD replication experiments."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from experiments.nsd_replication.betas import (
    fit_dir,
    inputs_digest,
    is_complete,
    read_fit,
    write_fit,
    zscore,
)
from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.inputs import load_ppdata, ppdata_brain
from experiments.nsd_replication.ladder import LEVELS, LSS_LEVELS, fit_ladder
from experiments.nsd_replication.metrics import (
    THRESHOLDS,
    decoding_accuracy,
    lagged_correlation,
    rdm,
    rdm_agreement,
    threshold_curves,
    voxel_reliability,
)
from experiments.nsd_replication.roi import load_roi
from experiments.nsd_replication.trials import (
    images_with,
    out_of_sample_images,
    repetition_array,
    trial_table,
)

SOURCES = ("ppdata",)
R4_THRESHOLDS = (0.0, 0.3)
R6_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4)
RSA_THRESHOLD = 0.0


# ---------------------------------------------------------------- fit stage


def _load(config, source, subject, session):
    if source != "ppdata":
        raise ValueError(f"source {source!r} has no fit stage; only ppdata is fitted")
    return load_ppdata(config, subject, session)


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _metadata(source, session, digest, fit):
    meta = dict(
        fit.record,
        source=source,
        session=session,
        inputs_digest=digest,
        boldtailor_version=version("boldtailor"),
        git_commit=_git_commit(),
    )
    return json.loads(json.dumps(meta, default=_jsonable))


def _check_digests(paths, digest):
    for path in paths:
        if not is_complete(path):
            continue
        stored = json.loads((path / "metadata.json").read_text())["inputs_digest"]
        if stored != digest:
            raise ValueError(
                f"inputs digest differs from the stored fit {path}; use --refit"
            )


def _write_levels(config, source, subject, session, data, fits):
    digest = inputs_digest(data)
    trials = trial_table(data.events, data.labels, session, config.image_column)
    for level, fit in fits.items():
        path = fit_dir(config.output_dir, source, subject, session, level)
        meta = _metadata(source, session, digest, fit)
        write_fit(path, fit.betas, trials, meta, fit.extras)


def fit_session(config, source, subject, session, levels, refit=False):
    paths = {
        level: fit_dir(config.output_dir, source, subject, session, level)
        for level in levels
    }
    data = _load(config, source, subject, session)
    if not refit:
        _check_digests(paths.values(), inputs_digest(data))
    missing = [lv for lv in levels if refit or not is_complete(paths[lv])]
    if missing:
        fits = fit_ladder(
            data, levels=missing, block_size=config.block_size, n_jobs=config.n_jobs
        )
        _write_levels(config, source, subject, session, data, fits)
    return list(paths.values())


# ------------------------------------------------------------ metrics stage


def _metrics_dir(config, source, subject=None):
    base = config.output_dir / "metrics" / source
    return base if subject is None else base / subject


def _level_betas(config, source, subject, level):
    """Per-session z-scored betas and trial tables for one level."""
    pairs = []
    for session in config.sessions:
        path = fit_dir(config.output_dir, source, subject, session, level)
        betas, trials, _ = read_fit(path)
        pairs.append((zscore(betas), trials))
    return pairs


def _subject_betas(config, source, subject, level):
    pairs = _level_betas(config, source, subject, level)
    betas = np.vstack([b for b, _ in pairs])
    trials = pd.concat([t for _, t in pairs], ignore_index=True)
    return betas, trials


def _reliabilities(betas, trials, images):
    """Per-level voxel reliability; all NaN when no image qualifies."""
    out = {}
    for level, values in betas.items():
        reps = repetition_array(values, trials, images)
        nan = np.full(values.shape[1], np.nan)
        out[level] = voxel_reliability(reps) if len(images) else nan
    return out


def _composite(reliabilities):
    return np.vstack(list(reliabilities.values())).mean(axis=0)


def _median_table(reliabilities, roi):
    rows = []
    for level, values in reliabilities.items():
        finite = values[roi & np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else np.nan
        rows.append((level, median))
    return pd.DataFrame(rows, columns=["version", "median"])


def _mask(roi, composite, threshold):
    return roi & np.isfinite(composite) & (composite >= threshold)


def _session_lags(config, source, subject, level, mask):
    frames = [
        lagged_correlation(betas, trials, mask)
        for betas, trials in _level_betas(config, source, subject, level)
    ]
    return (
        pd.concat(frames)
        .groupby("lag", as_index=False)
        .agg(mean_r=("mean_r", "mean"), n_pairs=("n_pairs", "sum"))
    )


def _r4_table(config, source, subject, levels, composite, roi, threshold):
    mask = _mask(roi, composite, threshold)
    frames = []
    for level in levels:
        lags = _session_lags(config, source, subject, level, mask)
        frames.append(lags.assign(version=level, threshold=threshold))
    columns = ["version", "threshold", "lag", "mean_r", "n_pairs"]
    return pd.concat(frames, ignore_index=True)[columns]


def _r6_table(reps, composite, roi):
    rows = []
    for level, values in reps.items():
        for t in R6_THRESHOLDS:
            result = decoding_accuracy(values, _mask(roi, composite, t))
            rows.append(dict(version=level, threshold=t, **result))
    return pd.DataFrame(rows)


def _repetitions(betas, trials, images):
    return {lv: repetition_array(b, trials, images) for lv, b in betas.items()}


def _write_tables(config, source, subject, tables):
    out = _metrics_dir(config, source, subject)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)


def _subject_roi(config, subject):
    brain = ppdata_brain(config, subject, config.sessions[0])
    return load_roi(config, subject, brain)


def subject_metrics(config, source, subject, levels):
    roi = _subject_roi(config, subject)
    loaded = {lv: _subject_betas(config, source, subject, lv) for lv in levels}
    betas = {lv: b for lv, (b, _) in loaded.items()}
    trials = loaded[levels[0]][1]
    images = images_with(trials, 3)
    rel = _reliabilities(betas, trials, images)
    composite = _composite(rel)
    oos = _reliabilities(betas, trials, out_of_sample_images(trials))
    tables = {
        "r1": threshold_curves(rel, roi),
        "r1_median": _median_table(rel, roi),
        "r3b": threshold_curves(oos, roi),
        "r6": _r6_table(_repetitions(betas, trials, images), composite, roi),
    }
    for t in R4_THRESHOLDS:
        tables[f"r4_t{t}"] = _r4_table(
            config, source, subject, levels, composite, roi, t
        )
    _write_tables(config, source, subject, tables)
    return tables


# ---------------------------------------------------------------- group RSA


def _subject_pattern(config, source, subject, level, images_for):
    roi = _subject_roi(config, subject)
    betas, trials = _subject_betas(config, source, subject, level)
    rel = voxel_reliability(repetition_array(betas, trials, images_with(trials, 3)))
    reps = repetition_array(betas, trials, images_for)
    return reps.mean(axis=0), _mask(roi, rel, RSA_THRESHOLD)


def _shared_images(config, source, subjects, level):
    sets = []
    for subject in subjects:
        trials = _subject_betas(config, source, subject, level)[1]
        sets.append(set(images_with(trials, 3)))
    return np.array(sorted(set.intersection(*sets)))


def group_rsa(config, source, subjects, level):
    shared = _shared_images(config, source, subjects, level)
    rdms = {}
    for subject in subjects:
        patterns, mask = _subject_pattern(config, source, subject, level, shared)
        rdms[subject] = rdm(patterns[:, mask])
    table = rdm_agreement(rdms).assign(version=level, n_images=len(shared))
    out = _metrics_dir(config, source)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"rsa_{level}.tsv", sep="\t", index=False)
    return table


# ----------------------------------------------------------------- cleanup


def discard_betas(config, source, subject, levels, sessions=None):
    for session in sessions or config.sessions:
        for level in levels:
            path = fit_dir(config.output_dir, source, subject, session, level)
            (path / "betas.npy").unlink(missing_ok=True)


# ---------------------------------------------------------------------- CLI


def _parser():
    parser = argparse.ArgumentParser(prog="nsd_replication.run")
    parser.add_argument("command", choices=("fit", "metrics"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", choices=SOURCES, default="ppdata")
    parser.add_argument("--levels", nargs="+", choices=LEVELS + LSS_LEVELS)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--discard-betas-after-metrics", action="store_true")
    return parser


def _run_fit(config, args, levels):
    for subject in config.subjects:
        for session in config.sessions:
            fit_session(config, args.source, subject, session, levels, args.refit)


def _run_metrics(config, args, levels):
    for subject in config.subjects:
        subject_metrics(config, args.source, subject, levels)
    if len(config.subjects) > 1:
        for level in levels:
            group_rsa(config, args.source, config.subjects, level)
    if args.discard_betas_after_metrics:
        for subject in config.subjects:
            discard_betas(config, args.source, subject, levels)


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    levels = tuple(args.levels or LEVELS)
    {"fit": _run_fit, "metrics": _run_metrics}[args.command](config, args, levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

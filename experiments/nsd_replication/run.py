"""Resumable fit and metrics stages of the NSD replication experiments."""

import argparse
from functools import cached_property
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
from experiments.nsd_replication.comparison import (
    VERSIONS,
    alignment_frame,
    alignment_row,
    check_floor,
    check_trial_tables,
    reliability_differences,
)
from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.figure_stage import make_figures
from experiments.nsd_replication.features import session_features
from experiments.nsd_replication.inputs import load_ppdata, ppdata_brain
from experiments.nsd_replication.ladder import LEVELS, LSS_LEVELS, fit_ladder
from experiments.nsd_replication.metric_tables import ensure_tables
from experiments.nsd_replication.metrics import (
    THRESHOLDS,
    decoding_accuracies,
    lagged_correlation,
    rdm,
    rdm_agreement,
    threshold_curves,
    voxel_reliability,
)
from experiments.nsd_replication.released import RELEASED, index_released
from experiments.nsd_replication.roi import load_roi
from experiments.nsd_replication.trials import (
    images_with,
    out_of_sample_images,
    repetition_array,
    trial_table,
)

SOURCES = ("ppdata", "released", "comparison")
R4_THRESHOLDS = (0.0, 0.3)
R6_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4)
RSA_THRESHOLDS = (0.0, 0.2, 0.4)
SUBJECT_TABLES = ("r1", "r1_median", "r3b", "r4_t0.0", "r4_t0.3", "r6")


# ---------------------------------------------------------------- fit stage


def _load(config, source, subject, session):
    if source != "ppdata":
        raise ValueError(f"source {source!r} has no fit stage; only ppdata is fitted")
    return load_ppdata(config, subject, session)


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
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


def _level_writer(config, source, subject, session, data):
    """A ``(level, fit)`` callback writing one level's fit to disk."""
    digest = inputs_digest(data)
    trials = trial_table(data.events, data.labels, session, config.image_column)

    def write(level, fit):
        path = fit_dir(config.output_dir, source, subject, session, level)
        meta = _metadata(source, session, digest, fit)
        write_fit(path, fit.betas, trials, meta, fit.extras)

    return write


def _write_levels(config, source, subject, session, data, fits):
    write = _level_writer(config, source, subject, session, data)
    for level, fit in fits.items():
        write(level, fit)


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
        fit_ladder(
            data,
            levels=missing,
            block_size=config.block_size,
            n_jobs=config.n_jobs,
            on_fit=_level_writer(config, source, subject, session, data),
        )
    return list(paths.values())


# ------------------------------------------------------------ metrics stage


def _metrics_dir(config, source, subject=None):
    base = config.output_dir / "metrics" / source
    return base if subject is None else base / subject


def _level_betas(config, source, subject, level, columns=None):
    """Per-session z-scored betas (optionally only ``columns``) and trials."""
    pairs = []
    for session in config.sessions:
        path = fit_dir(config.output_dir, source, subject, session, level)
        betas, trials, _ = read_fit(path)
        scored = zscore(betas)
        pairs.append((scored if columns is None else scored[:, columns], trials))
    return pairs


def _subject_betas(config, source, subject, level, columns=None):
    pairs = _level_betas(config, source, subject, level, columns)
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


class SubjectData:
    """Lazily loaded z-scored betas, trials and reliabilities per version.

    ``load(version)`` returns stacked ``(betas, trials)``; ``roi`` masks the
    loaded columns.
    """

    def __init__(self, load, versions, roi):
        self._load = load
        self.versions = tuple(versions)
        self.roi = roi

    @cached_property
    def _loaded(self):
        return {v: self._load(v) for v in self.versions}

    @cached_property
    def betas(self):
        return {v: b for v, (b, _) in self._loaded.items()}

    @cached_property
    def trials(self):
        return self._loaded[self.versions[0]][1]

    @cached_property
    def images(self):
        return images_with(self.trials, 3)

    @cached_property
    def reliabilities(self):
        return _reliabilities(self.betas, self.trials, self.images)

    @cached_property
    def out_of_sample(self):
        return _reliabilities(
            self.betas, self.trials, out_of_sample_images(self.trials)
        )

    @cached_property
    def composite(self):
        return _composite(self.reliabilities)


def _median_table(reliabilities, roi):
    rows = []
    for level, values in reliabilities.items():
        finite = values[roi & np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else np.nan
        rows.append((level, median))
    return pd.DataFrame(rows, columns=["version", "median"])


def _mask(roi, composite, threshold):
    return roi & np.isfinite(composite) & (composite >= threshold)


def _session_lags(betas, trials, mask):
    """Lag curve per session (in stacking order), averaged over sessions."""
    frames = []
    for session in trials["session"].unique():
        rows = (trials["session"] == session).to_numpy()
        frames.append(lagged_correlation(betas[rows], trials[rows], mask))
    return (
        pd.concat(frames)
        .groupby("lag", as_index=False)
        .agg(mean_r=("mean_r", "mean"), n_pairs=("n_pairs", "sum"))
    )


def _r4_table(data, threshold):
    mask = _mask(data.roi, data.composite, threshold)
    frames = [
        _session_lags(betas, data.trials, mask).assign(version=v, threshold=threshold)
        for v, betas in data.betas.items()
    ]
    columns = ["version", "threshold", "lag", "mean_r", "n_pairs"]
    return pd.concat(frames, ignore_index=True)[columns]


def _r6_table(data, n_jobs):
    masks = [_mask(data.roi, data.composite, t) for t in R6_THRESHOLDS]
    rows = []
    for version, betas in data.betas.items():
        reps = repetition_array(betas, data.trials, data.images)
        results = decoding_accuracies(reps, masks, n_jobs=n_jobs)
        rows += [
            dict(version=version, threshold=t, **r)
            for t, r in zip(R6_THRESHOLDS, results)
        ]
    return pd.DataFrame(rows)


def _table_builders(n_jobs):
    """Metric name -> builder(data), cheapest first."""
    builders = {
        "r1": lambda d: threshold_curves(d.reliabilities, d.roi),
        "r1_median": lambda d: _median_table(d.reliabilities, d.roi),
        "r3b": lambda d: threshold_curves(d.out_of_sample, d.roi),
    }
    for t in R4_THRESHOLDS:
        builders[f"r4_t{t}"] = lambda d, t=t: _r4_table(d, t)
    builders["r6"] = lambda d: _r6_table(d, n_jobs)
    return builders


def _write_tables(config, source, subject, tables):
    out = _metrics_dir(config, source, subject)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)


def _subject_roi(config, subject):
    brain = ppdata_brain(config, subject, config.sessions[0])
    return load_roi(config, subject, brain)


def subject_metrics(config, source, subject, levels, recompute=False):
    """Every metric on ROI columns only (betas are z-scored before slicing)."""
    columns = np.flatnonzero(_subject_roi(config, subject))
    data = SubjectData(
        lambda lv: _subject_betas(config, source, subject, lv, columns),
        levels,
        np.ones(len(columns), bool),
    )
    out = _metrics_dir(config, source, subject)
    return ensure_tables(out, _table_builders(config.n_jobs), data, recompute)


# ------------------------------------------------------- comparison stage


def _comparison_dir(config, subject):
    return _metrics_dir(config, "comparison", subject)


def _version_paths(config, subject, session):
    return {
        v: fit_dir(
            config.output_dir, v.split(":")[0], subject, session, v.split(":")[1]
        )
        for v in VERSIONS
    }


def _check_session_tables(config, subject):
    for session in config.sessions:
        paths = _version_paths(config, subject, session)
        tables = {v: pd.read_csv(p / "trials.tsv", sep="\t") for v, p in paths.items()}
        check_trial_tables(tables, session)


def _combined_betas(config, subject):
    loaded = {}
    for version in VERSIONS:
        source, level = version.split(":")
        loaded[version] = _subject_betas(config, source, subject, level)
    return loaded


def comparison_metrics(config, subject):
    _check_session_tables(config, subject)
    roi = _subject_roi(config, subject)
    loaded = _combined_betas(config, subject)
    trials = loaded[VERSIONS[0]][1]
    betas = {v: b for v, (b, _) in loaded.items()}
    rel = _reliabilities(betas, trials, images_with(trials, 3))
    tables = {"r1": threshold_curves(rel, roi), "r1_median": _median_table(rel, roi)}
    out = _comparison_dir(config, subject)
    _write_tables(config, "comparison", subject, tables)
    for level, diff in reliability_differences(rel).items():
        np.save(out / f"r1_difference_{level}.npy", diff)
    return tables


def _session_alignment(config, subject, session, roi):
    paths = _version_paths(config, subject, session)
    ours, trials, _ = read_fit(paths["ppdata:b1"])
    released, _, _ = read_fit(paths["released:b1"])
    return alignment_row(session, ours, released, trials, roi)


def alignment_table(config, subject):
    roi = _subject_roi(config, subject)
    rows = [_session_alignment(config, subject, s, roi) for s in config.sessions]
    table = alignment_frame(rows)
    _write_tables(config, "comparison", subject, {"alignment": table})
    check_floor(rows, subject, config.alignment_floor)
    return table


def run_comparison(config, subject):
    """Alignment first (written even on failure), then the comparison metrics."""
    alignment_table(config, subject)
    return comparison_metrics(config, subject)


# ---------------------------------------------------------------- group RSA


def _shared_images(trials_by_subject):
    sets = [set(images_with(t, 3)) for t in trials_by_subject.values()]
    return np.array(sorted(set.intersection(*sets)))


def _subject_inputs(config, source, subject, levels):
    loaded = {lv: _subject_betas(config, source, subject, lv) for lv in levels}
    betas = {lv: b for lv, (b, _) in loaded.items()}
    trials = loaded[levels[0]][1]
    rel = _reliabilities(betas, trials, images_with(trials, 3))
    return dict(betas=betas, trials=trials, composite=_composite(rel))


def _mean_patterns(inputs, images):
    return {
        lv: repetition_array(b, inputs["trials"], images).mean(axis=0)
        for lv, b in inputs["betas"].items()
    }


def _rsa_rows(patterns, masks, level, threshold):
    rdms = {s: rdm(patterns[s][level][:, masks[s](threshold)]) for s in patterns}
    return rdm_agreement(rdms).assign(threshold=threshold)


def _level_rsa(patterns, masks, level):
    frames = [_rsa_rows(patterns, masks, level, t) for t in RSA_THRESHOLDS]
    columns = ["threshold", "subject_a", "subject_b", "r"]
    return pd.concat(frames, ignore_index=True)[columns]


def group_rsa(config, source, subjects, levels):
    inputs = {s: _subject_inputs(config, source, s, levels) for s in subjects}
    shared = _shared_images({s: i["trials"] for s, i in inputs.items()})
    patterns = {s: _mean_patterns(i, shared) for s, i in inputs.items()}
    rois = {s: _subject_roi(config, s) for s in subjects}
    masks = {
        s: (lambda t, s=s: _mask(rois[s], inputs[s]["composite"], t)) for s in subjects
    }
    tables = {lv: _level_rsa(patterns, masks, lv) for lv in levels}
    out = _metrics_dir(config, source)
    out.mkdir(parents=True, exist_ok=True)
    for level, table in tables.items():
        table.to_csv(out / f"rsa_{level}.tsv", sep="\t", index=False)
    return pd.concat(tables, names=["version"])


# ----------------------------------------------------------------- cleanup


def _expected_tables(config, source, subject, levels, multiple_subjects):
    names = [f"{n}.tsv" for n in SUBJECT_TABLES]
    paths = [_metrics_dir(config, source, subject) / n for n in names]
    if multiple_subjects:
        paths += [_metrics_dir(config, source) / f"rsa_{lv}.tsv" for lv in levels]
    return paths


def _mark_discarded(path):
    meta_path = path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(
        json.dumps(dict(meta, betas_discarded=True), indent=2, sort_keys=True)
    )
    (path / "betas.npy").unlink(missing_ok=True)


def _check_comparison_done(config, source, subject):
    if source != "ppdata" or config.released_dir is None:
        return
    out = _comparison_dir(config, subject)
    missing = [n for n in ("alignment.tsv", "r1.tsv") if not (out / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"refusing to discard betas; missing comparison outputs {missing}; "
            "run `metrics --source comparison` first"
        )


def discard_betas(config, source, subject, levels, multiple_subjects=False):
    _check_comparison_done(config, source, subject)
    expected = _expected_tables(config, source, subject, levels, multiple_subjects)
    missing = [p.name for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"refusing to discard betas; missing metrics: {missing}"
        )
    for session in config.sessions:
        for level in levels:
            _mark_discarded(fit_dir(config.output_dir, source, subject, session, level))


# --------------------------------------------------------- feature stage

FEATURE_TABLES = ("gate", "rt", "hrf")


def _write_missing(config, subject, session, data, fits):
    paths = {
        lv: fit_dir(config.output_dir, "ppdata", subject, session, lv) for lv in fits
    }
    _check_digests(paths.values(), inputs_digest(data))
    missing = {lv: f for lv, f in fits.items() if not is_complete(paths[lv])}
    _write_levels(config, "ppdata", subject, session, data, missing)


def _session_features(config, subject, index, session, roi):
    """Seed ``index`` (the session's position in the config) for the null."""
    data = _load(config, "ppdata", subject, session)
    out = session_features(
        data, roi, index, session, block_size=config.block_size, n_jobs=config.n_jobs
    )
    _write_missing(config, subject, session, data, out["fits"])
    return out


def subject_features(config, subject):
    roi = _subject_roi(config, subject)
    outs = [
        _session_features(config, subject, i, s, roi)
        for i, s in enumerate(config.sessions)
    ]
    tables = {
        f"features_{name}": pd.concat([o[name] for o in outs], ignore_index=True)
        for name in FEATURE_TABLES
    }
    _write_tables(config, "ppdata", subject, tables)
    return tables


def _run_features(config, args, levels):
    if args.source != "ppdata":
        raise ValueError("feature experiments run on ppdata only")
    for subject in config.subjects:
        subject_features(config, subject)


# ---------------------------------------------------------------------- CLI


def _parser():
    parser = argparse.ArgumentParser(prog="nsd_replication.run")
    parser.add_argument("command", choices=("fit", "metrics", "features", "figures"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", choices=SOURCES, default="ppdata")
    parser.add_argument("--levels", nargs="+", choices=LEVELS + LSS_LEVELS)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--discard-betas-after-metrics", action="store_true")
    return parser


def _run_fit(config, args, levels):
    if args.source == "released":
        return _index_all_released(config)
    if args.source == "comparison":
        raise ValueError("comparison has no fit stage")
    for subject in config.subjects:
        for session in config.sessions:
            fit_session(config, args.source, subject, session, levels, args.refit)


def _index_all_released(config):
    for subject in config.subjects:
        for session in config.sessions:
            index_released(config, subject, session)


def _run_comparison(config):
    for subject in config.subjects:
        run_comparison(config, subject)


def _run_metrics(config, args, levels):
    if args.source == "comparison":
        return _run_comparison(config)
    for subject in config.subjects:
        subject_metrics(config, args.source, subject, levels, args.recompute)
    multiple = len(config.subjects) > 1
    if multiple:
        group_rsa(config, args.source, config.subjects, levels)
    if args.discard_betas_after_metrics:
        for subject in config.subjects:
            discard_betas(config, args.source, subject, levels, multiple)


def _run_figures(config, args, levels):
    make_figures(config)


def _default_levels(source):
    return tuple(RELEASED) if source == "released" else LEVELS


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    levels = tuple(args.levels or _default_levels(args.source))
    commands = {
        "fit": _run_fit,
        "metrics": _run_metrics,
        "features": _run_features,
        "figures": _run_figures,
    }
    commands[args.command](config, args, levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

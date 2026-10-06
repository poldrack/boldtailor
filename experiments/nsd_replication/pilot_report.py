"""Variability of the median ROI reliability across session subsets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.ladder import LEVELS, LSS_LEVELS
from experiments.nsd_replication.metrics import voxel_reliability
from experiments.nsd_replication.run import (
    _default_levels,
    _level_betas,
    _metrics_dir,
    _subject_roi,
)
from experiments.nsd_replication.trials import images_with, repetition_array


def _median_reliability(pairs, roi):
    betas = np.vstack([b for b, _ in pairs])
    trials = pd.concat([t for _, t in pairs], ignore_index=True)
    images = images_with(trials, 3)
    values = voxel_reliability(repetition_array(betas, trials, images))
    finite = values[roi & np.isfinite(values)]
    return float(np.median(finite))


def _leave_one_out(pairs, roi):
    subsets = [pairs[:i] + pairs[i + 1 :] for i in range(len(pairs))]
    return [_median_reliability(s, roi) for s in subsets]


def _split_sd(pairs, roi):
    odd, even = _median_reliability(pairs[::2], roi), _median_reliability(
        pairs[1::2], roi
    )
    return abs(odd - even) / np.sqrt(2)


def _version_row(config, source, subject, level, roi):
    pairs = _level_betas(config, source, subject, level)
    return {
        "version": level,
        "loso_sd": float(np.std(_leave_one_out(pairs, roi), ddof=1)),
        "split_sd": float(_split_sd(pairs, roi)),
        "median": _median_reliability(pairs, roi),
    }


def variability(config, source, levels, subject=None):
    subject = subject or config.subjects[0]
    roi = _subject_roi(config, subject)
    rows = [_version_row(config, source, subject, lv, roi) for lv in levels]
    return pd.DataFrame(rows, columns=["version", "loso_sd", "split_sd", "median"])


def _parser():
    parser = argparse.ArgumentParser(prog="nsd_replication.pilot_report")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", choices=("ppdata", "released"), default="ppdata")
    parser.add_argument("--levels", nargs="+", choices=LEVELS + LSS_LEVELS)
    return parser


def main(argv=None):
    """Print and write metrics/<source>/<subject>/variability.tsv per subject."""
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    levels = tuple(args.levels or _default_levels(args.source))
    for subject in config.subjects:
        table = variability(config, args.source, levels, subject)
        out = _metrics_dir(config, args.source, subject)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "variability.tsv", sep="\t", index=False)
        print(f"{subject}\n{table.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

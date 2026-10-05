"""Calibrate the constant offset between BIDS onsets and the ppdata time base."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.nsd_replication.config import load_config
from experiments.nsd_replication.inputs import load_ppdata
from experiments.nsd_replication.ladder import fit_ladder
from experiments.nsd_replication.released import (
    alignment_check,
    load_released,
    released_path,
)
from experiments.nsd_replication.roi import load_roi

CANDIDATES = np.round(np.arange(-2, 2.01, 1 / 3), 4)


def _n_trials(session):
    return sum(len(events) for events in session.events)


def _released_b1(config, subject, session, data):
    path = released_path(config, subject, session, "b1")
    name = f"{subject} {session} b1"
    return load_released(path, data.brain, _n_trials(data), name)


def _score_offset(config, subject, session, offset):
    config = replace(config, onset_offset=float(offset))
    data = load_ppdata(config, subject, session)
    fit = fit_ladder(
        data, levels=("b1",), block_size=config.block_size, n_jobs=config.n_jobs
    )
    roi = load_roi(config, subject, data.brain)
    released = _released_b1(config, subject, session, data)
    run_length = len(data.events[0])
    scores = alignment_check(fit["b1"].betas, released, run_length, roi)
    return {"onset_offset": float(offset), **scores}


def calibrate(config, subject, session, offsets):
    rows = [_score_offset(config, subject, session, o) for o in offsets]
    return pd.DataFrame(rows, columns=["onset_offset", "true_median", "null_median"])


def best_offset(table):
    return float(table.loc[table["true_median"].idxmax(), "onset_offset"])


def _parser():
    parser = argparse.ArgumentParser(prog="nsd_replication.calibrate_offset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    table = calibrate(config, args.subject, args.session, CANDIDATES)
    out = config.output_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.subject}_{args.session}_onset_offset.tsv"
    table.to_csv(out / name, sep="\t", index=False)
    print(table.to_string(index=False))
    print(f"best onset_offset: {best_offset(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

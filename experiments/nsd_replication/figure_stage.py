"""Figures stage: render PNGs from whichever metric tables exist."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.nsd_replication import figures
from experiments.nsd_replication.comparison import LEVELS

R2_LEVELS = ("b2", "b2-lib20")


def _read(path):
    return pd.read_csv(path, sep="\t") if path.is_file() else None


def _r4(table):
    return None if table is None else dict(tuple(table.groupby("version", sort=False)))


def _lss_rows(r1):
    keep = r1["version"].isin(["b4", "lss-assume", "lss-fit"])
    return r1[keep] if "lss-fit" in set(r1["version"]) else None


def _subject_figures(metrics):
    """name -> zero-argument builder, given a dict of loaded tables."""
    r1, r3b, r6 = metrics.get("r1"), metrics.get("r3b"), metrics.get("r6")
    rt, gate = metrics.get("features_rt"), metrics.get("features_gate")
    return {
        "r1": (r1, figures.fig_r1_curves),
        "r3b": (r3b, figures.fig_r1_curves),
        "r3_lss": (None if r1 is None else _lss_rows(r1), figures.fig_r3_lss),
        "r6": (r6, figures.fig_r6_decoding),
        "r4_t0.0": (_r4(metrics.get("r4_t0.0")), figures.fig_r4_lag),
        "r4_t0.3": (_r4(metrics.get("r4_t0.3")), figures.fig_r4_lag),
        "gate": (gate, figures.fig_gate),
        "modulators": (rt, figures.fig_modulators),
        "r2": (metrics.get("r2"), figures.fig_r2_hrf),
    }


def _load_subject_tables(metrics_dir):
    names = ("r1", "r3b", "r6", "r4_t0.0", "r4_t0.3", "features_rt", "features_gate")
    loaded = {n: _read(metrics_dir / f"{n}.tsv") for n in names}
    loaded["r2"] = _r2_summaries(metrics_dir)
    return {n: t for n, t in loaded.items() if t is not None}


def _r2_summaries(metrics_dir):
    """ROI summaries of R2, one row per HRF library level found."""
    found = [_read(metrics_dir / f"r2_{lv}_roi.tsv") for lv in R2_LEVELS]
    found = [t for t in found if t is not None]
    return pd.concat(found, ignore_index=True) if found else None


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render(items, out_dir, label):
    for name, (data, builder) in items.items():
        if data is None or len(data) == 0:
            print(f"skipped {label}/{name}: input tables missing")
            continue
        _save(builder(data), out_dir / f"{name}.png")


def subject_figures(config, subject):
    metrics = _load_subject_tables(config.output_dir / "metrics" / "ppdata" / subject)
    out = config.output_dir / "figures" / subject
    _render(_subject_figures(metrics), out, subject)


def _parity_table(config):
    frames = []
    for subject in config.subjects:
        median = _read(
            config.output_dir / "metrics" / "comparison" / subject / "r1_median.tsv"
        )
        if median is not None:
            frames.append(_subject_parity(subject, median))
    return pd.concat(frames, ignore_index=True) if frames else None


def _subject_parity(subject, median):
    values = dict(zip(median["version"], median["median"]))
    rows = [
        dict(
            subject=subject,
            level=lv,
            boldtailor=values[f"ppdata:{lv}"],
            released=values[f"released:{lv}"],
        )
        for lv in LEVELS
    ]
    return pd.DataFrame(rows)


def _rsa_tables(config):
    found = {}
    for level in LEVELS:
        table = _read(config.output_dir / "metrics" / "ppdata" / f"rsa_{level}.tsv")
        if table is not None:
            found[level] = table
    return found or None


def group_figures(config):
    out = config.output_dir / "figures" / "group"
    items = {
        "rsa": (_rsa_tables(config), figures.fig_r5_rsa),
        "parity": (_parity_table(config), figures.fig_parity),
    }
    _render(items, out, "group")


def make_figures(config):
    for subject in config.subjects:
        subject_figures(config, subject)
    group_figures(config)

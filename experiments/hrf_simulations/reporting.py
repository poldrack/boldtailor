"""Portable recovery summaries with failures and paired Monte Carlo comparisons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hrf_simulations.io import atomic_text, verify, write_json

GROUPS = [
    "session",
    "truth",
    "mechanism",
    "convention",
    "noise",
    "snr",
    "model",
    "estimator",
    "split",
]


def summarize(table):
    groups = GROUPS + [
        c
        for c in (
            "block",
            "regressor_scaling",
            "run_coefficient_sd",
            "gamma",
            "betas",
            "match_rt_rms",
            "generating_convention",
            "snr_level",
        )
        if c in table
    ]
    result = (
        table.groupby(groups, dropna=False)
        .agg(
            n=("failed", "size"),
            n_failed=("failed", "sum"),
            failure_rate=("failed", "mean"),
            n_recovered=("curve_rmse", "count"),
            curve_rmse_mean=("curve_rmse", "mean"),
            curve_rmse_sd=("curve_rmse", "std"),
            peak_absolute_error_mean=("peak_absolute_error", "mean"),
        )
        .reset_index()
    )
    result["curve_rmse_mcse"] = result.curve_rmse_sd / np.sqrt(result.n_recovered)
    result["failure_rate_mcse"] = np.sqrt(
        result.failure_rate * (1 - result.failure_rate) / result.n
    )
    return result


def paired_differences(table):
    index = ["id", "model", "convention", "split"]
    keys = ["session", "truth", "mechanism", "noise", "snr", "repeat"]
    wide = table.pivot(
        index=index + keys, columns="estimator", values="curve_rmse"
    ).reset_index()
    if "ar1" in wide and "iid" in wide:
        wide["ar1_minus_iid_rmse"] = wide.ar1 - wide.iid
    return wide


def half_stability(table, curves):
    rows = []
    for (model, estimator, convention), group in table.groupby(
        ["model", "estimator", "convention"]
    ):
        halves = group[group.split.isin(["half1", "half2"])].sort_values("split")
        if len(halves) != 2:
            continue
        a, b = curves[halves.index.to_numpy()]
        valid = np.isfinite(a).all() and np.isfinite(b).all()
        rows.append(
            dict(
                id=group.id.iloc[0],
                model=model,
                estimator=estimator,
                convention=convention,
                both_recovered=bool(valid),
                half_shape_correlation=(
                    float(np.corrcoef(a, b)[0, 1]) if valid else np.nan
                ),
            )
        )
    return rows


def _plot(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = table[(table.split == "full") & (~table.failed)]
    mechanisms = list(table.mechanism.unique())
    fig, axes = plt.subplots(
        1, len(mechanisms), figsize=(5 * len(mechanisms), 4), squeeze=False
    )
    for ax, mechanism in zip(axes[0], mechanisms, strict=True):
        data = full[full.mechanism == mechanism]
        if not data.empty:
            summary = data.groupby(["model", "estimator"]).curve_rmse.mean().unstack()
            summary.plot.bar(ax=ax)
        ax.set_title(mechanism)
        ax.set_ylabel("HRF RMSE (successful fits)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(
        "Descriptive recovery overview\nSee summary.tsv for individual conditions",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _noise_rows(manifest, generation):
    rows = []
    family = manifest["case"]["noise"]
    for i, (diag, donor) in enumerate(
        zip(
            generation["noise_diagnostics"],
            generation["donor_diagnostics"],
            strict=True,
        )
    ):
        rho = manifest["profile"]["rho"] if family == "ar1" else 0.0
        expected = (
            donor["acf"]
            if family == "spectral"
            else rho ** np.arange(1, len(diag["acf"]) + 1)
        )
        power = np.array(diag["periodogram"])
        donor_power = np.array(donor["periodogram"])
        rows.append(
            dict(
                id=manifest["case"]["id"],
                run=i,
                noise=family,
                variance=diag["variance"],
                acf1=diag["acf"][0],
                donor_variance=donor["variance"],
                donor_rho=donor["acf"][0],
                expected_variance=(
                    donor["variance"]
                    if family == "spectral"
                    else manifest["profile"]["variance"]
                ),
                acf_rmse=float(
                    np.sqrt(np.mean((np.array(diag["acf"]) - expected) ** 2))
                ),
                spectral_relative_power_error=(
                    float(
                        np.linalg.norm(power - donor_power)
                        / np.linalg.norm(donor_power)
                    )
                    if family == "spectral" and np.linalg.norm(donor_power) > 0
                    else np.nan
                ),
            )
        )
    return rows


def _expanded_reports(table, crossed, destination):
    if crossed:
        cross = pd.concat(crossed, ignore_index=True)
        cross.to_csv(destination / "crossed_scores.tsv.gz", sep="\t", index=False)
        keys = GROUPS + [
            c
            for c in (
                "evaluation_model",
                "block",
                "gamma",
                "betas",
                "match_rt_rms",
                "run_coefficient_sd",
                "regressor_scaling",
                "generating_convention",
            )
            if c in cross
        ]
        cross_summary = (
            cross.groupby(keys, dropna=False)
            .agg(
                n=("failed", "size"),
                n_recovered=("contrast_error", "count"),
                contrast_bias=("contrast_error", "mean"),
                contrast_sd=("contrast_error", "std"),
                known_hrf_bias=("known_hrf_contrast_error", "mean"),
                selection_bias=("selection_contrast_error", "mean"),
            )
            .reset_index()
        )
        cross_summary["contrast_mcse"] = cross_summary.contrast_sd / np.sqrt(
            cross_summary.n_recovered
        )
        cross_summary.to_csv(destination / "crossed_summary.tsv", sep="\t", index=False)
    if "curve_rmse_excess" in table:
        table.groupby(
            GROUPS
            + [
                c
                for c in (
                    "block",
                    "gamma",
                    "betas",
                    "match_rt_rms",
                    "run_coefficient_sd",
                    "regressor_scaling",
                )
                if c in table
            ],
            dropna=False,
        ).curve_rmse_excess.agg(["count", "mean", "std"]).reset_index().to_csv(
            destination / "excess_recovery.tsv", sep="\t", index=False
        )


def build_report(source, destination):
    source, destination = Path(source), Path(destination)
    paths = sorted(p.parent for p in source.glob("case-*/COMPLETE.json"))
    if not paths:
        raise ValueError("no completed simulation cases")
    tables, provenance, noise, shapes, stability = [], [], [], [], []
    crossed = []
    for folder in paths:
        record = verify(folder)
        table = pd.read_csv(folder / "scores.tsv", sep="\t")
        tables.append(table)
        if (folder / "crossed_scores.tsv").exists():
            crossed.append(pd.read_csv(folder / "crossed_scores.tsv", sep="\t"))
        provenance.append(dict(case=folder.name, hashes=record))
        generation = json.loads((folder / "generation.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        noise.extend(_noise_rows(manifest, generation))
        with np.load(folder / "hrfs.npz") as data:
            truth = data["truth"]
            stability.extend(half_stability(table, data["curves"]))
            for index, row in table.iterrows():
                if row["split"] == "full" and len(shapes) < 200:
                    shapes.append(
                        dict(
                            id=row["id"],
                            model=row["model"],
                            estimator=row["estimator"],
                            convention=row["convention"],
                            truth=truth.tolist(),
                            estimated=data["curves"][index].tolist(),
                        )
                    )
    destination.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    _expanded_reports(table, crossed, destination)
    atomic_text(
        destination / "summary.tsv", summarize(table).to_csv(sep="\t", index=False)
    )
    atomic_text(
        destination / "paired.tsv",
        paired_differences(table).to_csv(sep="\t", index=False),
    )
    atomic_text(
        destination / "noise.tsv", pd.DataFrame(noise).to_csv(sep="\t", index=False)
    )
    atomic_text(
        destination / "stability.tsv",
        pd.DataFrame(
            stability,
            columns=[
                "id",
                "model",
                "estimator",
                "convention",
                "both_recovered",
                "half_shape_correlation",
            ],
        ).to_csv(sep="\t", index=False),
    )
    table.to_csv(destination / "scores.tsv.gz", sep="\t", index=False)
    write_json(destination / "shapes.json", shapes)
    write_json(destination / "provenance.json", provenance)
    _plot(table, destination / "recovery.png")
    return table

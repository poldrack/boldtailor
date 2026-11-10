"""Empirical VTC noise profiles, with target-session exclusion at sampling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hrf_estimation.reml_extensions import _correct_noise_moments
from experiments.hrf_estimation.regularized_noise import shrink_rho
from experiments.hrf_estimation.scoring import residuals
from experiments.hrf_estimation.timing import sampling_interval
from experiments.hrf_simulations.design import reference_design
from experiments.hrf_simulations.io import (
    atomic_npz,
    atomic_text,
    complete,
    verify,
    write_json,
)


def _reference_fit(run):
    design = reference_design(run)
    beta, _, rank, _ = np.linalg.lstsq(design, run.signals, rcond=None)
    if len(design) <= rank:
        raise ValueError(
            "calibration reference requires positive residual degrees of freedom"
        )
    error = run.signals - design @ beta
    energy = np.sum(error[:-1] ** 2, axis=0)
    raw = np.divide(
        np.sum(error[1:] * error[:-1], axis=0),
        energy,
        out=np.full(len(energy), np.nan),
        where=energy > np.finfo(float).eps,
    )
    corrected = _correct_noise_moments(raw, design)
    corrected[~np.isfinite(raw)] = np.nan
    finite = corrected[np.isfinite(corrected)]
    variance_rho = float(np.clip(np.median(finite), -0.8, 0.8)) if len(finite) else 0.0
    u = np.linalg.svd(design, full_matrices=False)[0][:, :rank]
    lags = np.abs(np.arange(len(design))[:, None] - np.arange(len(design))[None, :])
    covariance = variance_rho**lags
    # E[e'e] = sigma² tr(M R), not sigma²(n-rank), for colored errors.
    variance_trace = len(design) - np.sum(u * (covariance @ u))
    return dict(
        design=design,
        beta=beta,
        rank=rank,
        residual=error,
        rho=corrected,
        variance=np.sum(error**2, axis=0) / variance_trace,
        variance_rho=variance_rho,
        variance_trace=float(variance_trace),
    )


def _session_records(runs, session, folder):
    if len(runs) < 2:
        raise ValueError("calibration needs at least two runs per session")
    if any(not np.array_equal(r.feature_ids, runs[0].feature_ids) for r in runs):
        raise ValueError("calibration requires matching vertices across runs")
    fits = [_reference_fit(r) for r in runs]
    rows, files = [], []
    for index, (run, fit) in enumerate(zip(runs, fits, strict=True)):
        n = len(run.confounds.columns)
        trained = np.mean(
            [
                f["beta"][len(r.confounds.columns) :]
                for j, (r, f) in enumerate(zip(runs, fits, strict=True))
                if j != index
            ],
            axis=0,
        )
        predicted = reference_design(run, include_junk=False)[:, n:] @ trained
        predicted, _ = residuals(run.confounds.to_numpy(float), predicted)
        snr = np.sqrt(np.mean(predicted**2, axis=0) / fit["variance"])
        name = f"session{session:02}_run{index:02}.npz"
        atomic_npz(
            folder / name,
            residual=fit["residual"],
            nuisance=fit["beta"][:n],
            names=np.asarray(run.confounds.columns, dtype=str),
            feature_ids=run.feature_ids,
        )
        files.append(name)
        for j, feature in enumerate(run.feature_ids):
            rows.append(
                dict(
                    session=session,
                    run_id=run.run_id,
                    feature_id=int(feature),
                    column=j,
                    artifact=name,
                    length=len(run.frame_times),
                    tr=sampling_interval(run.frame_times),
                    variance=fit["variance"][j],
                    rho=fit["rho"][j],
                    snr=snr[j],
                    rank=fit["rank"],
                    residual_dof=len(run.frame_times) - fit["rank"],
                    variance_rho=fit["variance_rho"],
                    variance_trace=fit["variance_trace"],
                )
            )
    return rows, files


def calibrate(sessions, destination, provenance):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COMPLETE.json").exists():
        raise FileExistsError("completed calibration exists; use a new destination")
    rows, files = [], []
    iterator = sessions.items() if hasattr(sessions, "items") else sessions
    for session, runs in iterator:
        records, names = _session_records(tuple(runs), int(session), folder)
        rows.extend(records)
        files.extend(names)
        print(f"CALIBRATED session {session}", flush=True)
    if not rows:
        raise ValueError("no calibration data")
    atomic_text(
        folder / "profiles.tsv", pd.DataFrame(rows).to_csv(sep="\t", index=False)
    )
    write_json(folder / "provenance.json", provenance)
    complete(folder, [*files, "profiles.tsv", "provenance.json"])


def select_donors(table, target_session, limit=64):
    if limit < 1:
        raise ValueError("donor limit must be positive")
    usable = table[
        (table.session != target_session)
        & np.isfinite(table.variance)
        & (table.variance > 0)
        & np.isfinite(table.rho)
        & np.isfinite(table.snr)
    ]
    if usable.empty:
        raise ValueError("no eligible non-target calibration profiles")
    stats = usable.groupby("feature_id")[["variance", "rho"]].median()
    # Rank in variance/rho jointly, rather than selecting on task recovery.
    score = stats.variance.rank(pct=True) + stats.rho.rank(pct=True)
    ordered = score.sort_values(kind="stable").index.to_numpy()
    chosen = ordered[
        np.unique(
            np.rint(np.linspace(0, len(ordered) - 1, min(limit, len(ordered)))).astype(
                int
            )
        )
    ]
    return usable[usable.feature_id.isin(chosen)].copy()


def _profile_rho(table, feature):
    pivot = table.pivot_table(
        index=["session", "run_id"], columns="feature_id", values="rho"
    )
    samples = pivot.to_numpy(float)
    pooled = float(np.clip(np.nanmedian(samples), -0.8, 0.8))
    if samples.shape[0] < 2:
        return pooled
    result = shrink_rho(samples, pooled)
    value = result["rho"][list(pivot.columns).index(feature)]
    return float(value) if np.isfinite(value) else pooled


def draw_profile(folder, target_session, runs, seed, donor_limit=64):
    folder = Path(folder)
    # The CLI verifies every artifact once before a batch; verify files selected
    # below as well so API use does not silently consume corrupted residuals.
    from experiments.hrf_simulations.io import file_hash

    record = json.loads((folder / "COMPLETE.json").read_text())
    if file_hash(folder / "profiles.tsv") != record["profiles.tsv"]:
        raise ValueError("corrupt calibration table")
    table = select_donors(
        pd.read_csv(folder / "profiles.tsv", sep="\t"), target_session, donor_limit
    )
    rng = np.random.default_rng(seed)
    feature = int(rng.choice(np.sort(table.feature_id.unique())))
    donor = table[table.feature_id == feature]
    profiles, sources = [], []
    for run in runs:
        compatible = donor[
            (donor.length == len(run.frame_times))
            & np.isclose(donor.tr, sampling_interval(run.frame_times))
        ]
        if compatible.empty:
            raise ValueError("no compatible donor run length/sampling interval")
        row = compatible.iloc[int(rng.integers(len(compatible)))]
        path = folder / row.artifact
        if file_hash(path) != record[row.artifact]:
            raise ValueError(f"corrupt calibration artifact: {path}")
        with np.load(path) as data:
            names = list(data["names"])
            if set(names) != set(run.confounds.columns):
                raise ValueError("no compatible nuisance-column mapping")
            indices = [names.index(name) for name in run.confounds.columns]
            profiles.append(
                dict(
                    residual=data["residual"][:, int(row.column)].copy(),
                    nuisance=data["nuisance"][indices, int(row.column)].copy(),
                )
            )
        sources.append(
            dict(
                artifact=row.artifact,
                run_id=row.run_id,
                session=int(row.session),
                hash=record[row.artifact],
            )
        )
    snrs = table.snr.to_numpy()
    snrs = snrs[np.isfinite(snrs) & (snrs > 0)]
    if not len(snrs):
        raise ValueError(
            "calibration has no positive SNR; explicitly choose fixed SNR controls"
        )
    return dict(
        feature_id=feature,
        rho=_profile_rho(table, feature),
        variance=float(donor.variance.median()),
        runs=profiles,
        sources=sources,
        calibration_sessions=sorted(map(int, table.session.unique())),
        calibration_provenance_hash=record["provenance.json"],
        calibration_table_hash=record["profiles.tsv"],
        snr_quantiles=np.quantile(snrs, [0.25, 0.5, 0.75]),
    )

"""Evaluate frozen HRFs with run-specific amplitudes refitted by OLS."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.hrf_estimation.design import build_design


@dataclass(frozen=True)
class ScoreResult:
    summary: pd.DataFrame
    per_run: pd.DataFrame


def residuals(design, signals):
    beta, _, rank, _ = np.linalg.lstsq(design, signals, rcond=None)
    return signals - design @ beta, rank


def kernel_groups(estimate):
    groups = {}
    for i, kernel in enumerate(estimate.kernels):
        if kernel is not None:
            key = (kernel.kind, kernel.parameters)
            groups.setdefault(key, (kernel, []))[1].append(i)
    return groups.values()


def score_hrfs(runs, estimate, rt_mode='amplitude'):
    if not runs or len(estimate.kernels) != len(estimate.feature_ids):
        raise ValueError('runs and one kernel per feature are required')
    rows = []
    for run in runs:
        if not np.array_equal(run.feature_ids, estimate.feature_ids):
            raise ValueError('feature identities/order differ from fitted HRFs')
        for kernel, indices in kernel_groups(estimate):
            parts = build_design(run, kernel, rt_mode)
            rows.extend(_run_scores(run, parts, indices))
    per_run = pd.DataFrame(rows)
    summary = _summarize(per_run, estimate)
    return ScoreResult(summary, per_run)


def _run_scores(run, parts, indices):
    y = run.signals[:, indices]
    null = np.column_stack([parts.nuisance, parts.junk])
    full = np.column_stack([null, parts.task])
    error, rank = residuals(full, y)
    nuisance_error, null_rank = residuals(null, y)
    c_error, _ = residuals(parts.nuisance, y)
    values = dict(sse_full=np.sum(error**2, axis=0),
                  sse_nuisance=np.sum(nuisance_error**2, axis=0),
                  sse_c=np.sum(c_error**2, axis=0),
                  sst=np.sum((y-y.mean(axis=0))**2, axis=0),
                  energy=np.sum(y**2, axis=0))
    return [dict(run_id=run.run_id, feature_id=int(run.feature_ids[index]),
                 rank_full=rank, rank_nuisance=null_rank, residual_dof=len(y)-rank,
                 **{name: value[j] for name, value in values.items()})
            for j, index in enumerate(indices)]


def _summarize(per_run, estimate):
    summary = pd.DataFrame(index=pd.Index(estimate.feature_ids, name='feature_id'))
    if per_run.empty:
        for name in ('sse_full', 'sse_nuisance', 'sse_c', 'sst', 'energy'):
            summary[name] = np.nan
    else:
        totals = per_run.groupby('feature_id')[['sse_full', 'sse_nuisance', 'sse_c', 'sst', 'energy']].sum()
        summary = summary.join(totals)
    tolerance = np.finfo(float).eps * 64 * summary.energy
    total_valid = summary.sst > tolerance
    task_valid = summary.sse_nuisance > tolerance
    summary['r2_full'] = (1-summary.sse_full/summary.sst).where(total_valid)
    summary['r2_task_partial'] = (1-summary.sse_full/summary.sse_nuisance).where(task_valid)
    summary['delta_r2_task'] = ((summary.sse_nuisance-summary.sse_full)/summary.sst).where(total_valid)
    summary['status'] = estimate.status
    summary.loc[~total_valid & summary.sst.notna(), 'status'] = 'zero_variance'
    return summary

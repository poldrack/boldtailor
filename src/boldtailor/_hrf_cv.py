"""Nuisance-projected sufficient statistics and leave-one-run-out HRF scores."""

import numpy as np
import pandas as pd


def _batched_products(run, yr, batch_size):
    """B for every candidate as (H, K, F), multiplying batch_size candidates at once."""
    n = len(run.library.candidates)
    b = np.empty((n, run.k, yr.shape[1]))
    for start in range(0, n, batch_size):
        ids = range(start, min(start + batch_size, n))
        stacked = np.concatenate([run.block(cid).x for cid in ids], axis=1)
        b[start : start + len(ids)] = (stacked.T @ yr).reshape(len(ids), run.k, -1)
    return b


def signal_statistics(runs, signals, batch_size):
    """Project Y once per run; form A, B, C per candidate and confound-only energy."""
    a_all, b_all, c_all, energy = [], [], [], []
    for run, y in zip(runs, signals, strict=True):
        yr = y - run.q @ (run.q.T @ y)
        # Numerical zero only: no statistical threshold or weak-signal fallback.
        tolerance = np.linalg.norm(y, axis=0) * len(y) * np.finfo(float).eps
        yr[:, np.linalg.norm(yr, axis=0) <= tolerance] = 0
        c0 = np.sum(yr**2, axis=0)
        n = len(run.library.candidates)
        a = np.stack([run.block(cid).a for cid in range(n)])
        c = np.tile(c0, (n, 1))
        for cid in range(n):
            qp = run.block(cid).qp
            if qp.shape[1]:
                residual = yr - qp @ (qp.T @ yr)
                c[cid] = np.sum(residual**2, axis=0)
        a_all.append(a)
        b_all.append(_batched_products(run, yr, batch_size))
        c_all.append(c)
        energy.append(c0)
    return np.array(a_all), np.array(b_all), np.array(c_all), np.array(energy)


def pooled_amplitude(a_sum, b_sum):
    """Solve pooled normal equations per candidate; singular ones get zeros and False."""
    s = np.linalg.svd(a_sum, compute_uv=False)
    ok = s[:, -1] > s[:, 0] * a_sum.shape[-1] * np.finfo(float).eps
    amplitude = np.zeros_like(b_sum)
    if ok.any():
        amplitude[ok] = np.linalg.solve(a_sum[ok], b_sum[ok])
    return amplitude, ok


def prediction_loss(a, b, c, amplitude):
    cross = 2 * np.einsum("hkf,hkf->hf", amplitude, b)
    fitted = np.einsum("hkf,hkj,hjf->hf", amplitude, a, amplitude)
    loss = c - cross + fitted
    magnitude = 2 * np.einsum("hkf,hkf->hf", np.abs(amplitude), np.abs(b))
    tolerance = 64 * np.finfo(float).eps * (c + magnitude + np.abs(fitted))
    if np.any(loss < -tolerance):
        raise ArithmeticError("negative HRF prediction SSE beyond roundoff tolerance")
    return np.maximum(loss, 0)


def loro_scores(a, b, c, energy):
    loss = np.zeros(c.shape[1:])
    eligible = np.ones(c.shape[1], dtype=bool)
    eligible &= ~np.any(np.all(a == 0, axis=(-2, -1)), axis=0)
    for r in range(len(a)):
        others = [i for i in range(len(a)) if i != r]
        # Explicit sums avoid cancellation when one run has much larger energy.
        amplitude, ok = pooled_amplitude(a[others].sum(axis=0), b[others].sum(axis=0))
        eligible &= ok
        loss += prediction_loss(a[r], b[r], c[r], amplitude)
    total = energy.sum(axis=0)
    scores = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=scores, where=total[None, :] > 0)
    scores = 1 - scores
    scores[~eligible] = -np.inf
    return scores


def choose_eligible(scores, runs):
    """Validate winners lazily; structural exclusions apply to every feature."""
    scores = scores.copy()
    checked = {}

    def check(cid):
        if cid not in checked:
            outcomes = [r.eligible(cid) for r in runs]
            checked[cid] = (
                all(ok for ok, _ in outcomes),
                "; ".join(
                    f"run {r}: {reason}"
                    for r, (ok, reason) in enumerate(outcomes)
                    if not ok
                ),
            )
            if not checked[cid][0]:
                scores[cid] = -np.inf
        return checked[cid][0]

    check(0)
    valid = np.any(np.isfinite(scores), axis=0)
    while True:
        best = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
        indices = np.full(scores.shape[1], -1, dtype=int)
        for v in np.flatnonzero(valid & np.isfinite(best)):
            tolerance = 64 * np.finfo(float).eps * max(1, abs(best[v]))
            indices[v] = np.flatnonzero(scores[:, v] >= best[v] - tolerance)[0]
        if all([check(int(cid)) for cid in np.unique(indices[indices >= 0])]):
            break
    if not any(ok for ok, _ in checked.values()):
        for cid in range(len(scores)):
            if check(cid):
                break
        else:
            raise ValueError("no eligible, estimable HRF candidate across all runs")
    if np.any(valid & (indices < 0)):
        raise ValueError("no eligible HRF with finite prediction score")
    table = pd.DataFrame(
        [
            dict(
                hrf_id=cid,
                eligible=checked.get(cid, (None, "unchecked"))[0],
                reason=checked.get(cid, (None, "unchecked"))[1],
            )
            for cid in range(len(scores))
        ]
    )
    return indices, scores, table

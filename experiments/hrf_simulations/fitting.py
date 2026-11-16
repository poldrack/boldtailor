"""Simulation adapters onto boldtailor's scalar REML and AR(1) estimates."""

from collections import OrderedDict
from hashlib import sha256

import numpy as np

from boldtailor import ar1
from boldtailor.reml import RemlGroup, select_random_effects_design
from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_estimation.scoring import residuals
from experiments.hrf_simulations.design import task_design, reference_design


class DesignCache:
    """Bounded, signal-independent cache; keys include actual timing/nuisance."""

    def __init__(self, max_entries=65536):
        self.entries = OrderedDict()
        self.max_entries = max_entries

    def get(self, run, key, builder):
        digest = sha256(run.events.to_csv(index=False).encode())
        digest.update(run.frame_times.tobytes())
        digest.update(run.confounds.to_numpy(float).tobytes())
        identity = (digest.hexdigest(), key)
        if identity not in self.entries:
            self.entries[identity] = builder()
            if len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        self.entries.move_to_end(identity)
        return self.entries[identity]


def _prepared_reference(run):
    return ar1.prepare_reference(reference_design(run))


def estimate_rho(runs, cache=None):
    """Pooled corrected AR(1) rho after projecting the shared reference design.

    The prepared reference (basis and expected lag ratio) is signal-independent
    and cached per run, so repeated fits pay only one projection per run.
    """
    cache = cache or DesignCache()
    references = [
        cache.get(r, "reference", lambda r=r: _prepared_reference(r)) for r in runs
    ]
    return float(ar1.estimate_rho([r.signals for r in runs], references).pooled)


def random_scales(parts, mode):
    if mode not in ("native", "unit_rms"):
        raise ValueError("unknown regressor scaling")
    count = parts[0].task.shape[1] + parts[0].junk.shape[1]
    if mode == "native":
        return np.ones(count)
    projected = [
        residuals(p.nuisance, np.column_stack([p.task, p.junk]))[0] for p in parts
    ]
    scale = np.sqrt(
        sum(np.sum(z * z, axis=0) for z in projected) / sum(len(z) for z in projected)
    )
    return np.where(scale < 1e-12, 1.0, scale)


def _check_inputs(runs, noise_mode, oracle_rho):
    if not runs or any(r.signals.shape[1] != 1 for r in runs):
        raise ValueError("fit requires one simulated vertex and at least one run")
    if noise_mode not in ("iid", "ar1"):
        raise ValueError("noise_mode must be iid or ar1")
    if oracle_rho is not None and noise_mode != "ar1":
        raise ValueError("oracle rho is only meaningful for AR1")


def _noise_rho(runs, noise_mode, oracle_rho, cache):
    if noise_mode == "iid":
        return 0.0
    return estimate_rho(runs, cache) if oracle_rho is None else float(oracle_rho)


def _candidate_parts(runs, candidate, model, convention, cache, design_builder):
    """Cached, signal-independent design parts of one candidate for every run."""
    key = (candidate.kind, candidate.parameters, model, convention, id(design_builder))
    return [
        cache.get(r, key, lambda r=r: design_builder(r, candidate, model, convention))
        for r in runs
    ]


def _random_designs(parts, scaling):
    scales = random_scales(parts, scaling)
    return [np.column_stack([p.task, p.junk]) / scales for p in parts]


def _fit_diagnostics(selection, scales):
    fit = selection.fit
    return dict(
        criterion=[float(fit.criterion[0])],
        variance_ratio=[float(fit.ratio[0])],
        sigma2=[float(fit.noise_variance[0])],
        at_zero=[fit.status[0] == "variance_zero"],
        at_upper=[bool(fit.at_upper[0])],
        unrefined=fit.unrefined.tolist(),
        any_at_upper=selection.any_at_upper.tolist(),
        any_unrefined=selection.any_unrefined.tolist(),
        provisional=selection.any_unrefined.tolist(),
        random_scales=[None if scales is None else scales.tolist()],
    )


def _run_diagnostics(runs, library, model, convention, noise_mode, rho, oracle_rho):
    return dict(
        rho=rho,
        oracle=oracle_rho is not None,
        model=model,
        convention=convention,
        noise_mode=noise_mode,
        training_run_ids=[r.run_id for r in runs],
        library_fingerprint=library.fingerprint,
    )


def _select(runs, library, parts, scaling, rho):
    # The fixed design is r.confounds while random_scales residualise on
    # p.nuisance; task_design sets nuisance = confounds, but a custom
    # design_builder must keep the two equal for the scales to be consistent.
    groups = [RemlGroup(r.signals, r.confounds.to_numpy(float)) for r in runs]
    candidates = (_random_designs(parts(k), scaling) for k in library.candidates)
    return select_random_effects_design(groups, candidates, rho=rho)


def _estimate(runs, kernel, state, diagnostics):
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        (kernel if state == "ok" else None,),
        (state,),
        tuple(r.run_id for r in runs),
        diagnostics,
        runs[0].brain_axis,
    )


def fit(
    runs,
    library,
    model,
    convention,
    noise_mode,
    oracle_rho=None,
    cache=None,
    settings=None,
    design_builder=task_design,
):
    """Select one vertex's library HRF by REML under iid or AR(1) noise."""
    _check_inputs(runs, noise_mode, oracle_rho)
    cache, settings = cache or DesignCache(), settings or {}
    scaling = settings.get("regressor_scaling", "native")
    rho = _noise_rho(runs, noise_mode, oracle_rho, cache)

    def parts(candidate):
        return _candidate_parts(
            runs, candidate, model, convention, cache, design_builder
        )

    selection = _select(runs, library, parts, scaling, rho)
    winner, state = int(selection.winner[0]), selection.fit.status[0]
    kernel = library.candidates[winner] if winner >= 0 else None
    diagnostics = _run_diagnostics(
        runs, library, model, convention, noise_mode, rho, oracle_rho
    )
    if state != "no_residual_signal":
        scales = None if kernel is None else random_scales(parts(kernel), scaling)
        diagnostics.update(_fit_diagnostics(selection, scales))
    return _estimate(runs, kernel, state, diagnostics)

"""Simulation adapters onto the public CV and REML HRF selectors."""

from collections import OrderedDict
from hashlib import sha256

import json

from boldtailor import ar1
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from boldtailor.provenance import identity_activity
from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_simulations.design import reference_design
from experiments.hrf_simulations.selection_inputs import selection_data


class DesignCache:
    """Bounded, signal-independent cache; keys include actual timing/nuisance."""

    def __init__(self, max_entries=65536):
        self.entries = OrderedDict()
        self.max_entries = max_entries

    def get(self, run, key, builder):
        digest = sha256(run.events.to_csv(index=False).encode())
        digest.update(run.frame_times.tobytes())
        digest.update(run.confounds.to_numpy(float).tobytes())
        digest.update(json.dumps(list(run.confounds.columns)).encode())
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


def _fit_diagnostics(result):
    """Keep likelihood estimates and numerical qualifications in their own fields."""
    diagnostics = result.reml
    selected = diagnostics.selection
    fit = selected.fit
    return dict(
        criterion=fit.criterion.tolist(),
        variance_ratio=fit.ratio.tolist(),
        sigma2=fit.noise_variance.tolist(),
        effect_variance=fit.effect_variance.tolist(),
        at_zero=[state == "variance_zero" for state in fit.status],
        at_upper=fit.at_upper.tolist(),
        unrefined=fit.unrefined.tolist(),
        any_at_upper=selected.any_at_upper.tolist(),
        any_unrefined=selected.any_unrefined.tolist(),
        provisional=selected.any_unrefined.tolist(),
        canonical_criterion=diagnostics.canonical_fit.criterion.tolist(),
        canonical_status=list(diagnostics.canonical_fit.status),
        delta_criterion=diagnostics.delta_criterion.tolist(),
        criterion_gap=selected.gap.tolist(),
        reference_hrf_id=diagnostics.reference_hrf_id,
        candidate_ids=diagnostics.candidate_ids.tolist(),
    )


def _cv_diagnostics(result):
    return dict(
        cv_r2=result.cv_r2.tolist(),
        canonical_cv_r2=result.canonical_cv_r2.tolist(),
        delta_cv_r2=result.delta_cv_r2.tolist(),
        provisional=[None],
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
    *,
    method="reml",
):
    """Identify one vertex's HRF with the current package selector.

    CV uses leave-one-run-out prediction. REML uses a fixed canonical mean and
    independent trial effects, with supplied or training-estimated noise rho.
    """
    _check_inputs(runs, noise_mode, oracle_rho)
    cache, settings = cache or DesignCache(), settings or {}
    if settings.get("regressor_scaling", "native") != "native":
        raise ValueError("regressor_scaling must be native for package identification")
    if method not in ("cv", "reml"):
        raise ValueError("method must be cv or reml")
    if method == "cv" and (noise_mode != "iid" or oracle_rho is not None):
        raise ValueError("CV does not support AR1 noise options")
    data, task_model = selection_data(runs, library, model, convention, cache)
    rho = _noise_rho(runs, noise_mode, oracle_rho, cache) if method == "reml" else None
    options = RemlHrfOptions(rho=rho) if method == "reml" else None
    result = select_hrfs(
        data,
        library=library,
        task_model=task_model,
        method=method,
        reml_options=options,
        run_labels=[r.run_id for r in runs],
    )
    winner = int(result.hrf_indices[0])
    state = (
        result.reml.selection.fit.status[0]
        if method == "reml"
        else "ok" if winner >= 0 else "no_eligible_hrf"
    )
    kernel = library.candidates[winner] if winner >= 0 else None
    diagnostics = _run_diagnostics(
        runs,
        library,
        model,
        convention,
        noise_mode if method == "reml" else None,
        rho,
        oracle_rho,
    )
    diagnostics.update(
        selection_method=method,
        task_model=task_model.to_dict(),
        package_selection=identity_activity(result.provenance),
    )
    diagnostics.update(
        _fit_diagnostics(result) if method == "reml" else _cv_diagnostics(result)
    )
    return _estimate(runs, kernel, state, diagnostics)

"""Regularized variable projection for a shared double-gamma HRF."""

import numpy as np
from scipy.optimize import minimize

from boldtailor.hrf_library import HrfCandidate, default_hrf_library
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features
from experiments.hrf_estimation.scoring import residuals

LOW = np.array([2.0, 8.0, 0.3, 0.5, 2.0])
HIGH = np.array([10.0, 24.0, 2.5, 4.0, 20.0])


def _encode(parameters):
    values = np.array(parameters[:5], dtype=float)
    values[2:] = np.log(values[2:])
    low, high = LOW.copy(), HIGH.copy()
    low[2:], high[2:] = np.log(low[2:]), np.log(high[2:])
    return (values - low) / (high - low)


def _decode(unit):
    low, high = LOW.copy(), HIGH.copy()
    low[2:], high[2:] = np.log(low[2:]), np.log(high[2:])
    values = low + np.asarray(unit) * (high - low)
    values[2:] = np.exp(values[2:])
    return (*values, 0.0, 36.0)


def _energy(runs, feature):
    energy = raw = 0.0
    for run in runs:
        y = run.signals[:, feature]
        error, _ = residuals(run.confounds.to_numpy(dtype=float), y)
        energy += np.sum(error**2)
        raw += np.sum(y**2)
    return energy if energy > 64 * np.finfo(float).eps * raw else 0.0


def profile_loss(runs, feature, parameters, rt_mode="amplitude"):
    energy = _energy(runs, feature)
    if energy == 0:
        return np.nan
    # Positive lobes must have modes, and the undershoot mode must follow response.
    a, b, c, d, _, _, _ = parameters
    if a <= c or b <= d or b - d <= a - c:
        return np.inf
    kernel = HrfCandidate(0, "double_gamma", tuple(parameters))
    sse = 0.0
    for run in runs:
        parts = build_design(run, kernel, rt_mode)
        matrix = np.column_stack([parts.nuisance, parts.junk, parts.task])
        error, _ = residuals(matrix, run.signals[:, feature])
        sse += np.sum(error**2)
    return float(sse / energy)


def _parameters(library):
    candidates = [
        (*c.parameters[:5], 0.0, 36.0)
        for c in library.candidates
        if c.kind != "glmsingle"
    ]
    candidates = [
        p for p in candidates if np.all(_encode(p) >= 0) and np.all(_encode(p) <= 1)
    ]
    return candidates


def _starts(runs, feature, library, settings, losses=None):
    candidates = _parameters(library)
    if not candidates:
        raise ValueError("initial library has no bounded double-gamma candidates")
    losses = losses if losses is not None else [
        profile_loss(runs, feature, p, settings.get("rt_mode", "amplitude"))
        for p in candidates
    ]
    order = np.argsort(losses)
    count = int(settings.get("max_starts", 4))
    chosen = [candidates[i] for i in order[: max(1, count - 1)]]
    canonical = (6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 36.0)
    if canonical not in chosen and len(chosen) < count:
        chosen.append(canonical)
    return chosen, _encode(candidates[order[0]])


def _optimize_feature(runs, feature, library, settings, prepared=None, losses=None):
    starts, reference = _starts(runs, feature, library, settings, losses)
    strength = settings["lambda"]

    def objective(unit):
        if prepared is None:
            loss = profile_loss(runs, feature, _decode(unit), settings.get("rt_mode", "amplitude"))
        else:
            from experiments.hrf_estimation.prepared import prepared_loss
            loss = prepared_loss(prepared, feature, _decode(unit))
        return (
            loss + strength * np.sum((unit - reference) ** 2)
            if np.isfinite(loss)
            else 1e10
        )

    initial = objective(reference)
    best, value, converged, calls = reference, initial, False, 0
    method = settings.get("optimizer", "L-BFGS-B")
    if method not in ("L-BFGS-B", "Powell"):
        raise ValueError("optimizer must be L-BFGS-B or Powell")
    options = {"maxiter": settings.get("maxiter", 500), "ftol": 1e-12}
    options["maxfun" if method == "L-BFGS-B" else "maxfev"] = settings.get(
        "maxfun", 5000
    )
    use_gradient = settings.get("gradient", "finite_difference") == "autodiff"
    function = _gradient_objective(prepared, feature, strength, reference) if use_gradient else objective
    for parameters in starts:
        result = minimize(
            function,
            _encode(parameters),
            method=method,
            jac=True if use_gradient else None,
            bounds=[(0.0, 1.0)] * 5,
            options=options,
        )
        calls += result.nfev
        if np.isfinite(result.fun) and result.fun <= value:
            best, value, converged = result.x, result.fun, bool(result.success)
    return _decode(best), value, initial, converged, calls


def fit_continuous(runs, settings):
    if not runs or any(
        not np.array_equal(r.feature_ids, runs[0].feature_ids) for r in runs
    ):
        raise ValueError("runs must have identical feature identities/order")
    validate_features(runs, runs[0].feature_ids, runs[0].brain_axis)
    settings = dict(settings)
    _validate_gradient_settings(settings)
    if "lambda" not in settings:
        from experiments.hrf_estimation.tuning import choose_regularization

        settings["lambda"] = choose_regularization(runs, settings)
    if not np.isfinite(settings["lambda"]) or settings["lambda"] < 0:
        raise ValueError("lambda must be finite and nonnegative")
    library = settings.get("library")
    if library is None:
        library = default_hrf_library()
    fitted = _fit_features(runs, library, settings)
    kernels, status, diagnostics = [], [], []
    for feature, result in enumerate(fitted):
        if result is None:
            kernels.append(None)
            status.append("no_signal")
            diagnostics.append((np.nan, np.nan, False, 0))
            continue
        parameters, value, initial, converged, calls = result
        kernels.append(HrfCandidate(feature, "double_gamma", parameters))
        status.append("ok" if converged else "not_converged")
        diagnostics.append((value, initial, converged, calls))
    details = {
        name: [row[i] for row in diagnostics]
        for i, name in enumerate(
            ("objective", "initial_objective", "converged", "function_evaluations")
        )
    }
    details.update(
        {
            "lambda": settings["lambda"],
            "optimizer": settings.get("optimizer", "L-BFGS-B"),
            "prepared": settings.get("prepared", True),
            "gradient": settings.get("gradient", "finite_difference"),
        }
    )
    if settings.get("gradient") == "autodiff":
        from experiments.hrf_estimation.autodiff import environment
        details["autodiff_environment"] = environment()
    details.update(
        _shape_diagnostics(runs, kernels, settings.get("rt_mode", "amplitude"))
    )
    status = [
        "rank_deficient" if rank is not None and rank < 3 else state
        for rank, state in zip(details["min_task_rank"], status)
    ]
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        tuple(kernels),
        tuple(status),
        tuple(r.run_id for r in runs),
        details,
        runs[0].brain_axis,
    )


def _shape_diagnostics(runs, kernels, mode):
    ranks, distances, lower, upper = [], [], [], []
    for kernel in kernels:
        if kernel is None:
            ranks.append(None)
            distances.append([None] * 5)
            lower.append([False] * 5)
            upper.append([False] * 5)
            continue
        task_ranks = []
        for run in runs:
            parts = build_design(run, kernel, mode)
            null = np.column_stack([parts.nuisance, parts.junk])
            task, _ = residuals(null, parts.task)
            task_ranks.append(int(np.linalg.matrix_rank(task)))
        unit = _encode(kernel.parameters)
        ranks.append(min(task_ranks))
        distances.append(np.minimum(unit, 1 - unit).tolist())
        lower.append((unit <= 0.02).tolist())
        upper.append((unit >= 0.98).tolist())
    return dict(
        min_task_rank=ranks,
        normalized_bound_distance=distances,
        near_lower_bound=lower,
        near_upper_bound=upper,
    )


def _one_feature(runs, feature, library, settings, prepared, losses):
    if _energy(runs, feature) == 0:
        return None
    return _optimize_feature(runs, feature, library, settings, prepared,
                             None if losses is None else losses[:, feature])


def _fit_features(runs, library, settings):
    from joblib import Parallel, delayed, parallel_config
    from experiments.hrf_estimation.prepared import prepare_profile, library_losses

    prepared = losses = None
    if settings.get("prepared", True):
        prepared = prepare_profile(runs, settings.get("rt_mode", "amplitude"))
        losses = library_losses(prepared, _parameters(library))
    jobs = int(settings.get("n_jobs", 1))
    runtime = {}
    if settings.get("gradient") == "autodiff" and settings.get("compilation_cache_dir"):
        from experiments.hrf_estimation.autodiff_runtime import configure_worker
        arguments = (settings.get("worker_memory_growth_bytes", int(3e8)),
                     settings["compilation_cache_dir"])
        runtime = dict(initializer=configure_worker, initargs=arguments)
        if jobs == 1:
            configure_worker(*arguments)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=jobs, verbose=settings.get("verbose", 0), **runtime)(
            delayed(_one_feature)(runs, f, library, settings, prepared, losses)
            for f in range(len(runs[0].feature_ids)))


def _validate_gradient_settings(settings):
    mode = settings.get("gradient", "finite_difference")
    if mode not in ("finite_difference", "autodiff"):
        raise ValueError("gradient must be finite_difference or autodiff")
    if mode == "autodiff":
        if not settings.get("prepared", True):
            raise ValueError("autodiff requires prepared=True")
        if settings.get("optimizer", "L-BFGS-B") != "L-BFGS-B":
            raise ValueError("autodiff is supported for L-BFGS-B only")


def _gradient_objective(prepared, feature, strength, reference):
    from experiments.hrf_estimation.autodiff import make_value_gradient
    evaluate = make_value_gradient(prepared, feature)
    def objective(unit):
        value, gradient = evaluate(unit)
        if not np.isfinite(value):
            return 1e10, np.zeros(5)
        return (value + strength*np.sum((unit-reference)**2),
                gradient + 2*strength*(unit-reference))
    return objective

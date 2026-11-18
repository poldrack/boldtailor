"""Library selection by the package's leave-one-run-out task prediction."""

from experiments.hrf_estimation.package_selection import estimate, select


def fit_library_cv(runs, library, settings):
    """Select one library HRF per feature with ``select_hrfs(method="cv")``.

    A feature without a finite prediction score gets no HRF. An unidentifiable
    task model is the package's error, raised before any fit.
    """
    if len(runs) < 2:
        raise ValueError("library CV needs at least two training runs")
    mode = settings.get("rt_mode", "amplitude")
    result = select(runs, library, mode, "cv")
    status = ["ok" if i >= 0 else "no_signal" for i in result.hrf_indices]
    diagnostics = dict(
        selected_cv_r2=result.cv_r2,
        canonical_cv_r2=result.canonical_cv_r2,
        delta_cv_r2=result.delta_cv_r2,
        selection="leave_one_run_out_prediction",
        rt_mode=mode,
    )
    return estimate(runs, library, result, status, diagnostics)

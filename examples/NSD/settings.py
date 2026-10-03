"""Resolve the NSD workflow notebook's settings, HRF library, and beta penalties."""

from collections.abc import Mapping

from boldtailor.hrf_library import (
    HrfLibrary,
    default_hrf_library,
    expanded_hrf_library,
    sobol_hrf_library,
)
from .notebook_paths import notebook_paths

RIDGE_MODES = ("fractional_cv", "cv", "fixed", "off")
DEFAULTS = dict(
    subject="sub-07",
    session="ses-nsd10",
    existing_results="reuse",  # "reuse", "overwrite", or "error"
    n_jobs=4,
    block_size=4096,
    max_grayordinates=None,
    encoding_mode="within_run",  # "absolute" reproduces the shared-intercept objective.
    ridge_mode="fractional_cv",
    ridge_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ridge_alphas=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    ridge_percentile=90.0,
    ridge_alpha=0.1,  # Used only in fixed mode.
    hrf_library="default",  # timing-space Sobol + GLMsingle; "sobol" or "expanded"
    hrf_n_samples=512,
    hrf_seed=0,
    hrf_parameters=None,  # Optional custom rows override the library choice.
    hrf_selection_rt=True,  # False scores HRFs without RT; the GLM keeps RT.
    surface_maps=True,
    surface_meshes=None,  # Optional {"left": path, "right": path} in fsLR order.
)


def _ridge_mode(config):
    """Explicit mode wins; legacy configs imply fixed/off (alpha) or cv (alphas)."""
    if "ridge_mode" in config:
        mode = config["ridge_mode"]
    elif "ridge_alpha" in config:
        mode = "off" if config["ridge_alpha"] is None else "fixed"
    elif "ridge_alphas" in config:
        mode = "cv"
    else:
        mode = DEFAULTS["ridge_mode"]
    if mode not in RIDGE_MODES:
        raise ValueError("ridge_mode must be fractional_cv, cv, fixed, or off")
    return mode


def resolve_settings(config: Mapping) -> dict:
    """Defaults, then the configuration, then expanded local paths and ridge mode.

    Paths come from the configuration, then NSD_* environment variables (see
    ``notebook_paths``). The configuration itself is not modified.
    """
    settings = {**DEFAULTS, **config}
    settings.update(notebook_paths(config))
    settings["ridge_mode"] = _ridge_mode(config)
    return settings


def build_hrf_library(settings):
    """Custom parameter rows, else the default, Sobol, or expanded grid library."""
    if settings["hrf_parameters"] is not None:
        return HrfLibrary.from_parameters(settings["hrf_parameters"])
    size, seed = settings["hrf_n_samples"], settings["hrf_seed"]
    if settings["hrf_library"] == "default":
        return default_hrf_library(size, seed=seed)
    if settings["hrf_library"] == "sobol":
        return sobol_hrf_library(size, seed=seed)
    if settings["hrf_library"] == "expanded":
        return expanded_hrf_library()
    raise ValueError("hrf_library must be 'default', 'sobol', or 'expanded'")


def beta_penalties(settings):
    """OLS always; fixed mode adds one positive ridge alpha."""
    penalties = {"OLS": 0.0}
    if settings["ridge_mode"] == "fixed":
        alpha = settings["ridge_alpha"]
        if alpha is None or alpha <= 0:
            raise ValueError("ridge_alpha must be positive in fixed mode")
        penalties["Ridge"] = alpha
    return penalties

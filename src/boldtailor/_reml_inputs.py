"""Input validation for the public scalar REML entry points."""

import numpy as np


def _check_finite(values, name, index):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} of group {index} contains non-finite values")


def _check_matrix(values, name, index):
    if np.ndim(values) != 2:
        raise ValueError(f"{name} of group {index} must be 2-D (scans, columns)")
    _check_finite(values, name, index)


def _check_group(group, index):
    _check_matrix(group.signals, "signals", index)
    _check_matrix(group.fixed_design, "fixed_design", index)
    if group.signals.shape[0] != group.fixed_design.shape[0]:
        raise ValueError(f"scan counts differ within group {index}")


def _check_random(group, random, label, index):
    if random is None:
        return
    _check_matrix(random, label, index)
    if np.shape(random)[0] != group.signals.shape[0]:
        raise ValueError(
            f"scan counts differ between signals and {label} of group {index}"
        )


def _as_random(random):
    return None if random is None else np.asarray(random, dtype=float)


def validate_groups(groups):
    """Raise ValueError unless the groups are finite, 2-D and share features."""
    if len(groups) == 0:
        raise ValueError("at least one group is required")
    for index, group in enumerate(groups):
        _check_group(group, index)
    if len({g.signals.shape[1] for g in groups}) != 1:
        raise ValueError("feature counts differ across groups")
    if groups[0].signals.shape[1] == 0:
        raise ValueError("signals must have at least one feature")


def checked_random(groups, random_design, label):
    """One validated float random design (or None) per group from any iterable."""
    designs = [_as_random(z) for z in random_design]
    if len(designs) != len(groups):
        raise ValueError(f"{label} must have one entry per group")
    for index, (group, random) in enumerate(zip(groups, designs, strict=True)):
        _check_random(group, random, label, index)
    return designs


def feature_rho(rho, n_features):
    """Broadcast rho to one stationary value per feature."""
    values = np.asarray(rho, dtype=float)
    if values.ndim == 0:
        values = np.full(n_features, float(values))
    if values.shape != (n_features,):
        raise ValueError("rho must be a scalar or have shape (features,)")
    if not np.all(np.abs(values) < 1):
        raise ValueError("rho must be finite with |rho| < 1")
    return values

"""Map an independently defined surface ROI onto CIFTI columns."""

import numpy as np
from scipy.spatial import cKDTree


def vtc_indices(brain_axis, labels_by_hemisphere):
    mask = np.zeros(len(brain_axis), dtype=bool)
    for structure, side in [
        ("CIFTI_STRUCTURE_CORTEX_LEFT", "L"),
        ("CIFTI_STRUCTURE_CORTEX_RIGHT", "R"),
    ]:
        labels = np.asarray(labels_by_hemisphere[side])
        expected = brain_axis.nvertices.get(structure)
        if labels.ndim != 1 or labels.size != expected:
            raise ValueError(
                f"{side}: ROI size must match the full surface vertex count"
            )
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("ROI labels must be a binary mask")
        selected = brain_axis.name == structure
        mask[selected] = labels[brain_axis.vertex[selected]].astype(bool)
    if not mask.any():
        raise ValueError("VTC ROI is empty on the CIFTI axis")
    return np.flatnonzero(mask)


def resample_mask(source_sphere, target_sphere, source_mask):
    """Nearest-vertex categorical mapping on aligned registered spheres."""
    source = _unit_sphere(source_sphere)
    target = _unit_sphere(target_sphere)
    mask = np.asarray(source_mask)
    if mask.shape != (len(source),) or not np.isin(mask, [0, 1]).all():
        raise ValueError("source mask must be binary and match source sphere size")
    _, nearest = cKDTree(source).query(target)
    return mask[nearest].astype(bool)


def _unit_sphere(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("sphere coordinates must be finite N by 3 arrays")
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii <= 0):
        raise ValueError("sphere points must have positive radii")
    return points / radii[:, None]

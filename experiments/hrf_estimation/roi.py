"""Map an independently defined surface ROI onto CIFTI columns."""

import numpy as np


def vtc_indices(brain_axis, labels_by_hemisphere):
    mask = np.zeros(len(brain_axis), dtype=bool)
    for structure, side in [('CIFTI_STRUCTURE_CORTEX_LEFT', 'L'),
                            ('CIFTI_STRUCTURE_CORTEX_RIGHT', 'R')]:
        labels = np.asarray(labels_by_hemisphere[side])
        expected = brain_axis.nvertices.get(structure)
        if labels.ndim != 1 or labels.size != expected:
            raise ValueError(f'{side}: ROI size must match the full surface vertex count')
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError('ROI labels must be a binary mask')
        selected = brain_axis.name == structure
        mask[selected] = labels[brain_axis.vertex[selected]].astype(bool)
    if not mask.any():
        raise ValueError('VTC ROI is empty on the CIFTI axis')
    return np.flatnonzero(mask)

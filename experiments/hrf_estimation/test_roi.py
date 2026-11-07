"""A permuted axis must map through hemisphere and surface vertex IDs."""

import importlib

import nibabel as nib
import numpy as np
import pytest


def test_vtc_mapping_uses_both_hemisphere_and_vertex_id():
    left = nib.cifti2.BrainModelAxis.from_surface([4, 1, 3], 6, 'CortexLeft')
    right = nib.cifti2.BrainModelAxis.from_surface([1, 4], 6, 'CortexRight')
    labels = {'L': np.array([0, 1, 0, 0, 1, 0]), 'R': np.array([0, 0, 0, 0, 1, 0])}
    fn = importlib.import_module('experiments.hrf_estimation.roi').vtc_indices
    np.testing.assert_array_equal(fn(left + right, labels), [0, 1, 4])
    with pytest.raises(ValueError, match='empty'):
        fn(left + right, {side: np.zeros(6) for side in labels})
    with pytest.raises(ValueError, match='size'):
        fn(left + right, {'L': np.ones(5), 'R': np.ones(6)})

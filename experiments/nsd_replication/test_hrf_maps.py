import numpy as np
import pytest

from boldtailor.hrf_library import glmsingle_hrf_library
from experiments.nsd_replication.hrf_maps import session_consistency


@pytest.fixture(scope="module")
def library():
    return glmsingle_hrf_library()


def test_session_consistency_per_feature(library):
    indices = np.array([[1, 2, 3], [1, 2, 4]])
    table = session_consistency(library, indices, ["ses-01", "ses-02"])
    assert len(table) == 3
    assert table.mean_pairwise_r.iloc[:2].tolist() == pytest.approx([1.0, 1.0])
    assert table.mean_pairwise_r.iloc[2] < 1
    assert {"mean_pairwise_r", "mean_canonical_baseline"} <= set(table.columns)

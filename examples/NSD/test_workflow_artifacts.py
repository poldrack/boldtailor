"""Shared NSD artifacts: derivative descriptions and RT extraction."""

from importlib.metadata import version
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.NSD.workflow_artifacts import dataset_description
from examples.NSD.workflow_files import reaction_times


def test_dataset_description_is_a_boldtailor_derivative():
    artifact = dataset_description("NSD single-trial models")
    assert artifact.path == "dataset_description.json"
    assert json.loads(artifact.payload) == {
        "Name": "NSD single-trial models",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "boldtailor", "Version": version("boldtailor")}],
    }


def _run(**columns):
    return SimpleNamespace(events=pd.DataFrame(columns, index=range(3)))


def test_reaction_times_keep_clean_columns_and_require_them_by_default():
    rt = reaction_times([_run(response_time=[0.5, np.nan, 1.0])])
    np.testing.assert_allclose(rt[0], [0.5, np.nan, 1.0])
    with pytest.raises(AttributeError):
        reaction_times([_run(onset=[0, 1, 2])])


def test_missing_ok_reaction_times_tolerate_absent_or_text_values():
    runs = [_run(onset=[0, 1, 2]), _run(response_time=["0.5", "n/a", 2])]
    rt = reaction_times(runs, missing_ok=True)
    np.testing.assert_array_equal(np.isnan(rt[0]), [True, True, True])
    np.testing.assert_allclose(rt[1], [0.5, np.nan, 2.0])
    assert all(values.dtype == float for values in rt)

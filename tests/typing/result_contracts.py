"""Static regression for normalized options, sentinels, and serialized activities."""

from typing import assert_type
from collections.abc import Iterator

import numpy as np
from boldtailor._deprecation import UNSET
from boldtailor.data import AnalysisData
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import task_delta_r2_prepared
from boldtailor.provenance import ProvenanceRecord
from boldtailor.results import AnalysisResult
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs


def check_optional_inputs(data: AnalysisData, prepared: PreparedDesignAnalysis,
                          result: AnalysisResult, record: ProvenanceRecord) -> None:
    """Omitted values and activity records have explicit, usable contracts."""
    task_delta_r2_prepared(prepared, result, contrasts=UNSET,
                          noise_model=UNSET, model_metadata=UNSET)
    fit_single_trials(data, hrf_model=UNSET)
    fit_selected_hrfs(data, hrf_selection=UNSET, selection=UNSET)
    assert_type(record.activity_records(), list[dict[str, object]])
    options = RemlHrfOptions(rho=np.array([0.1, 0.2]))
    assert_type(options.subset([1]), RemlHrfOptions)

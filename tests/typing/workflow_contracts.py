"""Static contracts for workflow arrays, metadata, and optional settings."""

from typing import assert_type
import numpy as np
from boldtailor.workflow.analysis import fit_glms, fit_beta_series
from boldtailor.workflow.settings import _check_positive_int


def check_workflow_outputs(runs, root, blocks, model) -> None:
    """Workflow outputs retain array types alongside heterogeneous metadata."""
    glm = fit_glms(runs, root, blocks, model)
    assert_type(glm["effects"], np.ndarray)
    assert_type(glm["provenance"], list[dict[str, object]])
    beta = fit_beta_series(runs, root, blocks, task_model=model.task_model)
    assert_type(beta["betas"], list[np.ndarray])
    assert_type(beta["r2"], np.ndarray)
    assert_type(_check_positive_int(3, "count"), int)
    assert_type(_check_positive_int(None, "count", optional=True), int | None)

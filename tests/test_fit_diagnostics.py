import numpy as np
import pytest


def test_nested_ols_tolerance_keeps_roundoff_but_rejects_material_loss():
    from boldtailor._fit_diagnostics import validate_nested_ols_delta

    validate_nested_ols_delta(np.array([-1e-12, 0.0, 0.5]))
    with pytest.raises(ValueError, match="nested OLS monotonicity violated"):
        validate_nested_ols_delta(np.array([-1.1e-12]))


def test_contrast_metadata_preserves_expression_weights_and_ownership():
    from boldtailor.model import contrast_metadata

    weights = {"face": 1.0, "house": -1.0}
    actual = contrast_metadata({"difference": weights, "expression": "face - house"})
    weights["face"] = 99
    assert actual == {
        "difference": {"kind": "weights", "weights": {"face": 1.0, "house": -1.0}},
        "expression": {"kind": "expression", "value": "face - house"},
    }

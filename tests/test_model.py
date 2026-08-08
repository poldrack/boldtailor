import numpy as np
import pytest

from boldtailor.model import ModelSpec


def test_model_spec_owns_contrasts_and_confounds():
    weights = {"face": 1.0, "house": -1.0}
    contrasts = {"face_gt_house": weights}
    confounds = ["trans_x", "trans_y"]

    model = ModelSpec(contrasts=contrasts, confounds=confounds)
    weights["face"] = 99.0
    contrasts["face_gt_house"] = "face"
    confounds[0] = "changed"

    assert dict(model.contrasts["face_gt_house"]) == {
        "face": 1.0,
        "house": -1.0,
    }
    assert model.confounds == ("trans_x", "trans_y")
    with pytest.raises(TypeError):
        model.contrasts["face_gt_house"]["face"] = 2.0


def test_model_spec_accepts_symbolic_contrast():
    model = ModelSpec(contrasts={"face_gt_house": "face - house"})

    assert model.contrasts["face_gt_house"] == "face - house"
    assert model.noise_model == "ar1"
    assert model.min_onset == -24.0


@pytest.mark.parametrize("noise_model", ["ar0", "ar", "ar2", "ar-1", "white"])
def test_model_spec_rejects_unknown_noise_model(noise_model):
    with pytest.raises(ValueError, match="'ols' or 'ar1'"):
        ModelSpec(contrasts={"face": "face"}, noise_model=noise_model)


def test_model_spec_rejects_positional_contrast_vector():
    with pytest.raises(ValueError, match="semantic"):
        ModelSpec(contrasts={"omnibus": np.eye(2)})


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"": 1.0},
        {"face": np.nan},
        {"face": 0.0, "house": 0.0},
    ],
)
def test_model_spec_rejects_invalid_weight_mapping(weights):
    with pytest.raises(ValueError, match="contrast"):
        ModelSpec(contrasts={"invalid": weights})


@pytest.mark.parametrize("weight", [True, False, np.bool_(True), np.bool_(False)])
def test_model_spec_rejects_boolean_weight(weight):
    with pytest.raises(ValueError, match="contrast"):
        ModelSpec(contrasts={"invalid": {"face": weight}})


@pytest.mark.parametrize(
    "option", ["high_pass", "drift_order", "oversampling", "min_onset"]
)
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_model_spec_rejects_boolean_design_option(option, value):
    with pytest.raises(ValueError, match=option):
        ModelSpec(contrasts={"face": "face"}, **{option: value})


def test_model_spec_rejects_empty_contrasts():
    with pytest.raises(ValueError, match="at least one contrast"):
        ModelSpec(contrasts={})


def test_model_spec_rejects_nonfinite_min_onset():
    with pytest.raises(ValueError, match="min_onset"):
        ModelSpec(contrasts={"face": "face"}, min_onset=np.nan)

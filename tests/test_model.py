import numpy as np
import pytest

from boldtailor.model import (
    ModelSpec,
    TaskModel,
    model_identity,
    nuisance_model_settings,
)


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


_FACE = {"contrasts": {"face": "face"}}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contrasts": {}}, "at least one contrast"),
        ({**_FACE, "noise_model": "ar2"}, "'ols' or 'ar1'"),
        ({"contrasts": {"omnibus": np.eye(2)}}, "semantic"),
        ({"contrasts": {"invalid": {}}}, "at least one weight"),
        ({"contrasts": {"invalid": {"": 1.0}}}, "invalid regressor name"),
        ({"contrasts": {"invalid": {"face": np.nan}}}, "must be finite"),
        ({"contrasts": {"invalid": {"face": 0.0, "house": 0.0}}}, "nonzero weight"),
        ({"contrasts": {"invalid": {"face": True}}}, "must be numeric"),
        ({"contrasts": {"invalid": {"face": np.bool_(False)}}}, "must be numeric"),
        ({**_FACE, "high_pass": True}, "high_pass"),
        ({**_FACE, "high_pass": np.bool_(False)}, "high_pass"),
        ({**_FACE, "drift_order": True}, "drift_order"),
        ({**_FACE, "oversampling": False}, "oversampling"),
        ({**_FACE, "min_onset": True}, "min_onset"),
        ({**_FACE, "min_onset": np.nan}, "min_onset"),
    ],
)
def test_model_spec_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelSpec(**kwargs)


def test_noise_model_message_has_no_phase_reference():
    with pytest.raises(ValueError, match="'ols' or 'ar1'") as error:
        ModelSpec(**_FACE, noise_model="ar2")
    assert "Phase" not in str(error.value)


_TASK_MODEL_HRF = "task_model requires hrf_model 'spm' or 'glover'"


def test_model_spec_accepts_task_model_with_spm_or_glover():
    from boldtailor.model import Modulator, TaskModel

    task_model = TaskModel((Modulator("response_time", missing="indicator"),))
    for hrf in ("spm", "glover"):
        model = ModelSpec(
            contrasts={"task": {"task": 1}}, hrf_model=hrf, task_model=task_model
        )
        assert model.task_model == task_model
    assert ModelSpec(contrasts={"task": {"task": 1}}).task_model is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hrf_model": "spm + derivative", "task_model": TaskModel()}, _TASK_MODEL_HRF),
        (
            {
                "hrf_model": "glover + derivative + dispersion",
                "task_model": TaskModel(),
            },
            _TASK_MODEL_HRF,
        ),
        ({"hrf_model": None, "task_model": TaskModel()}, _TASK_MODEL_HRF),
        ({"hrf_model": "fir", "task_model": TaskModel()}, _TASK_MODEL_HRF),
        ({"hrf_model": "spm", "task_model": "nsd"}, "TaskModel"),
    ],
)
def test_model_spec_rejects_invalid_task_model(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelSpec(contrasts={"task": {"task": 1}}, **kwargs)


def test_model_identity_reports_selected_hrf_without_a_lazy_import():
    import boldtailor.fit as fit_module

    spec = ModelSpec(contrasts={"c": "a"}, task_model=TaskModel())
    identity = model_identity(spec, hrf_model={"kind": "selected"})
    assert identity.activity["hrf_model"] == {"kind": "selected"}
    assert identity.activity["task_model"] == spec.task_model.to_dict()
    assert identity.activity["task_model_fingerprint"] == spec.task_model.fingerprint
    assert identity.fingerprint == identity.activity
    assert not hasattr(fit_module, "_model_provenance")


def test_model_identity_keeps_the_models_own_hrf_by_default():
    spec = ModelSpec(contrasts={"c": "a"}, hrf_model="spm")
    assert model_identity(spec).activity["hrf_model"] == "spm"


def test_nuisance_model_settings_describe_the_nuisance_only_model():
    spec = ModelSpec(contrasts={"c": "a"}, confounds=["motion"])
    settings = nuisance_model_settings(spec)
    assert settings["events"] is False
    assert settings["confounds"] == ["motion"]
    assert settings["noise_model"] == "ols"


def test_fit_modules_import_without_a_cycle():
    import subprocess
    import sys

    code = "import boldtailor._hrf_glm, boldtailor.fit, boldtailor.model"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_prepared_shares_the_run_count_validator_with_data():
    import boldtailor.data as data_module
    import boldtailor.prepared as prepared_module

    assert prepared_module._validate_run_count is data_module._validate_run_count

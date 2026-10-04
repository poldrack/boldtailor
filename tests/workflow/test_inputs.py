"""Session loading, modulator detection, and the GLM model for any task."""

import numpy as np
import pandas as pd
import pytest

from boldtailor.design import expand_events
from boldtailor.model import Modulator, TaskModel
from boldtailor.workflow import inputs

TT = Modulator("trial_type", kind="categorical", levels=("0", "1"))


def test_detection_uses_rt_and_categorical_trial_type(events):
    model = inputs.detect_task_model([events, events])
    assert model == TaskModel((Modulator("response_time", missing="indicator"), TT))
    assert model.regressor_names == ("task", "response_time", "trial_type[1]")
    without_rt = inputs.detect_task_model(
        [events.drop(columns="response_time"), events]
    )
    assert without_rt == TaskModel((TT,))
    assert inputs.detect_task_model([events[["onset", "duration"]]]) == TaskModel()


def test_string_and_mixed_dtype_trial_types_become_one_level_set(events):
    words = events.assign(
        trial_type=["face", "house", "scrambled"] * (len(events) // 3)
    )
    model = inputs.detect_task_model([words, words])
    assert model.regressor_names[-2:] == ("trial_type[house]", "trial_type[scrambled]")
    as_text = events.assign(trial_type=events.trial_type.astype(str))
    assert (
        inputs.detect_task_model([events, as_text]).regressor_names[-1]
        == "trial_type[1]"
    )


def test_single_level_trial_type_is_left_out_with_a_note(events):
    flat = events.assign(trial_type="face")
    assert inputs.detect_task_model([flat]).regressor_names == ("task", "response_time")
    assert inputs.task_model_notes([flat]) == [inputs.TRIAL_TYPE_NOTE]


def test_explicit_categorical_resolves_levels_and_checks_reference(events):
    wanted = (Modulator("trial_type", kind="categorical", reference="1"),)
    model = inputs.detect_task_model([events, events], wanted)
    assert model.regressor_names == ("task", "trial_type[0]")
    bad = (Modulator("trial_type", kind="categorical", reference="7"),)
    with pytest.raises(inputs.InputError, match="trial_type.*7"):
        inputs.detect_task_model([events], bad)


def test_run_missing_a_level_is_an_input_error_naming_the_run(four_runs, settings_for):
    root, _ = four_runs
    path = sorted((root / "sub-07" / "ses-nsd10" / "func").glob("*run-02_events.tsv"))[
        0
    ]
    table = pd.read_csv(path, sep="\t")
    table.assign(trial_type=0).to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="run-02.*trial_type.*'1'"):
        inputs.load_session(settings_for(root))


def test_run_summary_counts_each_level(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    summary = inputs.run_summary(runs, model)
    assert {"n_trial_type_0", "n_trial_type_1"} <= set(summary.columns)
    assert (summary.n_trial_type_0 + summary.n_trial_type_1 == summary.trials).all()


def test_explicit_modulators_must_exist_in_every_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run 2.*stimulus_id"):
        inputs.detect_task_model([events.assign(stimulus_id=1), events], wanted)
    assert inputs.detect_task_model(
        [events.assign(stimulus_id=1)] * 2, wanted
    ) == TaskModel(wanted)


def test_load_session_detects_the_model_and_builds_the_glm(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    glm = inputs.glm_model(runs, model)
    assert (
        glm.task_model == model and glm.noise_model == "ols" and glm.drift_model is None
    )
    assert set(glm.contrasts) == {"task", "response_time", "trial_type[1]"}
    assert list(inputs.run_summary(runs, model).columns)[:5] == [
        "run",
        "trials",
        "scans",
        "retained_scans",
        "dropped_scans",
    ]


def test_task_only_session_has_no_rt_columns(four_runs, settings_for):
    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        table = pd.read_csv(path, sep="\t")
        table.drop(columns=["response_time", "trial_type"]).to_csv(
            path, sep="\t", index=False
        )
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    assert model == TaskModel()
    assert set(inputs.glm_model(runs, model).contrasts) == {"task"}
    summary = inputs.run_summary(runs, model)
    assert "mean_rt_seconds" not in summary.columns


def test_hrf_only_loading_needs_two_runs_not_two_per_parity(dataset, settings_for):
    root, *_ = dataset
    assert len(inputs.load_session(settings_for(root), hrf_only=True)) == 2
    with pytest.raises(ValueError, match="two odd and two even"):
        inputs.load_session(settings_for(root))


@pytest.mark.parametrize("column,value", [("trial_type", 2), ("trial_type", np.nan)])
def test_invalid_glm_covariates_fail_explicitly(events, column, value):
    model = inputs.detect_task_model([events])
    events.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        inputs.validate_glm_events(events, model)


def test_run_without_the_declared_levels_is_an_input_error(four_runs, settings_for):
    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.assign(trial_type=2).to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match=r"run-\d+.*trial_type"):
        inputs.load_session(settings_for(root))


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_nonpositive_rt_becomes_missing_with_indicator(
    four_runs, settings_for, missing
):
    root, _ = four_runs
    path = next(root.rglob("*run-01_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[1, "response_time"] = missing
    table.to_csv(path, sep="\t", index=False)
    runs = inputs.load_session(settings_for(root))
    run = next(r for r in runs if r.number == 1)
    assert np.isnan(run.events.response_time.iloc[1])
    assert np.isfinite(run.events.response_time.drop(index=1)).all()
    model = inputs.detect_task_model([r.events for r in runs])
    expanded = expand_events(run.events, model)
    indicator = expanded.loc[
        expanded.trial_type == "missing_response_time", "modulation"
    ]
    np.testing.assert_array_equal(
        indicator, (np.arange(len(run.events)) == 1).astype(float)
    )


def test_glm_requires_observed_rt_to_estimate_rt_effect(events):
    model = inputs.detect_task_model([events])
    events["response_time"] = np.nan
    with pytest.raises(ValueError, match="response_time.*positive.*finite"):
        inputs.validate_glm_events(events, model)


def test_glm_does_not_treat_malformed_rt_text_as_missing(events):
    model = inputs.detect_task_model([events])
    events["response_time"] = events.response_time.astype(object)
    events.loc[1, "response_time"] = "invalid"
    with pytest.raises(ValueError, match="response_time"):
        inputs.validate_glm_events(events, model)


def test_trimming_keeps_acquisition_times_and_matches_confounds(
    four_runs, settings_for
):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    data = inputs.load_block(runs, root, [0, 2], model)
    assert [len(t) for t in data.frame_times] == [95, 94, 93, 95]
    for run, y, dropped in zip(runs, data.signals, (1, 2, 3, 1), strict=True):
        np.testing.assert_allclose(
            run.frame_times, 0.775 + np.arange(dropped, 96) * 1.6
        )
        np.testing.assert_allclose(y, np.asarray(run.image.dataobj)[dropped:, [0, 2]])
        assert not any(c.startswith("non_steady") for c in run.confounds)
        assert run.events.onset.iloc[0] == 8.0
    assert len({tuple(r.confounds.columns) for r in runs}) == 1
    assert sum(c.startswith("a_comp_cor") for c in runs[0].confounds) == 6
    assert data.provenance.metadata_fingerprint is not None
    assert (
        data.provenance.sources[2].signal.annotations["retained_frame_indices"][0] == 3
    )


def test_interior_nonsteady_flag_is_rejected(four_runs, settings_for):
    root, prep = four_runs
    path = next(prep.rglob("*run-01*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[12, "non_steady_state_outlier00"] = 1
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="leading|contiguous"):
        inputs.load_session(settings_for(root))


def test_selection_task_model_switch_drops_only_rt(events):
    model = inputs.detect_task_model([events])
    assert inputs.selection_task_model(model, True) == model
    without_rt = inputs.selection_task_model(model, False)
    assert without_rt == TaskModel((TT,))
    assert without_rt.is_subset_of(model)
    with pytest.raises(ValueError, match="include_rt"):
        inputs.selection_task_model(model, 1)


def test_missing_modulator_column_in_a_session_is_an_input_error(
    four_runs, settings_for
):
    root, _ = four_runs
    settings = settings_for(root, modulators=(Modulator("stimulus_id"),))
    with pytest.raises(inputs.InputError, match="stimulus_id"):
        inputs.load_session(settings)


def test_detection_takes_any_multilevel_trial_type_without_a_note(events):
    for values in (["face", "house"] * 3, [0, 1, 2, 0, 1, 2]):
        tables = [events, events.assign(trial_type=values)]
        model = inputs.detect_task_model(tables)
        assert model.modulators[-1].kind == "categorical"
        assert inputs.task_model_notes(tables) == []


def test_detection_keeps_binary_trial_type_written_as_text(events):
    text = events.assign(trial_type=events.trial_type.astype(str))
    assert TT in inputs.detect_task_model([text]).modulators
    assert inputs.task_model_notes([text]) == []


def test_automatic_trial_type_with_missing_values_is_an_input_error(
    four_runs, settings_for
):
    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[0, "trial_type"] = np.nan
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="run-02.*trial_type.*missing"):
        inputs.load_session(settings_for(root))


def test_notes_are_only_for_automatic_detection(events):
    strings = events.assign(trial_type=["face", "house"] * 3)
    assert inputs.task_model_notes([strings], modulators=()) == []
    assert inputs.task_model_notes([strings], modulators=(Modulator("x"),)) == []


def test_explicit_modulator_errors_name_the_bids_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(inputs.InputError, match="run-02.*stimulus_id"):
        inputs.detect_task_model(
            [events.assign(stimulus_id=1), events], wanted, labels=["run-01", "run-02"]
        )


def test_explicit_trial_type_stays_strict_and_names_the_run(four_runs, settings_for):
    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.assign(trial_type=["face", "house"] * 3).to_csv(path, sep="\t", index=False)
    settings = settings_for(root, modulators=(Modulator("trial_type"),))
    with pytest.raises(inputs.InputError, match="run-02.*trial_type"):
        inputs.load_session(settings)


def test_trimming_errors_name_the_bids_run(four_runs, settings_for):
    root, prep = four_runs
    path = next(prep.rglob("*run-03*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table["non_steady_state_outlier00"] = table.non_steady_state_outlier00.astype(float)
    table.loc[0, "non_steady_state_outlier00"] = 0.5
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="run-03.*binary"):
        inputs.load_session(settings_for(root))


def test_no_modulators_means_a_task_only_model(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root, modulators=()))
    assert inputs.detect_task_model([r.events for r in runs], ()) == TaskModel()


REMEDY = "--modulator trial_type:categorical,indicator"


def test_auto_trial_type_with_missing_rows_names_run_count_and_remedy(
    four_runs, settings_for
):
    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[[0, 1], "trial_type"] = np.nan
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError) as raised:
        inputs.load_session(settings_for(root))
    message = str(raised.value)
    assert "run-02" in message and "2 missing" in message
    assert REMEDY in message and "--no-modulators" in message


def test_auto_trial_type_blocked_design_gets_the_remedy(four_runs, settings_for):
    root, _ = four_runs
    for number, label in ((1, "a"), (2, "b")):
        path = next(root.rglob(f"*run-0{number}_events.tsv"))
        table = pd.read_csv(path, sep="\t")
        table.assign(trial_type=label).to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="no 'trial_type' trials") as raised:
        inputs.load_session(settings_for(root))
    assert REMEDY in str(raised.value)


def test_explicit_modulators_get_no_automatic_detection_hint(four_runs, settings_for):
    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[0, "trial_type"] = np.nan
    table.to_csv(path, sep="\t", index=False)
    settings = settings_for(
        root, modulators=(Modulator("trial_type", kind="categorical"),)
    )
    with pytest.raises(inputs.InputError) as raised:
        inputs.load_session(settings)
    assert "detected automatically" not in str(raised.value)


def test_boolean_trial_type_error_names_run_and_column(four_runs, settings_for):
    root, _ = four_runs
    path = next(root.rglob("*run-03_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.assign(trial_type=[True, False] * 3).to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="run-03.*trial_type"):
        inputs.load_session(settings_for(root))


def test_numeric_trial_type_on_strings_suggests_categorical(four_runs, settings_for):
    root, _ = four_runs
    from tests.workflow.synthetic_bids import face_house_events, rewrite_events

    rewrite_events(root, face_house_events)
    settings = settings_for(root, modulators=(Modulator("trial_type"),))
    with pytest.raises(inputs.InputError, match="numeric.*:categorical"):
        inputs.load_session(settings)

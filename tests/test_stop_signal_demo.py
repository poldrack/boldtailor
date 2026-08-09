import io
import importlib
import json
from pathlib import Path
import warnings

import nibabel as nib
import nbformat
import numpy as np
import pandas as pd
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec
from boldtailor.publication import Artifact, publish_artifact_set
from examples.stop_signal_demo import (
    common_brain_mask,
    estimate_signal_memory_gib,
    load_run,
    make_masker,
    run_sources,
    whole_brain_image,
)

TRIAL_TYPES = (
    "go_success",
    "go_failure",
    "stop_success",
    "stop_failure",
)
CONFOUNDS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "framewise_displacement",
)
NOTEBOOK = Path(__file__).parents[1] / "examples" / "stop_signal_demo.ipynb"


def _demo_module():
    return importlib.import_module("examples.stop_signal_demo")


def _notebook_configuration():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    configuration_cell = next(
        cell for cell in notebook.cells if cell.cell_type == "code"
    )
    namespace = {}
    exec(compile(configuration_cell.source, NOTEBOOK.name, "exec"), namespace)
    return namespace


def _discover(root, session="ses-02"):
    return _demo_module().discover_run_inputs(
        root,
        root / "derivatives" / "fmri_25.2.0",
        subject="sub-s4",
        session=session,
        task="stopSignal",
        run="run-01",
        space="MNI152NLin2009cAsym",
        resolution=2,
    )


def test_discover_run_inputs_matches_all_entities(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)

    assert isinstance(inputs, _demo_module().RunInputs)
    assert inputs.session == "ses-02"
    assert inputs.events.name.endswith("run-01_events.tsv")
    assert "space-MNI152NLin2009cAsym_res-2" in inputs.bold.name
    assert inputs.mask.name.endswith("desc-brain_mask.nii.gz")
    assert inputs.confounds.name.endswith("desc-confounds_timeseries.tsv")


def test_discover_run_inputs_rejects_missing_file(stop_signal_bids_dataset):
    _discover(stop_signal_bids_dataset).events.unlink()

    with pytest.raises(FileNotFoundError, match="events.*exactly one"):
        _discover(stop_signal_bids_dataset)


def test_discover_run_inputs_rejects_duplicate_file(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    duplicate = inputs.events.with_name(
        inputs.events.name.replace("events", "copy_events")
    )
    duplicate.write_bytes(inputs.events.read_bytes())

    with pytest.raises(ValueError, match="events.*exactly one"):
        _discover(stop_signal_bids_dataset)


def _loaded_runs(root):
    inputs = tuple(_discover(root, session) for session in ("ses-02", "ses-04"))
    mask_image = common_brain_mask(inputs)
    masker = make_masker(mask_image)
    runs = tuple(
        load_run(
            item,
            masker,
            trial_types=TRIAL_TYPES,
            confound_names=CONFOUNDS,
        )
        for item in inputs
    )
    return inputs, mask_image, masker, runs


@pytest.fixture
def example_result(stop_signal_bids_dataset):
    inputs, _, _, runs = _loaded_runs(stop_signal_bids_dataset)
    rng = np.random.default_rng(20260808)
    confounds = []
    for run in runs:
        values, _ = np.linalg.qr(
            rng.standard_normal((len(run.frame_times), len(CONFOUNDS)))
        )
        confounds.append(pd.DataFrame(values, columns=CONFOUNDS))
    data = from_arrays(
        [run.signals for run in runs],
        [run.events for run in runs],
        frame_times=[run.frame_times for run in runs],
        confounds=confounds,
        sources=[run_sources(item, stop_signal_bids_dataset) for item in inputs],
        provenance_metadata={"example": "stop-signal"},
    )
    model = ModelSpec(
        contrasts={
            "successful_inhibition": "stop_success - stop_failure",
            "stop_vs_go": "(stop_success + stop_failure) - go_success",
            "go_success_vs_baseline": "go_success",
        },
        confounds=CONFOUNDS,
        noise_model="ols",
    )
    return inputs, fit(data, model)


def test_common_brain_mask_intersects_runs(stop_signal_bids_dataset):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session)
        for session in ("ses-02", "ses-04")
    )
    second = nib.load(inputs[1].mask)
    values = np.asarray(second.dataobj).copy()
    values[0, 0, 0] = 0
    nib.save(
        nib.Nifti1Image(values, second.affine, second.header),
        inputs[1].mask,
    )

    mask_image = common_brain_mask(inputs)

    assert mask_image.shape == (7, 7, 7)
    assert int(np.asarray(mask_image.dataobj).sum()) == 342
    np.testing.assert_allclose(mask_image.affine, nib.load(inputs[0].mask).affine)


def test_common_brain_mask_rejects_mismatched_affine(stop_signal_bids_dataset):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session)
        for session in ("ses-02", "ses-04")
    )
    second = nib.load(inputs[1].mask)
    affine = second.affine.copy()
    affine[0, 3] += 1.0
    nib.save(
        nib.Nifti1Image(np.asarray(second.dataobj), affine, second.header),
        inputs[1].mask,
    )

    with pytest.raises(ValueError, match="masks must share shape and affine"):
        common_brain_mask(inputs)


def test_masker_preserves_whole_brain_values(stop_signal_bids_dataset):
    _, mask_image, masker, runs = _loaded_runs(stop_signal_bids_dataset)

    assert int(np.asarray(mask_image.dataobj).sum()) == 343
    assert runs[0].signals.shape == (80, 343)
    assert runs[1].signals.shape == (88, 343)
    assert masker.standardize is False
    assert masker.detrend is False
    assert masker.smoothing_fwhm is None
    assert masker.low_pass is None
    assert masker.high_pass is None
    assert masker.reports is False
    np.testing.assert_allclose(np.diff(runs[0].frame_times), 1.5)
    assert set(runs[0].events.trial_type) <= set(TRIAL_TYPES)
    assert tuple(runs[0].confounds) == CONFOUNDS
    assert runs[0].confounds.iloc[0].framewise_displacement == 0.0
    assert np.isfinite(runs[0].signals).all()


def test_load_run_rejects_bold_mask_geometry_mismatch(stop_signal_bids_dataset):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session)
        for session in ("ses-02", "ses-04")
    )
    masker = make_masker(common_brain_mask(inputs))
    image = nib.load(inputs[1].bold)
    affine = image.affine.copy()
    affine[1, 3] += 1.0
    nib.save(
        nib.Nifti1Image(np.asarray(image.dataobj), affine, image.header),
        inputs[1].bold,
    )

    with pytest.raises(ValueError, match="BOLD image must match common mask geometry"):
        load_run(
            inputs[1],
            masker,
            trial_types=TRIAL_TYPES,
            confound_names=CONFOUNDS,
        )


def test_whole_brain_image_round_trips_mask_values(stop_signal_bids_dataset):
    _, mask_image, masker, _ = _loaded_runs(stop_signal_bids_dataset)
    values = np.arange(343, dtype=float)

    image = whole_brain_image(values, masker)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="boolean values for 'standardize' will be deprecated.*",
            category=FutureWarning,
        )
        restored = masker.transform(image)

    np.testing.assert_array_equal(restored, values)
    assert image.shape == mask_image.shape
    np.testing.assert_allclose(image.affine, mask_image.affine)


def test_signal_memory_estimate_uses_float64_storage():
    estimate = estimate_signal_memory_gib((80, 88), 343)

    assert estimate == pytest.approx(
        80 * 343 * 8 / 2**30 + 88 * 343 * 8 / 2**30
    )


def test_load_run_rejects_confound_length_mismatch(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t").iloc[:-1]
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    masker = make_masker(common_brain_mask((inputs,)))

    with pytest.raises(ValueError, match="confounds.*80 rows"):
        load_run(inputs, masker, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS)


def test_load_run_rejects_unexpected_nonfinite_confound(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t")
    frame.loc[3, "trans_x"] = np.nan
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    masker = make_masker(common_brain_mask((inputs,)))

    with pytest.raises(ValueError, match="non-finite.*trans_x"):
        load_run(inputs, masker, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS)


def test_run_sources_records_dataset_relative_inputs(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)

    sources = run_sources(inputs, stop_signal_bids_dataset)

    assert sources.signal.uri.startswith("derivatives/fmri_25.2.0/")
    assert sources.signal.byte_size == inputs.bold.stat().st_size
    assert sources.signal.annotations["mask"]["uri"].endswith("brain_mask.nii.gz")
    assert sources.events.uri.startswith("sub-s4/ses-02/")
    assert sources.confounds.uri.endswith("desc-confounds_timeseries.tsv")
    assert sources.signal.modified_at.endswith("Z")


def test_load_run_rejects_event_beyond_acquisition(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.events, sep="\t")
    frame.loc[0, ["onset", "duration"]] = [119.5, 1.0]
    frame.to_csv(inputs.events, sep="\t", index=False)
    masker = make_masker(common_brain_mask((inputs,)))

    with pytest.raises(ValueError, match="event timing exceeds acquisition"):
        load_run(inputs, masker, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS)


def test_result_artifacts_are_deterministic_valid_metadata(
    example_result, stop_signal_bids_dataset, tmp_path
):
    _, result = example_result
    options = {
        "subject": "sub-s4",
        "task": "stopSignal",
        "sessions": ["ses-02", "ses-04"],
        "roi": {"center_mni": [48.0, 16.0, 20.0], "radius_mm": 6.0},
    }
    result_artifacts = _demo_module().result_artifacts

    first = result_artifacts(
        result,
        subject="sub-s4",
        task="stopSignal",
        sessions=("ses-02", "ses-04"),
        configuration=options,
    )
    second = result_artifacts(
        result,
        subject="sub-s4",
        task="stopSignal",
        sessions=("ses-02", "ses-04"),
        configuration=options,
    )

    assert first == second
    assert {item.path for item in first} == {
        "reports/sub-s4_ses-02_task-stopSignal_desc-design_matrix.tsv",
        "reports/sub-s4_ses-04_task-stopSignal_desc-design_matrix.tsv",
        "reports/sub-s4_task-stopSignal_desc-roi_contrasts.tsv",
        "reports/sub-s4_task-stopSignal_desc-example_config.json",
    }
    assert all(isinstance(item.payload, bytes) for item in first)
    artifacts = {item.path: item for item in first}
    for index, session in enumerate(("ses-02", "ses-04")):
        path = f"reports/sub-s4_{session}_task-stopSignal_desc-design_matrix.tsv"
        design = pd.read_csv(io.BytesIO(artifacts[path].payload), sep="\t")
        expected = result.design_matrices[index]
        assert design.columns[0] == "frame_time"
        assert list(design.columns[1:]) == list(expected.columns)
        np.testing.assert_allclose(design.frame_time, expected.index)
        np.testing.assert_allclose(design.iloc[:, 1:], expected)

    contrasts = pd.read_csv(
        io.BytesIO(
            artifacts["reports/sub-s4_task-stopSignal_desc-roi_contrasts.tsv"].payload
        ),
        sep="\t",
    )
    assert list(contrasts.columns) == [
        "contrast",
        "n_features",
        "mean_effect",
        "mean_z",
        "max_abs_z",
        "min_one_sided_p",
    ]
    assert list(contrasts.contrast) == list(result.contrast_names)
    np.testing.assert_array_equal(
        contrasts.n_features,
        [result.effect(name).size for name in result.contrast_names],
    )
    np.testing.assert_allclose(
        contrasts.mean_effect,
        [np.mean(result.effect(name)) for name in result.contrast_names],
    )
    np.testing.assert_allclose(
        contrasts.mean_z,
        [np.mean(result.z_score(name)) for name in result.contrast_names],
    )
    np.testing.assert_allclose(
        contrasts.max_abs_z,
        [np.max(np.abs(result.z_score(name))) for name in result.contrast_names],
    )
    np.testing.assert_allclose(
        contrasts.min_one_sided_p,
        [np.min(result.one_sided_p_value(name)) for name in result.contrast_names],
    )

    config_text = artifacts[
        "reports/sub-s4_task-stopSignal_desc-example_config.json"
    ].payload.decode("utf-8")
    assert json.loads(config_text) == options
    assert str(stop_signal_bids_dataset.resolve()) not in config_text

    published = publish_artifact_set(tmp_path / "published", first)
    assert len(published) == len(first)
    assert all(path.is_file() for path in published)


def test_publication_destination_defaults_to_temp(stop_signal_bids_dataset, tmp_path):
    publication_destination = _demo_module().publication_destination

    destination = publication_destination(
        stop_signal_bids_dataset,
        persistent=False,
        temporary_parent=tmp_path,
    )

    assert destination.parent == tmp_path
    assert destination.name.startswith("boldtailor-")


def test_persistent_destination_is_restricted(stop_signal_bids_dataset, tmp_path):
    publication_destination = _demo_module().publication_destination
    expected = stop_signal_bids_dataset / "derivatives" / "boldtailor"

    assert (
        publication_destination(stop_signal_bids_dataset, persistent=True)
        == expected.resolve()
    )

    with pytest.raises(ValueError, match="derivatives/boldtailor"):
        publication_destination(
            stop_signal_bids_dataset,
            persistent=True,
            requested=tmp_path / "outside",
        )


@pytest.mark.parametrize(
    "symlink_path",
    ("derivatives", "derivatives/boldtailor"),
)
def test_persistent_destination_rejects_symlink_components(tmp_path, symlink_path):
    publication_destination = _demo_module().publication_destination
    bids_root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    bids_root.mkdir()
    outside.mkdir()

    target = bids_root / symlink_path
    target.parent.mkdir(exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        publication_destination(bids_root, persistent=True)


def test_notebook_contains_the_complete_feature_story():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    for phrase in (
        "Run-specific designs",
        "Structured logging and provenance",
        "BIDS provenance projection",
        "Transactional publication",
        "stop_success - stop_failure",
    ):
        assert phrase in source


def test_notebook_configuration_defaults_to_complete_real_sessions(monkeypatch):
    monkeypatch.delenv("BOLDTAILOR_SESSIONS", raising=False)

    configuration = _notebook_configuration()

    assert configuration["SESSIONS"] == ("ses-06", "ses-08")


def test_notebook_configuration_normalizes_session_override(monkeypatch):
    monkeypatch.setenv("BOLDTAILOR_SESSIONS", " ses-02, ses-04 ")

    configuration = _notebook_configuration()

    assert configuration["SESSIONS"] == ("ses-02", "ses-04")


@pytest.mark.parametrize(
    "selection",
    ("", "ses-02", "ses-02,,ses-04", "ses-02,ses-04,ses-06"),
)
def test_notebook_configuration_rejects_invalid_session_override(
    monkeypatch, selection
):
    monkeypatch.setenv("BOLDTAILOR_SESSIONS", selection)

    with pytest.raises(ValueError, match="exactly two non-empty sessions"):
        _notebook_configuration()


@pytest.mark.parametrize(
    "working_directory",
    (NOTEBOOK.parents[1], NOTEBOOK.parent),
    ids=("repository-root", "notebook-directory"),
)
def test_notebook_executes_against_fixture(
    stop_signal_bids_dataset, tmp_path, monkeypatch, working_directory
):
    monkeypatch.setenv("BOLDTAILOR_BIDS_ROOT", str(stop_signal_bids_dataset))
    monkeypatch.setenv("BOLDTAILOR_SESSIONS", "ses-02,ses-04")
    monkeypatch.setenv("BOLDTAILOR_TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
    )

    try:
        executed = client.execute()
    except CellExecutionError as error:
        pytest.fail(str(error))

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", ())
    )
    assert tuple(tmp_path.glob("boldtailor-*"))


def test_readme_links_real_data_notebook():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "examples/stop_signal_demo.ipynb" in readme


def test_protected_source_paths_include_all_inputs(stop_signal_bids_dataset):
    protected_source_paths = _demo_module().protected_source_paths
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04")
    )

    protected = protected_source_paths(inputs)

    assert set(protected) == {
        path
        for item in inputs
        for path in (item.events, item.bold, item.mask, item.confounds)
    }


def test_publication_refuses_overlap_with_protected_source(
    stop_signal_bids_dataset,
):
    protected_source_paths = _demo_module().protected_source_paths
    inputs = (_discover(stop_signal_bids_dataset),)
    artifact = Artifact(
        inputs[0].events.name,
        b"onset\tduration\ttrial_type\n0\t1\tgo_success\n",
    )

    with pytest.raises(ValueError, match="source"):
        publish_artifact_set(
            inputs[0].events.parent,
            (artifact,),
            source_paths=protected_source_paths(inputs),
        )

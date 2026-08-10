import ast
import gzip
import hashlib
import importlib
import io
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
from boldtailor.fit import fit, task_delta_r2
from boldtailor.model import ModelSpec
from boldtailor.publication import Artifact, publish_artifact_set
from boldtailor.results import make_task_delta_r2_result
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
CONTRAST_EXPRESSIONS = {
    "successful_inhibition": "stop_success - stop_failure",
    "stop_vs_go": "(stop_success + stop_failure) - go_success",
    "go_success_vs_baseline": "go_success",
}
MASK_METADATA = {
    "strategy": "intersection",
    "shape": [7, 7, 7],
    "affine": [
        [2.0, 0.0, 0.0, 42.0],
        [0.0, 2.0, 0.0, 10.0],
        [0.0, 0.0, 2.0, 14.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "voxel_count": 343,
}
MASKER_SETTINGS = {
    "standardize": False,
    "detrend": False,
    "smoothing_fwhm": None,
    "low_pass": None,
    "high_pass": None,
    "reports": False,
}
RESOURCE_ASSUMPTIONS = {
    "minimum_memory_gib": 32,
    "feature_chunking": False,
    "estimated_signal_memory_gib": 0.00042933225631713867,
}
TRANSFORMED_SIGNALS = [
    {"session": "ses-02", "shape": [80, 343], "dtype": "float64"},
    {"session": "ses-04", "shape": [88, 343], "dtype": "float64"},
]
PLOT_AUDIT_PREFIX = "BOLDTAILOR_PLOT_AUDIT="
DELTA_AUDIT_PREFIX = "BOLDTAILOR_DELTA_AUDIT="
DISPLAY_AUDIT_PREFIX = "BOLDTAILOR_DISPLAY_AUDIT="
DISPLAY_AUDIT_SETUP = """
_display_audit_records = []
_real_display = display


def _shares_raw_signal_memory(value):
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        array = value.to_numpy(copy=False)
    else:
        return False
    return any(
        np.shares_memory(array, run.signals) for run in loaded_runs
    )


def _display_footprint(value):
    if isinstance(value, np.ndarray):
        return int(value.size), _shares_raw_signal_memory(value)
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return int(value.size), _shares_raw_signal_memory(value)
    if isinstance(value, dict):
        children = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        children = tuple(value)
    else:
        return 0, False
    footprints = tuple(_display_footprint(child) for child in children)
    nested_sizes = tuple(size for size, _ in footprints)
    return (
        max((len(children), *nested_sizes), default=0),
        any(is_raw_signal for _, is_raw_signal in footprints),
    )


def _record_display(*objects, **kwargs):
    _display_audit_records.extend(_display_footprint(value) for value in objects)
    return _real_display(*objects, **kwargs)


display = _record_display
"""
PLOT_AUDIT_SETUP = """
import inspect as _inspect
import json as _json

_plot_stat_map_records = []
_real_plot_stat_map = plotting.plot_stat_map


def _record_plot_stat_map(stat_map_img, *args, **kwargs):
    call = _inspect.signature(_real_plot_stat_map).bind(
        stat_map_img, *args, **kwargs
    )
    call.apply_defaults()
    _plot_stat_map_records.append(
        {"image": stat_map_img, "arguments": call.arguments}
    )
    return _real_plot_stat_map(stat_map_img, *args, **kwargs)


plotting.plot_stat_map = _record_plot_stat_map
"""
PLOT_AUDIT_REPORT = f"""
_plot_audit = []
_common_mask_values = (
    np.asarray(common_mask.dataobj, dtype=bool)
    if "common_mask" in globals()
    else None
)
for _record in _plot_stat_map_records:
    _arguments = _record["arguments"]
    _matches = []
    _matches_aggregate = False
    _matches_task_delta = False
    _minimum = None
    _all_nonnegative = False
    if _common_mask_values is not None:
        _values = np.asarray(_record["image"].dataobj)[_common_mask_values]
        _minimum = float(_values.min())
        _all_nonnegative = bool(np.all(_values >= 0.0))
        _matches = [
            _name
            for _name in result.contrast_names
            if np.allclose(_values, result.z_score(_name))
        ]
        _matches_aggregate = bool(np.allclose(_values, result.r2))
        if "task_delta" in globals():
            _matches_task_delta = bool(
                np.allclose(_values, task_delta.delta_r2)
            )
    _plot_audit.append(
        {{
            "matched_z_scores": _matches,
            "matched_aggregate_r2": _matches_aggregate,
            "matched_task_delta_r2": _matches_task_delta,
            "minimum": _minimum,
            "all_nonnegative": _all_nonnegative,
            "threshold": _arguments["threshold"],
            "vmin": _arguments["vmin"],
            "colorbar": _arguments["colorbar"],
            "cmap": _arguments["cmap"],
            "symmetric_cbar": _arguments["symmetric_cbar"],
            "title": _arguments["title"],
        }}
    )
print("{PLOT_AUDIT_PREFIX}" + _json.dumps(_plot_audit, sort_keys=True))
_delta_audit = {{
    "raw_min_delta_r2": float(task_delta.raw_min),
    "negative_voxel_count": int(task_delta.negative_voxel_count),
    "mean_delta_r2": float(task_delta.delta_r2.mean()),
    "max_delta_r2": float(task_delta.delta_r2.max()),
}}
print("{DELTA_AUDIT_PREFIX}" + _json.dumps(_delta_audit, sort_keys=True))
_analysis_display_count = len(_display_audit_records)
"""
DISPLAY_AUDIT_REPORT = f"""
_display_audit = {{
    "display_count": _analysis_display_count,
    "max_array_or_table_elements": max(
        (size for size, _ in _display_audit_records), default=0
    ),
    "contains_raw_signal_memory": any(
        is_raw_signal for _, is_raw_signal in _display_audit_records
    ),
}}
print("{DISPLAY_AUDIT_PREFIX}" + _json.dumps(_display_audit, sort_keys=True))
"""


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


def _instrument_notebook_plots(notebook):
    design_index = next(
        index for index, cell in enumerate(notebook.cells) if cell.id == "design-fit"
    )
    notebook.cells.insert(
        design_index,
        nbformat.v4.new_code_cell(DISPLAY_AUDIT_SETUP),
    )
    results_index = next(
        index for index, cell in enumerate(notebook.cells) if cell.id == "results"
    )
    notebook.cells.insert(
        results_index,
        nbformat.v4.new_code_cell(PLOT_AUDIT_SETUP),
    )
    notebook.cells.insert(
        results_index + 2,
        nbformat.v4.new_code_cell(PLOT_AUDIT_REPORT),
    )
    notebook.cells.append(nbformat.v4.new_code_cell(DISPLAY_AUDIT_REPORT))


def _plot_audit(executed):
    return _runtime_audit(executed, PLOT_AUDIT_PREFIX)


def _delta_audit(executed):
    return _runtime_audit(executed, DELTA_AUDIT_PREFIX)


def _display_audit(executed):
    return _runtime_audit(executed, DISPLAY_AUDIT_PREFIX)


def _runtime_audit(executed, prefix):
    audit_lines = [
        line
        for cell in executed.cells
        for output in cell.get("outputs", ())
        if output.get("output_type") == "stream"
        for line in output.get("text", "").splitlines()
        if line.startswith(prefix)
    ]
    assert len(audit_lines) == 1
    return json.loads(audit_lines[0].removeprefix(prefix))


def _assert_display_contract(executed):
    audit = _display_audit(executed)
    assert audit["display_count"] == 11
    assert audit["max_array_or_table_elements"] <= 256
    assert audit["contains_raw_signal_memory"] is False

    relevant_cells = {
        cell.id: cell for cell in executed.cells if cell.id in {"design-fit", "results"}
    }
    assert set(relevant_cells) == {"design-fit", "results"}
    rendered_cells = {
        cell_id: _rendered_cell_text(cell) for cell_id, cell in relevant_cells.items()
    }
    for rendered_text in rendered_cells.values():
        rendered_text_size = len(rendered_text)
        assert rendered_text_size <= 20_000
    design_rendered = rendered_cells["design-fit"]
    assert "nuisance_design_columns" in design_rendered
    for field in (
        "raw_min_delta_r2",
        "negative_voxel_count",
        "mean_delta_r2",
        "max_delta_r2",
    ):
        assert field in design_rendered


def _rendered_cell_text(cell):
    return "\n".join(
        str(output.get("text", "")) + str(output.get("data", {}).get("text/plain", ""))
        for output in cell.get("outputs", ())
    )


def _assert_compact_variance_display_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.id == "design-fit")
    display_calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "display"
    ]
    assert any(
        len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "variance_partition"
        for call in display_calls
    )
    for call in display_calls:
        displayed_nodes = (
            *call.args,
            *(keyword.value for keyword in call.keywords),
        )
        assert not any(
            (isinstance(node, ast.Attribute) and node.attr == "signals")
            or (isinstance(node, ast.Name) and node.id == "signals")
            for displayed in displayed_nodes
            for node in ast.walk(displayed)
        )


def _assert_plot_contract(executed):
    audit = _plot_audit(executed)

    assert len(audit) == 5
    for call, contrast_name in zip(audit[:3], CONTRAST_EXPRESSIONS, strict=True):
        assert call["matched_z_scores"] == [contrast_name]
        assert call["matched_aggregate_r2"] is False
        assert call["matched_task_delta_r2"] is False
        assert call["threshold"] is None
        assert call["colorbar"] is True

    aggregate = audit[3]
    assert aggregate["matched_z_scores"] == []
    assert aggregate["matched_aggregate_r2"] is True
    assert aggregate["matched_task_delta_r2"] is False
    assert aggregate["threshold"] is None
    assert aggregate["colorbar"] is True
    assert aggregate["cmap"] == "viridis"
    assert aggregate["symmetric_cbar"] is False

    delta = audit[4]
    assert delta["matched_z_scores"] == []
    assert delta["matched_aggregate_r2"] is False
    assert delta["matched_task_delta_r2"] is True
    assert delta["minimum"] >= 0.0
    assert delta["all_nonnegative"] is True
    assert delta["threshold"] is None
    assert delta["vmin"] == 0
    assert delta["colorbar"] is True
    assert delta["cmap"] == "magma"
    assert delta["symmetric_cbar"] is False
    assert "delta" in delta["title"].lower()
    assert "clipped at zero" in delta["title"].lower()


def _assert_published_metadata(published, bids_root, expected_delta):
    configuration_path = (
        published / "reports/sub-s4_task-stopSignal_desc-example_config.json"
    )
    configuration = json.loads(configuration_path.read_text())
    expected_shared = {
        "mask": MASK_METADATA,
        "masker": MASKER_SETTINGS,
        "resources": RESOURCE_ASSUMPTIONS,
        "transformed_signals": TRANSFORMED_SIGNALS,
        "contrasts": CONTRAST_EXPRESSIONS,
    }
    variance = configuration.pop("variance_partition")
    assert configuration == {
        "subject": "sub-s4",
        "task": "stopSignal",
        "sessions": ["ses-02", "ses-04"],
        **expected_shared,
    }
    assert variance["definition"] == "full_r2 - nuisance_r2"
    assert variance["clip_below_zero"] is True
    assert variance["nuisance_model"] == {
        "events": False,
        "confounds": list(CONFOUNDS),
        "drift_model": "cosine",
        "high_pass": 0.01,
        "drift_order": 1,
        "noise_model": "ar1",
    }
    assert variance["raw_min_delta_r2"] == pytest.approx(
        expected_delta["raw_min_delta_r2"]
    )
    assert variance["negative_voxel_count"] == expected_delta["negative_voxel_count"]
    assert variance["mean_delta_r2"] == pytest.approx(expected_delta["mean_delta_r2"])
    assert variance["max_delta_r2"] == pytest.approx(expected_delta["max_delta_r2"])
    assert set(variance) == {
        "definition",
        "clip_below_zero",
        "nuisance_model",
        "raw_min_delta_r2",
        "negative_voxel_count",
        "mean_delta_r2",
        "max_delta_r2",
    }

    provenance = json.loads((published / "logs/boldtailor_provenance.json").read_text())
    normalized = next(
        activity
        for activity in provenance["activities"]
        if activity["name"] == "normalize"
    )
    assert normalized["metadata"] == {
        "example": "two-session stop-signal whole-brain",
        "sessions": ["ses-02", "ses-04"],
        **expected_shared,
    }
    fitted = next(
        activity for activity in provenance["activities"] if activity["name"] == "fit"
    )
    assert fitted["model"]["contrasts"] == {
        name: {"kind": "expression", "value": expression}
        for name, expression in CONTRAST_EXPRESSIONS.items()
    }
    assert fitted["model"]["noise_model"] == "ar1"
    assert provenance["activities"][-1]["name"] == "task_delta_r2"

    images = tuple(sorted(published.glob("images/*.nii.gz")))
    assert len(images) == 11
    assert (
        published
        / "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz"
    ).is_file()

    shareable_text = "\n".join(
        path.read_text()
        for path in sorted(published.rglob("*"))
        if path.is_file() and path.suffix != ".gz" and ".boldtailor" not in path.parts
    )
    assert str(bids_root.resolve()) not in shareable_text


def _execute_notebook(
    bids_root,
    temporary_root,
    monkeypatch,
    working_directory,
    *,
    instrument_plots=False,
):
    monkeypatch.setenv("BOLDTAILOR_BIDS_ROOT", str(bids_root))
    monkeypatch.setenv("BOLDTAILOR_SESSIONS", "ses-02,ses-04")
    monkeypatch.setenv("BOLDTAILOR_TEMP_ROOT", str(temporary_root))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if instrument_plots:
        _instrument_notebook_plots(notebook)
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
    rendered = "\n".join(
        str(output.get("text", output.get("data", {}).get("text/plain", "")))
        for cell in executed.cells
        for output in cell.get("outputs", ())
    )
    published = next(temporary_root.glob("boldtailor-*"))
    return executed, rendered, published


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
    inputs, mask_image, masker, runs = _loaded_runs(stop_signal_bids_dataset)
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
    result = fit(data, model)
    return inputs, mask_image, masker, result, task_delta_r2(data, model, result)


def test_common_brain_mask_intersects_runs(stop_signal_bids_dataset):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04")
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
        _discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04")
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
    for run in runs:
        assert run.signals.dtype == np.float64
        assert run.signals.flags.owndata
        assert not run.signals.flags.writeable


def test_shared_masker_preserves_feature_order_across_runs(
    stop_signal_bids_dataset,
):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04")
    )
    spatial_pattern = np.arange(343, dtype=np.float32).reshape((7, 7, 7))
    for item in inputs:
        image = nib.load(item.bold)
        values = np.asarray(image.dataobj).copy()
        values[..., 0] = spatial_pattern
        nib.save(nib.Nifti1Image(values, image.affine, image.header), item.bold)

    masker = make_masker(common_brain_mask(inputs))
    runs = tuple(
        load_run(
            item,
            masker,
            trial_types=TRIAL_TYPES,
            confound_names=CONFOUNDS,
        )
        for item in inputs
    )

    np.testing.assert_array_equal(runs[0].signals[0], np.arange(343))
    np.testing.assert_array_equal(runs[1].signals[0], runs[0].signals[0])


def test_load_run_rejects_bold_mask_geometry_mismatch(stop_signal_bids_dataset):
    inputs = tuple(
        _discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04")
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

    assert estimate == pytest.approx(80 * 343 * 8 / 2**30 + 88 * 343 * 8 / 2**30)


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
    _, mask_image, masker, result, comparison = example_result
    options = {
        "subject": "sub-s4",
        "task": "stopSignal",
        "sessions": ["ses-02", "ses-04"],
        "space": "MNI152NLin2009cAsym",
        "resolution": 2,
    }
    result_artifacts = _demo_module().result_artifacts

    first = result_artifacts(
        result,
        masker,
        mask_image,
        subject="sub-s4",
        task="stopSignal",
        sessions=("ses-02", "ses-04"),
        space="MNI152NLin2009cAsym",
        resolution=2,
        configuration=options,
        task_delta=comparison,
    )
    second = result_artifacts(
        result,
        masker,
        mask_image,
        subject="sub-s4",
        task="stopSignal",
        sessions=("ses-02", "ses-04"),
        space="MNI152NLin2009cAsym",
        resolution=2,
        configuration=options,
        task_delta=comparison,
    )

    assert first == second
    expected_images = (
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-common_mask.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-successfulInhibition_stat-effect_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-successfulInhibition_stat-z_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-stopVsGo_stat-effect_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-stopVsGo_stat-z_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-goSuccessVsBaseline_stat-effect_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_contrast-goSuccessVsBaseline_stat-z_statmap.nii.gz",
        "images/sub-s4_ses-02_task-stopSignal_space-MNI152NLin2009cAsym_res-2_stat-r2_statmap.nii.gz",
        "images/sub-s4_ses-04_task-stopSignal_space-MNI152NLin2009cAsym_res-2_stat-r2_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-aggregate_stat-r2_statmap.nii.gz",
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz",
    )
    assert len(expected_images) == 11
    expected_paths = {
        "reports/sub-s4_ses-02_task-stopSignal_desc-design_matrix.tsv",
        "reports/sub-s4_ses-04_task-stopSignal_desc-design_matrix.tsv",
        "reports/sub-s4_task-stopSignal_desc-wholebrain_contrasts.tsv",
        "reports/sub-s4_task-stopSignal_desc-example_config.json",
        "reports/sub-s4_task-stopSignal_desc-image_manifest.tsv",
        *expected_images,
    }
    assert {item.path for item in first} == expected_paths
    assert not any("roi" in item.path for item in first)
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
            artifacts[
                "reports/sub-s4_task-stopSignal_desc-wholebrain_contrasts.tsv"
            ].payload
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

    expected_values = {
        expected_images[1]: result.effect("successful_inhibition"),
        expected_images[2]: result.z_score("successful_inhibition"),
        expected_images[3]: result.effect("stop_vs_go"),
        expected_images[4]: result.z_score("stop_vs_go"),
        expected_images[5]: result.effect("go_success_vs_baseline"),
        expected_images[6]: result.z_score("go_success_vs_baseline"),
        expected_images[7]: result.run_r2[0],
        expected_images[8]: result.run_r2[1],
        expected_images[9]: result.r2,
        expected_images[10]: comparison.delta_r2,
    }
    common = np.asarray(mask_image.dataobj, dtype=bool)
    assert np.all(comparison.delta_r2 >= 0.0)
    for path in expected_images:
        image = nib.Nifti1Image.from_bytes(gzip.decompress(artifacts[path].payload))
        assert image.shape == (7, 7, 7)
        np.testing.assert_allclose(image.affine, mask_image.affine)
        if path == expected_images[0]:
            np.testing.assert_array_equal(image.dataobj, mask_image.dataobj)
        else:
            np.testing.assert_allclose(
                np.asarray(image.dataobj)[common], expected_values[path]
            )

    manifest = pd.read_csv(
        io.BytesIO(
            artifacts["reports/sub-s4_task-stopSignal_desc-image_manifest.tsv"].payload
        ),
        sep="\t",
    )
    assert list(manifest.columns) == [
        "relative_path",
        "media_type",
        "byte_size",
        "sha256",
    ]
    assert tuple(manifest.relative_path) == expected_images
    assert set(manifest.media_type) == {"application/gzip"}
    np.testing.assert_array_equal(
        manifest.byte_size,
        [len(artifacts[path].payload) for path in expected_images],
    )
    assert list(manifest.sha256) == [
        hashlib.sha256(artifacts[path].payload).hexdigest() for path in expected_images
    ]

    published = publish_artifact_set(tmp_path / "published", first)
    assert len(published) == len(first)
    assert all(path.is_file() for path in published)


def test_result_artifacts_serializes_clipped_task_delta_values(example_result):
    _, mask_image, masker, result, comparison = example_result
    nuisance_r2 = comparison.nuisance_r2.copy()
    nuisance_r2[0] = comparison.full_r2[0] + 0.25
    negative_comparison = make_task_delta_r2_result(
        full_r2=comparison.full_r2,
        nuisance_r2=nuisance_r2,
        nuisance_designs=comparison.nuisance_design_matrices,
        provenance=comparison.provenance,
    )

    artifacts = _demo_module().result_artifacts(
        result,
        masker,
        mask_image,
        subject="sub-s4",
        task="stopSignal",
        sessions=("ses-02", "ses-04"),
        space="MNI152NLin2009cAsym",
        resolution=2,
        configuration={},
        task_delta=negative_comparison,
    )

    path = (
        "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_"
        "desc-taskDelta_stat-r2_statmap.nii.gz"
    )
    image = nib.Nifti1Image.from_bytes(
        gzip.decompress(
            {artifact.path: artifact for artifact in artifacts}[path].payload
        )
    )
    serialized = np.asarray(image.dataobj)[np.asarray(mask_image.dataobj, dtype=bool)]
    assert negative_comparison.raw_delta_r2[0] == pytest.approx(-0.25)
    assert serialized[0] == 0.0
    assert np.all(serialized >= 0.0)
    np.testing.assert_allclose(serialized, negative_comparison.delta_r2)


@pytest.mark.parametrize("alteration", ("shape", "voxel_support", "affine"))
def test_result_artifacts_rejects_common_mask_mismatch(example_result, alteration):
    _, mask_image, masker, result, comparison = example_result
    values = np.asarray(mask_image.dataobj).copy()
    affine = mask_image.affine.copy()
    if alteration == "shape":
        values = values[:-1]
    elif alteration == "voxel_support":
        values[0, 0, 0] = 0
    else:
        affine[0, 3] += 1.0
    changed_mask = nib.Nifti1Image(values, affine, mask_image.header)

    with pytest.raises(ValueError, match="common mask must match fitted masker"):
        _demo_module().result_artifacts(
            result,
            masker,
            changed_mask,
            subject="sub-s4",
            task="stopSignal",
            sessions=("ses-02", "ses-04"),
            space="MNI152NLin2009cAsym",
            resolution=2,
            configuration={},
            task_delta=comparison,
        )


def test_result_artifacts_rejects_task_delta_length_mismatch(example_result):
    _, mask_image, masker, result, comparison = example_result
    short_comparison = make_task_delta_r2_result(
        full_r2=np.zeros(comparison.delta_r2.size - 1),
        nuisance_r2=np.zeros(comparison.delta_r2.size - 1),
        nuisance_designs=comparison.nuisance_design_matrices,
        provenance=comparison.provenance,
    )

    with pytest.raises(
        ValueError, match="task delta r-squared values must match mask voxel count"
    ):
        _demo_module().result_artifacts(
            result,
            masker,
            mask_image,
            subject="sub-s4",
            task="stopSignal",
            sessions=("ses-02", "ses-04"),
            space="MNI152NLin2009cAsym",
            resolution=2,
            configuration={},
            task_delta=short_comparison,
        )


def test_result_artifacts_rejects_incomplete_spatial_context(example_result):
    _, _, masker, result, comparison = example_result

    with pytest.raises(ValueError, match="masker, common mask, space, and resolution"):
        _demo_module().result_artifacts(
            result,
            masker,
            None,
            subject="sub-s4",
            task="stopSignal",
            sessions=("ses-02", "ses-04"),
            space="MNI152NLin2009cAsym",
            resolution=2,
            configuration={},
            task_delta=comparison,
        )


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


def test_notebook_configuration_defaults_to_complete_real_sessions(monkeypatch):
    monkeypatch.delenv("BOLDTAILOR_SESSIONS", raising=False)

    configuration = _notebook_configuration()

    assert configuration["SESSIONS"] == ("ses-06", "ses-08")
    assert configuration["MASK_STRATEGY"] == "intersection"
    assert configuration["MINIMUM_MEMORY_GIB"] == 32
    assert not any("ROI" in name.upper() for name in configuration)


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


def test_notebook_design_fit_displays_compact_variance_summary():
    _assert_compact_variance_display_source()


@pytest.mark.parametrize(
    "working_directory",
    (NOTEBOOK.parents[1], NOTEBOOK.parent),
    ids=("repository-root", "notebook-directory"),
)
def test_notebook_executes_against_fixture(
    stop_signal_bids_dataset, tmp_path, monkeypatch, working_directory
):
    executed, rendered, published = _execute_notebook(
        stop_signal_bids_dataset,
        tmp_path,
        monkeypatch,
        working_directory,
        instrument_plots=True,
    )

    _assert_plot_contract(executed)
    _assert_display_contract(executed)
    assert "successful_inhibition" in rendered
    assert "stop_vs_go" in rendered
    assert "go_success_vs_baseline" in rendered
    assert "common_voxel_count" in rendered
    assert "estimated_signal_memory_gib" in rendered
    assert "clipped at zero" in rendered
    assert "descriptive variance accounting" in rendered
    assert (
        "Maps are descriptive, unthresholded, and do not imply "
        "multiple-comparison-corrected inference."
    ) in rendered
    assert "published_count" in rendered
    assert str(stop_signal_bids_dataset.resolve()) not in rendered
    assert len(tuple(published.glob("images/*.nii.gz"))) == 11
    assert (
        published
        / "images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz"
    ).is_file()
    assert (
        published / "reports/sub-s4_task-stopSignal_desc-image_manifest.tsv"
    ).is_file()
    assert not any("roi" in path.name.lower() for path in published.rglob("*"))


def test_notebook_publishes_complete_private_metadata(
    stop_signal_bids_dataset, tmp_path, monkeypatch
):
    executed, rendered, published = _execute_notebook(
        stop_signal_bids_dataset,
        tmp_path,
        monkeypatch,
        NOTEBOOK.parents[1],
        instrument_plots=True,
    )

    _assert_published_metadata(
        published,
        stop_signal_bids_dataset,
        _delta_audit(executed),
    )
    assert str(stop_signal_bids_dataset.resolve()) not in rendered


def test_readme_links_real_data_notebook():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    notebook_line = next(
        line
        for line in readme.splitlines()
        if "examples/stop_signal_demo.ipynb" in line
    )

    assert "whole-brain" in notebook_line.lower()


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

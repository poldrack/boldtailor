# Stop-Signal Real-Data Notebook Implementation Plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, executable two-session notebook that demonstrates Boldtailor modeling, diagnostics, logging, provenance projection, and transactional derivative publication with real stop-signal fMRIPrep data.

**Architecture:** Keep the substantive Boldtailor calls visible in `examples/stop_signal_demo.ipynb`. Put only example-specific discovery, bounded ROI extraction, source metadata, report serialization, and output-path safeguards in `examples/stop_signal_demo.py`; test that helper with a generated two-session BIDS/fMRIPrep fixture and execute the notebook against the fixture with `nbclient`.

**Tech Stack:** Python 3.12, uv, pytest, NumPy, pandas, nibabel, Nilearn, nbformat, nbclient, ipykernel, Jupyter notebooks.

## Global Constraints

- Use `uv` for package management and `uv run` for every local command.
- Follow strict RED-GREEN-Refactor: commit failing tests before the corresponding implementation and never weaken a test merely to pass.
- Keep functions short, modular, and single-purpose.
- Every `__init__.py` must remain completely empty; do not add an initializer under `examples/`.
- The installed package gains no public BIDS-loading API; all adapter code remains under `examples/`.
- Default dataset root: `/Users/poldrack/data_unsynced/rdoc_fmri`.
- Default real-data selection: `sub-s4`, `ses-06` and `ses-08`, task `stopSignal`,
  run `run-01`, MNI152NLin2009cAsym resolution 2.
- `BOLDTAILOR_SESSIONS` may override the default with exactly two non-empty,
  comma-separated session labels after trimming surrounding whitespace. The
  synthetic notebook smoke test sets it to `ses-02,ses-04` for its fixture.
- Default publication is temporary. Persistent publication is restricted to `<dataset>/derivatives/boldtailor` and refuses collisions unless overwrite is explicitly enabled.
- Notebook tests must not require outputs to be absent; an executed notebook with outputs may later be committed.
- Do not silently truncate BOLD, events, confounds, or mismatched runs.

## File Map

- Create `examples/stop_signal_demo.py`: example-only data and publication helpers.
- Create `examples/stop_signal_demo.ipynb`: the narrative real-data workflow.
- Create `tests/conftest.py`: reusable two-session BIDS/fMRIPrep fixture.
- Create `tests/test_stop_signal_demo.py`: helper behavior and notebook smoke tests.
- Modify `pyproject.toml`: add notebook execution dependencies to the dev group.
- Modify `uv.lock`: lock the new dev dependencies.
- Modify `README.md`: link the real-data example without importing it as package API.

---

### Task 1: Exact BIDS/fMRIPrep Input Discovery

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_stop_signal_demo.py`
- Create: `examples/stop_signal_demo.py`

**Interfaces:**
- Produces: `RunInputs(session: str, events: Path, bold: Path, mask: Path, confounds: Path)`.
- Produces: `discover_run_inputs(bids_root: Path, fmriprep_root: Path, *, subject: str, session: str, task: str, run: str, space: str, resolution: int) -> RunInputs`.
- Consumes: only filesystem paths and BIDS entity strings.

- [ ] **Step 1: Add the generated two-session fixture**

Create `tests/conftest.py`. The fixture must use functions rather than a fixture class and must write the smallest data needed by both helper and notebook tests:

```python
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


SESSIONS = ("ses-02", "ses-04")
AFFINE = np.array(
    [[2.0, 0.0, 0.0, 42.0],
     [0.0, 2.0, 0.0, 10.0],
     [0.0, 0.0, 2.0, 14.0],
     [0.0, 0.0, 0.0, 1.0]]
)


def _events(session):
    trial_types = [
        "go_success", "stop_success", "stop_failure", "go_failure",
        "go_success", "stop_success", "stop_failure", "go_success",
    ]
    if session == "ses-04":
        trial_types.remove("go_failure")
    return pd.DataFrame(
        {
            "onset": np.arange(5.0, 5.0 + 10.0 * len(trial_types), 10.0),
            "duration": np.ones(len(trial_types)),
            "trial_type": trial_types,
        }
    )


def _confounds(n_scans):
    values = np.linspace(-0.1, 0.1, n_scans)
    frame = pd.DataFrame(
        {
            "trans_x": values,
            "trans_y": values[::-1],
            "trans_z": values * 0.5,
            "rot_x": values * 0.1,
            "rot_y": values * 0.2,
            "rot_z": values * 0.3,
            "framewise_displacement": np.abs(values),
        }
    )
    frame.loc[0, "framewise_displacement"] = np.nan
    return frame


@pytest.fixture
def stop_signal_bids_dataset(tmp_path):
    root = tmp_path / "rdoc_fmri"
    derivative = root / "derivatives" / "fmri_25.2.0"
    rng = np.random.default_rng(20260808)
    for index, session in enumerate(SESSIONS):
        n_scans = 80 + index * 8
        raw_func = root / "sub-s4" / session / "func"
        derivative_func = derivative / "sub-s4" / session / "func"
        raw_func.mkdir(parents=True)
        derivative_func.mkdir(parents=True)
        stem = f"sub-s4_{session}_task-stopSignal_run-01"
        _events(session).to_csv(
            raw_func / f"{stem}_events.tsv", sep="\t", index=False
        )
        shape = (7, 7, 7, n_scans)
        signal = rng.normal(1000.0, 3.0, shape).astype(np.float32)
        image = nib.Nifti1Image(signal, AFFINE)
        image.header.set_zooms((2.0, 2.0, 2.0, 1.5))
        nib.save(
            image,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        )
        mask = nib.Nifti1Image(np.ones(shape[:3], dtype=np.uint8), AFFINE)
        nib.save(
            mask,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        )
        _confounds(n_scans).to_csv(
            derivative_func / f"{stem}_desc-confounds_timeseries.tsv",
            sep="\t",
            index=False,
        )
    (root / "dataset_description.json").write_text(
        '{"Name":"fixture","BIDSVersion":"1.11.1"}\n'
    )
    return root
```

- [ ] **Step 2: Write discovery tests before creating the helper**

Create `tests/test_stop_signal_demo.py` with a namespace-package import and focused function tests:

```python
import importlib
from pathlib import Path

import pytest


def _demo_module():
    return importlib.import_module("examples.stop_signal_demo")


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
    duplicate = inputs.events.with_name(inputs.events.name.replace("events", "copy_events"))
    duplicate.write_bytes(inputs.events.read_bytes())

    with pytest.raises(ValueError, match="events.*exactly one"):
        _discover(stop_signal_bids_dataset)
```

- [ ] **Step 3: Run the RED tests and confirm the missing-module failure**

Run:

```bash
uv run pytest tests/test_stop_signal_demo.py -q
```

Expected: the three tests are collected and fail from their test bodies with
`ModuleNotFoundError: No module named 'examples.stop_signal_demo'`.

- [ ] **Step 4: Commit the RED tests**

```bash
uv run git add tests/conftest.py tests/test_stop_signal_demo.py
uv run git commit -m "test: specify stop-signal input discovery"
```

- [ ] **Step 5: Implement exact discovery**

Create `examples/stop_signal_demo.py`. Keep the public helper small and use one private exact-match function:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunInputs:
    session: str
    events: Path
    bold: Path
    mask: Path
    confounds: Path


def discover_run_inputs(
    bids_root: Path,
    fmriprep_root: Path,
    *,
    subject: str,
    session: str,
    task: str,
    run: str,
    space: str,
    resolution: int,
) -> RunInputs:
    raw_func = Path(bids_root) / subject / session / "func"
    derivative_func = Path(fmriprep_root) / subject / session / "func"
    stem = f"{subject}_{session}_task-{task}_{run}"
    return RunInputs(
        session=session,
        events=_one(raw_func.glob(f"{stem}*_events.tsv"), "events"),
        bold=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-preproc_bold.nii.gz"
            ),
            "BOLD",
        ),
        mask=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-brain_mask.nii.gz"
            ),
            "mask",
        ),
        confounds=_one(
            derivative_func.glob(f"{stem}_desc-confounds_timeseries.tsv"),
            "confounds",
        ),
    )


def _one(paths, role: str) -> Path:
    matches = tuple(sorted(paths))
    if not matches:
        raise FileNotFoundError(f"{role} discovery expected exactly one file")
    if len(matches) != 1:
        raise ValueError(f"{role} discovery expected exactly one file")
    return matches[0]
```

- [ ] **Step 6: Run focused tests and formatting**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run black examples/stop_signal_demo.py tests/conftest.py tests/test_stop_signal_demo.py
uv run pytest tests/test_stop_signal_demo.py -q
```

Expected: 3 tests pass after formatting.

- [ ] **Step 7: Commit the GREEN implementation**

```bash
uv run git add examples/stop_signal_demo.py tests/conftest.py tests/test_stop_signal_demo.py
uv run git commit -m "feat: discover stop-signal example inputs"
```

---

### Task 2: Bounded ROI Loading, Validation, and Source Metadata

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: `RunInputs` from Task 1.
- Produces: `LoadedRun(inputs, signals, events, confounds, frame_times, voxel_indices, spatial_shape, affine, tr)`.
- Produces: `common_roi_voxels(inputs: Sequence[RunInputs], *, center_mni: tuple[float, float, float], radius_mm: float) -> np.ndarray`.
- Produces: `load_run(inputs: RunInputs, voxel_indices: np.ndarray, *, trial_types: Sequence[str], confound_names: Sequence[str]) -> LoadedRun`.
- Produces: `roi_image(values: np.ndarray, loaded: LoadedRun) -> nib.Nifti1Image`.
- Produces: `run_sources(inputs: RunInputs, bids_root: Path) -> RunSources`.

- [ ] **Step 1: Add failing ROI, confound, timing, and provenance tests**

Append tests that require the full behavior rather than only shape checks:

```python
import nibabel as nib
import numpy as np
import pandas as pd

from examples.stop_signal_demo import (
    common_roi_voxels,
    load_run,
    roi_image,
    run_sources,
)


TRIAL_TYPES = (
    "go_success", "go_failure", "stop_success", "stop_failure"
)
CONFOUNDS = (
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "framewise_displacement",
)


def _loaded_runs(root):
    inputs = tuple(_discover(root, session) for session in ("ses-02", "ses-04"))
    voxels = common_roi_voxels(
        inputs, center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )
    return inputs, voxels, tuple(
        load_run(
            item,
            voxels,
            trial_types=TRIAL_TYPES,
            confound_names=CONFOUNDS,
        )
        for item in inputs
    )


def test_load_run_extracts_common_bounded_roi(stop_signal_bids_dataset):
    _, voxels, runs = _loaded_runs(stop_signal_bids_dataset)

    assert voxels.ndim == 2 and voxels.shape[1] == 3
    assert len(voxels) > 1
    assert runs[0].signals.shape == (80, len(voxels))
    assert runs[1].signals.shape == (88, len(voxels))
    np.testing.assert_allclose(np.diff(runs[0].frame_times), 1.5)
    assert set(runs[0].events.trial_type) <= set(TRIAL_TYPES)
    assert tuple(runs[0].confounds) == CONFOUNDS
    assert runs[0].confounds.iloc[0].framewise_displacement == 0.0
    assert np.isfinite(runs[0].signals).all()


def test_roi_image_restores_values_to_spatial_coordinates(stop_signal_bids_dataset):
    _, voxels, runs = _loaded_runs(stop_signal_bids_dataset)
    values = np.arange(len(voxels), dtype=float)

    image = roi_image(values, runs[0])

    restored = image.get_fdata()[tuple(voxels.T)]
    np.testing.assert_array_equal(restored, values)
    assert image.shape == runs[0].spatial_shape


def test_load_run_rejects_confound_length_mismatch(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t").iloc[:-1]
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="confounds.*80 rows"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )


def test_load_run_rejects_unexpected_nonfinite_confound(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t")
    frame.loc[3, "trans_x"] = np.nan
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="non-finite.*trans_x"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )


def test_run_sources_records_dataset_relative_inputs(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)

    sources = run_sources(inputs, stop_signal_bids_dataset)

    assert sources.signal.uri.startswith("derivatives/fmri_25.2.0/")
    assert sources.signal.byte_size == inputs.bold.stat().st_size
    assert sources.signal.annotations["mask"]["uri"].endswith("brain_mask.nii.gz")
    assert sources.events.uri.startswith("sub-s4/ses-02/")
    assert sources.confounds.uri.endswith("desc-confounds_timeseries.tsv")
    assert sources.signal.modified_at.endswith("Z")
```

Add these explicit boundary tests:

```python
def test_common_roi_rejects_nonoverlap(stop_signal_bids_dataset):
    inputs = (_discover(stop_signal_bids_dataset),)

    with pytest.raises(ValueError, match="ROI does not overlap"):
        common_roi_voxels(
            inputs, center_mni=(500.0, 500.0, 500.0), radius_mm=1.0
        )


def test_load_run_rejects_event_beyond_acquisition(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.events, sep="\t")
    frame.loc[0, ["onset", "duration"]] = [119.5, 1.0]
    frame.to_csv(inputs.events, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="event timing exceeds acquisition"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )
```

- [ ] **Step 2: Run RED and commit tests before helper changes**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify bounded stop-signal loading"
```

Expected: failures report missing `common_roi_voxels`, `load_run`, `roi_image`, and
`run_sources`.

- [ ] **Step 3: Implement the immutable loaded-run record and common ROI**

Add imports for `datetime`, `timezone`, `mimetypes`, `Sequence`, nibabel, NumPy,
pandas, and Boldtailor provenance types. Implement:

```python
@dataclass(frozen=True, slots=True)
class LoadedRun:
    inputs: RunInputs
    signals: np.ndarray
    events: pd.DataFrame
    confounds: pd.DataFrame
    frame_times: np.ndarray
    voxel_indices: np.ndarray
    spatial_shape: tuple[int, int, int]
    affine: np.ndarray
    tr: float


def common_roi_voxels(inputs, *, center_mni, radius_mm):
    if not inputs:
        raise ValueError("ROI requires at least one run")
    masks = [nib.load(item.mask) for item in inputs]
    shape = masks[0].shape
    affine = masks[0].affine
    if any(mask.shape != shape or not np.allclose(mask.affine, affine) for mask in masks):
        raise ValueError("run masks must share shape and affine")
    indices = np.indices(shape).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, indices)
    in_sphere = np.linalg.norm(world - np.asarray(center_mni), axis=1) <= radius_mm
    in_all_masks = np.logical_and.reduce(
        [np.asarray(mask.dataobj).reshape(-1) > 0 for mask in masks]
    )
    voxels = indices[in_sphere & in_all_masks]
    if not len(voxels):
        raise ValueError("ROI does not overlap every run mask")
    return _immutable_array(voxels, dtype=int)
```

Implement `_immutable_array` as an owned NumPy copy with `write=False`.

- [ ] **Step 4: Implement bounded signal extraction and tabular validation**

Use a bounding-box slice, not `get_fdata()` on the 4D BOLD image:

```python
def _extract_signals(image, voxels):
    lower = voxels.min(axis=0)
    upper = voxels.max(axis=0) + 1
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    block = np.asarray(image.dataobj[slices + (slice(None),)], dtype=float)
    local = voxels - lower
    signals = block[tuple(local.T) + (slice(None),)].T
    if not np.isfinite(signals).all():
        raise ValueError("ROI signals must be finite")
    return _immutable_array(signals, dtype=float)
```

Implement `load_run` with short private functions `_model_events`,
`_selected_confounds`, and `_frame_times`. Required behavior:

- Validate `image.shape[:3]` contains every voxel and obtain TR from
  `image.header.get_zooms()[3]`.
- Filter events to the requested `trial_type` values and retain exactly `onset`,
  `duration`, and `trial_type`.
- Reject an empty modeled-event table and any event with
  `onset + duration > n_scans * tr`.
- Require all requested confound columns and exactly `n_scans` rows.
- Replace only a first-row missing `framewise_displacement` with `0.0`; reject every
  other non-finite selected value and identify its column in the error.
- Set frame times to `np.arange(n_scans, dtype=float) * tr`.
- Own immutable copies of arrays and owned DataFrame copies.

Implement `roi_image` by validating a one-dimensional finite values array of exactly
`len(loaded.voxel_indices)`, filling a zero-valued 3D array, and returning
`nib.Nifti1Image(data, loaded.affine)`.

- [ ] **Step 5: Implement complete source metadata**

Use this shape so the mask participates in the provenance fingerprint even though
`RunSources` has only signal, events, and confounds roles:

```python
def run_sources(inputs: RunInputs, bids_root: Path) -> RunSources:
    mask = _source_metadata(inputs.mask, bids_root)
    signal = _source_ref(
        inputs.bold,
        bids_root,
        role="signal",
        annotations={"mask": mask},
    )
    return RunSources(
        signal=signal,
        events=_source_ref(inputs.events, bids_root, role="events"),
        confounds=_source_ref(inputs.confounds, bids_root, role="confounds"),
    )
```

`_source_metadata` returns dataset-relative POSIX `uri`, `media_type`, `byte_size`,
and a UTC `modified_at` ending in `Z`. Use
`datetime.fromtimestamp(stat.st_mtime, timezone.utc)`. Map `.nii.gz` to
`application/gzip` and TSV to `text/tab-separated-values`; do not rely on a
platform-specific MIME database for those extensions.

- [ ] **Step 6: Run focused tests, refactor, and commit GREEN**

```bash
uv run black examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run pytest tests/test_stop_signal_demo.py -q
uv run git diff --check
uv run git add examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run git commit -m "feat: load bounded stop-signal ROI data"
```

Expected: all helper tests pass and every added function remains focused on one
validation or transformation.

---

### Task 3: Deterministic Reports and Safe Destination Selection

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: `AnalysisResult`, `LoadedRun`, and `RunInputs`.
- Produces: `result_artifacts(result: AnalysisResult, *, subject: str, task: str, sessions: Sequence[str], configuration: Mapping[str, object]) -> tuple[Artifact, ...]`.
- Produces: `publication_destination(bids_root: Path, *, persistent: bool, requested: Path | None = None, temporary_parent: Path | None = None) -> Path`.
- Produces: `protected_source_paths(inputs: Sequence[RunInputs]) -> tuple[Path, ...]`.

- [ ] **Step 1: Add RED tests for deterministic bytes and path safety**

Create a small fitted two-run result inside a pytest fixture using the fixture data,
`from_arrays`, and `ModelSpec(noise_model="ols")`. Use only contrasts whose
regressors occur in both runs:

```python
from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec
from boldtailor.publication import Artifact, publish_artifact_set


@pytest.fixture
def example_result(stop_signal_bids_dataset):
    inputs, _, runs = _loaded_runs(stop_signal_bids_dataset)
    data = from_arrays(
        [run.signals for run in runs],
        [run.events for run in runs],
        frame_times=[run.frame_times for run in runs],
        confounds=[run.confounds for run in runs],
        sources=[run_sources(item, stop_signal_bids_dataset) for item in inputs],
        provenance_metadata={"example": "stop-signal"},
    )
    model = ModelSpec(
        contrasts={
            "successful_inhibition": "stop_success - stop_failure",
            "stop_vs_go": "stop_success - go_success",
        },
        confounds=CONFOUNDS,
        noise_model="ols",
    )
    return inputs, fit(data, model)
```

Then add:

```python
def test_result_artifacts_are_deterministic_valid_metadata(example_result, tmp_path):
    _, result = example_result
    options = {
        "subject": "sub-s4",
        "task": "stopSignal",
        "sessions": ["ses-02", "ses-04"],
        "roi": {"center_mni": [48.0, 16.0, 20.0], "radius_mm": 6.0},
    }

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
    published = publish_artifact_set(tmp_path / "published", first)
    assert len(published) == len(first)
    assert all(path.is_file() for path in published)


def test_publication_destination_defaults_to_temp(stop_signal_bids_dataset, tmp_path):
    destination = publication_destination(
        stop_signal_bids_dataset,
        persistent=False,
        temporary_parent=tmp_path,
    )

    assert destination.parent == tmp_path
    assert destination.name.startswith("boldtailor-")


def test_persistent_destination_is_restricted(stop_signal_bids_dataset, tmp_path):
    expected = stop_signal_bids_dataset / "derivatives" / "boldtailor"
    assert publication_destination(
        stop_signal_bids_dataset, persistent=True
    ) == expected.resolve()

    with pytest.raises(ValueError, match="derivatives/boldtailor"):
        publication_destination(
            stop_signal_bids_dataset,
            persistent=True,
            requested=tmp_path / "outside",
        )


def test_protected_source_paths_include_all_inputs(stop_signal_bids_dataset):
    inputs = tuple(_discover(stop_signal_bids_dataset, s) for s in ("ses-02", "ses-04"))

    protected = protected_source_paths(inputs)

    assert set(protected) == {
        path
        for item in inputs
        for path in (item.events, item.bold, item.mask, item.confounds)
    }


def test_publication_refuses_overlap_with_protected_source(
    stop_signal_bids_dataset,
):
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
```

- [ ] **Step 2: Run RED and commit the new tests**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify stop-signal report publication"
```

Expected: failures report missing report and destination helpers.

- [ ] **Step 3: Implement deterministic report artifacts**

Add imports for `json`, `Mapping`, `tempfile`, `AnalysisResult`, and `Artifact`.
Implement one private serializer per format:

```python
def _tsv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        sep="\t", index=False, float_format="%.10g", lineterminator="\n"
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
```

For each contrast, summarize `n_features`, `mean_effect`, `mean_z`, `max_abs_z`, and
`min_one_sided_p`. Preserve `result.contrast_names` order. For each design matrix,
prepend frame times with
`design.rename_axis("frame_time").reset_index()`. Serialize the provided
configuration only; it must not contain an absolute dataset root.

- [ ] **Step 4: Implement destination and source-path safeguards**

```python
def publication_destination(
    bids_root,
    *,
    persistent,
    requested=None,
    temporary_parent=None,
):
    root = Path(bids_root).resolve()
    if not persistent:
        parent = None if temporary_parent is None else Path(temporary_parent)
        return Path(tempfile.mkdtemp(prefix="boldtailor-", dir=parent)).resolve()
    expected = (root / "derivatives" / "boldtailor").resolve()
    destination = expected if requested is None else Path(requested).resolve()
    if destination != expected:
        raise ValueError("persistent output must be dataset derivatives/boldtailor")
    return destination


def protected_source_paths(inputs):
    return tuple(
        path
        for item in inputs
        for path in (item.events, item.bold, item.mask, item.confounds)
    )
```

- [ ] **Step 5: Verify metadata preflight and commit GREEN**

Run the RED-authored publication assertion with the full focused test file:

```bash
uv run black examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run pytest tests/test_stop_signal_demo.py -q
uv run git diff --check
uv run git add examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run git commit -m "feat: serialize stop-signal example reports"
```

Expected: all focused tests pass, including publication metadata validation.

---

### Task 4: Executable Narrative Notebook

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_stop_signal_demo.py`
- Create: `examples/stop_signal_demo.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes all helper interfaces from Tasks 1-3 and the existing public module APIs
  `boldtailor.data.from_arrays`, `boldtailor.model.ModelSpec`, `boldtailor.fit.fit`,
  `boldtailor.bids_provenance.project_bids_provenance`, `boldtailor.publication.Artifact`,
  and `boldtailor.publication.publish_artifact_set`.
- Produces an executable notebook; no installed Python API.

- [ ] **Step 1: Add notebook execution dependencies with uv**

```bash
uv add --dev nbclient ipykernel
```

Confirm `pyproject.toml` lists both packages in the `dev` dependency group and
`uv.lock` is updated. This is test infrastructure, not notebook implementation.

- [ ] **Step 2: Add RED notebook contract and smoke tests**

Append imports for `nbformat`, `NotebookClient`, and `CellExecutionError`. Add:

```python
NOTEBOOK = Path(__file__).parents[1] / "examples" / "stop_signal_demo.ipynb"


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


def test_notebook_executes_against_fixture(
    stop_signal_bids_dataset, tmp_path, monkeypatch
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
        resources={"metadata": {"path": str(NOTEBOOK.parents[1])}},
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
```

Do not assert that outputs or execution counts are empty. Add this README test:

```python
def test_readme_links_real_data_notebook():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "examples/stop_signal_demo.ipynb" in readme
```

- [ ] **Step 3: Run RED and commit tests plus test dependencies**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run git add pyproject.toml uv.lock tests/test_stop_signal_demo.py
uv run git commit -m "test: specify executable stop-signal notebook"
```

Expected: helper tests pass; notebook tests fail because the notebook does not exist
and README does not link it.

- [ ] **Step 4: Create the notebook with concise narrative cells**

Create `examples/stop_signal_demo.ipynb` as nbformat v4 using `apply_patch`. Include
the following sections and substantive code. Markdown must explain what is being
shown and must explicitly label the right inferior frontal ROI as illustrative.

1. **Configuration** — first code cell:

```python
import os
from pathlib import Path

BIDS_ROOT = Path(
    os.environ.get("BOLDTAILOR_BIDS_ROOT", "/Users/poldrack/data_unsynced/rdoc_fmri")
)
FMRIPREP_ROOT = BIDS_ROOT / "derivatives" / "fmri_25.2.0"
SUBJECT = "sub-s4"
DEFAULT_SESSIONS = ("ses-06", "ses-08")


def _configured_sessions():
    selection = os.environ.get("BOLDTAILOR_SESSIONS")
    if selection is None:
        return DEFAULT_SESSIONS
    sessions = tuple(value.strip() for value in selection.split(","))
    if len(sessions) != 2 or not all(sessions):
        raise ValueError(
            "BOLDTAILOR_SESSIONS must select exactly two non-empty sessions"
        )
    return sessions


SESSIONS = _configured_sessions()
TASK = "stopSignal"
RUN = "run-01"
SPACE = "MNI152NLin2009cAsym"
RESOLUTION = 2
ROI_CENTER_MNI = (48.0, 16.0, 20.0)
ROI_RADIUS_MM = 6.0
PERSIST_DERIVATIVES = False
OVERWRITE_EXISTING = False
TEMPORARY_PARENT = os.environ.get("BOLDTAILOR_TEMP_ROOT")
```

2. **Imports and discovery** — import helpers and module-qualified Boldtailor APIs;
discover both sessions and display only session labels and basenames, not absolute
paths.

3. **Bounded ROI loading** — compute common voxels, load both runs, plot condition
counts, and report only shapes, TRs, and scan counts.

4. **Array ingestion**:

```python
analysis_data = boldtailor.data.from_arrays(
    [run.signals for run in loaded_runs],
    [run.events for run in loaded_runs],
    frame_times=[run.frame_times for run in loaded_runs],
    confounds=[run.confounds for run in loaded_runs],
    sources=[run_sources(item, BIDS_ROOT) for item in inputs],
    provenance_metadata={
        "example": "two-session stop-signal ROI",
        "sessions": list(SESSIONS),
        "roi": {"center_mni": list(ROI_CENTER_MNI), "radius_mm": ROI_RADIUS_MM},
    },
)
```

Because every package initializer is empty, import modules explicitly:

```python
import boldtailor.bids_provenance as bids_provenance
import boldtailor.data as data
import boldtailor.fit as fit
import boldtailor.model as model
import boldtailor.publication as publication
```

Use `data.from_arrays`, `model.ModelSpec`, and so on; never add initializer exports.

5. **Run-specific designs and fit** — construct:

```python
model_spec = model.ModelSpec(
    contrasts={
        "successful_inhibition": "stop_success - stop_failure",
        "stop_vs_go": "stop_success - go_success",
    },
    confounds=CONFOUNDS,
    noise_model="ar1",
)
result = fit.fit(analysis_data, model_spec)
```

Show each run's design-column tuple without requiring the tuples to differ. Plot
both design matrices with Nilearn.

6. **Results and diagnostics** — show `contrast_names`; compact summaries of effect,
variance, stat, z-score, one-sided p-value, `run_r2`, and `r2`; design provenance;
warnings; execution ID; and analysis fingerprint. Reconstruct the
`successful_inhibition` z-score with `roi_image` and plot it with Nilearn.

7. **Structured logging and provenance** — display selected event fields from
`result.provenance.events` and explain execution identity versus deterministic
analysis fingerprinting. Do not dump the full raw provenance record.

8. **BIDS provenance projection**:

```python
projected = bids_provenance.project_bids_provenance(
    result.provenance,
    dataset_name="Boldtailor stop-signal example derivatives",
    label="boldtailor",
)
```

Display projected relative paths and byte sizes only.

9. **Transactional publication** — merge projected bytes and report artifacts,
resolve the destination, and publish:

```python
artifacts = tuple(publication.Artifact(path, payload) for path, payload in projected.items())
artifacts += result_artifacts(
    result,
    subject=SUBJECT,
    task=TASK,
    sessions=SESSIONS,
    configuration=shareable_configuration,
)
destination = publication_destination(
    BIDS_ROOT,
    persistent=PERSIST_DERIVATIVES,
    temporary_parent=TEMPORARY_PARENT,
)
published = publication.publish_artifact_set(
    destination,
    artifacts,
    source_paths=protected_source_paths(inputs),
    overwrite=OVERWRITE_EXISTING,
)
```

Display destination basename, artifact-relative paths, and counts. Explain that
setting `PERSIST_DERIVATIVES=True` writes only to
`<dataset>/derivatives/boldtailor`.

- [ ] **Step 5: Link the notebook from README**

Add a short `Examples` section linking
`[Two-session stop-signal real-data notebook](../../../examples/stop_signal_demo.ipynb)`.
State that its default path is local and can be overridden with
`BOLDTAILOR_BIDS_ROOT`.

- [ ] **Step 6: Run GREEN notebook and focused tests**

```bash
uv run black examples/stop_signal_demo.py tests/conftest.py tests/test_stop_signal_demo.py
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest tests/test_stop_signal_demo.py -q -W error
uv run git diff --check
```

Expected: all helper, contract, README, and notebook execution tests pass both ways.
The notebook file may contain zero or nonzero outputs; neither state affects tests.

- [ ] **Step 7: Commit notebook implementation**

```bash
uv run git add examples/stop_signal_demo.ipynb examples/stop_signal_demo.py README.md tests/conftest.py tests/test_stop_signal_demo.py
uv run git commit -m "docs: add stop-signal real-data notebook"
```

Do not include temporary published derivatives or an executed real-data copy.

---

### Task 5: Real-Data and Repository Verification

**Files:**
- Verify only; modify code only through a new RED-GREEN cycle if verification exposes
  a defect.

**Interfaces:**
- Consumes the completed notebook and the repaired dataset at
  `/Users/poldrack/data_unsynced/rdoc_fmri`.
- Produces verification evidence, not a new API.

- [ ] **Step 1: Execute the notebook against the real data without saving outputs**

Run from the repository root so `examples.stop_signal_demo` is importable:

```bash
BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri \
MPLBACKEND=Agg \
uv run python -c 'from pathlib import Path; import nbformat; from nbclient import NotebookClient; path=Path("examples/stop_signal_demo.ipynb"); notebook=nbformat.read(path, as_version=4); NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata":{"path":str(Path.cwd())}}).execute(); print("real notebook execution passed")'
```

Expected: exit 0 and `real notebook execution passed`. Confirm the notebook reports
temporary publication and does not create
`/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor`.

- [ ] **Step 2: Run focused and complete test suites**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest -q
uv run pytest -q -W error
```

Expected: all tests pass with no warnings promoted to failures.

- [ ] **Step 3: Run formatting, dependency, build, and repository checks**

```bash
uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv build
uv run rm -rf dist src/boldtailor.egg-info
uv run pytest tests/test_repository_contracts.py -q
uv run python -c 'from pathlib import Path; files=list(Path("src").rglob("__init__.py")); bad=[str(path) for path in files if path.read_bytes()]; print(f"checked {len(files)} initializers"); raise SystemExit(bool(bad))'
uv run git diff --check
uv run git status --short
```

Expected: every command exits 0, all initializers are empty, and status contains no
generated build, notebook-execution, or derivative artifacts.

- [ ] **Step 4: Request final code review**

Invoke `superpowers:requesting-code-review`. The review must check spec coverage,
strict tests-before-implementation history, bounded-memory extraction, path safety,
privacy-safe notebook display, and both synthetic and real execution evidence.

- [ ] **Step 5: Commit only review-driven fixes through RED-GREEN cycles**

If review identifies a defect, first commit a focused failing pytest case, then the
minimal fix, rerun Steps 1-3, and request follow-up review. If review finds no defect,
do not create an empty commit.

# Whole-Brain Stop-Signal Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stop-signal notebook's single-ROI analysis with an in-memory, two-session whole-brain analysis restricted to the intersection of the fMRIPrep brain masks.

**Architecture:** Build one explicitly non-preprocessing `NiftiMasker` from the intersection mask, transform both BOLD runs into a shared feature axis, and pass those arrays through the existing Boldtailor API. Reconstruct whole-brain contrast and R-squared images with the fitted masker, serialize deterministic NIfTI derivatives, and retain the existing safe publication and provenance flow.

**Tech Stack:** Python 3.12+, uv, pytest, NumPy, pandas, nibabel, Nilearn 0.14.x, nbformat/nbclient, Matplotlib.

## Global Constraints

- Every local command uses `uv` or `uv run`.
- Every `__init__.py` remains completely empty.
- Use pytest functions and fixtures; do not add test classes.
- Enforce RED-GREEN-Refactor literally: commit behavioral tests before implementation, run the RED test and record its expected failure, then implement without weakening the test.
- The notebook contains no ROI mode, sphere center, radius, ROI bounding-box extraction, or ROI-specific artifact name.
- Use the intersection of the `ses-06` and `ses-08` fMRIPrep brain masks as the real-data feature space.
- Use one fitted `NiftiMasker` with `standardize=False`, `detrend=False`, `smoothing_fwhm=None`, `low_pass=None`, `high_pass=None`, and `reports=False`.
- Do not chunk, downsample, truncate, smooth, filter, detrend, or standardize the signals; assume at least 32 GiB of memory.
- Retain run-specific design matrices and AR(1) fitting.
- Fit exactly `successful_inhibition`, `stop_vs_go`, and `go_success_vs_baseline` with the expressions specified below.
- Display unthresholded z maps descriptively; do not imply multiple-comparison-corrected inference.
- Temporary publication remains the default. Persistent publication remains restricted to `<dataset>/derivatives/boldtailor` with existing path, overlap, and symlink protections.
- Notebook outputs may be committed; tests must not require the notebook to be output-free.
- Never modify `/Users/poldrack/data_unsynced/rdoc_fmri` or create `/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor` during default verification.

---

## File Structure

- Modify `examples/stop_signal_demo.py`: replace ROI extraction/reconstruction with common-mask `NiftiMasker` helpers and add deterministic whole-brain image artifacts.
- Modify `examples/stop_signal_demo.ipynb`: replace the ROI narrative and execution cells with whole-brain masking, fitting, maps, provenance, and publication.
- Modify `tests/test_stop_signal_demo.py`: specify whole-brain helper, artifact, provenance, and notebook behavior using the existing synthetic BIDS fixture.
- Modify `README.md`: describe the example as a whole-brain analysis.
- Modify `docs/superpowers/specs/2026-08-09-whole-brain-notebook-design.md` only if implementation reveals a genuine specification error; do not silently change requirements.

---

### Task 1: Common-Mask Extraction and Reconstruction

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: existing `RunInputs`, discovery, event/confound validation, and source metadata.
- Produces:
  - `common_brain_mask(inputs: Sequence[RunInputs]) -> nib.Nifti1Image`
  - `make_masker(mask_image: nib.Nifti1Image) -> NiftiMasker`
  - `load_run(inputs: RunInputs, masker: NiftiMasker, *, trial_types: Sequence[str], confound_names: Sequence[str]) -> LoadedRun`
  - `whole_brain_image(values: np.ndarray, masker: NiftiMasker) -> nib.Nifti1Image`
  - `estimate_signal_memory_gib(scan_counts: Sequence[int], voxel_count: int) -> float`

- [ ] **Step 1: Replace ROI fixture setup with failing whole-brain behavior tests**

Change the focused-test imports and `_loaded_runs` helper to the new interface:

```python
from examples.stop_signal_demo import (
    common_brain_mask,
    estimate_signal_memory_gib,
    load_run,
    make_masker,
    run_sources,
    whole_brain_image,
)


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
```

Replace the ROI extraction and reconstruction tests with these behavioral cases:

```python
def test_common_brain_mask_intersects_runs(stop_signal_bids_dataset):
    inputs = tuple(_discover(stop_signal_bids_dataset, session) for session in ("ses-02", "ses-04"))
    second = nib.load(inputs[1].mask)
    values = np.asarray(second.dataobj).copy()
    values[0, 0, 0] = 0
    nib.save(nib.Nifti1Image(values, second.affine, second.header), inputs[1].mask)

    mask_image = common_brain_mask(inputs)

    assert mask_image.shape == (7, 7, 7)
    assert int(np.asarray(mask_image.dataobj).sum()) == 342
    np.testing.assert_allclose(mask_image.affine, nib.load(inputs[0].mask).affine)


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
    assert np.isfinite(runs[0].signals).all()


def test_whole_brain_image_round_trips_mask_values(stop_signal_bids_dataset):
    _, mask_image, masker, _ = _loaded_runs(stop_signal_bids_dataset)
    values = np.arange(343, dtype=float)

    image = whole_brain_image(values, masker)
    restored = masker.transform(image)[0]

    np.testing.assert_array_equal(restored, values)
    assert image.shape == mask_image.shape
    np.testing.assert_allclose(image.affine, mask_image.affine)


def test_signal_memory_estimate_uses_float64_storage():
    estimate = estimate_signal_memory_gib((80, 88), 343)
    assert estimate == pytest.approx(80 * 343 * 8 / 2**30 + 88 * 343 * 8 / 2**30)
```

Add tests that save a changed affine into the second mask and into one BOLD image and assert messages containing `masks must share shape and affine` and `BOLD image must match common mask geometry`. Retain the existing confound, event, source, and protected-path tests, updating only the `load_run` setup to use the fitted masker. Delete tests whose only requirement is ROI non-overlap or bounding-box behavior because the approved requirement removed those behaviors.

Update `example_result` to fit all 343 features and exactly:

```python
contrasts={
    "successful_inhibition": "stop_success - stop_failure",
    "stop_vs_go": "(stop_success + stop_failure) - go_success",
    "go_success_vs_baseline": "go_success",
}
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest tests/test_stop_signal_demo.py -k "common_brain or masker or whole_brain or signal_memory" -q
```

Expected: collection or test failures because `common_brain_mask`, `make_masker`, `whole_brain_image`, and `estimate_signal_memory_gib` do not exist and `load_run` still requires ROI indices.

- [ ] **Step 3: Commit the RED tests before any helper implementation**

```bash
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify whole-brain example loading"
```

- [ ] **Step 4: Implement the common mask and explicitly configured masker**

In `examples/stop_signal_demo.py`, import `NiftiMasker` and replace the ROI helpers with short functions following this structure:

```python
from nilearn.maskers import NiftiMasker


def common_brain_mask(inputs: Sequence[RunInputs]) -> nib.Nifti1Image:
    if not inputs:
        raise ValueError("common brain mask requires at least one run")
    masks = tuple(nib.load(item.mask) for item in inputs)
    _require_same_geometry(masks, "run masks must share shape and affine")
    common = np.logical_and.reduce(
        [np.asarray(mask.dataobj, dtype=bool) for mask in masks]
    )
    if not common.any():
        raise ValueError("run mask intersection must contain at least one voxel")
    header = masks[0].header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(common.astype(np.uint8), masks[0].affine, header)


def make_masker(mask_image: nib.Nifti1Image) -> NiftiMasker:
    return NiftiMasker(
        mask_img=mask_image,
        standardize=False,
        detrend=False,
        smoothing_fwhm=None,
        low_pass=None,
        high_pass=None,
        reports=False,
    ).fit()
```

Add `_require_same_geometry(images, message)` and `_require_bold_geometry(image, masker)`. The latter compares `image.shape[:3]` and `image.affine` to `masker.mask_img_` before transforming. Add the masker-backed `load_run` path, require a finite two-dimensional `n_scans x n_voxels` result, and return an owned immutable array. Retain the existing ROI dispatch and `LoadedRun` spatial fields only as a temporary sequencing bridge so the unchanged notebook remains executable; Task 3 removes them in the same RED-GREEN cycle that migrates the notebook.

Implement reconstruction and memory estimation:

```python
def whole_brain_image(values: np.ndarray, masker: NiftiMasker) -> nib.Nifti1Image:
    restored = np.asarray(values, dtype=float)
    voxel_count = int(np.asarray(masker.mask_img_.dataobj, dtype=bool).sum())
    if restored.ndim != 1 or len(restored) != voxel_count:
        raise ValueError("whole-brain values must be one-dimensional and match mask voxel count")
    if not np.isfinite(restored).all():
        raise ValueError("whole-brain values must be finite")
    return masker.inverse_transform(restored)


def estimate_signal_memory_gib(
    scan_counts: Sequence[int], voxel_count: int
) -> float:
    if not scan_counts or any(count <= 0 for count in scan_counts) or voxel_count <= 0:
        raise ValueError("memory estimate requires positive scan and voxel counts")
    return float(sum(scan_counts) * voxel_count * 8 / 2**30)
```

Do not add compatibility aliases or expand the legacy ROI behavior. Its temporary bridge is deleted in Task 3 because the final workflow intentionally removes ROI analysis.

- [ ] **Step 5: Run focused and full tests to verify GREEN**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest -q -W error
```

Expected: focused tests and the full warning-strict suite pass with no warnings promoted to failures.

- [ ] **Step 6: Commit the helper implementation**

```bash
uv run git add examples/stop_signal_demo.py
uv run git commit -m "feat: load common-mask whole-brain signals"
```

---

### Task 2: Whole-Brain Image Derivatives

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: Task 1's fitted masker, common mask, `LoadedRun` values, and three-contrast `AnalysisResult`.
- Produces:
  - `result_artifacts(result: AnalysisResult, masker: NiftiMasker, common_mask: nib.Nifti1Image, *, subject: str, task: str, sessions: Sequence[str], space: str, resolution: int, configuration: Mapping[str, object]) -> tuple[Artifact, ...]`
  - deterministic gzip-compressed NIfTI artifacts and `desc-image_manifest.tsv`.

- [ ] **Step 1: Write failing derivative tests**

Change the artifact fixture to return `inputs, mask_image, masker, result`. Call `result_artifacts` with the Task 2 signature and assert the exact image paths:

```python
expected_images = {
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
}
```

Assert the report is named
`reports/sub-s4_task-stopSignal_desc-wholebrain_contrasts.tsv`, no artifact path
contains `roi`, and the manifest is
`reports/sub-s4_task-stopSignal_desc-image_manifest.tsv`.

For every expected image, decompress and load the payload independently:

```python
import gzip

image = nib.Nifti1Image.from_bytes(gzip.decompress(artifacts[path].payload))
assert image.shape == (7, 7, 7)
np.testing.assert_allclose(image.affine, mask_image.affine)
```

Read the image manifest and assert its literal columns are
`relative_path`, `media_type`, `byte_size`, and `sha256`; its paths equal
`expected_images`; every media type is `application/gzip`; byte sizes equal the
actual payload sizes; and SHA-256 values equal independently computed digests.
Call `result_artifacts` twice and retain the existing equality assertion to
prove gzip output is deterministic.

- [ ] **Step 2: Run artifact tests and verify RED**

```bash
uv run pytest tests/test_stop_signal_demo.py -k "result_artifacts" -q
```

Expected: FAIL because the old signature emits only ROI TSV/config reports and no NIfTI images or manifest.

- [ ] **Step 3: Commit the RED artifact tests**

```bash
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify whole-brain image derivatives"
```

- [ ] **Step 4: Implement deterministic NIfTI images and manifest**

Add `gzip` and `hashlib` imports. Keep helpers short:

```python
def _nifti_bytes(image: nib.Nifti1Image) -> bytes:
    return gzip.compress(image.to_bytes(), compresslevel=9, mtime=0)


def _image_artifact(path: str, image: nib.Nifti1Image) -> Artifact:
    return Artifact(path, _nifti_bytes(image))


def _image_manifest(
    images: Sequence[Artifact], *, subject: str, task: str
) -> Artifact:
    frame = pd.DataFrame(
        [
            {
                "relative_path": image.path,
                "media_type": "application/gzip",
                "byte_size": len(image.payload),
                "sha256": hashlib.sha256(image.payload).hexdigest(),
            }
            for image in images
        ]
    )
    return Artifact(
        f"reports/{subject}_task-{task}_desc-image_manifest.tsv",
        _tsv_bytes(frame),
    )
```

Use the exact contrast-label mapping from the spec. Construct mask, six contrast,
two run-R², and aggregate-R² artifacts with `whole_brain_image`. Build the image
manifest only from those ten image artifacts, then return design TSVs, the
whole-brain contrast TSV, shareable JSON configuration, image artifacts, and
manifest in stable order.

Until Task 3 migrates the notebook call site, retain the old artifact behavior
only when all four new spatial-context arguments are omitted. Reject every
partial combination. Task 3 deletes this narrow sequencing bridge.

- [ ] **Step 5: Verify focused and warning-strict GREEN**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest -q -W error
```

- [ ] **Step 6: Commit the derivative implementation**

```bash
uv run git add examples/stop_signal_demo.py
uv run git commit -m "feat: serialize whole-brain result maps"
```

---

### Task 3: Executable Whole-Brain Notebook

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-2 helper functions and the existing Boldtailor modules.
- Produces: a complete whole-brain notebook that runs from repository-root and notebook-directory kernels, plus accurate README documentation.

- [ ] **Step 1: Write failing notebook behavior tests**

Extend the parameterized synthetic notebook execution test. After execution,
collect displayed text and inspect the temporary publication directory:

```python
rendered = "\n".join(
    str(output.get("text", output.get("data", {}).get("text/plain", "")))
    for cell in executed.cells
    for output in cell.get("outputs", ())
)
published = next(tmp_path.glob("boldtailor-*"))

assert "go_success_vs_baseline" in rendered
assert "common_voxel_count" in rendered
assert "estimated_signal_memory_gib" in rendered
assert "descriptive, unthresholded" in rendered
assert len(tuple(published.glob("images/*.nii.gz"))) == 10
assert (published / "reports/sub-s4_task-stopSignal_desc-image_manifest.tsv").is_file()
assert not any("roi" in path.name.lower() for path in published.rglob("*"))
```

Change the configuration test to assert `MASK_STRATEGY == "intersection"`,
`MINIMUM_MEMORY_GIB == 32`, and the absence of ROI configuration keys. Update
the README test to require the phrase `whole-brain` near the notebook link.
Replace the old source-phrase contract with behavior assertions from the
executed notebook; do not add tests that merely grep implementation source.
Also assert that loaded whole-brain signals are owned, write-protected float64
arrays and that applying the shared masker to the same spatial pattern in both
runs yields identical feature ordering.

- [ ] **Step 2: Run notebook tests and verify RED**

```bash
uv run pytest tests/test_stop_signal_demo.py -k "notebook or readme" -q
```

Expected: FAIL because the notebook still executes the ROI path, lacks the third contrast and whole-brain outputs, and publishes no images.

- [ ] **Step 3: Commit the RED notebook tests**

```bash
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify whole-brain notebook story"
```

- [ ] **Step 4: Replace notebook configuration and loading cells**

Keep the existing import fallback that supports both launch directories. Replace
ROI constants with:

```python
MASK_STRATEGY = "intersection"
MINIMUM_MEMORY_GIB = 32
PERSIST_DERIVATIVES = False
OVERWRITE_EXISTING = False
TEMPORARY_PARENT = os.environ.get("BOLDTAILOR_TEMP_ROOT")
```

Import `common_brain_mask`, `make_masker`, `estimate_signal_memory_gib`, and
`whole_brain_image`. Build one common mask and one fitted masker, then load both
runs through that masker. Display a compact table/dictionary containing
`mask_shape`, `common_voxel_count`, run scan counts, signal shapes, and
`estimated_signal_memory_gib`. State in Markdown that the analysis assumes at
least 32 GiB and does not chunk features.

After the notebook has migrated to the masker-backed interface, delete the
temporary ROI dispatch, `common_roi_voxels`, `roi_image`, ROI-only extraction
helpers, and the legacy `LoadedRun` spatial fields from
`examples/stop_signal_demo.py`. Also make the whole-brain `result_artifacts`
spatial context mandatory and delete its all-omitted legacy branch. No
compatibility aliases or legacy publication paths remain in the final workflow.

- [ ] **Step 5: Replace model, result, and plotting cells**

Use exactly:

```python
model_spec = model.ModelSpec(
    contrasts={
        "successful_inhibition": "stop_success - stop_failure",
        "stop_vs_go": "(stop_success + stop_failure) - go_success",
        "go_success_vs_baseline": "go_success",
    },
    confounds=CONFOUNDS,
    noise_model="ar1",
)
result = fit.fit(analysis_data, model_spec)
```

Retain the compact summaries for every result accessor. Plot
`whole_brain_image(result.z_score(name), masker)` once per contrast with
`threshold=None` and a colorbar. Plot `whole_brain_image(result.r2, masker)` as
the aggregate fit-quality map with a sequential colormap and non-symmetric
colorbar. In the plotting code cell, display the literal text
`Maps are descriptive, unthresholded, and do not imply multiple-comparison-corrected inference.`
so the executed-notebook behavior test can verify the disclosure.

- [ ] **Step 6: Update shareable configuration, provenance, and publication cells**

The shareable configuration and `provenance_metadata` must include literal,
JSON-safe values for:

```python
mask_metadata = {
    "strategy": "intersection",
    "shape": list(common_mask.shape),
    "affine": common_mask.affine.tolist(),
    "voxel_count": common_voxel_count,
}
masker_settings = {
    "standardize": False,
    "detrend": False,
    "smoothing_fwhm": None,
    "low_pass": None,
    "high_pass": None,
    "reports": False,
}
resource_assumptions = {
    "minimum_memory_gib": 32,
    "feature_chunking": False,
    "estimated_signal_memory_gib": estimated_signal_memory_gib,
}
```

Include transformed signal shapes and dtypes, but never `BIDS_ROOT`. Call the
Task 2 `result_artifacts` signature with masker, common mask, space, and
resolution. Preserve projected BIDS provenance, protected source paths,
temporary-default destination, optional exact persistent destination,
transactional publication, and privacy-safe relative-path displays.

- [ ] **Step 7: Update README and remove all ROI narrative**

Describe the linked notebook as a two-session common-mask whole-brain example.
Remove ROI wording from notebook Markdown, output labels, filenames, and code.
Do not require empty notebook outputs and do not clear outputs merely for test
convenience.

- [ ] **Step 8: Run notebook, focused, and full tests**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest -q
uv run pytest -q -W error
uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
```

Expected: two working-directory notebook executions pass, all image artifacts
round-trip, the full suite is warning-clean, Black reports no changes, and the
lock is unchanged.

- [ ] **Step 9: Commit notebook and README implementation**

```bash
uv run git add examples/stop_signal_demo.ipynb README.md
uv run git commit -m "docs: demonstrate whole-brain stop-signal analysis"
```

---

### Task 4: Real-Data and Repository Verification

**Files:**
- Verify only. If verification exposes a defect, start a new RED test commit before modifying implementation.

**Interfaces:**
- Consumes: the completed notebook and read-only dataset at `/Users/poldrack/data_unsynced/rdoc_fmri`.
- Produces: verification evidence and a reviewed, clean feature branch.

- [ ] **Step 1: Guard the persistent destination before execution**

```bash
uv run python -c 'from pathlib import Path; target=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor"); print(f"publication_exists_before={target.exists()}"); raise SystemExit(target.exists())'
```

Expected: `publication_exists_before=False` and exit 0.

- [ ] **Step 2: Execute the real whole-brain notebook from its own directory**

Run with a long timeout because the approved design fits about 289,425 voxels
in memory without chunking:

```bash
BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri \
MPLBACKEND=Agg \
uv run python -c 'from pathlib import Path; import nbformat; from nbclient import NotebookClient; path=Path("examples/stop_signal_demo.ipynb"); notebook=nbformat.read(path, as_version=4); executed=NotebookClient(notebook, timeout=7200, kernel_name="python3", resources={"metadata":{"path":str(path.parent.resolve())}}).execute(); rendered="\n".join(str(output.get("text", output.get("data", {}).get("text/plain", ""))) for cell in executed.cells for output in cell.get("outputs", [])); required=("ses-06", "ses-08", "successful_inhibition", "stop_vs_go", "go_success_vs_baseline", "common_voxel_count", "estimated_signal_memory_gib", "published_count"); missing=[value for value in required if value not in rendered]; print({"missing":missing}); raise SystemExit(bool(missing))'
```

Expected: exit 0 and `{'missing': []}`.

- [ ] **Step 3: Prove no persistent derivative was created**

Repeat Step 1 with the label `publication_exists_after`; expected `False` and exit 0.

- [ ] **Step 4: Run all repository gates**

```bash
uv run pytest tests/test_stop_signal_demo.py -q
uv run pytest -q
uv run pytest -q -W error
uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv build
uv run rm -rf dist src/boldtailor.egg-info
uv run pytest tests/test_repository_contracts.py -q
uv run python -c 'from pathlib import Path; files=list(Path("src").rglob("__init__.py")); bad=[str(path) for path in files if path.read_bytes()]; print(f"checked {len(files)} initializers"); raise SystemExit(bool(bad))'
uv run git diff --check
uv run git status --short
```

Expected: all commands exit 0, all initializers are byte-empty, and only
disposable `__pycache__` directories may appear after tests. Remove only cache
directories generated by this verification, then require an empty status.

- [ ] **Step 5: Request final whole-branch review**

Use `superpowers:requesting-code-review` over the branch merge base through
HEAD. Require review of strict test-first history, mask intersection and
geometry safety, explicit no-preprocessing masker settings, 32 GiB/no-chunking
disclosure, all three contrasts, deterministic NIfTI maps and manifest,
privacy-safe provenance, publication path protections, and synthetic plus real
execution evidence.

- [ ] **Step 6: Route any review fix through a new RED-GREEN cycle**

For each real Critical or Important finding, first add and commit a failing
focused pytest case, then implement the minimal fix in a separate commit, rerun
Steps 1-4, and request scoped re-review. Do not create an empty commit if the
review is clean.

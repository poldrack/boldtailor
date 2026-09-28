# Task-Attributable Delta R-Squared Implementation Plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **Historical plan:** The statistical definition in this completed plan was
> superseded on 2026-08-11 by the amendment in
> `docs/superpowers/specs/2026-08-09-task-delta-r2-design.md`. New work must use
> nested OLS for the full and nuisance diagnostic fits.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an aggregate whole-brain map and deterministic derivative for the nonnegative increase in R-squared from nuisance-only to complete task modeling.

**Architecture:** Compile nuisance-only run designs through a focused design API, fit them through shared GLM/R-squared internals, and compare their pooled aggregate R-squared with a fingerprint-matched complete result. Return an immutable provenance-bearing result that the notebook displays and publishes as an eleventh deterministic NIfTI.

**Tech Stack:** Python 3.12+, uv, NumPy, pandas, Nilearn 0.14.x, nibabel, nbformat/nbclient, pytest, Black.

## Global Constraints

- Use `uv` for package management and `uv run` for every local command.
- Every `__init__.py` remains byte-empty.
- Use committed RED-GREEN-Refactor; tests fail and are committed before production changes.
- Keep functions short and responsibilities narrow.
- Define `raw_delta_r2 = full_r2 - nuisance_r2` and publish `delta_r2 = maximum(raw_delta_r2, 0)`.
- Record raw minimum and clipped-voxel count even though the map is clipped.
- Publish one aggregate delta image only; no per-session or nuisance-only image.
- Preserve AR(1), the complete model and its three contrasts, common-mask geometry, and explicit no-preprocessing masker settings.
- Preserve temporary-default and exact persistent-path, symlink, overwrite, source-protection, and transaction safety.
- Assume at least 32 GiB RAM; add no chunking, resampling, truncation, smoothing, detrending, filtering, or standardization.
- The main checkout has user-owned edits to `examples/stop_signal_demo.ipynb`, `pyproject.toml`, and `uv.lock`. Execute in a new isolated worktree and never alter those files there. Preserve a recoverable copy of the executed notebook before later integration.

---

### Task 1: Nuisance-Only Design Compilation

**Files:**
- Modify: `tests/test_design.py`
- Modify: `src/boldtailor/design.py`

**Interfaces:**
- Consumes: `AnalysisData`, `ModelSpec`, existing confound selection and Nilearn design options.
- Produces: `compile_nuisance_designs(data: AnalysisData, model: ModelSpec) -> tuple[CompiledDesign, ...]`.

- [ ] **Step 1: Write failing nuisance-design tests**

Import `compile_nuisance_designs` and add:

```python
def test_compile_nuisance_designs_excludes_events_and_retains_nuisance(inputs):
    events, _ = inputs
    confounds = pd.DataFrame(
        {
            "trans_x": np.linspace(-1.0, 1.0, 80),
            "unused": np.linspace(0.5, -0.5, 80),
        }
    )
    data = from_arrays(
        np.zeros((80, 2)), events, tr=2.0, confounds=confounds
    )
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("trans_x",),
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ar1",
    )

    compiled = compile_nuisance_designs(data, model)[0]

    assert "face" not in compiled.matrix
    assert "house" not in compiled.matrix
    assert "trans_x" in compiled.matrix
    assert "unused" not in compiled.matrix
    assert "constant" in compiled.matrix
    assert any(column.startswith("drift_") for column in compiled.matrix)
    assert compiled.excluded_event_count == 0
    assert compiled.min_onset_cutoff == -24.0
```

Add a parity test with `drift_model=None`:

```python
expected = make_first_level_design_matrix(
    data.frame_times[0],
    events=None,
    hrf_model=None,
    drift_model=None,
    high_pass=model.high_pass,
    drift_order=model.drift_order,
    add_regs=confounds[["trans_x"]],
    min_onset=model.min_onset,
    oversampling=model.oversampling,
)
assert_frame_equal(compiled.matrix, expected)
```

Parameterize the existing missing/nonfinite selected-confound tests over
`compile_designs` and `compile_nuisance_designs`.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/test_design.py -k nuisance -q
```

Expected: collection fails because `compile_nuisance_designs` is absent.

- [ ] **Step 3: Commit RED tests**

```bash
uv run git add tests/test_design.py
uv run git commit -m "test: specify nuisance-only design compilation"
```

- [ ] **Step 4: Implement the compiler**

```python
def compile_nuisance_designs(
    data: AnalysisData, model: ModelSpec
) -> tuple[CompiledDesign, ...]:
    return tuple(
        _compile_nuisance_run(
            data.frame_times[run], data.confounds[run], model, run
        )
        for run in range(data.n_runs)
    )


def _compile_nuisance_run(
    frame_times: np.ndarray,
    confounds: pd.DataFrame,
    model: ModelSpec,
    run: int,
) -> CompiledDesign:
    selected = _select_confounds(confounds, model.confounds, run)
    matrix = _make_nuisance_matrix(frame_times, selected, model, run)
    _validate_design_matrix(matrix, run)
    return CompiledDesign(
        matrix=matrix,
        excluded_event_count=0,
        min_onset_cutoff=float(frame_times[0] + model.min_onset),
    )
```

Refactor duplicate-column/finite checks into `_validate_design_matrix`.
`_make_nuisance_matrix` calls `make_first_level_design_matrix` with
`events=None`, `hrf_model=None`, selected confounds, and the model's exact
drift/high-pass/order/min-onset/oversampling options. Contextualize errors as
`run {run} nuisance design compilation failed: ...`.

- [ ] **Step 5: Verify GREEN and commit production**

```bash
uv run pytest tests/test_design.py -q -W error
uv run pytest -q -W error
uv run black --check src tests
uv run git add src/boldtailor/design.py
uv run git commit -m "feat: compile nuisance-only designs"
```

---

### Task 2: Delta-R-Squared Result, Fit, Logging, and Provenance

**Files:**
- Modify: `tests/test_fit.py`
- Modify: `tests/test_logging.py`
- Modify: `src/boldtailor/results.py`
- Modify: `src/boldtailor/fit.py`

**Interfaces:**
- Consumes: Task 1 compiler, fingerprinted `AnalysisData`, matching `ModelSpec`, and complete `AnalysisResult`.
- Produces `TaskDeltaR2Result` and `task_delta_r2(data, model, full_result) -> TaskDeltaR2Result`.

- [ ] **Step 1: Write failing fit/result tests**

Create a sourced AR(1) fixture with at least 80 scans, alternating task events,
one varying confound, and signals generated from the complete Nilearn design
plus seeded noise. Fit the complete model and assert:

```python
comparison = task_delta_r2(data, model, full_result)

np.testing.assert_allclose(
    comparison.raw_delta_r2,
    comparison.full_r2 - comparison.nuisance_r2,
)
np.testing.assert_allclose(
    comparison.delta_r2,
    np.maximum(comparison.raw_delta_r2, 0.0),
)
assert np.all(comparison.full_r2 > comparison.nuisance_r2)
assert comparison.negative_voxel_count == int(
    np.count_nonzero(comparison.raw_delta_r2 < 0.0)
)
assert comparison.raw_min == pytest.approx(comparison.raw_delta_r2.min())
```

For all four arrays, assert one-dimensional equal shapes, float64, owned
storage, and rejection of `setflags(write=True)`. Mutate a returned nuisance
design and prove a later accessor is unchanged.

Test the result factory directly with full/nuisance values producing raw
`[-0.25, 0.0, 0.3]`; require clipped `[0.0, 0.0, 0.3]`, count `1`, and minimum
`-0.25`. Add errors for a changed data/model full result and for anonymous
provenance, matching `full result does not match data and model` and
`requires fingerprintable`.

Assert the final provenance activity:

```python
activity = comparison.provenance.activities[-1]
assert activity["name"] == "task_delta_r2"
assert activity["parent_analysis_id"] == full_result.provenance.analysis_fingerprint
assert activity["definition"] == "full_r2 - nuisance_r2"
assert activity["clip_below_zero"] is True
assert activity["nuisance_model"]["events"] is False
assert activity["nuisance_model"]["noise_model"] == "ar1"
assert activity["diagnostics"]["negative_voxel_count"] == comparison.negative_voxel_count
```

- [ ] **Step 2: Write failing structured-log tests**

Follow `_structured_records` in `tests/test_logging.py`. Success emits
`task_delta_r2_started` then `task_delta_r2_completed` with the comparison ID.
A mismatch emits `task_delta_r2_started` then `task_delta_r2_failed`, followed
by a context-free sentinel event.

- [ ] **Step 3: Verify and commit RED**

```bash
uv run pytest tests/test_fit.py tests/test_logging.py -k delta_r2 -q
uv run git add tests/test_fit.py tests/test_logging.py
uv run git commit -m "test: specify task delta r-squared comparison"
```

Expected: collection fails because the result/function do not exist.

- [ ] **Step 4: Implement the immutable result**

Add frozen `TaskDeltaR2Result` with private arrays, diagnostics, nuisance
designs, and provenance. Its factory validates equal, finite, nonempty 1-D
inputs and uses `immutable_float_array`:

```python
raw = immutable_float_array(full - nuisance)
clipped = immutable_float_array(np.maximum(raw, 0.0))
```

Store exact negative count/raw minimum and deep-copy nuisance designs on input
and access.

- [ ] **Step 5: Refactor shared GLM/R-squared internals**

Extract rank/DOF validation, `run_glm`, prediction, and sums of squares from
`_fit_run` into a short helper returning labels, regression results, residual
sum, and total sum. Keep complete contrasts and results unchanged. Add `_R2Fit`
for nuisance residual/total sums.

- [ ] **Step 6: Implement matching and comparison**

Compute the expected complete analysis ID through the same `_model_provenance`
and `_analysis_id` path as `fit`. Require a non-`None` expected ID and equality
with `full_result.provenance.analysis_fingerprint`. Validate run/feature
dimensions and finite complete R-squared.

Compile nuisance designs, fit each run with `model.noise_model`, pool residual
and total sums, and calculate nuisance R-squared using `_r2_from_sums`. Reject
nonfinite results and construct `TaskDeltaR2Result`.

- [ ] **Step 7: Implement logging and provenance**

Derive a deterministic comparison ID from the parent analysis ID and the
comparison definition/settings. Emit started/completed/failed events under a
fresh execution context. Extend provenance with:

```python
{
    "name": "task_delta_r2",
    "stage": "fit",
    "parent_analysis_id": parent_id,
    "definition": "full_r2 - nuisance_r2",
    "clip_below_zero": True,
    "nuisance_model": {
        "events": False,
        "confounds": list(model.confounds),
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "noise_model": model.noise_model,
    },
    "runs": nuisance_run_diagnostics,
    "diagnostics": {
        "raw_min": raw_min,
        "negative_voxel_count": negative_count,
        "mean_delta_r2": float(delta.mean()),
        "max_delta_r2": float(delta.max()),
    },
}
```

Never serialize signal/design values or absolute paths.

- [ ] **Step 8: Verify GREEN and commit production**

```bash
uv run pytest tests/test_fit.py tests/test_logging.py -k delta_r2 -q -W error
uv run pytest -q -W error
uv run black --check src tests
uv run git diff --check
uv run git add src/boldtailor/results.py src/boldtailor/fit.py
uv run git commit -m "feat: compare task and nuisance model r-squared"
```

---

### Task 3: Deterministic Delta-R-Squared Image

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: `TaskDeltaR2Result`, fitted common masker, and artifact helpers.
- Produces: eleventh `desc-taskDelta_stat-r2_statmap.nii.gz` artifact.

- [ ] **Step 1: Write failing derivative tests**

Make `example_result` return a real comparison. Require keyword
`task_delta=comparison` in `result_artifacts`. Add exact path:

```python
"images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz"
```

Require exactly 11 images. Decompress/load the new payload, assert mask geometry,
and compare masked values with `comparison.delta_r2`; all must be nonnegative.
Verify deterministic bytes and exact manifest path, size, media type, and hash.

Pass a mask with changed voxel support or affine to the original masker and
require `common mask must match fitted masker`.

- [ ] **Step 2: Verify and commit RED**

```bash
uv run pytest tests/test_stop_signal_demo.py -k result_artifacts -q
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify delta r-squared derivative"
```

Expected: signature/image-count failures.

- [ ] **Step 3: Implement validation and image**

Require `task_delta: TaskDeltaR2Result` as a keyword in `result_artifacts`.
Validate common mask and `masker.mask_img_` shape, affine, and boolean support;
validate delta length. Append after aggregate complete R-squared:

```python
_image_artifact(
    f"{stem}_desc-taskDelta_stat-r2_statmap.nii.gz",
    whole_brain_image(task_delta.delta_r2, masker),
)
```

Let the existing manifest include it without special casing.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run pytest tests/test_stop_signal_demo.py -k result_artifacts -q -W error
uv run pytest -q -W error
uv run black --check src tests examples/stop_signal_demo.py
uv run git add examples/stop_signal_demo.py
uv run git commit -m "feat: serialize task delta r-squared map"
```

---

### Task 4: Notebook Display and Shareable Provenance

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.ipynb`

**Interfaces:**
- Consumes: core comparison and Task 3 artifact keyword.
- Produces: fifth map, summary, comparison provenance projection, and 11-image publication.

- [ ] **Step 1: Write failing notebook tests**

Extend runtime plot audit to require exactly five calls: three z maps, complete
aggregate R-squared, and delta R-squared. The delta call requires
`threshold=None`, `vmin=0`, `cmap="magma"`, `symmetric_cbar=False`, colorbar,
and title containing `delta` and `clipped at zero`. Its values match the
comparison and are nonnegative.

Require rendered fields:

```text
mean_delta_r2
max_delta_r2
raw_min_delta_r2
negative_voxel_count
clipped at zero
descriptive variance accounting
```

Inspect published configuration:

```python
variance = configuration["variance_partition"]
assert variance["definition"] == "full_r2 - nuisance_r2"
assert variance["clip_below_zero"] is True
assert variance["nuisance_model"]["events"] is False
assert variance["nuisance_model"]["noise_model"] == "ar1"
```

Require final projected activity `task_delta_r2`, no absolute BIDS root, exactly
11 image files, and the delta path.

- [ ] **Step 2: Verify and commit RED**

```bash
uv run pytest tests/test_stop_signal_demo.py -k notebook -q
uv run git add tests/test_stop_signal_demo.py
uv run git commit -m "test: specify task delta r-squared notebook story"
```

Expected: both cwd executions fail with four plots/10 images/no comparison.

- [ ] **Step 3: Fit comparison and create metadata**

After the complete fit:

```python
task_delta = fit.task_delta_r2(analysis_data, model_spec, result)
variance_partition = {
    "definition": "full_r2 - nuisance_r2",
    "clip_below_zero": True,
    "nuisance_model": {
        "events": False,
        "confounds": list(CONFOUNDS),
        "drift_model": model_spec.drift_model,
        "high_pass": model_spec.high_pass,
        "drift_order": model_spec.drift_order,
        "noise_model": model_spec.noise_model,
    },
    "raw_min_delta_r2": task_delta.raw_min,
    "negative_voxel_count": task_delta.negative_voxel_count,
    "mean_delta_r2": float(task_delta.delta_r2.mean()),
    "max_delta_r2": float(task_delta.delta_r2.max()),
}
shareable_configuration["variance_partition"] = variance_partition
```

Display nuisance columns and the compact summary without paths/raw signals.

- [ ] **Step 4: Plot and disclose**

```python
plotting.plot_stat_map(
    whole_brain_image(task_delta.delta_r2, masker),
    threshold=None,
    vmin=0,
    colorbar=True,
    cmap="magma",
    symmetric_cbar=False,
    title="Task-attributable delta R-squared (clipped at zero)",
)
plt.show()
```

Display: `Delta R-squared is descriptive variance accounting, not inferential evidence.`

- [ ] **Step 5: Project and publish comparison**

Project `task_delta.provenance` rather than `result.provenance`. Pass
`task_delta=task_delta` to `result_artifacts`; keep destination/protection logic
unchanged.

- [ ] **Step 6: Verify GREEN and commit**

```bash
uv run pytest tests/test_stop_signal_demo.py -k notebook -q -W error
uv run pytest tests/test_stop_signal_demo.py -q -W error
uv run pytest -q -W error
uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv run git diff --check
uv run git add examples/stop_signal_demo.ipynb
uv run git commit -m "docs: display task delta r-squared"
```

Do not require or clear notebook outputs.

---

### Task 5: Real-Data and Whole-Branch Verification

**Files:**
- Verify only; defects require a new committed RED test before production edits.

**Interfaces:**
- Consumes: completed branch and read-only `/Users/poldrack/data_unsynced/rdoc_fmri`.
- Produces: reviewed integration-ready branch and verification evidence.

- [ ] **Step 1: Guard persistent output before execution**

```bash
uv run python -c 'from pathlib import Path; target=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor"); print(f"publication_exists_before={target.exists()}"); raise SystemExit(target.exists())'
```

Expected: `False`, exit 0.

- [ ] **Step 2: Run real notebook**

From `examples/`, use the real BIDS root, `MPLBACKEND=Agg`, task-specific
Jupyter directories in `/private/tmp`, and `NotebookClient(timeout=7200)`.
Require rendered sessions, three contrast names, all four delta summary fields,
`clipped at zero`, and `published_count`. Before the temporary directory is
released, require exactly 11 images including the delta path.

- [ ] **Step 3: Guard persistent output after execution**

Expected: `publication_exists_after=False`, exit 0.

- [ ] **Step 4: Run repository gates**

```bash
uv run pytest tests/test_stop_signal_demo.py -q -W error
uv run pytest -q -W error
uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv build --out-dir /private/tmp/boldtailor-task-delta-r2-dist
uv run pytest tests/test_repository_contracts.py -q -W error
uv run python -c 'from pathlib import Path; files=list(Path("src").rglob("__init__.py")); bad=[str(path) for path in files if path.read_bytes()]; print(f"checked {len(files)} initializers; nonempty={bad}"); raise SystemExit(bool(bad))'
uv run git diff --check
uv run git status --short
```

Record whether `src/boldtailor.egg-info` exists before build; remove only a
newly generated exact directory. Do not delete user caches/main changes.

- [ ] **Step 5: Request whole-branch review**

Use `superpowers:requesting-code-review` from merge base through HEAD. Audit
TDD history, nuisance semantics, pooled AR(1) R-squared, fingerprint matching,
clipping, immutable results, logs/provenance privacy, fifth plot, exact 11-image
determinism, path safety, both cwd executions, and real-data evidence.

- [ ] **Step 6: Route findings through RED-GREEN**

Each real Critical/Important finding first receives a committed failing focused
test, then a separate minimal production fix, full re-verification, and scoped
re-review. Do not create an empty fix commit for a clean review.

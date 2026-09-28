# Single-trial CIFTI maps and RT diagnostics implementation plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** Use `superpowers:executing-plans` for direct implementation, or `superpowers:subagent-driven-development` if the user explicitly selects that execution method. Complete each task's RED–commit–GREEN–refactor cycle before continuing.

**Goal:** Fit independent trial amplitudes with OLS and optional fixed ridge, publish NSD CIFTI maps, and assess RT associations without repetition-based tuning.

**Architecture:** Reuse `AnalysisData` and add a small pure numerical single-trial entry point. Keep trial-design compilation, solving, and immutable results separate. Extend the NSD example for imaging and behavioral diagnostics; preserve the current conventional example.

**Tech stack:** Python >=3.12, uv, NumPy, pandas, Nilearn, NiBabel, matplotlib, pytest. Existing dependencies suffice.

**Spec:** `docs/superpowers/specs/2026-09-26-single-trial-rt-design.md`.

**Status:** Plan only. No estimator, tests, or analysis outputs have been implemented for this feature.

## Global constraints

- Use `uv run` for local commands and uv for dependencies.
- Write and commit failing pytest tests before implementation.
- Every `__init__.py` stays completely empty.
- Prefer short functions and test functions with fixtures.
- Fitting performs no filesystem I/O; publication uses `publish_artifact_set`.
- Canonical SPM HRF, oversampling 50, explicit frame times, native signal units.
- RT and image repetition identity never influence fitting or tuning.
- OLS is the reference; optional normalized ridge alpha is fixed by the caller.
- NSD comparison alpha is predeclared as 0.1; it is not selected using RT.
- No automatic HRF/denoising/ridge selection or inferential RT p-values.

## Review focus

1. Trial identities survive unsorted event rows and feature blocks: Tasks 1 and 5.
2. Nuisance coefficients remain unpenalized and beta units are restored: Task 2.
3. Invalid RT affects diagnostics only; constant features produce NaN: Tasks 2–3.
4. Run means cannot manufacture RT correlations, and even runs cannot affect vertex selection: Task 3.
5. New output names preserve prior results; failed publication exposes no partial successful set: Tasks 4–5.

## Task 1: Compile a real single-trial design

**Files:** Create `src/boldtailor/_single_trial_design.py` and `tests/test_single_trial_design.py`.

**Interfaces:** Internal `compile_trial_run(events, frame_times, confounds, run_label)` returns a trial matrix, nuisance matrix, and trial table. Trial columns are named by trial ID. Nuisance columns are supplied confounds plus `constant`. A later wrapper adds global run/trial indices.

- [ ] Write tests proving: distinct columns for repeated image IDs; original event row order retained; sub-TR onsets and variable positive durations preserved; RT changes leave matrices identical; zero-duration events have an explicit supported impulse convention; invalid timing, duplicate/reserved names, and unsupported event columns with no sampled response fail clearly.

Use a direct HRF convolution as the oracle, not the production compiler:

```python
def test_subtr_timing_and_repeat_identity():
    events = pd.DataFrame({
        "onset": [12.0, 8.0], "duration": [3.0, 3.0],
        "73k_id": [99, 99], "response_time": [0.5, 2.0],
    })
    times = 0.775 + 1.6 * np.arange(40)
    x, n, trials = compile_trial_run(events, times, pd.DataFrame(index=range(40)), "run-01")
    assert trials.event_index.tolist() == [0, 1]
    assert trials.trial_id.tolist() == ["run-01_trial-0001", "run-01_trial-0002"]
    assert list(n) == ["constant"]
    expected, _ = compute_regressor(np.array([[12.0], [3.0], [1.0]]), "spm", times)
    np.testing.assert_allclose(x.iloc[:, 0], expected[:, 0])
    changed = events.assign(response_time=[10.0, np.nan])
    x2, _, _ = compile_trial_run(changed, times, pd.DataFrame(index=range(40)), "run-01")
    np.testing.assert_array_equal(x, x2)
```

- [ ] Run `uv run pytest tests/test_single_trial_design.py -q -W error`; confirm behavior fails because the compiler is absent. Commit tests with `uv run git add tests/test_single_trial_design.py` and `uv run git commit -m "test: specify single-trial designs"`.
- [ ] Implement per-event convolution and deterministic row mapping. Read only onset/duration when constructing X; copy all event metadata to the table. Positive durations use unit-height boxcars; zero durations use Nilearn's documented impulse convention with a narrowly handled expected warning. Generate no extra drifts; never add the old stimulus or RT column.

```python
condition = np.array([[onset], [duration], [1.0]])
column, _ = compute_regressor(condition, "spm", frame_times, oversampling=50)
```

- [ ] Run the task tests, refactor short validation/convolution functions, then commit implementation. Do not change expected results to accommodate timing errors.

## Task 2: Fit OLS/fixed ridge and return immutable, provenance-backed results

**Files:** Create `src/boldtailor/single_trial.py`, `src/boldtailor/_single_trial_fit.py`, `src/boldtailor/single_trial_results.py`, and `tests/test_single_trial.py`. Reuse, without changing their semantics, `data.py`, `_arrays.py`, `provenance.py`, and `logging.py`.

**Interfaces:** Public `fit_single_trials(data: AnalysisData, *, ridge_alpha: float = 0.0, run_labels: Sequence[str] | None = None) -> SingleTrialResult`. Internal `fit_trial_run(x, nuisance, signals, alpha)` returns an internal `TrialRunFit` with beta/nuisance coefficients, full/nuisance SSE, SST, design rank, and condition number. `SingleTrialResult` exposes `run_betas`, `trial_table`, `design_matrices`, `run_full_r2`, `run_nuisance_r2`, `full_r2`, `nuisance_r2`, `delta_r2`, `diagnostics`, `ridge_alpha`, and `provenance`.

- [ ] Write numerical tests against NumPy least squares and independent augmented least squares. Include alpha zero/positive/negative/nonfinite; shifted nuisance columns; changing RT/image IDs; missing RT; constant signals; residual task rank deficiency; run-specific nuisance columns; unequal run variances; source identity and mutable-access attempts. Tests must inspect actual outputs, not private delegation.

```python
def test_ridge_penalizes_trials_and_restores_units():
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    n = np.ones((4, 1))
    y = 10.0 + x @ np.array([[2.], [4.]])
    fitted = fit_trial_run(x, n, y, alpha=0.1)
    np.testing.assert_allclose(fitted.betas[:, 0], [2 / 1.1, 4 / 1.1])
    np.testing.assert_allclose(fitted.nuisance_betas, [[10.]])
```

- [ ] Run `uv run pytest tests/test_single_trial.py -q -W error`, confirm RED, and commit only the tests before writing the estimator.
- [ ] Implement validation and the numerical kernel. For alpha zero call `nilearn.glm.first_level.run_glm(..., noise_model="ols")` and extract trial coefficients. For alpha positive use the following algebra with an SVD-derived nuisance basis Q; rank thresholds follow the ordinary matrix-rank tolerance:

```python
xr = x - q @ (q.T @ x)
yr = y - q @ (q.T @ y)
scale = np.linalg.norm(xr, axis=0)
xs = xr / scale
u, s, vt = np.linalg.svd(xs, full_matrices=False)
w = (vt.T * (s / (s * s + alpha))) @ (u.T @ yr)
beta = w / scale[:, None]
gamma = np.linalg.lstsq(nuisance, y - x @ beta, rcond=None)[0]
prediction = x @ beta + nuisance @ gamma
```

Reject unidentifiable task columns and nonpositive residual degrees of freedom before either path. Compute nuisance-only SSE independently. Calculate R² from pooled sums, not averages; retain raw differences. Features constant within a run contribute zero SST/SSE and have NaN trial betas for that run; an entirely constant feature has NaN pooled metrics.

- [ ] Assemble immutable results using `immutable_float_array` and defensive table/dictionary copies. Fit each run independently. Add lifecycle events and canonical provenance with source metadata, labeled design fingerprints, numerical versions, run labels, HRF configuration, nuisance names, alpha, normalization, and beta units. Use the prepared-design hashing pattern so materially different designs cannot share fit identity. Do not log signals, design values, RTs, coefficients, or absolute paths.
- [ ] Run both task test files and `uv run pytest -q -W error`; commit the implementation only after GREEN. Unrelated pre-existing failures must be named rather than hidden.

## Task 3: Add RT diagnostics and independent vertex selection

**Files:** Create `examples/NSD/rt_diagnostics.py` and `examples/NSD/test_rt_diagnostics.py`.

**Interfaces:** `correlate_rt(beta_runs, rt_runs, *, run_numbers)` returns a dictionary with `all`, `odd`, `even` (feature arrays), `per_run` (run × feature), and `counts` (matching arrays in a nested dictionary). `select_vertices(odd_r, cortical_mask, *, count=5)` returns grayordinate indices. `scatter_artifact(beta_runs_by_model, rt_runs, run_numbers, vertices, path)` returns an in-memory PNG `Artifact`; it plots only even-run points, demeaned within run, with separate estimator panels.

- [ ] Write tests for confounding by run mean, missing/nonpositive RT, constant beta/RT, partial feature missingness, empty odd/even partitions, reproducible tie-breaking, cortical-only selection, and selection unchanged after arbitrary even-run edits. Estimation tests from Task 2 verify that diagnostic exclusions never remove fitted trials.

```python
def test_correlation_removes_run_mean_confounding():
    betas = [np.array([[3.], [2.], [1.]]), np.array([[13.], [12.], [11.]])]
    rt = [np.array([1., 2., 3.]), np.array([11., 12., 13.])]
    result = correlate_rt(betas, rt, run_numbers=[1, 2])
    for split in ("all", "odd", "even"):
        np.testing.assert_allclose(result[split], [-1.])
    np.testing.assert_array_equal(result["counts"]["all"], [6])
```

- [ ] Run `uv run pytest examples/NSD/test_rt_diagnostics.py -q -W error`, observe RED, and commit the failing tests.
- [ ] Implement matched finite/positive-RT masking per feature, then center within each run. Pool centered dot products for the requested partition:

```python
r = sum_cross / np.sqrt(sum_rt_squared * sum_beta_squared)
```

Use NaN when counts or variance are insufficient; do not produce p-values. Use absolute odd-run OLS correlation for selection, with numeric grayordinate index as a deterministic tiebreaker. Missing cortical candidates produce fewer plots, including zero, rather than fabricated vertices. Build matplotlib figures in memory and close figures explicitly.
- [ ] Run the diagnostic tests plus Task 2 RT-independence tests; commit after GREEN.

## Task 4: Connect the NSD inputs and publish CIFTI outputs

**Files:** Create `examples/NSD/nsd_single_trial.py`, `examples/NSD/single_trial_artifacts.py`, and `examples/NSD/test_nsd_single_trial.py`; update `examples/NSD/nsd_cifti.py` only to extract a reusable input loader, and update `examples/NSD/README.md`.

**Interfaces:** Extract `load_inputs(run: RunInputs)` to return CIFTI image, raw events, selected confounds, and frame times without constructing task regressors. Existing `_load_run` calls this loader and retains current conventional behavior. New `run_single_trial_analysis(bids_root, fmriprep_root, output_root, *, subject="sub-07", session="ses-nsd10", ridge_alpha=None, block_size=4096)` always fits OLS and optionally one positive fixed alpha. `single_trial_artifacts(...)` consumes run beta arrays, trial table, maps, RT diagnostics, imaging axes, model metadata, and provenance, and returns a complete tuple of `Artifact` values.

- [ ] Build a fixture containing two small real CIFTIs, event tables (including repeated IDs and invalid RT), known trial signals, and valid nuisance TSV/JSON files. Assert trial scalar names and rows, exact BrainModelAxis equality, OLS plus one ridge output, RT maps/counts, untouched existing conventional output, pre-fit collision detection, and rejection of mismatched axes or missing runs.

```python
def test_trial_map_order_and_spatial_axis(mini_nsd, tmp_path):
    root, prep, brain, expected_ids = mini_nsd
    paths = run_single_trial_analysis(root, prep, tmp_path / "out", ridge_alpha=0.1)
    beta_path = next(p for p in paths if "run-01" in p.name
                     and "singletrialOLS" in p.name and p.name.endswith("_betas.dscalar.nii"))
    image = nib.load(beta_path)
    assert image.header.get_axis(1) == brain
    assert image.header.get_axis(0).name.tolist() == expected_ids[0]
```

- [ ] Run `uv run pytest examples/NSD/test_nsd_single_trial.py -q -W error`, verify RED, and commit tests before extracting the loader or creating the runner.
- [ ] Implement the loader extraction, reuse `discover_runs` and `select_confounds`, and retain the existing 12 tests for the conventional NSD example. Read signals in feature blocks. For each block create `AnalysisData` with onset/duration events and selected confounds, call the new estimator, and place results into per-run float32 output arrays. Keep all trials together within every run fit. Compute RT checks separately using the same trial ordering and fixed model settings.
- [ ] Add argparse flags matching the interface; default output stays `derivatives/boldtailor`. Name artifacts with `desc-singletrialOLS` or `desc-singletrialRidge`; include alpha in the latter's sidecar. Save per-run trial `dscalar.nii`, a session trial TSV, model-specific full/confounds/delta R² maps, all/odd/even and per-run RT correlation/count maps, selected-vertex TSV, and even-run scatterplots. Preserve existing dataset-level metadata. Store canonical block/model provenance under unique filenames. Use JSON-safe diagnostic values; represent undefined spatial values as NaN in CIFTIs.
- [ ] Assemble the entire in-memory artifact set before calling `publish_artifact_set`, passing all source paths. Add one publication-failure integration test using the established publication-test injection pattern; reuse the core's extensive rollback tests rather than duplicating them. Run existing and new NSD tests and commit after GREEN.

## Task 5: Validate the complete feature and document the NSD run

**Files:** Extend `tests/test_single_trial.py` and `examples/NSD/test_nsd_single_trial.py`; update `examples/NSD/README.md` with actual verification outcomes and interpretation.

- [ ] Add acceptance checks for block-size invariance, nuisance-only features, and a simulation containing real RT-associated trial variability plus AR(1) noise. These verify behavior already specified by Tasks 1–4 and may pass immediately. Any bug they uncover needs a committed failing regression test before a fix. Generate amplitudes from known values; use RT only to evaluate the resulting maps. Do not assert that ridge always beats OLS or that real NSD correlations exceed a threshold.

```python
# Use the same miniature dataset, run configuration, and fixed alpha.
# Different output roots avoid intentional collision protection.
small = run_single_trial_analysis(root, prep, out_small, ridge_alpha=0.1, block_size=1)
large = run_single_trial_analysis(root, prep, out_large, ridge_alpha=0.1, block_size=4096)
# Match maps by relative output filename; compare data and both CIFTI axes.
```

- [ ] Complete any newly exposed fixes through RED–GREEN. Run `uv run pytest tests examples/NSD -q -W error`, `uv run black --check` with the changed Python paths, and `uv run git diff --check`.
- [ ] Run the real analysis using the predeclared alpha, without looking at RT to choose it:

```bash
uv run python examples/NSD/nsd_single_trial.py --ridge-alpha 0.1
```

Use all 12 runs from the previous analysis and preserve the existing output files. Request the required filesystem escalation for writing to the external volume. Confirm 750 total trial scalars from the input tables rather than hard-coding that count into the algorithm.

- [ ] Independently reconstruct at least 25 sampled grayordinates from the saved trial and nuisance designs using float64 NumPy least squares and augmented least squares. Check beta values, native units, CIFTI axes, trial IDs, pooled R², and independently recomputed RT correlations. Record peak memory and elapsed time; the complete in-memory beta artifact set is an expected memory cost.
- [ ] Inspect the even-run scatterplots and compare map reproducibility descriptively. Report weak or absent associations honestly. Document that native-unit beta scales can differ across runs, temporal correlations affect diagnostic interpretation, and these figures are not significance tests or an RT-optimized model.
- [ ] Obtain one final code review, resolve substantive findings, rerun affected checks, and commit only feature files. Report the output directory, settings, test counts, and verification limits.

## Plan self-review

- The five review concerns each have an assigned test task.
- OLS uses the existing numerical engine; ridge gets an independent augmented-system oracle.
- Alpha is a normalized penalty, not a GLMsingle fractional-ridge fraction.
- Beta estimation does not depend on RT or repeated-image identity.
- Raw R² differences use the actual selected estimator, not an implicit OLS refit.
- All/odd/even correlation maps and selected-vertex plots have distinct, explicit roles.
- CIFTI trial axes, source-event mapping, immutable results, provenance, and publication are covered.
- No new inference, tuning framework, storage format, or runtime dependency is required.

## Execution handoff

Direct implementation in the current session is the simplest execution option
for these closely connected tasks. Review this plan before starting the first
test cycle. This planning step itself makes no changes to the numerical code or
previously published NSD results.

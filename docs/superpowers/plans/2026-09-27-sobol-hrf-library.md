# Sobol HRF library implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this approved change. Follow RED–GREEN and commit tests before implementation.

**Goal:** Make the approved 512-sample Sobol HRF preview available through the API and NSD notebook.

**Architecture:** Add a factory returning the existing `HrfLibrary`, preserving canonical SPM at index zero and all downstream fitting behavior. The notebook chooses Sobol by default and retains grid/custom choices. Existing exports preserve the exact parameter table, curves, fingerprint, and settings.

**Tech stack:** Python, SciPy QMC, NumPy, pytest, Jupyter; existing dependencies only.

**Spec:** User approved the seed-0 preview on 2026-09-27: 512 scrambled Sobol samples in the six continuous parameter ranges of the old grid, plus canonical SPM. Custom durations remain 36 seconds. Coverage is uniform in normalized parameter coordinates, not necessarily in waveform distance.

## Global constraints

- Use `uv run` for Python and tests; keep every `__init__.py` empty.
- Write and commit failing tests before implementation.
- Preserve unrelated changes and notebook outputs.
- Keep `expanded_hrf_library()` unchanged for earlier analyses.
- Do not launch a full NSD refit or overwrite existing derivatives.

## Review focus

- Exact seed-0 preview identity, including NumPy RNG convention.
- Reject non-power-of-two counts and noninteger/negative seeds.
- Custom parameter rows take precedence over sampler settings.
- Exported settings and exact library support reproducibility.
- Notebook preview executes without starting HRF fitting.

## Task 1: Public factory

Files: `src/boldtailor/hrf_library.py`, new `tests/test_sobol_hrf_library.py`.

Interface: `sobol_hrf_library(n_samples=512, *, seed=0) -> HrfLibrary`.
Bounds in parameter order: (3,6), (10,16), (0.5,1.5), (0.5,2.5), (2,8), (0,2).

- [ ] Test default count, exact canonical anchor, normalization, preview reference row, stratification, reproducibility, different seeds, NumPy integer arguments, and invalid counts/seeds.
- [ ] Run `uv run --no-cache --no-sync pytest tests/test_sobol_hrf_library.py -q -W error -p no:cacheprovider`; expect missing factory failures. Commit tests.
- [ ] Validate integers, draw `qmc.Sobol(d=6, scramble=True, rng=int(seed)).random_base2(int(n_samples).bit_length()-1)`, scale bounds, append duration 36, pass to `HrfLibrary.from_parameters`.
- [ ] Run existing and new library tests; expect all passing.

## Task 2: Notebook and documentation

Files: `examples/NSD/nsd_workflow.ipynb`, `examples/NSD/test_nsd_workflow.py`, `examples/NSD/README.md`, `README.md`, `docs/user-guide.md`, `docs/api.md`.

Interface: settings `hrf_library="sobol"`, `hrf_n_samples=512`, `hrf_seed=0`; existing non-None `hrf_parameters` overrides the factory choice. `hrf_library="expanded"` reproduces the old grid. Invalid names raise ValueError.

- [ ] Add executed notebook preview tests for defaults/overrides/grid/custom/invalid choice. Parameterize the full synthetic CIFTI workflow over custom and small Sobol libraries; verify saved settings and reconstruct exact curves from the saved table.
- [ ] Run and commit failing tests before notebook edits.
- [ ] Separate library generation/plot from selection into independent cells. Use colored curves by full-HRF time to peak, black canonical reference, full time range. Update settings and plain-language documentation, including bounds and parameter-space versus waveform-space coverage.
- [ ] Execute both full synthetic notebook cases and complete suite with warnings as errors. Independently compare the new default factory to the approved preview table.
- [ ] Request a fresh code review, integrate the tested change into the main checkout while preserving user notebook outputs, then verify the integrated suite.

## Execution notes

The user explicitly said “looks great, proceed”; implementation is authorized without another planning approval. Reuse `/private/tmp/boldtailor-hrf-selection` at baseline `f239ba6`. Baseline notebook test needs Jupyter runtime access outside the sandbox. The main checkout contains unrelated stop-signal changes, which must remain untouched.

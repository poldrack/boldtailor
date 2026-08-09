# Task 1 Senior Review

## Scope and verdict

- Range reviewed: `d8050aa7105a35c41f492419353ad2d3774aecf2..e0cef1663f8067d83c185426110b604ad44939b8`
- TDD verdict: **PASS**
- Spec verdict: **PASS**
- Quality verdict: **PASS with one Minor test-coverage finding**
- Readiness: **READY for Task 2**

No Critical or Important findings were identified.

## Findings

### Minor

1. The focused tests do not directly lock down two properties that the implementation supplies: owned/write-protected signal arrays and identical feature ordering across runs. The approved design explicitly names cross-run feature ordering as focused coverage (`docs/superpowers/specs/2026-08-09-whole-brain-notebook-design.md:145-150`), while the main extraction test asserts shapes, masker settings, metadata, and finiteness only (`tests/test_stop_signal_demo.py:185-201`). The production path correctly copies to float64 and clears the writeable flag (`examples/stop_signal_demo.py:340-356`); an independent in-memory probe returned `dtype=float64`, `OWNDATA=True`, and `WRITEABLE=False`. Because both runs are transformed by the same fitted masker and the round-trip test verifies mask-space ordering, this is a regression-coverage gap rather than a current production defect. Add explicit assertions in a future test-first change when the notebook migration removes the temporary ROI path.

## TDD audit

The history has the required order and content:

1. `9e2bf11` changes only `tests/test_stop_signal_demo.py`. Its new imports reference APIs absent from its parent, so the recorded collection RED (`ImportError`, exit 2) is credible and appropriate.
2. `37eb1f8` changes only the round-trip test before production. The correction from `masker.transform(image)[0]` to `masker.transform(image)` is justified: an independent probe against the installed Nilearn returned a one-dimensional `(8,)` result for a reconstructed three-dimensional image. Its narrow suppression of the exact Nilearn `standardize=False` deprecation warning is necessary for warning-strict compatibility with the mandated setting.
3. `e55663d` is the first production commit and changes only `examples/stop_signal_demo.py`.
4. `e0cef16` adds only the implementation report.

The report also records the warning-strict RED caused by Nilearn's exact `standardize=False` FutureWarning, followed by GREEN. No production change appears in either test commit.

## Specification and implementation audit

- `common_brain_mask` rejects an empty input sequence, validates all mask shapes and affines, computes a logical intersection, rejects an empty result, and preserves first-mask geometry with uint8 data/header (`examples/stop_signal_demo.py:89-101`).
- `make_masker` fits once with exactly `standardize=False`, `detrend=False`, `smoothing_fwhm=None`, `low_pass=None`, `high_pass=None`, and `reports=False` (`examples/stop_signal_demo.py:104-113`).
- `load_run` validates BOLD spatial shape and affine before transformation (`examples/stop_signal_demo.py:157-173`, `321-328`). Transformed signals must be two-dimensional, have one row per scan and one column per common-mask voxel, and contain only finite values (`examples/stop_signal_demo.py:340-356`).
- Signal storage is an owned immutable float64 copy (`examples/stop_signal_demo.py:305-308`, `340-356`), confirmed independently at runtime.
- `whole_brain_image` validates one-dimensional length and finiteness and reconstructs with the same fitted masker (`examples/stop_signal_demo.py:176-185`). The test correction is dimensionally correct and round-trips all mask values (`tests/test_stop_signal_demo.py:227-242`).
- Memory estimation uses the sum of positive scan counts, positive voxel count, and eight-byte float64 storage (`examples/stop_signal_demo.py:199-205`).
- The example fixture fits all 343 common-mask features and exactly the three approved contrast expressions (`tests/test_stop_signal_demo.py:118-145`).
- Mask and BOLD affine mismatch tests assert the required messages (`tests/test_stop_signal_demo.py:168-182`, `204-224`). Existing event, confound, source, publication-path, and protected-source tests remain present.
- The narrow production warning filter is confined to the mandated masker transform and matches only the known Nilearn FutureWarning (`examples/stop_signal_demo.py:343-349`). The full suite passes under `-W error`.

## Authorized sequencing exception and scope

The retained `LoadedRun` spatial fields, ROI helpers, and non-`NiftiMasker` dispatch are the controller-authorized temporary sequencing exception (`examples/stop_signal_demo.py:31-41`, `116-140`, `150-156`, `188-196`, `359-404`). `_load_roi_run` is the prior ROI body extracted without expanded public behavior, and `_mask_voxel_indices` supplies accurate values for the temporarily required fields. I found no additional compatibility alias, fallback, downsampling, resampling, truncation, or chunking behavior. Task 3 must still remove this exception after migrating the notebook.

Only the expected test, example helper, and report files changed in the reviewed range. No `__init__.py` changed; `src/boldtailor/__init__.py` remains zero bytes. `git diff --check` passed. Existing untracked `__pycache__/` directories were left untouched as required.

## Fresh verification

```text
uv run pytest tests/test_stop_signal_demo.py -q -W error
30 passed in 6.38s
exit 0

uv run pytest -q -W error
217 passed in 8.33s
exit 0
```

Both commands emitted only shell/uv environment setup text outside pytest's warning handling; no Python warning was promoted to an error.

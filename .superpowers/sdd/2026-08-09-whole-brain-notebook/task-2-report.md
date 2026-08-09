# Task 2 Implementation Report

## Commit range

- Base: `7016713ab1c1e95329afc18e278132b2b5b18377`
- Implementation head before this report commit:
  `7166422ef2573adc86bee0909d807a475d95dc54`
- Initial test-only commit: `fbcb96161d0c6beb8a3a1a0e33bcf33d95fe2f25`
  (`test: specify whole-brain image derivatives`)
- Sequencing-boundary test-only commit:
  `720b851577eddebdd8f18a709a79f1b4f6501755`
  (`test: reject partial image artifact context`)
- Production commit: `7166422ef2573adc86bee0909d807a475d95dc54`
  (`feat: serialize whole-brain result maps`)
- The report commit is recorded in the final handoff because a commit cannot
  contain its own identifier.

## RED evidence

The required focused derivative command was run after changing only the tests:

```text
uv run pytest tests/test_stop_signal_demo.py -k "result_artifacts" -q
exit 1
1 failed, 29 deselected in 1.18s
TypeError: result_artifacts() got an unexpected keyword argument 'space'
```

This was the expected failure of the old ROI-only artifact interface. The tests
were committed alone as `fbcb961` before the production file was committed.

The complete Task 2 implementation initially exposed a sequencing contradiction.
The Task 2 brief requires the new mandatory masker, common-mask, space, and
resolution inputs, leaves notebook migration to Task 3, and also requires the
existing notebook-execution tests and full suite to remain green. The unchanged
notebook still uses the exact old call shape. Executing those tests produced:

```text
uv run pytest tests/test_stop_signal_demo.py -k "notebook_executes_against_fixture" -q
exit 1
2 failed, 28 deselected in 7.10s
TypeError: result_artifacts() missing 2 required positional arguments:
'masker' and 'common_mask'
```

Work stopped and the evidence was reported before choosing among editing the
Task 3 notebook early, adding a compatibility branch, or accepting a non-green
suite. The controller authorized a narrow sequencing bridge: preserve only the
exact old behavior when all four spatial inputs are absent, require all four
together for whole-brain output, and delete the bridge during Task 3.

A focused test was then added to prevent a partial spatial context from entering
either branch:

```text
uv run pytest tests/test_stop_signal_demo.py -k "incomplete_spatial_context" -q
exit 1
1 failed, 30 deselected in 1.12s
AttributeError: 'NoneType' object has no attribute 'to_bytes'
```

The expected contract is a specific `ValueError`, so this was a genuine RED.
That test was committed alone as `720b851` before the guard implementation.

## GREEN evidence

The whole-brain derivative behavior passed after deterministic image and
manifest serialization was implemented:

```text
uv run pytest tests/test_stop_signal_demo.py -k "result_artifacts" -q
1 passed, 29 deselected in 1.10s
```

After the authorized sequencing guard was implemented, the whole-brain path,
partial-context rejection, and both unchanged notebook execution variants passed
together under warning-strict execution:

```text
uv run pytest tests/test_stop_signal_demo.py \
  -k "result_artifacts or notebook_executes_against_fixture" -q -W error
4 passed, 27 deselected in 6.59s
```

Fresh verification immediately before production commit `7166422`:

```text
uv run pytest tests/test_stop_signal_demo.py -q
31 passed in 6.81s

uv run pytest -q -W error
218 passed in 8.65s

uv run git diff --check
exit 0
```

The suite increased from the accepted 217-test baseline to 218 tests because the
partial-context boundary test was added. No `__init__.py` file was modified.

## Files changed

- `tests/test_stop_signal_demo.py`
- `examples/stop_signal_demo.py`
- `.superpowers/sdd/2026-08-09-whole-brain-notebook/task-2-report.md`

The SDD progress ledger, real dataset, main checkout, and every `__init__.py`
were left untouched. Pre-existing untracked `__pycache__/` directories were also
left untouched.

## Requirement coverage

- Produces exactly ten image artifacts in fixed order: common mask, effect and z
  maps for three contrasts, two run-specific R-squared maps, and aggregate
  R-squared.
- Uses the approved stable contrast labels `successfulInhibition`, `stopVsGo`,
  and `goSuccessVsBaseline` in the exact derivative paths.
- Reconstructs effect, z, run-R-squared, and aggregate-R-squared values with the
  fitted whole-brain masker and serializes the supplied common-mask image.
- Uses gzip level 9 with `mtime=0`; repeated artifact construction is byte-for-byte
  equal.
- Independently decompresses and loads every payload in tests, verifies the
  `(7, 7, 7)` shape and common affine, and checks all masked image values against
  their corresponding analysis result vectors.
- Renames the contrast report to `desc-wholebrain_contrasts.tsv` and verifies no
  whole-brain artifact path contains `roi`.
- Emits `desc-image_manifest.tsv` with literal `relative_path`, `media_type`,
  `byte_size`, and `sha256` columns in image-artifact order.
- Records `application/gzip` for every image and verifies sizes and SHA-256
  digests independently against the actual immutable payload bytes.
- Preserves design matrices and privacy-safe, sorted JSON configuration before
  image artifacts and the manifest in stable return order.
- Rejects a partial image-artifact context with a specific `ValueError`.

## Concerns and scoped deviation

The unchanged Task 3 notebook calls `result_artifacts` without the newly required
masker, common mask, space, or resolution. The controller authorized the same
narrow sequencing principle used in Task 1: when and only when all four values
are absent, the helper temporarily returns the exact pre-existing design TSVs,
ROI contrast TSV, and JSON configuration. Supplying any subset raises a
`ValueError`; supplying all four selects the exact Task 2 whole-brain artifact
path.

This branch exists only to keep the pre-existing notebook execution tests green
without editing the notebook outside Task 2 scope. Task 3 must migrate the
notebook to the complete spatial signature and then delete this legacy artifact
branch and its optional signature defaults. No ROI loading helper or dispatch was
removed or otherwise expanded in Task 2.

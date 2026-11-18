# Addressing the package's Pyright findings

The first step should be to correct shared type definitions and make existing
validation visible to the type checker. This is not a request to rewrite the
numerical methods or change their outputs.

## Baseline

The boolean-negation fix was merged into main at `6d6f77a`. A subsequent
package-only scan with Pyright 1.1.414 found **317 errors and 2 warnings**.
It analyzed 82 files under `src/boldtailor` and `tests/typing`, using basic
mode, a Python 3.13 target, and the project's installed dependencies.

These are static diagnostics, not 319 established runtime defects. Several
diagnostics arise from one shared helper or from incomplete library typing.
The broader scan of tracked Python files also included tests, examples, and
experiments; those findings are outside the package counts below.

| Diagnostic group | Count |
| --- | ---: |
| Argument types | 159 |
| Attribute access | 74 |
| Optional values: indexing, attributes, arithmetic, iteration | 35 |
| Function calls and overloads | 16 |
| Indexing | 14 |
| Operators | 8 |
| Return types | 6 |
| Other type and assignment issues | 5 |
| Deprecation warnings | 2 |

The largest concentrations are `workflow/beta_series.py` (56),
`workflow/analysis.py` (31), and `hrf_selection.py` (26).
These counts measure reported diagnostics, not independent defects or effort.

## 1. Establish a reproducible check

Add Pyright as a pinned development dependency using uv, and commit a
configuration that initially includes the package and the small static
regression directory. Start with basic mode.

Use Python 3.12 as the primary target, matching the package's declared minimum
and current CI. The baseline above used 3.13, so regenerate it when adopting
the configuration rather than treating its counts as fixed.

Record the Pyright version, Python target, and installed dependency versions.
For example, the eventual CI command would be `uv run pyright`, alongside
pytest in `.github/workflows/tests.yml`. The configuration and dependency
change have not been made as part of this assessment.

Initially, prevent new diagnostics while reducing the existing baseline.
Compare diagnostic identity (file, rule, and message), not only the total
count: removing one error must not hide a different new error.
Do not make the existing 317 errors an immediately failing repository-wide
gate. Modules can become mandatory clean checks as they are corrected.

## 2. Correct the shared array helpers

Start with `_arrays.py`. Its dtype parameters have an unannotated default of
`np.float64`. Pyright consequently treats legitimate boolean and integer
dtype arguments as errors in many callers. Twenty-two current diagnostics
name `readonly_array`, `copy_array_fields_in_place`, or
`copy_array_tuple_fields_in_place`.

Annotate accepted dtype inputs explicitly, using `numpy.typing.DTypeLike`
where that matches the existing API. Give array-returning helpers explicit
return types. Where callers need a specific dtype, use a small number of
accurate overloads or a scalar-type parameter rather than declaring every
result to be floating point.

Preserve copying, C order, and read-only ownership. Type annotations do not
prove array shape or ownership, so retain the existing runtime checks and
tests. This should be a small change with many downstream benefits.

## 3. Make alternative result states explicit

Several warnings reflect valid runtime distinctions that Pyright cannot infer.

- `RemlHrfOptions.rho` is a float or an array. A test of `np.ndim(rho)`
  does not narrow that union for Pyright. After construction has normalized
  the value, use an explicit array type check in `validate` and `subset`.
- HRF results contain CV arrays or REML diagnostics. Branching on `method`
  does not automatically prove which optional fields are present. Add narrow,
  typed accessors or explicit local checks. Preserve the public result API;
  separate result classes are not required for this cleanup.
- Ridge results vary by regularization method. In `workflow/beta_series.py`,
  the separate `fractional` flag does not narrow a heterogeneous dictionary's
  `selection` value. Give the tuning result a defined structure and narrow
  the actual selection object before reading `ridge_fraction` or
  `ridge_alpha`.
- `prepared_fit.py` has parameters annotated as ordinary values but defaulted
  to `UNSET`. Include the sentinel type in the parameter types, then normalize
  and validate before calling functions that require concrete values.

For each branch, determine whether construction already guarantees the
invariant. If so, expose that invariant to the checker. If not, add the missing
validation and a failing runtime test before implementing the correction.
Do not insert assertions merely to hide a reachable invalid state.

## 4. Type the workflow and provenance records

Unannotated mixed dictionaries account for many workflow errors. For example,
`workflow/analysis.py` builds an options dictionary and passes it with
`**options`; Pyright cannot associate each value with its named parameter.

Use explicit keyword arguments for short calls. For records shared between
functions, use a small `TypedDict` or dataclass. Likely targets are selection
options, tuning results, split bundles, and saved provenance records.
Do not introduce a new record class for every temporary dictionary.

Serialization also needs accurate value types. For example,
`Modulator.to_dict()` starts with strings and later adds a list and possibly
`None`. Annotating the dictionary at construction avoids incorrect
string-only inference. At deserialization boundaries, validate values read
from `Mapping[str, object]` before constructing typed objects.

Preserve JSON fields, output names, and provenance identity. Test round trips
and invalid inputs where validation changes. Do not use `Any` throughout the
workflow to suppress these diagnostics.

## 5. Handle third-party library boundaries

The current scan sometimes infers a pandas scalar where the code expects a
Series. This affects event conversion in `_task_design.py`, workflow input
loading, and `trial_encoding.py`. Nibabel and Nilearn boundaries also produce
unknown attributes or overly broad `object` results.

First annotate DataFrame/Series parameters and return values at these
boundaries. Then assess compatible pandas stubs against the pinned pandas
version. Stubs can help, but adding them may also expose additional findings.

For Nilearn contrast objects, refine the existing protocols to describe the
operations actually used, including contrast arithmetic where needed.
For nibabel, prefer imports from the defining module when that matches its
public API. Use local stubs or a narrow cast only when the runtime contract is
known and third-party typing remains incomplete. Explain each such exception;
do not disable an entire diagnostic category.

## 6. Review the remaining numerical and deprecation findings

Some array reductions are inferred as an integer or a scalar because Python's
`sum` starts from zero. Inspect these individually and state the intended
array return type. Preserve summation order where changing it could affect
roundoff-sensitive calculations. Keep scalar logical negation distinct from
NumPy's elementwise logical operations.

The two remaining warnings are in `_fit_lifecycle.py` and `logging.py`.
They concern context-manager return annotations written as `Iterator`.
Use the appropriate `Generator[YieldType, None, None]` annotation.
The warning does not mean that `contextlib.contextmanager` itself should be
removed.

Investigate remaining operator and index errors as possible runtime defects.
Only dismiss a finding after tracing the value's construction and validation.

## Suggested sequence and validation

Use separate, reviewable changes in this order:

1. Reproducible configuration and baseline.
2. Array/dtype helper contracts and context-manager annotations.
3. Optional results, scalar/array alternatives, and sentinel handling.
4. Workflow and provenance record types.
5. Third-party interfaces and remaining numerical diagnostics.
6. A clean package-level CI gate; stricter checks can follow module by module.

For typing-only corrections, add focused static regressions like
`tests/typing/reml_masks.py`. For behavior changes, use pytest and the
repository's tests-first commit requirement. Run affected modules and direct
consumers, plus the integration suite before commits. Run the full default
suite before merging and after broad shared-helper changes.

Completion means a reproducible package scan with no unexplained basic-mode
errors or warnings, passing numerical and workflow tests, unchanged supported
scientific behavior, and documented narrow exceptions where library typing
requires them. It does not mean suppressing errors, making every value
`Any`, or extending the first pass to all historical experiments.

The amount of work should be reassessed after the shared helpers and result
states are typed. Diagnostic counts alone are not a reliable time estimate.


## Completed first pass: shared contracts and result states

The first pass reduced the same Python 3.13/basic-mode package scan from
317 errors to **157 errors**, with the same two deprecation warnings. It now
covers 86 files, including the added static regressions and workflow records.
Comparison by file, diagnostic rule, and message found no new diagnostics.
All 35 optional-value diagnostics were removed. These counts do not imply
that 160 runtime bugs were fixed.

Changes include:

- Dtype-aware array helpers and generic block-worker results.
- Array reductions that preserve Python's sequential summation order and
  return arrays, including zero-dimensional arrays; empty input is rejected.
- Explicit omitted-argument types and typed keyword-alias resolution.
- Narrowing of normalized REML correlations, fitted GLM/trial-design variants,
  CV/REML map fields, and fractional/shared-alpha tuning selections.
- Typed workflow mappings for selection, fitted blocks, tuning, and evaluation.
- Owned activity dictionaries and validated provenance deserialization.
- Explicit encoding metadata fields and normalized settings invariants.

The public numerical methods and serialized field names are unchanged.
The two narrow joblib argument suppressions in `parallel.py` document accepted
integer options that its inferred annotations incorrectly describe as strings.
The casts in keyword-alias resolution follow identity checks against the
omitted-value singleton.

### Repeat the clean regression check

From the repository root, with the project environment already installed:

```sh
uv run --no-project --with pyright==1.1.414 pyright   --project tests/typing --pythonpath .venv/bin/python
```

The committed configuration targets Python 3.12 and checks the static fixtures
plus ten corrected core/workflow modules. This check passes with zero errors
and warnings. Pytest does not collect the `tests/typing` fixtures; run this
command separately. Pyright remains an ephemeral tool dependency, and no
repository-wide typing gate has been enabled.

The remaining package findings include third-party pandas, NumPy, Nilearn,
and nibabel interfaces; other serialization boundaries; and the two
context-manager annotations. Those require a separate pass. The full package
is not yet type-clean.


Validation for this pass: the full default suite passed 2,570 tests (9 skipped,
2 expected failures). After the final review fixes, the affected array,
denoising, settings, REML workflow, beta-series, and encoding checks passed
360 tests; alias-consumer checks passed 144; and the final integration suite
passed 23. The earlier complete workflow run passed 309 tests. The full suite
was not repeated after those final targeted fixes.

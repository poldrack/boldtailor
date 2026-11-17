# API guide for human code review

[Complete module/function inventory](api-review/README.md) · [API options and returns](api.md) · [User guide](user-guide.md) · [Developer guide](development.md)

This guide follows the code from inputs to scientific estimates to saved outputs.
Use it to choose a review route, run a small example, and locate the functions
that implement each step. The linked inventory covers **all 76 Python modules
and 1,193 explicit definitions** in `src/boldtailor`, including private helpers,
methods, properties, and nested functions. Each module page lists signatures,
source docstrings, class fields, source links, and available direct-call examples
in package code and tests. This is a source snapshot dated **2026-11-15**, not a
claim that the scientific methods have been independently validated.

Start here for the architecture; use [api.md](api.md) for parameter explanations
and result schemas, then the inventory for individual functions. Research code
in `experiments/` and notebook helpers in `examples/` are outside this package
inventory. The top-level `boldtailor` initializer exports the primary public
names; everything else is imported from its defining module, and the
`boldtailor.workflow` initializer is empty.

## The objects that cross module boundaries

| Object | What it represents | Shape / important contract |
| --- | --- | --- |
| `AnalysisData` | Aligned runs, raw events, confounds, timing and source records | Signals are scans × features, with the same feature order in every run; event onsets/durations are seconds |
| `ModelSpec` | Conventional contrasts, nuisance/drift settings, noise and HRF choices | Contrasts are expressions or named weight mappings; t contrasts only |
| `TaskModel`, `Modulator` | Shared task regressor semantics for HRF selection and task-model GLMs | Separate from raw single-trial design columns; numeric/categorical modulator handling matters |
| `PreparedDesignAnalysis` | Signals with caller-supplied labeled design matrices | Every column has a role; fitting does not add or transform regressors |
| `HrfLibrary`, `HrfCandidate` | Ordered candidate kernels and their identities | Candidate zero is canonical SPM; candidate kernels have peak one |
| `HrfSelectionResult` | An HRF choice per feature and its selection diagnostics | Preserve feature order/signature; selection scores are not independent evaluation |
| `AnalysisResult`, `HrfAnalysisResult` | Conventional contrast statistics and R² | Contrast maps are feature vectors; selected-HRF designs are reconstructed by run/HRF group |
| `SingleTrialResult` | One beta per original event row | `run_betas[run]` is trials × features; `design` owns or reconstructs the design |
| `CandidateScores` | Encoding scores across candidate penalties | `cv_r2` is candidates × features; `regularization` distinguishes the objective |
| `RemlGroup`, `RemlFit`, `RemlSelection` | Fixed/random-design variance estimation | One random design per group; variance/status maps are per feature |
| `ProvenanceRecord`, `Artifact` | Analysis identity and in-memory output payloads | Computation and file publication have separate boundaries |

Owned numerical result arrays are ordinary NumPy arrays marked read-only.
Use `.copy()` before editing. Public table accessors generally return defensive
copies; private fields are not an editing interface. Feature indices and run
indices are zero-based; BIDS run numbers and display labels are distinct.

## Choose a review route

Read the public entry point first, then its numerical helpers, then its result
container and tests. Names in the following tables link to complete module pages.
The test links are starting points, not the full validation suite.

### 1. Conventional GLMs and prepared designs

| Read in order | Functions to follow and their usage |
| --- | --- |
| [data](api-review/data.md), [model](api-review/model.md) | `from_arrays` validates and owns runs; `ModelSpec` describes contrasts and model settings; `run_labels_for` standardizes run labels |
| [design](api-review/design.md), [_task_design](api-review/_task_design.md) | `compile_designs` and `compile_nuisance_designs` build labeled per-run matrices; `kernel_task_columns` handles a supplied peak-one kernel |
| [fit](api-review/fit.md), [_conventional](api-review/_conventional.md) | `fit(data, model)` estimates each run and combines contrasts; a supplied `hrf_selection` dispatches to grouped fitting |
| [_fit_diagnostics](api-review/_fit_diagnostics.md), [results](api-review/results.md) | `task_delta_r2` compares nested OLS full/nuisance fits; result builders own contrast and diagnostic arrays |
| [prepared](api-review/prepared.md), [prepared_fit](api-review/prepared_fit.md) | `PreparedDesignAnalysis.from_arrays`, `fit_prepared`, and `task_delta_r2_prepared` are the route for externally constructed designs |

**Review questions.** Are the intended event columns and confounds actually in
the design? Are early events excluded as recorded? Conventional contrasts use
equal-run fixed effects, not precision weighting. Diagnostic task ΔR² uses nested
OLS even when contrast inference uses AR(1). Check constant-feature NaNs and the
identity checks that connect a diagnostic to its parent fit. The prepared route
must preserve the caller's matrix and enforce its column-role contract.

Start with [test_fit.py](../tests/test_fit.py),
[test_multirun.py](../tests/test_multirun.py),
[test_prepared_fit.py](../tests/test_prepared_fit.py), and
[test_fit_diagnostics.py](../tests/test_fit_diagnostics.py).

### 2. HRF construction, selection and selected-HRF GLMs

| Read in order | Functions to follow and their usage |
| --- | --- |
| [hrf_library](api-review/hrf_library.md) | `default_hrf_library`, `timing_hrf_library`, `sobol_hrf_library`, `expanded_hrf_library`, and `glmsingle_hrf_library` construct candidate sets; parameter/timing conversion functions distinguish gamma parameters from realized curve timing |
| [hrf_selection](api-review/hrf_selection.md), [_hrf_design](api-review/_hrf_design.md), [_hrf_cv](api-review/_hrf_cv.md) | `select_hrfs` selects by leave-one-run-out task prediction; `evaluate_hrf_split` selects on training runs and evaluates frozen choices on test runs |
| [hrf_results](api-review/hrf_results.md), [_hrf_assignment](api-review/_hrf_assignment.md) | Inspect score schemas, feature identity, undefined choices and assignment validation |
| [_hrf_glm_design](api-review/_hrf_glm_design.md), [_hrf_glm](api-review/_hrf_glm.md), [hrf_glm_results](api-review/hrf_glm_results.md) | Follow `fit(..., hrf_selection=selection)` into grouped conventional estimation and `group_design(run, hrf_id)` reconstruction |
| [reliability](api-review/reliability.md) | `compare_hrfs` and `curve_correlations` describe agreement of selected curves on the stored time grid |

**Review questions.** Are task amplitudes learned only on the intended training
runs? Are selection and evaluation distinguished? Are feature signatures, task
model compatibility, and HRF normalization consistent when reusing a selection?
Curve agreement is Pearson correlation without temporal shifting, not an ICC.
Undefined choices must stay in their original spatial positions as NaNs.

Start with [test_hrf_cv.py](../tests/test_hrf_cv.py),
[test_hrf_selection.py](../tests/test_hrf_selection.py),
[test_hrf_glm.py](../tests/test_hrf_glm.py), and
[test_hrf_timing.py](../tests/test_hrf_timing.py).

### 3. Single-trial estimation and encoding-guided regularization

| Read in order | Functions to follow and their usage |
| --- | --- |
| [single_trial](api-review/single_trial.md), [_single_trial_design](api-review/_single_trial_design.md) | `fit_single_trials` uses one fixed HRF; `fit_selected_hrfs` uses a prior spatial assignment; each raw event gets its own column |
| [_single_trial_fit](api-review/_single_trial_fit.md), [_selected_hrf_fit](api-review/_selected_hrf_fit.md), [_fractional_ridge](api-review/_fractional_ridge.md) | Inspect nuisance projection, trial scaling, penalty application and restoration of feature order |
| [single_trial_results](api-review/single_trial_results.md) | Follow beta arrays, trial metadata and shared versus reconstructed selected-HRF designs |
| [trial_encoding](api-review/trial_encoding.md) | `trial_predictors` builds aligned trial covariates; `evaluate_trial_encoding` fits training slopes and scores held-out trials |
| [ridge_selection](api-review/ridge_selection.md), [fractional_ridge](api-review/fractional_ridge.md), [_ridge_cv](api-review/_ridge_cv.md) | `score_ridge_candidates` / `select_ridge_penalty` choose one alpha across features; `score_fraction_candidates` / `select_ridge_fractions` choose a fraction at each feature |
| [ridge_results](api-review/ridge_results.md), [diagnostics](api-review/diagnostics.md) | Inspect score shapes and selection masks; `one_sample_t` and `correlate_rt` summarize fitted betas |

**Review questions.** Nuisance columns and the intercept are unpenalized. Fixed
alpha acts on normalized projected trial columns; fractional ridge constrains
coefficient norm in the raw projected basis. Fractional CV uses fixed OLS targets;
normalized-alpha CV uses candidate-regularized targets. Default encoding removes
within-run offsets for scoring, but predictions must not depend on test outcomes.
A shared-alpha selection must reduce over the complete intended feature set, not
separately over spatial blocks. Supplied HRF libraries require nested selection
inside the training folds. Repeated stimuli are not required.

Start with [test_single_trial.py](../tests/test_single_trial.py),
[test_fractional_cv.py](../tests/test_fractional_cv.py),
[test_ridge_cv.py](../tests/test_ridge_cv.py), and
[test_within_run_cv_isolation.py](../tests/test_within_run_cv_isolation.py).

### 4. Scalar REML and AR(1)

| Read in order | Functions to follow and their usage |
| --- | --- |
| [ar1](api-review/ar1.md), [_ar1_moments](api-review/_ar1_moments.md) | `whiten` transforms rows; `prepare_reference` caches design work; `estimate_rho` estimates autocorrelation using a reference design |
| [reml](archive/api-review/reml.md) | `fit_reml(groups, random_design, rho=...)` estimates one random-effect variance ratio; `select_random_effects_design` compares candidate random designs using the same groups; includes input validation, fixed-space projection, pooled restricted likelihood, and bounded variance-ratio search |
| [_numerics](api-review/_numerics.md) | Review shared rank, projection and least-squares conventions used by numerical paths |

**Review questions.** `rho` is supplied to REML, not estimated inside it. Each
group has scans × features signals, scans × fixed-columns `fixed_design`, and
a scans × random-columns design (or `None`). Groups share the feature axis.
`noise_variance` is marginal noise variance; `effect_variance` is `ratio *
noise_variance`. Do not read `ratio` without `status`: zero variance, no signal,
unidentifiable and numerically unresolved cases have distinct meanings.
`at_upper` and `unrefined` mark search limitations. For candidate selection,
`winner == -1` means no eligible winner; only `ok` candidates enter the race.
`gap` is runner-up minus best criterion when available, not a probability.
`any_at_upper` / `any_unrefined` include competing candidates, not just the winner.

Start with [test_reml_oracle.py](../tests/test_reml_oracle.py),
[test_reml_scientific_properties.py](../tests/test_reml_scientific_properties.py),
[test_reml_scientific_regressions.py](../tests/test_reml_scientific_regressions.py),
[test_reml_selection.py](../tests/test_reml_selection.py), and
[test_ar1.py](../tests/test_ar1.py).

### 5. Optional task-guided denoising

Read [denoising](archive/api-review/denoising.md): `select_denoising` selects temporal
noise PCs and `with_denoising` appends them to the baseline confounds of the same
analysis (including the noise pool/PCA, PC-count selection, significance gate, and
source-identity checks), using [_mixture_threshold](archive/api-review/_mixture_threshold.md)
for the automatic tail threshold and
[denoising_results](archive/api-review/denoising_results.md) for identity and outputs.

**Review questions.** HRFs, the pool and its components use all supplied runs;
count-selection scores are not independent performance estimates. The significance
gate is a Boldtailor addition to GLMdenoise. This path is opt-in through the array
API and is not part of `boldtailor run`. Check PC-count limits, empty/degenerate
pools and source-identity enforcement before appending confounds.

Start with [test_denoising.py](../tests/test_denoising.py),
[test_denoising_cv.py](../tests/test_denoising_cv.py), and
[test_denoising_gate.py](../tests/test_denoising_gate.py).

### 6. Ownership, provenance, lifecycle and publication

| Modules | Usage and review focus |
| --- | --- |
| [_arrays](api-review/_arrays.md), [_scalars](api-review/_scalars.md) | Array ownership/copying, type validation and immutable result construction |
| [provenance](api-review/provenance.md), [_software](api-review/_software.md), [bids_provenance](api-review/bids_provenance.md) | `SourceRef`, `RunSources`, fingerprints and `extend_provenance`; `project_bids_provenance` produces in-memory metadata bytes |
| [logging](api-review/logging.md), [_fit_lifecycle](api-review/_fit_lifecycle.md) | Context binding and start/completion/failure events around computation |
| [publication](api-review/publication.md) | `Artifact` holds an output; `publish_artifact_set` takes the writer lock, stages outputs, and rolls back failures |
| [_constants](api-review/_constants.md), [_deprecation](api-review/_deprecation.md) | Shared numerical defaults and old-keyword handling |

**Review questions.** Do fingerprints change for scientifically meaningful
changes? Does a table accessor expose mutable internal state? Does an exception
leave understandable lifecycle records? Publication uses a sibling control
directory and backups; inspect the [publication guarantees](publication-migration.md)
before assuming crash durability or filesystem-wide transaction semantics.

Start with [test_arrays.py](../tests/test_arrays.py),
[test_provenance.py](../tests/test_provenance.py),
[test_fit_lifecycle.py](../tests/test_fit_lifecycle.py), and
[test_publication.py](../tests/test_publication.py).

### 7. BIDS workflow, CIFTI outputs and presentation

| Read in order | Functions to follow and their usage |
| --- | --- |
| [cli](api-review/cli.md), [workflow.settings](api-review/workflow-settings.md) | `main`, `build_parser`, `settings_from_args`, `WorkflowSettings` resolve a session's settings |
| [workflow.run](api-review/workflow-run.md) | `describe_inputs` is the dry-run path; `run_workflow` orchestrates fitting and publication |
| [workflow.files](api-review/workflow-files.md), [workflow.inputs](api-review/workflow-inputs.md), [task_detection](api-review/task_detection.md) | Discover matching inputs, load runs, choose confounds, preserve timing, detect task regressors and form feature blocks |
| [workflow.analysis](api-review/workflow-analysis.md), [workflow.beta_series](api-review/workflow-beta_series.md), [parallel](api-review/parallel.md) | Blockwise public-API calls, penalty tuning and outer evaluation; `map_blocks` bounds dispatch and preserves order |
| [cifti](api-review/cifti.md), [workflow.artifacts](api-review/workflow-artifacts.md), [workflow.outputs](api-review/workflow-outputs.md) | Preserve spatial axes, construct in-memory artifacts and publish all stages together |
| [workflow.summaries](api-review/workflow-summaries.md), [workflow.plots](api-review/workflow-plots.md), [workflow.surfaces](api-review/workflow-surfaces.md), [workflow.report](api-review/workflow-report.md) | Summarize fitted results, render figures and embed them in an HTML report |

**Review questions.** This workflow expects BIDS events and fMRIPrep fsLR-91k
CIFTIs. Confirm acquisition times after removing non-steady-state scans, confound
selection, odd/even BIDS run splits, and restoration of excluded features to NaN.
The similarly named `workflow.analysis.select_hrfs` is a blockwise adapter to the
core selection API. Settings construction resolves paths; `run_workflow` writes
outputs. Plotting functions should consume results without refitting.

Start with [workflow tests](../tests/workflow), especially
[test_synthetic_bids.py](../tests/workflow/test_synthetic_bids.py),
[test_analysis.py](../tests/workflow/test_analysis.py), and
[test_outputs.py](../tests/workflow/test_outputs.py).

## Runnable examples for the numerical entry points

Run these snippets in order in a Python session (`uv run python`) or combine them
into a script and run `uv run python your_script.py`. They use small synthetic
arrays to demonstrate contracts, not to validate scientific recovery. No files
are written and no external dataset is required.

### Construct runs and fit a conventional contrast

```python
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec

rng = np.random.default_rng(7)
events = pd.DataFrame({
    "onset": [10., 30., 50., 70., 90., 110.],
    "duration": [1.] * 6,
    "trial_type": ["face", "house"] * 3,
})
signals = [100 + rng.normal(size=(80, 3)) for _ in range(3)]
data = from_arrays(signals, [events.copy() for _ in signals], tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1., "house": -1.}},
    hrf_model="spm", noise_model="ols",
)
result = fit(data, model)
print(result.effect("face_gt_house").shape)  # (3,)
print(result.design_matrices[0].columns)
```

To use a prebuilt design, construct `PreparedDesignAnalysis.from_arrays` with
`column_roles` and pass it to `fit_prepared`; see the
[prepared-design API](api.md#prepared-designs) for role and diagnostic requirements.

### Select HRFs and fit one beta per trial

```python
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs

# One candidate keeps the demonstration quick; use a larger library in practice.
library = HrfLibrary.from_parameters([], origin={"kind": "review_example"})
selection = select_hrfs(data, library=library, feature_signature="example-order")
trials = fit_selected_hrfs(
    data, hrf_selection=selection, feature_signature="example-order",
    ridge_fraction=0.5,
)
print(trials.run_betas[0].shape)  # (6, 3)
print(trials.trial_table[["trial_id", "run_index", "event_index"]])
```

The signature is a caller-supplied spatial identity, not a substitute for aligning
the inputs. Use a signature derived from real spatial metadata for real data.
For a conventional selected-HRF fit, pass the same selection and signature to
`fit(data, model, hrf_selection=..., feature_signature=...)`.

### Score and choose fractional regularization

```python
from boldtailor.fractional_ridge import (
    score_fraction_candidates, select_ridge_fractions,
)

predictors = [pd.DataFrame({"face": [1., 0.] * 3}) for _ in signals]
scores = score_fraction_candidates(data, predictors, fractions=[1., 0.5])
chosen = select_ridge_fractions(scores)
print(scores.cv_r2.shape)  # (2, 3)
print(chosen.ridge_fraction)  # one selected fraction per feature
```

Here `library=None` uses canonical SPM. These scores choose a penalty; to estimate
performance independently, keep outer test runs out of this scoring call. Refit
with `fit_single_trials(..., ridge_fraction=chosen.ridge_fraction)` for the same
canonical HRF. Do not silently combine a canonical-HRF tuning result with a
different selected HRF and describe it as the same tuning procedure.

### Estimate a random-effect variance with REML

```python
from boldtailor.ar1 import estimate_rho
from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design

fixed = [np.ones((y.shape[0], 1)) for y in signals]
random = [rng.normal(size=(y.shape[0], 4)) for y in signals]
groups = [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]
rho = estimate_rho(signals, fixed)
variance_fit = fit_reml(groups, random, rho=rho.pooled)
print(variance_fit.status)
print(variance_fit.ratio.shape)  # (3,)
comparison = select_random_effects_design(groups, [random], rho=rho.pooled)
print(comparison.winner)  # candidate 0 or -1 if no eligible winner
```

This illustrates the call sequence with a reference intercept design. A real
analysis must justify its reference and random designs. Inspect `status`,
`at_upper`, `unrefined`, and the selection-wide flags before interpreting estimates.

## How to use the inventory during review

Open the [module index](api-review/README.md), choose a module, and jump to its
function or class. Signatures include source defaults and annotations; class
fields explain result storage. Properties are marked `@property` and are read
as attributes, not called. Dataclass `field(...)` declarations are construction
metadata, not literal user arguments. Deprecated aliases appear because they
remain in source; prefer the current names used above.

Docstrings are reproduced from source and may contain reStructuredText markup.
Where a definition has no docstring, its page says so and links to its body and
available call sites instead of inventing a contract. The handwritten routes and
existing API guide supply the higher-level usage context. Direct-call links are
static examples, not a complete runtime call graph. Nested definitions are shown
under their enclosing function and cannot be imported as module-level APIs.

For each reviewed path, record the intended estimand, input invariants, undefined
cases, training/test boundaries, and evidence from the relevant tests. Run a
focused suite with `uv run pytest tests/test_<area>.py`; use the
[developer guide](development.md#environment-and-checks) for broader checks.
When source changes, refresh signatures, definition counts and line links in
this snapshot alongside the affected narrative.

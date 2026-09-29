# NSD CIFTI analysis

This example analyzes every run in one Natural Scenes Dataset session using
fMRIPrep CIFTI time series. Start with the
[full workflow notebook](nsd_workflow.ipynb) for a guided analysis, or use the
scripts below for individual analyses.

| Analysis | What you get |
| --- | --- |
| Stimulus presentation plus response-time modulation | Full, confounds-only, and task-added R² maps |
| Single-trial model with the canonical SPM HRF | One beta map per presentation, OLS/fixed-ridge comparisons, and RT diagnostics |
| Single-trial model with an HRF selected per grayordinate | Optimized beta series, HRF parameter maps, independent prediction checks, and odd/even HRF maps |
| Matched conventional GLMs with canonical and optimized HRFs (notebook) | Task, response-time, and trial-type contrast maps, R², and optimized-minus-canonical R² |

See the project [user guide](../../docs/user-guide.md) for the underlying models
and the [API reference](../../docs/api.md) for using arrays directly.

## Data needed

The scripts expect BIDS event tables and matching fMRIPrep files for every run:

- `*_space-fsLR_den-91k_bold.dtseries.nii` and BOLD JSON sidecars.
- Confound TSV and JSON files.
- Event onsets, durations, and `response_time` for the RT analyses.

All runs need the same CIFTI spatial axis. The fMRIPrep directory must be
inside the BIDS root. Event durations and acquisition offsets are read from
the inputs; onsets are not rounded to whole volumes.

The defaults analyze `sub-07/ses-nsd10`, read `derivatives/fmriprep-25.2.5`,
and write to `derivatives/boldtailor` under the selected BIDS root.
That session has 12 runs, 750 trials, and 91,282 grayordinates. The data are
not included in this repository.

## Full workflow notebook

Open [nsd_workflow.ipynb](nsd_workflow.ipynb) in your notebook editor after
running `uv sync --group dev`, and select the checkout's `.venv` Python kernel.
Set `NSD_BIDS_ROOT` before starting the notebook kernel, then edit the first
code cell's scientific and execution settings and run all cells. For example:

```bash
export NSD_BIDS_ROOT=/path/to/NSD/BIDS
```

Alternatively, add a cell before setup defining
`NSD_CONFIG = {"bids_root": "/path/to/NSD/BIDS"}`. The session-reliability
notebook uses `HRF_RELIABILITY_CONFIG` instead. Both accept `fmriprep_root`
and `output_root` explicitly, or through `NSD_FMRIPREP_ROOT` and
`NSD_OUTPUT_ROOT`. Explicit dictionary paths override environment values;
otherwise derivative defaults follow the BIDS root. Missing BIDS configuration
raises an actionable error before loading or writing data. These helpers do
not read `.env` files. Standalone scripts retain their existing command-line
defaults; supply path flags for your data.

The
default uses all grayordinates and four workers. Set `max_grayordinates=128`
for a quick run through every stage using all runs and a small spatial subset.
The HRF reliability stages require at least two odd and two even runs;
optimized-HRF ridge CV needs at least three in each half. This
notebook requires matching nuisance column names and order after trimming;
it rejects mismatches before fitting.

The notebook walks through input inspection, two conventional GLMs, optimized
HRF selection, odd/even reliability, canonical and optimized single-trial
models, encoding-guided ridge CV, RT checks, and export. The conventional GLMs
fit three predictors jointly, without orthogonalization:

- `task`: unit amplitude for every presentation.
- `response_time`: seconds, centered within each run.
- `trial_type`: binary codes 0/1, centered within each run; the coefficient is
  type 1 minus type 0, controlling for RT.

By default, the notebook samples 512 continuous parameter combinations with
a scrambled Sobol sequence (seed 0), then adds canonical SPM for 513 HRFs.
This is the same parameter range as the original grid described below, with
36-second custom curves. Sampling balances coverage in parameter space;
different parameter combinations can still produce similar HRF shapes.
Set `hrf_n_samples` to a power of two and `hrf_seed` to a nonnegative integer.
The library cell plots every HRF, colored by time to peak, and runs separately
from the following selection cell so you can inspect it before fitting.

For the previous grid, set `hrf_library="expanded"`. Non-None
`hrf_parameters` overrides the library choice with your own parameter rows.
The metadata JSON saves these settings; the library TSV and NPZ save the
exact parameter values and curves used. When changing libraries, rerun
selection and all subsequent models with a new output root: HRF IDs are only
meaningful together with their original library.

Both GLMs use the same scans, motion/aCompCor/cosine columns, run intercepts,
and OLS settings. The notebook removes leading flagged nonsteady volumes
from every analysis and keeps original acquisition times and event onsets.
The scripts below instead include nonsteady indicator regressors and all
scans. Their R² values therefore need not agree with the notebook's values.

Results use `desc-notebook...` filenames under the chosen derivative root,
so existing script results are preserved. Reruns require a new output root.
Each conventional `stat-rsquared.dscalar.nii` contains three maps: full R²,
confounds-only R², and task-added ΔR². The
`desc-notebookGLMComparison_stat-deltarsquared.dscalar.nii` map contains
optimized minus canonical full R². This is a descriptive comparison using
the fitted data; independent half-session prediction scores are saved separately.

The notebook also saves contrast effects/variances/t/z, all/odd/even HRF
parameters and indices, the complete HRF library with time to peak, per-run
beta series, trial tables, grouped designs, scan times, provenance, and plots.
The final cells explain how to read the outputs and reuse the saved HRFs.
Section 7b compares full HRF curves at each grayordinate: odd versus even,
odd versus canonical SPM, and even versus canonical SPM. These are Pearson
correlations across the library's complete 0.1-second time grid, including
the undershoot, without shifting curves to align peaks. The canonical
comparisons provide a baseline for shared HRF shape. The three correlations
are saved in `desc-notebookHRFReliability_stat-curvecorrelation.dscalar.nii`.
The plots use the same grayordinates for every comparison, requiring both
half-session HRFs to be defined. This section can run from existing
`library` and `hrf_maps` variables without refitting.

Spatially unprocessed or undefined grayordinates remain NaN on the original
CIFTI axis. Beta-series RT plots are descriptive; all-run optimized HRFs use
both halves of the session.
Beta-series fitting prints one start and one completion message per model,
including when blocks run in parallel.
See the [notebook validation record](../../docs/validation/nsd-notebook.md)
for the fixture and real-data checks.

### Example helpers and the core library

NSD file discovery, CIFTI loading/export, surfaces, and plotting stay under
`examples/NSD/`; they are not installed as part of `boldtailor`. Run these
notebooks from a repository checkout with the development dependencies.
The core accepts arrays and scientific model specifications, independently
of the dataset and imaging format.

The notebooks keep model specification, HRF library construction, selection,
CV settings, and fit calls visible. `workflow_plots.py` and
`session_hrf_plots.py` handle figure details from already computed results;
they return figures without fitting, displaying, or writing files.
`notebook_paths.py` supplies local paths. Figure names and scientific exports
are unchanged. Direct helper tests cover plotted values and missing-data masks,
while notebook execution tests exercise fitting and export on synthetic CIFTI data.

### Cortical surface figures

The workflow notebook shows left/right lateral and medial views of full-model
BOLD R² for both conventional GLMs and beta-series models, plus signed beta–RT
correlations across all runs. The plots use existing fitted arrays, so the new
cells can run in an active notebook kernel without refitting models. R² includes
the confounds and pools errors across runs; it is distinct from the held-out
trial-encoding score. RT correlations center betas and RT within each run and
remain descriptive, particularly when RT helped tune ridge strength.

By default, the notebook uses the subject's `space-fsLR_den-32k_midthickness`
surfaces from fMRIPrep. Set `surface_meshes={"left": path, "right": path}` in
the settings to use another matching fsLR mesh, such as an inflated surface.
Native FreeSurfer and fsaverage meshes are not interchangeable with fsLR.
Multiple anatomical matches require explicit paths. No template is downloaded.
Set `surface_maps=False` to disable the figures; if neither matching bilateral
surface pair nor explicit paths are available, plotting is skipped with a message.

Values are placed using the CIFTI vertex IDs. Missing/unprocessed vertices and
triangles touching them are gray, including the medial wall; subcortical values
are omitted. R² uses a common 0–1 range, extended below zero if needed. RT uses a
shared symmetric range across the displayed models. No significance threshold
is applied. The final save cell writes `desc-notebookGLMR2Surface_plot.png`,
`desc-notebookBetaR2Surface_plot.png`, and `desc-notebookRTSurface_plot.png` beside
the CIFTI outputs. Resolved surface paths are retained in notebook metadata.

### Descriptive beta-series activation maps

Section 8d tests the mean trial beta against zero at every grayordinate for
each fitted beta-series model, including OLS and the selected ridge mode.
It pools finite trials across runs with equal weight per trial, without
within-run centering, and computes `t = mean / (sample_sd / sqrt(n))` with
`n - 1` degrees of freedom. **Trial betas are assumed independent.** This is
a descriptive activation-style summary relative to the fitted model baseline,
not an explicit task-versus-rest contrast. It ignores covariance among trial
estimates, shrinkage, and HRF/ridge selection uncertainty.

The notebook displays t histograms and, when matching surfaces are available,
signed cortical t maps using a shared symmetric scale. No significance threshold
is applied. For each model, `desc-notebook<model>_stat-activation.dscalar.nii`
contains `mean_beta`, `t`, `p_uncorrected` (two-sided), `n_trials`, and `df`.
Fewer than two finite trials or zero sample variance gives NaN t/p; no finite
trials gives NaN in all maps. The metadata records these assumptions and that
no multiple-comparison correction is applied. This cell uses existing fits;
rerun the save cell with a new output root to export the maps and figures.
See the [activation-map validation record](../../docs/validation/nsd-beta-activation-2026-09-28.md)
for numerical and notebook execution checks.

### Ridge selection and held-out encoding

The notebook defaults to `ridge_mode="fractional_cv"`, with candidate fractions
0.1 through 1. Each grayordinate selects the fraction maximizing its own pooled
held-out encoding R². Ties favor the larger fraction. The encoding model uses
separate training-run intercepts and shared binary-trial-type and RT slopes;
no repeated images are needed. The default `encoding_mode="within_run"` removes
each held-out run/feature mean residual for scoring only. Set
`NSD_CONFIG["encoding_mode"] = "absolute"` to reproduce shared-intercept fitting
and uncentered scoring. Inner and outer evaluations use the same mode.
HRFs are selected on the training runs within each fold using mean-stimulus
prediction. Training betas use the candidate fraction; validation targets are fixed OLS
betas under the training-selected HRFs.

A fraction of 1 gives OLS. Smaller fractions shrink the coefficient norm in the
raw trial-coefficient basis after nuisance projection. Confounds remain unpenalized,
and the exported betas retain native units. The chosen fraction is fixed across
runs, while the corresponding alpha is computed per run and grayordinate.
OLS estimates define both the fraction reference and the validation targets.

Odd-run tuning evaluates on even runs, then the roles reverse. Separate all-run
tuning supplies the final beta images. Canonical and optimized HRF models tune
independently. Missing predictors exclude encoding rows only. Locations need
finite scores for every candidate; undefined locations remain NaN.

Use `ridge_fractions` to change the candidate grid. `ridge_percentile` controls
a descriptive summary curve and does not select the fractions. The notebook
also plots the counts of grayordinates choosing each fraction.

The older modes remain available:

- `ridge_mode="cv"`: choose one shared alpha using the spatial percentile;
  `ridge_alphas` defaults to `[0, .001, .01, .1, 1, 10, 100]`.
- `ridge_mode="fixed"`: use a positive `ridge_alpha`.
- `ridge_mode="off"`: OLS alone.

An older injected `NSD_CONFIG` with only `ridge_alpha` retains fixed/off behavior.
An override with `ridge_alphas` and no mode retains shared-alpha CV.

Fractional beta files use `desc-notebookCanonicalTrialFractionalCV` and
`desc-notebookOptimizedTrialFractionalCV`. Tuning and outer-evaluation descriptors
contain `CanonicalFractionalCV` or `OptimizedFractionalCV`, followed by `Odd`,
`Even`, `All`, `OddToEven`, or `EvenToOdd`. Shared-alpha outputs retain `RidgeCV`.

| File suffix | Contents |
| --- | --- |
| `_scores.tsv` | Candidate fractions, descriptive percentile R², and counts selected |
| `_stat-ridgefraction.dscalar.nii` | Selected fraction per grayordinate, for tuning scopes and outer/final fits |
| `_stat-ridgealpha.dscalar.nii` | One implied-alpha map per run for outer/final fits |
| `_stat-encodingcvr2.dscalar.nii` | One inner-CV encoding R² map per candidate |
| `_stat-selectedencodingr2.dscalar.nii` | Each grayordinate's selected inner-CV score |
| `_stat-scoringmask.dscalar.nii` | Common eligibility mask across candidates |
| `_stat-foldhrfindex.dscalar.nii`, `_folds.npz` | Inner-training HRFs and validation SSE/SST |
| `_stat-encodingpredictionr2.dscalar.nii` | Independent outer-test encoding R² |
| `_stat-coefficients.dscalar.nii` | Training reference level (`task`), trial-type, and RT effects |
| `_predictions.dscalar.nii`, `_targets.dscalar.nii` | Original-order training-reference predictions (without scoring offsets) and OLS scoring targets (regularized targets for shared-alpha CV) |
| `_loss.npz` | SSE/SST, training-run intercepts and predictor means, scoring offsets, and ordered run labels |
| `_metadata.json`, `_provenance.json` | Run splits, transforms, norm definitions, sources and linked tuning identities |

`desc-notebookFractionalCV_predictors.tsv` saves exact predictors, original trial
IDs and excluded rows. `desc-notebookFractionalCV_tuning.png` saves the descriptive
score curves. The complete HRF library is saved with the notebook outputs.
In Python,
candidate scores expose `grid` and `regularization`; beta results expose
designs through `design.matrices`. Saved artifact keys remain unchanged; see
the [result migration guide](../../docs/result-migration.md).

RT and trial type help choose shrinkage, so final all-run RT correlations are
descriptive. Outer fractional scores assess prediction of noisy OLS beta targets fixed
across fractions within each split and HRF model. Targets can differ between
canonical and optimized HRFs or between splits. These scores do not measure
recovery of unobserved ground-truth responses. Conventional GLMs and the
across-session HRF reliability notebook are unchanged.

See the [fractional-ridge validation record](../../docs/validation/nsd-fractional-ridge.md)
for automated tests and the bounded real-data audit.

## HRF reliability across sessions

Use [nsd_session_hrf_reliability.ipynb](nsd_session_hrf_reliability.ipynb) to
compare independently estimated HRFs for `sub-07`, `ses-nsd10` through
`ses-nsd19`. All sessions use the same 513-candidate Sobol library by default.
The notebook fits only HRF selection, using all runs within each session and
the same confounds and nonsteady-volume trimming as the full workflow.
HRF selection needs only stimulus timing, so trials with missing reaction
times are retained; RT and trial type do not enter this selection model.

Completed session estimates are saved immediately under each session's `func`
directory. Reruns reuse them when the input identities, library, settings,
coverage, software versions, and spatial axis match. Matching outputs from
the full workflow can also be reused; add other derivative roots to
`reuse_roots` if needed. Older grid estimates are refitted separately. A
changed input or library receives a new cache name, and damaged caches are
recomputed. Source identity uses file paths, sizes, and modification times,
plus the prepared event/confound/timing values; it does not hash every BOLD
file's contents.

The primary outputs are full-HRF Pearson correlations at each grayordinate:
45 session-pair maps and 10 session-to-canonical maps. Each pair's canonical
baseline is the mean of its two session-to-SPM correlations. Summary maps
contain mean pairwise correlation, mean matched canonical baseline, their
mean difference, and the contributing session/pair counts. The notebook also
saves sample SD maps for the six varied HRF parameters and full-HRF time to
peak. Missing estimates remain NaN; counts indicate available data. These
are descriptive shape and parameter comparisons, not ICCs or tests treating
the 45 overlapping session pairs as independent observations.

Across-session CIFTIs, tables, figures, and metadata go under `sub-07/func/`
with `desc-sessionHRFReliability...` names. Per-session caches use
`desc-sessionHRF...` names and contain the exact library and selection
provenance. Sessions run sequentially, with `n_jobs` parallel blocks within
each session. Set `max_grayordinates=128` for a check across all ten sessions;
use `None` for the full spatial axis. This notebook does not fit beta series
or depend on ridge tuning.

## Run an analysis

From the repository root, install the example dependencies:

```bash
uv sync --group dev
```

For a stimulus-plus-RT model:

```bash
uv run python examples/NSD/nsd_cifti.py \
  --bids-root /path/to/NSD/BIDS \
  --fmriprep-root /path/to/NSD/BIDS/derivatives/fmriprep-25.2.5 \
  --output-root /path/to/NSD/BIDS/derivatives/boldtailor-conventional
```

For canonical-HRF beta series, with both OLS and fixed ridge:

```bash
uv run python examples/NSD/nsd_single_trial.py \
  --bids-root /path/to/NSD/BIDS \
  --ridge-alpha 0.1 \
  --output-root /path/to/NSD/BIDS/derivatives/boldtailor-singletrial
```

For optimized-HRF beta series and the HRF diagnostics:

```bash
uv run python examples/NSD/nsd_single_trial.py \
  --bids-root /path/to/NSD/BIDS \
  --hrf-library expanded --ridge-alpha 0.1 --n-jobs 4 \
  --output-root /path/to/NSD/BIDS/derivatives/boldtailor-expanded
```

Replace the example paths with your own. Omit path flags to use the local
defaults above. Existing output files are protected: use a new output root
when rerunning an analysis or comparing settings.

| Option | Default | Use |
| --- | --- | --- |
| `--bids-root` | `/Volumes/extdata1/NSD/BIDS` | Raw BIDS directory |
| `--fmriprep-root` | `<bids-root>/derivatives/fmriprep-25.2.5` | Preprocessed inputs |
| `--output-root` | `<bids-root>/derivatives/boldtailor` | Results directory |
| `--subject`, `--session` | `sub-07`, `ses-nsd10` | Select the session |
| `--block-size` | `4096` | Grayordinates processed together |
| `--hrf-library` | `canonical` | Single-trial script: choose `canonical` or `expanded` |
| `--ridge-alpha` | Omitted | Single-trial script: add fixed ridge to the OLS fit; must be positive |
| `--n-jobs` | `1` | Single-trial script: number of worker processes |

More workers trade memory for speed. A four-worker benchmark on this dataset
was about 2.5 times faster than serial fitting for 16,384 grayordinates, using
about 12.8 GB summed process memory. This is a measured example, not a memory
limit; details are in the [validation record](../../docs/validation/nsd-session.md#parallel-fitting-benchmark).
Use fewer workers or a smaller block size if memory is limited. Python callers
using several workers should call the runner inside an
`if __name__ == "__main__":` block.

## Model settings

Every model includes the same nuisance regressors, selected separately per run:

- 24 motion columns: six rigid-body parameters, derivatives, squares, and squared derivatives.
- The six retained combined-mask aCompCor components with the largest explained variance.
- All fMRIPrep cosine high-pass columns and non-steady-state indicators.
- A run intercept.

The cosine columns provide high-pass adjustment in the model. The scripts do
not additionally smooth, rescale, or filter the signals. They use the CIFTI TR
and each BOLD JSON's `StartTime`. Only missing first-volume motion derivatives
are replaced with zero; other missing selected confounds are errors.

The conventional model has a stimulus regressor and an RT amplitude modulator,
centered within each run. Both use the SPM HRF with 50-fold oversampling. The
RT column is not orthogonalized against the stimulus regressor. This model
requires a valid response time for every event.

Single-trial models instead fit one amplitude per event row. RT and image IDs
are retained as metadata. Missing RT does not remove a trial from the model.
OLS is always fitted; `--ridge-alpha` adds one fixed ridge fit. Nuisance
coefficients are unpenalized and betas retain the input signal scale.

## How optimized HRFs are selected

The expanded library has 649 candidates: canonical SPM plus 648 double-gamma
HRFs varying in delay, dispersion, undershoot, and onset. Each kernel is
normalized to sum to one. Selection predicts a shared mean stimulus response
across runs; final beta estimation allows every trial its own amplitude.

Three selections serve different purposes:

| Selection | Runs used | Purpose |
| --- | --- | --- |
| All-run HRF | Leave-one-run-out CV across the whole session | Final OLS and ridge beta series |
| Odd-run HRF | CV within odd-numbered runs | Independent prediction/RT checks on even runs; reliability comparison |
| Even-run HRF | CV within even-numbered runs | Reliability comparison with odd-run HRFs |

Within each selection, prediction errors are summed across folds before
choosing one HRF per grayordinate. Parameter maps contain that winner's
parameters, not averages of parameters across held-out runs. The all-run HRF
uses information from every run and is not independent of the final beta fits.

For the independent even-run prediction score, both the HRF and mean amplitude
are learned from odd runs and held fixed. Odd/even reliability selections use
only their own half's BOLD, while sharing timing/confound eligibility checks
across all runs. At least two runs are needed within a selection set. Unavailable
splits and undefined grayordinates have NaN outputs and recorded reasons.

## Finding your results

Outputs are placed in `<output-root>/<subject>/<session>/func/`. The tables
below omit the common subject/session/task/spatial prefixes.

### Conventional model

| Suffix | Contents |
| --- | --- |
| `desc-full_stat-rsquared.dscalar.nii` | Pooled full-model R² |
| `desc-confounds_stat-rsquared.dscalar.nii` | Pooled nuisance-only R² |
| `desc-task_stat-deltarsquared.dscalar.nii` | Full minus nuisance R² |
| `run-XX_desc-full_design.tsv` | Run design and frame times |
| `desc-model_metadata.json`, `desc-blocks_provenance.json` | Model settings and analysis records |

The conventional script exports fit diagnostics, not contrast or predicted
time-series images. Use the array API for contrasts.

### Beta series and RT diagnostics

Descriptors identify the model: `singletrialOLS` / `singletrialRidge` for the
canonical HRF, or `hrfOptOLS` / `hrfOptRidge` for optimized HRFs.

| Filename component or suffix | Contents |
| --- | --- |
| `run-XX_..._betas.dscalar.nii` | One scalar map per event, with trial IDs as map names |
| `_trials.tsv` | Event metadata and the mapping from trial IDs to image rows |
| `_stat-fullrsquared.dscalar.nii` | Pooled R² for that estimator |
| `_stat-confoundsrsquared.dscalar.nii` | Pooled nuisance-only OLS R² |
| `_stat-deltarsquared.dscalar.nii` | Full minus nuisance R² |
| `_stat-hrfdeltarsquared.dscalar.nii` | Optimized minus canonical full R²; expanded mode only |
| `_stat-rtcorrelation.dscalar.nii`, `_stat-rtcount.dscalar.nii` | RT correlations and counts: all, odd, even, then individual runs |
| `_metadata.json`, `_provenance.json` | Settings, numerical diagnostics, and fit records |

Canonical mode also saves `desc-singletrialOLS_selectedvertices.tsv` and
`desc-singletrialOLS_scatter.png` for the odd-to-even RT check. With ridge
enabled, the scatterplot compares OLS and ridge at the same selected vertices.

Canonical models save a design TSV per run. Expanded models save grouped
designs in `run-XX_desc-hrfSelection_designs.npz`, since different grayordinates
use different HRFs. The [developer guide](../../docs/development.md#nsd-exports-and-parallel-execution)
describes how to reconstruct those matrices.

### HRF maps

These use `desc-hrfSelection`:

| Statistic / suffix | Contents |
| --- | --- |
| `stat-hrfindex`, `stat-hrfparameters` | All-run selected IDs and parameters |
| `stat-oddhrfindex`, `stat-oddhrfparameters` | Odd-run selected IDs and parameters |
| `stat-evenhrfindex`, `stat-evenhrfparameters` | Even-run selected IDs and parameters |
| `stat-selectioncvr2`, `stat-canonicalcvr2`, `stat-deltacvr2` | All-run HRF selection score, canonical score, and difference |
| `stat-testr2`, `stat-canonicaltestr2`, `stat-deltatestr2` | Independent even-run prediction scores from odd-trained models |
| `_library.tsv`, `_curves.npz`, `_library.png` | HRF parameters, sampled curves, and library plot |
| `_folds.tsv`, `_eligibility.tsv` | Fold memberships and candidate design checks |
| `_metadata.json`, `_provenance.json` | Selection and prediction records |
| `_splitmetadata.json`, `_splitprovenance.json` | Odd/even selection settings and records |
| `_selectedvertices.tsv`, `_scatter.png`, `_selectedhrfs.png`, `_rt_provenance.json` | Independent RT check and selected-vertex HRFs |

All `stat-*` entries in this table are `.dscalar.nii` files. Parameter files
contain eight named maps: response delay, undershoot delay, response dispersion,
undershoot dispersion, response/undershoot ratio, onset delay, duration, and
`peak_time`. Peak time uses the maximum of the full HRF on its saved 0.1-second
grid, including the onset delay and undershoot. It differs from the response-delay
parameter.

Compare matching maps in the odd/even parameter images to assess reliability.
The numeric HRF IDs are categorical labels; use the library table to interpret
them. The standalone expanded-HRF script exports parameter maps for comparison.
The full workflow notebook computes odd/even curve correlations, and the
session-reliability notebook computes across-session curve correlations and
parameter variability. None of these outputs is an ICC.
Standalone all-run response-delay and time-to-peak files in the development
dataset were additional exports; new runs include both quantities in the
parameter images above.

## Interpreting the diagnostics

Full-model R² pools residual and total sums of squares across independently
fitted runs. Its denominator is the sum of within-run signal variance.
`deltarsquared` measures extra variance explained beyond confounds.
`hrfdeltarsquared` compares optimized and canonical HRFs using the same OLS or
ridge estimator. Positive values favor the optimized fit; negative values are
retained. These are all in-sample diagnostics.

The selection and independent-test scores instead measure prediction of a
mean stimulus response after nuisance adjustment. They have a different
denominator from full-model R². The winning selection-CV score was used to
choose among HRFs; use the separate test maps for independent prediction.

RT correlations use finite betas and positive, finite RTs, centered within each
run. Plot vertices are selected using canonical-OLS odd-run correlations;
scatterplots show even-run trials at those locations. Expanded plots use
odd-selected HRFs. Production all-run RT maps are descriptive and may differ
from these independent checks. No p-values are reported. These script-based
checks do not tune HRFs or ridge penalties; the notebook's encoding CV option
uses RT and trial type to tune ridge, with separate outer-test evaluation.

Measured results, numerical audits, and timings for the development session
are in [NSD validation](../../docs/validation/nsd-session.md). Development and
testing instructions are in the [developer guide](../../docs/development.md).

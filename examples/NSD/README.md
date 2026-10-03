# NSD CIFTI analysis

These notebooks analyze Natural Scenes Dataset sessions using fMRIPrep CIFTI
time series. Start with the [full workflow notebook](nsd_workflow.ipynb); the
other two notebooks reuse or extend its results. Older
[command-line scripts](#command-line-scripts) remain for a few script-only exports.

| Notebook | What you get |
| --- | --- |
| [Full workflow](nsd_workflow.ipynb) | Matched canonical and optimized-HRF GLMs (task, RT, trial type) with contrast, R², and optimized-minus-canonical R² maps; per-grayordinate HRFs with odd/even reliability; OLS, fixed-ridge, or ridge-CV beta series; RT correlations; activation maps |
| [Session HRF reliability](nsd_session_hrf_reliability.ipynb) | HRFs selected separately per session, with across-session curve agreement and parameter variability |
| [Multi-session comparison](nsd_multisession.ipynb) | Fits missing sessions with the full workflow, then summarizes paired optimized-minus-canonical changes across sessions |

See the project [user guide](../../docs/user-guide.md) for the underlying models
and the [API reference](../../docs/api.md) for using arrays directly.

## Data needed

The notebooks expect BIDS event tables and matching fMRIPrep files for every run:

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
Set `NSD_BIDS_ROOT` before starting the notebook kernel, then edit the
`overrides` in the first code cell and run all cells. Every setting and its
default is listed in `DEFAULTS` in [settings.py](settings.py);
`resolve_settings` combines them with your overrides and the data paths.
For example:

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
not read `.env` files.

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
fit these predictors jointly, without orthogonalization:

- `task`: unit amplitude for every presentation.
- `response_time`: positive finite seconds, centered over observed RTs within
  each run; unavailable RTs receive zero modulation.
- `trial_type`: binary codes 0/1, uncentered; the coefficient is type 1 minus
  type 0, controlling for RT, and the task coefficient is the response on
  trial_type 0 trials.
- `missing_response_time`: one for nonfinite or nonpositive RT, zero otherwise;
  included only in runs with unavailable RTs and convolved with the same HRF.

All presentations remain in the task and trial-type regressors. The extra
indicator allows the mean response on unavailable-RT trials to differ; the
task coefficient refers to observed-RT trial_type 0 trials at the run's mean
observed RT. Runs with no observed RT cannot estimate an RT effect
and are rejected. Complete runs retain their original three-predictor design.
The indicator is saved in the event tables and design matrices; contrast maps
remain task, RT, and trial type. Conventional task ΔR² includes the indicator's
contribution. HRF selection and beta estimation retain all trials; RT scoring
continues to exclude unavailable RTs.

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

Results use `desc-notebook...` filenames under the chosen derivative root.
Set `existing_results` in the notebook:

- `"reuse"` (default): reload saved results and skip fitting; regenerate notebook
  plots without modifying saved files.
- `"overwrite"`: refit and replace matching notebook outputs.
- `"error"`: stop before fitting if notebook outputs already exist.

With no saved results, all three modes fit and save normally. Reuse requires
matching analysis settings, run lengths, and spatial axes. It reads the saved
analysis rather than checking raw-file contents for changes; refit with
`"overwrite"` after changing input data. Incomplete saved results require a refit.

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

NSD file discovery (`workflow_files.py`), settings (`settings.py`), artifact
writers (`workflow_artifacts.py`), surfaces, and plotting stay under
`examples/NSD/`; they are not installed as part of `boldtailor`. Run these
notebooks from a repository checkout with the development dependencies.
Generic helpers the notebooks share live in the package: `boldtailor.cifti`
(dense-scalar export and checked reloads, cortical projection, feature
identity), `boldtailor.diagnostics` (one-sample t maps, RT correlations),
`boldtailor.reliability` (HRF curve agreement), and `boldtailor.parallel`
(bounded process batches).
The core accepts arrays and scientific model specifications, independently
of the dataset and imaging format.

The notebooks keep model specification, HRF library construction, selection,
CV settings, and fit calls visible. `boldtailor.workflow.plots` and
`session_hrf_plots.py` handle figure details from already computed results;
they return figures without fitting, displaying, or writing files.
`notebook_paths.py` supplies local paths. Figure names and scientific exports
are unchanged. Direct helper tests cover plotted values and missing-data masks,
while notebook execution tests exercise fitting and export on synthetic CIFTI data.

### Cortical surface figures

The workflow notebook shows left/right lateral and medial views of full-model
BOLD R² for conventional GLMs and task ΔR² (full minus confound-only R²) for
beta-series models, plus signed beta–RT
correlations across all runs. The plots use existing fitted arrays, so the new
cells can run in an active notebook kernel without refitting models. Both R²
components pool errors across runs using the same within-run total variance.
Task ΔR² measures the additional variance explained by trial regressors beyond
confounds; it is an in-sample measure, distinct from the held-out
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
are omitted. R² and ΔR² use a common 0–1 range, extended below zero if needed.
ΔR² uses a black–red–orange–yellow–white heat scale with a square-root color
progression to emphasize smaller values; colorbar
ticks remain in original ΔR² units. RT uses a
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
signed cortical t maps using a shared −10 to +10 scale, with values outside that
range shown in the endpoint colors. No significance threshold
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
HRFs are selected on the training runs within each fold using task-model
prediction. Inner-fold HRF selection inside the ridge cross-validation uses the
package default task-only model; the final beta series use HRFs selected with
the NSD task model. Training betas use the candidate fraction; validation targets are fixed OLS
betas under the training-selected HRFs.

Set `hrf_selection_rt=False` in the notebook settings to select HRFs without
the RT regressor. Selection then scores task and uncentered trial type only,
while the GLM still fits task, RT, and trial type; the selected-HRF fit accepts
this because the selection task model is a subset of the GLM's. The exported
`hrf_selection` metadata string and the session HRF cache identity record
which model selected the HRFs, and saved results with a different setting are
refitted rather than reused.

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
designs through `design.matrices` (canonical) or `design.matrix(run, hrf)`
(selected HRFs). Saved artifact keys remain unchanged; see
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

For combined HRF **and beta-series** comparisons, use
[nsd_multisession.ipynb](nsd_multisession.ipynb). It defaults to `sub-07`, sessions
10–19, and canonical versus optimized OLS and fractional-CV models. Set paths in
`NSD_MULTI_CONFIG` or the same `NSD_BIDS_ROOT` environment variable used above.
The notebook automatically runs the full single-session workflow for missing
sessions, sequentially in separate kernels. Completed sessions are reused;
each new session is saved before the next begins. This can be a long computation
with the full Sobol library and fractional CV. Set `fit_missing=False` to require
existing results instead. `analysis_config` supplies fitting overrides; otherwise
scientific settings inherit from the first completed session. Conflicting
settings, axes, or libraries fail explicitly rather than mixing analyses.

The notebook compares full HRF curves, without temporal shifts, for every session
pair at each grayordinate. It also computes **within-session optimized-minus-
canonical differences** in mean trial beta, descriptive task t, task-added ΔR²,
signed within-run beta–RT r, and absolute beta–RT r. Aggregate means give each
session equal weight and require two matched finite session values. Exported
maps include the number of contributing sessions, SD of changes, and fraction
with positive changes. A configurable grayordinate view shows its full HRFs and
canonical/optimized metric trajectories across sessions.

Section 3's cortical HRF map shows the mean **between-session correlation minus
matched canonical baseline**, with a zero-centered diverging scale. Each pair's
baseline averages its two session-to-canonical correlations. Positive values
indicate shape consistency above that baseline, not improved BOLD prediction.

Section 3a maps **mean HRF time-to-peak in seconds**, averaging valid selected-HRF
peak times equally across sessions (including canonical selections). Peak times
come from the full library curves, not their response-delay parameters or the
peak of a mean curve. One valid session is sufficient; no valid sessions gives
NaN. The sequential cortical map is saved as a PNG, and
`desc-multisessionHRF_stat-peaktime.dscalar.nii` stores both the mean and count.

Section 3b loads saved conventional GLM effect maps to show **mean task amplitude**
and **mean RT slope** across sessions. Each figure contains canonical, optimized,
and optimized-minus-canonical maps on a shared signed scale. Both models use the
same finite sessions at each location, with at least two required. Task effects
are in native signal units and RT slopes in native signal units per second;
these are descriptive coefficients, not t statistics. The corresponding means,
deltas, SDs, positive fractions, and session counts are exported as
`desc-multisessionGLM_stat-...` CIFTIs. Loading also validates the GLM scalar names
and spatial axes and includes those input maps in the source-file hashes.

OLS isolates HRF changes from ridge regularization; fractional-CV differences
also reflect separately selected penalties. RT contributes to tuning, so larger
RT associations are descriptive rather than independent evidence of improved
prediction. Mean-beta changes retain native units and can reflect session signal
scaling. Differences of descriptive t statistics are not themselves t tests.
Aggregate CIFTIs, TSVs, figures, and JSON source hashes use
`sub-07/func/*desc-multisession*`; reruns replace only these aggregate files.

For the smaller **HRF-only** workflow:

Use [nsd_session_hrf_reliability.ipynb](nsd_session_hrf_reliability.ipynb) to
compare independently estimated HRFs for `sub-07`, `ses-nsd10` through
`ses-nsd19`. All sessions use the same 513-candidate Sobol library by default.
The notebook fits only HRF selection, using all runs within each session and
the same confounds and nonsteady-volume trimming as the full workflow.
By default this notebook selects HRFs with the full task model (task, centered
RT, uncentered trial type); pass `include_rt=False` for stimulus-timing-only
selection. Trials with missing reaction times are retained.

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

## Run the notebooks

From the repository root, install the example dependencies:

```bash
uv sync --group dev
export NSD_BIDS_ROOT=/path/to/NSD/BIDS
```

Then open a notebook and select the checkout's `.venv` kernel. To run the
workflow notebook non-interactively, inject a configuration cell defining
`NSD_CONFIG` before the setup cell (the multi-session notebook does this for
each missing session). Keys and defaults are those of `DEFAULTS` in
[settings.py](settings.py); `resolve_settings(NSD_CONFIG)` returns the
settings the notebook uses:

| Setting | Default | Use |
| --- | --- | --- |
| `bids_root` | `NSD_BIDS_ROOT` | Raw BIDS directory (required) |
| `fmriprep_root` | `<bids_root>/derivatives/fmriprep-25.2.5` | Preprocessed inputs; or `NSD_FMRIPREP_ROOT` |
| `output_root` | `<bids_root>/derivatives/boldtailor` | Results directory; or `NSD_OUTPUT_ROOT` |
| `subject`, `session` | `sub-07`, `ses-nsd10` | Select the session |
| `block_size`, `max_grayordinates` | `4096`, `None` | Grayordinates processed together; optional spatial subset |
| `n_jobs` | `4` | Worker processes for feature blocks |
| `ridge_mode` | `fractional_cv` | `fractional_cv`, `cv` (global alpha), `fixed` (`ridge_alpha`), or `off` |
| `hrf_library` | `sobol` | `sobol` (`hrf_n_samples`, `hrf_seed`) or `expanded`; `hrf_parameters` rows override both |
| `existing_results` | `reuse` | `reuse`, `overwrite`, or `error` |

More workers trade memory for speed. A four-worker benchmark on this dataset
was about 2.5 times faster than serial fitting for 16,384 grayordinates, using
about 12.8 GB summed process memory. This is a measured example, not a memory
limit; details are in the [validation record](../../docs/validation/nsd-session.md#parallel-fitting-benchmark).
Use fewer workers or a smaller block size if memory is limited.

## Model settings

Every model includes the same nuisance regressors, selected separately per run:

- 24 motion columns: six rigid-body parameters, derivatives, squares, and squared derivatives.
- The six retained combined-mask aCompCor components with the largest explained variance.
- All fMRIPrep cosine high-pass columns.
- A run intercept.

Leading volumes flagged as non-steady-state are removed rather than modeled.
The cosine columns provide high-pass adjustment in the model. The notebooks do
not additionally smooth, rescale, or filter the signals. They use the CIFTI TR
and each BOLD JSON's `StartTime`. Only missing first-volume motion derivatives
are replaced with zero; other missing selected confounds are errors.

Single-trial models fit one amplitude per event row. RT and image IDs are
retained as metadata. Missing RT does not remove a trial from the model. OLS
is always fitted; `ridge_mode` adds fixed, alpha-CV, or fractional-CV ridge
fits. Nuisance coefficients are unpenalized and betas retain the input signal
scale.

## How optimized HRFs are selected

The expanded library has 649 candidates: canonical SPM plus 648 double-gamma
HRFs varying in delay, dispersion, undershoot, and onset. Every event's predicted response is scaled to a peak of one (kernels are
also stored at unit peak). A beta is therefore the peak BOLD response to that
presentation in signal units, independent of TR, oversampling, and event
duration, and comparable across grayordinates with different selected HRFs.
This matches GLMsingle's convention. Nilearn derivative and FIR bases and
user-supplied kernels are passed to Nilearn unchanged and keep its
sum-to-one scaling. Canonical and optimized betas therefore share a scale. Selection predicts each held-out run's task-model response
(task, centered RT, uncentered trial type) from the other runs; final beta estimation allows every trial its own amplitude.

Three selections serve different purposes:

| Selection | Runs used | Purpose |
| --- | --- | --- |
| All-run HRF | Leave-one-run-out CV across the whole session | Optimized GLM and final beta series |
| Odd-run HRF | CV within odd-numbered runs | Independent prediction on even runs; reliability comparison |
| Even-run HRF | CV within even-numbered runs | Independent prediction on odd runs; reliability comparison |

Within each selection, prediction errors are summed across folds before
choosing one HRF per grayordinate. Parameter maps contain that winner's
parameters, not averages of parameters across held-out runs. The all-run HRF
uses information from every run and is not independent of the final beta fits.

For each independent prediction score (odd to even, even to odd), both the HRF
and mean amplitude are learned from the training half and held fixed. Odd/even reliability selections use
only their own half's BOLD, while sharing timing/confound eligibility checks
across all runs. At least two runs are needed within a selection set. Unavailable
splits and undefined grayordinates have NaN outputs and recorded reasons.

## Finding your results

Outputs are placed in `<output_root>/<subject>/<session>/func/` with
`desc-notebook...` descriptors. The table omits the common
subject/session/task/spatial prefixes. All `stat-*` entries are `.dscalar.nii`
files on the original CIFTI axis; undefined grayordinates are NaN.

| Descriptor / suffix | Contents |
| --- | --- |
| `CanonicalGLM`, `OptimizedGLM` `_stat-effects`, `_stat-variances`, `_stat-t`, `_stat-z` | Task, RT, and trial-type contrast maps |
| `<model>_stat-rsquared` | Full, confounds-only, and task-added R² (GLMs and beta-series models) |
| `GLMComparison_stat-deltarsquared` | Optimized minus canonical full R² |
| `HRFAll`, `HRFOdd`, `HRFEven` `_stat-selection`, `_stat-hrfparameters` | Selected HRF IDs and CV scores; selected-HRF parameters including `peak_time` |
| `HRFOddToEven`, `HRFEvenToOdd` `_stat-prediction` | Independent selected, canonical, and Δ test R² |
| `HRFReliability_stat-curvecorrelation` | Odd/even and half/canonical full-curve correlations |
| `HRF_library.tsv`, `HRF_library.npz`, `HRF_provenance.json` | Exact library parameters and curves; selection records |
| `run-XX_..._<model>_betas` | One map per event, with trial IDs as map names |
| `<model>_trials.tsv`, `<model>_stat-rtcorrelation`, `<model>_stat-activation` | Trial table; all/odd/even RT correlations; one-sample t maps |
| `<model>_designs.npz`, `<model>_provenance.json` | Design matrices with column names and frame times, per run and HRF; fit records |
| `_runs.tsv`, `run-XX_..._events.tsv`, `run-XX_..._confounds.tsv` | Run summary, expanded events, and retained confounds with original frame indices |
| `_metadata.json`, `<figure>_plot.png` | Settings, model definitions, and figures |

Beta-series models are `CanonicalTrialOLS` and `OptimizedTrialOLS`, plus
`...TrialRidge`, `...TrialRidgeCV`, or `...TrialFractionalCV` for the chosen
`ridge_mode`. Ridge-CV tuning outputs are described in
[ridge selection](#ridge-selection-and-held-out-encoding). Session reliability
and multi-session outputs are described in
[HRF reliability across sessions](#hrf-reliability-across-sessions).

## Command-line scripts

The scripts predate the notebooks and are retained because they export some
results the notebooks do not: a stimulus-plus-RT-only conventional GLM,
per-run RT correlation and count maps, beta-series optimized-minus-canonical
R² (`hrfdeltarsquared`), selected-vertex RT tables and scatterplots, and
selected-HRF curve plots. They keep the nonsteady-state indicator regressors
and all scans, so their R² values need not agree with the notebooks'. They
use the same discovery and loading helpers (`workflow_files.py`) and the
package's diagnostics, CIFTI, and parallel helpers.

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

Replace the example paths with your own. Omit `--bids-root` only when
`NSD_BIDS_ROOT` is set; the other path flags default as below. Existing output files are protected: use a new output root
when rerunning an analysis or comparing settings.

| Option | Default | Use |
| --- | --- | --- |
| `--bids-root` | `NSD_BIDS_ROOT` | Raw BIDS directory (required) |
| `--fmriprep-root` | `NSD_FMRIPREP_ROOT`, else `<bids-root>/derivatives/fmriprep-25.2.5` | Preprocessed inputs |
| `--output-root` | `NSD_OUTPUT_ROOT`, else `<bids-root>/derivatives/boldtailor` | Results directory |
| `--subject`, `--session` | `sub-07`, `ses-nsd10` | Select the session |
| `--block-size` | `4096` | Grayordinates processed together |
| `--hrf-library` | `canonical` | Single-trial script: choose `canonical` or `expanded` |
| `--ridge-alpha` | Omitted | Single-trial script: add fixed ridge to the OLS fit; must be positive |
| `--n-jobs` | `1` | Single-trial script: number of worker processes |

The conventional model has a stimulus regressor and an RT amplitude modulator,
centered within each run. Both use the SPM HRF with 50-fold oversampling. The
RT column is not orthogonalized against the stimulus regressor. This model
requires a valid response time for every event.

Single-trial models instead fit one amplitude per event row. RT and image IDs
are retained as metadata. Missing RT does not remove a trial from the model.
OLS is always fitted; `--ridge-alpha` adds one fixed ridge fit. Nuisance
coefficients are unpenalized and betas retain the input signal scale.

### Script outputs

Outputs are placed in `<output-root>/<subject>/<session>/func/`. The tables
below omit the common subject/session/task/spatial prefixes.

#### Conventional model

| Suffix | Contents |
| --- | --- |
| `desc-full_stat-rsquared.dscalar.nii` | Pooled full-model R² |
| `desc-confounds_stat-rsquared.dscalar.nii` | Pooled nuisance-only R² |
| `desc-task_stat-deltarsquared.dscalar.nii` | Full minus nuisance R² |
| `run-XX_desc-full_design.tsv` | Run design and frame times |
| `desc-model_metadata.json`, `desc-blocks_provenance.json` | Model settings and analysis records |

The conventional script exports fit diagnostics, not contrast or predicted
time-series images. Use the array API for contrasts.

#### Beta series and RT diagnostics

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

#### HRF maps

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
Task-added ΔR² measures extra variance explained beyond confounds.
`GLMComparison_stat-deltarsquared` compares optimized and canonical HRFs in
the same GLM. Positive values favor the optimized fit; negative values are
retained. These are all in-sample diagnostics.

The selection and independent-test scores instead measure prediction of the
task-model response after nuisance adjustment. They have a different
denominator from full-model R². The winning selection-CV score was used to
choose among HRFs; use the separate test maps for independent prediction.

RT correlations use finite betas and positive, finite RTs, centered within each
run. The RT plot's grayordinate is selected using canonical-OLS odd-run
correlations; the scatterplots show even-run trials at that location. All-run
RT maps are descriptive. No p-values are reported. With
`hrf_selection_rt=True`, RT enters HRF selection as a task-model regressor;
the encoding CV options use RT and trial type to tune ridge, with separate
outer-test evaluation.

Measured results, numerical audits, and timings for the development session
are in [NSD validation](../../docs/validation/nsd-session.md). Development and
testing instructions are in the [developer guide](../../docs/development.md).

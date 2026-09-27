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

The defaults analyze `sub-07/ses-nsd10` under `/Volumes/extdata1/NSD/BIDS`,
read `derivatives/fmriprep-25.2.5`, and write to `derivatives/boldtailor`.
That session has 12 runs, 750 trials, and 91,282 grayordinates. The data are
not included in this repository.

## Full workflow notebook

Open [nsd_workflow.ipynb](nsd_workflow.ipynb) in your notebook editor after
running `uv sync --group dev`, and select the checkout's `.venv` Python kernel.
Edit the first cell's paths and execution settings, then run all cells. The
default uses all grayordinates and four workers. Set `max_grayordinates=128`
for a quick run through every stage using all runs and a small spatial subset.
The reliability stages require at least two odd and two even runs. This
notebook requires matching nuisance column names and order after trimming;
it rejects mismatches before fitting.

The notebook walks through input inspection, two conventional GLMs, optimized
HRF selection, odd/even reliability, canonical and optimized single-trial
models, optional fixed ridge, RT checks, and export. The conventional GLMs
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
them. The current pipeline does not calculate reliability coefficients for you.
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
from these independent checks. No p-values are reported and RT does not tune
HRFs or ridge penalties.

Measured results, numerical audits, and timings for the development session
are in [NSD validation](../../docs/validation/nsd-session.md). Development and
testing instructions are in the [developer guide](../../docs/development.md).

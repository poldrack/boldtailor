# NSD single-session CIFTI example

`nsd_cifti.py` fits all 12 runs of `sub-07/ses-nsd10` with the current
boldtailor prepared-design API. The example loads and writes CIFTIs with
NiBabel; it does not require a CIFTI adapter in the package.

From this directory, using the repository's uv environment:

```bash
uv run python nsd_cifti.py
```

The defaults read `/Volumes/extdata1/NSD/BIDS` and its
`derivatives/fmriprep-25.2.5` directory, then publish results under
`/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor`. Override paths with
`--bids-root`, `--fmriprep-root`, and `--output-root`. The fMRIPrep directory
must remain within the BIDS root so source provenance can use dataset-relative
paths. Relative overrides are resolved from the working directory. `--subject`, `--session`,
and `--block-size` are also available. All raw NSD runs in the selected session
must have matching fsLR 91k CIFTIs, events, confounds, and JSON sidecars.
Existing result files are protected from overwriting; use a new output directory
when rerunning an analysis.

## Model

Each run has independent coefficients for:

- All stimulus presentations, using the recorded onsets and 3-second durations.
- A response-time amplitude modulator in seconds, centered within each run.
  Both task regressors use the SPM canonical HRF, sampled at 50-fold
  oversampling. The RT column is not orthogonalized against stimulus presentation.
- The six rigid-body motion parameters, their temporal derivatives, their
  squares, and the squared derivatives (24 columns).
- The six retained **combined-mask** aCompCor components with greatest
  explained variance, selected from each confound JSON. In this session these
  are `a_comp_cor_00` through `a_comp_cor_05`.
- Every fMRIPrep cosine high-pass regressor and non-steady-state indicator.
- A run intercept.

Frame times use the CIFTI TR and the BOLD JSON `StartTime`, accounting for
slice-timing correction. These data use 188 volumes with TR 1.6 seconds;
`StartTime` is 0.774 seconds for runs 02, 09, 11, and 12, and 0.775 seconds
for the others. Each run's exact JSON value is used. The example does not
smooth, scale, or separately filter the signals.
Only missing first-volume motion derivatives are replaced with zero; missing
response times or other selected confounds cause an explicit error.

The cosine regressors implement high-pass adjustment inside the model and
match the preprocessing used for aCompCor. See the
[fMRIPrep output documentation](https://fmriprep.org/en/latest/outputs.html)
for its cosine, aCompCor, and `StartTime` conventions, and
[Nilearn's regressor documentation](https://nilearn.github.io/dev/modules/generated/nilearn.glm.first_level.compute_regressor.html)
for HRF convolution.

## R² and outputs

The full model is fit with `fit_prepared(..., noise_model="ols")`.
`task_delta_r2_prepared` compares it with the nested model containing the same
nuisance columns and intercepts. These are **in-sample OLS** fit diagnostics.
For each grayordinate, pooled R² is:

```text
1 - sum_run(SSE_run) / sum_run(sum_time((y_run - mean_time(y_run))²))
```

This pools residual and total sums of squares across independently fit runs;
it is not an arithmetic average of run-wise R². Run mean differences are
excluded from the denominator. ΔR² is `full R² - confounds-only R²`, expressing
the additional fraction of original within-run signal variance explained
jointly by the stimulus and RT terms. It is not partial R² or a cross-validated
score. Undefined values at grayordinates with no within-run variance are NaN.

Results are saved in `sub-07/ses-nsd10/func/`:

| Filename suffix | Content |
| --- | --- |
| `desc-full_stat-rsquared.dscalar.nii` | Pooled full-model R² |
| `desc-confounds_stat-rsquared.dscalar.nii` | Pooled confounds-only R² |
| `desc-task_stat-deltarsquared.dscalar.nii` | Raw full minus confounds-only R² |
| `run-XX_desc-full_design.tsv` | Run design, including frame times |
| `desc-model_metadata.json` | Model specification, runs, software versions, definitions |
| `desc-blocks_provenance.json` | Canonical boldtailor provenance for each fitted block |

Each map has a JSON sidecar and preserves the input BrainModel axis. A derivative
`dataset_description.json` is created if needed. Publication uses boldtailor's
transactional `publish_artifact_set`. No time-series predictions or contrast
maps are exported. Feature blocks bound memory use without changing the fit.

## Tests

```bash
uv run pytest test_nsd_cifti.py -q
```

Tests generate real miniature CIFTI datasets and compare the saved maps with
independent NumPy least-squares calculations, including unequal run variances,
different run means, and an all-zero grayordinate.

## Single-trial betas and independent RT checks

The simple single-trial alternative uses a fixed canonical SPM HRF and fits
one amplitude per event row. Repeated image IDs remain separate trials.
There is no GLMsingle repeated-stimulus cross-validation, adaptive HRF,
data-driven denoising selection, or behavioral tuning.

From this directory:

```bash
uv run python nsd_single_trial.py --ridge-alpha 0.1
```

OLS is always fitted. Omitting `--ridge-alpha` produces only OLS; a positive
value adds one fixed ridge comparison. The NSD comparison uses the predeclared
value **0.1**, not an RT-selected value or a fractional-ridge fraction. The
same path, subject, session, and block-size flags as the conventional example
are supported. Use a new output root to rerun or compare another fixed alpha.

`--n-jobs N` enables parallel feature-block fitting in both single-trial modes;
the default `1` retains serial execution. For example:

```bash
uv run python nsd_single_trial.py --hrf-library expanded --ridge-alpha 0.1 \
  --n-jobs 4 --output-root /Volumes/extdata1/NSD/BIDS/derivatives/boldtailor-parallel
```

Each process fits a separate grayordinate block using all runs, preserving the
same cross-validation folds. Worker processes limit numerical-library threads
to one to avoid competing thread pools. The parent assembles blocks in their
original order and publishes only after every fit succeeds. At most `N` blocks
are dispatched together; each process keeps its own timing/design cache and
loads only its assigned CIFTI columns. RAM grows with the number of workers,
while the parent still retains the complete output maps. The metadata's
`Execution` field records the requested/effective workers, block size, backend,
and worker thread limit. A single available block uses the serial path.

The Python API accepts `run_single_trial_analysis(..., n_jobs=4)`. Scripts that
use multiple processes should call it inside `if __name__ == "__main__":`.
Worker counts must be positive integers. No automatic core or memory sizing is
performed. Existing output files remain protected; use a fresh output root.

On the 16-core, 128-GiB development machine, a real-data benchmark using all
12 runs, all 649 HRFs, and 16,384 grayordinates measured:

| Workers | Fitting time | Speedup | Sampled peak process-tree RSS |
| --- | ---: | ---: | ---: |
| 1 | 186.19 s | 1.00× | 5.43 GB |
| 2 | 104.66 s | 1.78× | 9.19 GB |
| 4 | 74.22 s | 2.51× | 12.77 GB |

Four workers are a reasonable starting point on this machine. These timings
include process startup and cold design caches, OLS/ridge fitting, canonical
comparisons, and block-level odd/even checks. They exclude final artifact
assembly/publication and cover four 4096-feature blocks, not the entire
session. RSS is the sampled sum across the parent and children (decimal GB);
shared pages may be counted more than once. Every benchmark map and trial beta
matched serial execution exactly. See the
[parallel validation record](../../docs/superpowers/validation/2026-09-26-nsd-parallel.md).

The package entry point is:

```python
from boldtailor.single_trial import fit_single_trials

result = fit_single_trials(data, ridge_alpha=0.1)  # data is AnalysisData
betas = result.run_betas  # one trials × features array per run
```

The model uses raw event onsets and durations, the same selected nuisance
regressors as above, and a run intercept. It keeps sub-TR timing and models
positive durations as unit-height boxcars; zero-duration events use Nilearn's
sampled impulse convention. RT and image identity are metadata only. Missing
or nonpositive RT does not exclude a trial from fitting.

For ridge, trial columns and signals are projected off the nuisance span;
trial columns are normalized to unit Euclidean norm before applying the
penalty. Betas are then restored to native signal units, and nuisance
coefficients are fitted without a penalty. Rank-deficient residual trial
designs and designs without residual degrees of freedom are rejected.

New files use `desc-singletrialOLS` or `desc-singletrialRidge`:

| Suffix | Content |
| --- | --- |
| `run-XX_..._betas.dscalar.nii` | Trial beta maps, with trial IDs on the ScalarAxis |
| `run-XX_..._design.tsv` | Trial columns, confounds, intercept, and frame times |
| `_trials.tsv` | Original event metadata plus global trial index, run index, event index, and trial ID |
| `_stat-fullrsquared.dscalar.nii` | Actual estimator's pooled in-sample R² |
| `_stat-confoundsrsquared.dscalar.nii` | Pooled nuisance-only OLS R² |
| `_stat-deltarsquared.dscalar.nii` | Raw full minus nuisance R² |
| `_stat-rtcorrelation.dscalar.nii` | RT correlations: all, odd, even, then individual runs |
| `_stat-rtcount.dscalar.nii` | Matching valid trial counts |
| `_metadata.json`, `_provenance.json` | Settings, numerical diagnostics, and block provenance |
| `desc-singletrialOLS_selectedvertices.tsv` | Up to five cortical vertices selected using odd-run OLS |
| `desc-singletrialOLS_scatter.png` | Even-run scatterplots at those same vertices for both estimators |

Trial map row `event_index` (zero-based within run) matches its row in the
trial table; `trial_index` is zero-based across the session. Spatial axes
are copied exactly from the input. Constant run signals have NaN trial betas
and contribute zero SSE/SST to pooled R². Existing conventional outputs and
dataset metadata are preserved. The complete artifact set is assembled in
memory before transactional publication, so beta outputs remain a memory
cost even though fitting uses feature blocks.

For each feature, RT and beta are centered within each run using the same
finite-beta/finite-positive-RT mask. Correlations pool these centered values;
fewer than three valid trials or zero variance gives NaN. Numeric BIDS run
parity defines odd and even sets. Vertices are selected by absolute odd-run
OLS correlation, with lower grayordinate indices breaking ties; plots show
only even-run values, centered within run. These are descriptive reality
checks, not significance tests. Native beta scales can differ across runs,
and temporal dependence affects interpretation. Strong RT associations are
not guaranteed, and the checks never select HRFs or penalties.

```bash
uv run pytest test_nsd_cifti.py test_nsd_single_trial.py test_rt_diagnostics.py -q -W error
```

### Verified sub-07/ses-nsd10 run

The 2026-09-26 run fitted all 12 runs, 750 trials, and 91,282 grayordinates
with OLS and fixed ridge alpha 0.1. It published 66 new files under
`/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor/sub-07/ses-nsd10/func/`.
Checksums confirmed that all 21 pre-existing files were preserved.

| Pooled diagnostic (median across defined grayordinates) | OLS | Ridge 0.1 |
| --- | ---: | ---: |
| Full R² | 0.818498 | 0.799512 |
| Confounds-only R² | 0.569213 | 0.569213 |
| ΔR² | 0.222574 | 0.205088 |

There are 525 grayordinates with undefined pooled R². Single-trial models
have many more task coefficients than the conventional model, so increased
in-sample R² alone is not evidence of better generalization.

The five vertices selected by odd-run OLS had even-run correlations
`0.163, 0.164, 0.172, -0.154, 0.173`; fixed ridge at the same vertices gave
`0.225, 0.217, 0.222, -0.139, 0.187`. All retained their odd-run sign.
The spatial correlation between odd/even cortical RT maps was 0.657 for OLS
and 0.750 for ridge. These are modest, directionally consistent RT associations,
not significance estimates or proof that ridge is generally superior. The
scatterplots show a sparse long-RT tail; no outliers were removed or used to
choose a penalty.

An independent NumPy audit reconstructed 30 sampled grayordinates (including
the five selected vertices) across every run using least squares and augmented
least squares. Maximum absolute beta differences were below `7.7e-6` native
units, and pooled R² errors below `3e-8`, consistent with float32 storage.
The audit also checked exact imaging axes, trial identities, RT correlations,
valid-trial counts, and odd-only selection. The measured run took 100.57 seconds
with 2.57 GB maximum resident memory (decimal GB, macOS `/usr/bin/time -l`).

## Expanded HRF selection across runs

```bash
uv run python nsd_single_trial.py --hrf-library expanded --ridge-alpha 0.1
```

This mode selects one HRF per grayordinate from 649 candidates: the exact
existing Nilearn SPM kernel (ID 0), plus the supplied GLMsingle notebook's
648 double-gamma combinations. Kernels have discrete sum one. The library
table lists every parameter and ID; exported curves use 0.1-second sampling,
while convolution uses TR/50 and the original acquisition offsets.

Selection uses one stimulus-presentation regressor per candidate and run.
For each held-out run, the common mean amplitude is learned on the other
runs and kept fixed for prediction. Nuisances are projected out within each
run. Scores pool held-out squared errors and nuisance-adjusted signal energy,
without standardizing runs. Negative scores remain visible. This requires
the mean response to transfer between runs; it requires neither repeated
images nor equal responses to individual presentations. Final trial betas
are free amplitudes estimated by OLS or fixed ridge using the same HRF map.

The best all-run CV score is a **selection statistic**. A separate evaluation
selects HRFs within odd runs, learns the mean amplitude from those runs, and
predicts even runs with both fixed. It needs at least two odd runs and one
even run; otherwise the metadata records its absence and test maps are NaN.
RT does not enter HRF or ridge selection. The independent RT plot chooses
vertices from canonical OLS odd-run correlations, then fits their even-run
trial betas using odd-selected HRFs. Production RT maps use all-run selected
HRFs and are descriptive. Neither correlation plot provides significance tests.

New outputs coexist with the earlier models:

| Descriptor / suffix | Content |
| --- | --- |
| `desc-hrfOptOLS`, `desc-hrfOptRidge` | Trial betas, trial mapping, pooled in-sample full/confound/ΔR², descriptive RT maps and provenance |
| `desc-hrfOptOLS_stat-hrfdeltarsquared`, `desc-hrfOptRidge_stat-hrfdeltarsquared` | Per-grayordinate optimized full R² minus matching canonical full R², with JSON sidecars |
| `desc-hrfSelection_stat-hrfindex` | Stable all-run HRF IDs; NaN where undefined |
| `...stat-hrfparameters` | Selected parameters and peak times |
| `...stat-selectioncvr2`, `canonicalcvr2`, `deltacvr2` | All-run selection score and canonical comparison |
| `...stat-oddhrfindex` | HRFs selected within odd runs |
| `...stat-testr2`, `canonicaltestr2`, `deltatestr2` | Independent even-run mean-prediction scores |
| `...library.tsv`, `curves.npz`, `library.png` | Candidate identities and sampled HRFs |
| `run-XX_desc-hrfSelection_designs.npz` | Each used HRF's float64 trial matrix, shared nuisance matrix, frame times and column IDs |
| `...folds.tsv`, `eligibility.tsv`, `metadata.json`, `provenance.json` | Exact folds, lazy structural checks, definitions and source identities |
| `...selectedvertices.tsv`, `scatter.png`, `selectedhrfs.png`, `rt_provenance.json` | Independent RT diagnostics and the odd-run HRFs used |

Load design archives with `np.load(path, allow_pickle=False)`. For HRF ID `h`,
the full design is `np.column_stack([saved[f"hrf_{h}"], saved["nuisance"]])`.
`trial_columns`, `nuisance_columns` and `hrf_ids` are string arrays. No candidate
may discard trials: proposed winners must support every trial with positive
residual degrees of freedom. Eligibility is checked lazily; `unchecked` is
distinct from eligible or excluded. Numerically zero nuisance-adjusted signals
have no selected HRF. No weak-signal fallback or significance threshold is used.

The `hrfdeltarsquared` maps compare **single-trial in-sample** fits pooled
across all runs: `optimized_full_r2 - canonical_full_r2`. Positive values
favor the optimized HRF, negative values favor the canonical HRF, and NaN
means either score is undefined or the canonical design is ineligible.
Expanded runs fit the canonical baseline with the same trials, confounds,
and estimator (OLS or the same fixed ridge alpha); no prior canonical output
is required. These maps differ from `deltarsquared` (task versus confounds)
and `deltatestr2` (independent mean-prediction comparison). Their sidecars
record the formula, fitting provenance and any unavailable-comparison reason.

The array API is available from `boldtailor.hrf_library`,
`boldtailor.hrf_selection` and `boldtailor.single_trial`:

```python
from boldtailor.hrf_library import expanded_hrf_library
from boldtailor.hrf_selection import select_hrf, evaluate_hrf_split
from boldtailor.single_trial import fit_selected_hrfs

library = expanded_hrf_library()
selection = select_hrf(data, library=library, feature_signature=axis_signature)
result = fit_selected_hrfs(data, selection=selection, feature_signature=axis_signature)
evaluation = evaluate_hrf_split(data, library=library, train_runs=[0, 2],
                                test_runs=[1, 3], feature_signature=axis_signature)
```

The NSD wrapper hashes the ordered BrainModel axis and block indices. Array
callers may omit the signature, but must then preserve feature order themselves.
Results expose grouped designs keyed by `(run_index, hrf_id)`, because different
features can have different designs. Existing canonical-only calls are unchanged.

If the canonical trial model is structurally ineligible in an odd run, expanded
fitting can still succeed. The independent RT diagnostic then has no selected
vertices, and metadata records `IndependentRTAvailable=False` and the reason.
Design archives contain the HRFs used in all-run production fits. A candidate
used only in the independent RT diagnostic may require reconstruction from the
exported library, frame times, trial timing, and shared nuisance matrix.

### Verified expanded sub-07/ses-nsd10 run

The 2026-09-26 expanded analysis fitted all 12 runs, 750 trials, and 91,282
grayordinates. It published 72 new files in the same derivatives directory;
all 87 earlier files were checksum-preserved. Runtime was 720.54 seconds
(12.01 minutes), with 6.39 GB peak resident memory. The run used 4096-feature
blocks and candidate batches of 32, without changing the library or ridge
penalty after evaluation.

| Median nuisance-adjusted mean-prediction R² | Selected HRF | Canonical SPM | Paired ΔR² |
| --- | ---: | ---: | ---: |
| All-run selection CV (used to choose HRFs) | 0.005495 | 0.000952 | 0.003686 |
| Independent even runs (odd-trained HRF and amplitude) | 0.002465 | 0.001078 | 0.000924 |

Each column is a separate median; the median paired difference need not equal
the difference of medians. Independent prediction improved at 57.93% of defined
grayordinates. There were 525 undefined HRFs and 622 distinct selected candidates.
The improvement is small and varies across locations; the larger selection-CV
gain is not an independent estimate of predictive benefit.

| Median pooled single-trial diagnostic | Optimized OLS | Optimized ridge 0.1 |
| --- | ---: | ---: |
| Full R² | 0.812488 | 0.799636 |
| Confounds-only R² | 0.569213 | 0.569213 |
| ΔR² | 0.219646 | 0.207429 |

The mean-prediction selection target does not maximize single-trial in-sample
R²; the optimized OLS median is slightly lower than the canonical OLS median.
At the same five canonical odd-RT-selected vertices, independent even-run
correlations were `0.229, 0.197, 0.197, -0.102, 0.150` for optimized OLS and
`0.251, 0.247, 0.213, -0.107, 0.135` for optimized ridge. Three associations
strengthened and two weakened relative to their canonical counterparts;
all five retained their odd-run sign. These are descriptive checks.

An independent audit reconstructed 32 grayordinates across every run using
direct convolution, explicit held-out predictions, and NumPy ordinary/augmented
least squares. Maximum beta error was `3.80e-6` native units; R² errors were
below `3e-8`, consistent with float32 storage. It also verified the saved
designs, exact CIFTI axes, all 750 trial identities, RT vertex selection and
even-run correlations. Library and diagnostic plots were visually checked.
See the [validation record](../../docs/superpowers/validation/2026-09-26-hrf-selection.md)
for audit details, implementation decisions, and the remaining archive limitation.

The subsequently added `hrfdeltarsquared` maps were generated from the saved
full-model R² images without refitting. Matching axes, all 750 trial rows,
confounds, source runs, and estimator settings were verified first. Every
grayordinate was checked against direct subtraction, with 525 undefined values
per map. Median optimized-minus-canonical R² was −0.002160 for OLS (34.36%
positive) and +0.001326 for ridge (59.61% positive). The two maps and their JSON
sidecars preserve all 160 files present before this addition.

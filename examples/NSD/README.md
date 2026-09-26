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
slice-timing correction. For these data they are `0.775 + 1.6 * arange(188)`
seconds. The example does not smooth, scale, or separately filter the signals.
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

# NSD Replication Protocol

## 1. Windows and subjects

- **Pilot:** sub-07, ses-nsd10–ses-nsd19 (10 sessions).
- **Primary replication:** subjects 01–04, ses-nsd01–ses-nsd10 (10 sessions each).
- **Extension:** subjects 05, 06, 08, ses-nsd01–ses-nsd10 (10 sessions each).

## 2. Metric definitions

*Metric definitions are taken from the paper (PMC9708069, checked 2026-10-05). Each metric is z-scored per grayordinate within the session.*

### R1. Voxel test-retest reliability

Betas are z-scored per grayordinate within each session. Use images with ≥ 3 presentations in the window, taking each image's first three.

- **Reliability:** the mean over k ∈ {1, 2, 3} of the Pearson r, across images, between presentation k and the mean of the other two.
  - *Deviation:* the paper lists two of the three splits. We use all three, which is the "all possible unique split-halves" the paper describes.
- **Masks:** composite reliability (the mean over the compared versions) ≥ t, for t = −0.2:0.05:0.6, within the ROI.
- **Plotted:** for each version and t, the mean over masked grayordinates of (version reliability − composite).
- **Reported:** the median ROI reliability per version.

### R3B. Out-of-sample restriction

Restrict to images whose first three presentations fall in three distinct sessions. This follows the paper's rule of removing repeats inside one GLMsingle partition (one session).

### R4. Pattern correlation over time

For each session, the trial × trial Pearson correlation of ROI patterns, restricted to grayordinates with composite reliability ≥ 0 and ≥ 0.3.

- **Summary:** the mean correlation over same-run trial pairs at lag L = 1…15 trials, with L × 4 s on the x-axis, averaged over sessions.
- *Deviation:* the paper shows full session RSMs. We summarise within runs, because between-run time gaps vary.

### R5. RDM correlation across subjects

For each subject, the RDM is 1 − Pearson r between repetition-averaged ROI patterns of the shared images. Score each subject pair by the Pearson r of the RDMs' upper triangles. Grayordinate masks: composite ≥ 0, 0.2 and 0.4.

- *Deviation:* the paper used 241 images shared with BOLD5000. We use the NSD images that every primary subject saw at least three times in the window.

### R6. Image decoding

Linear SVM (`sklearn.svm.LinearSVC`, default C, one-vs-rest) over images with ≥ 3 presentations. Three folds: train on two presentations, test on the third. Report accuracy and chance (1 / number of classes). Masks: composite ≥ 0, 0.1, 0.2, 0.3 and 0.4, skipped when fewer than 10 grayordinates remain.

### R2. HRF consistency

For each session, b2's HRF index map. Consistency is measured with `boldtailor.reliability.compare_hrfs` over the window's sessions.

## 3. Ladder and settings

### Boldtailor ladder

All fits are per session and use the public API, not `run_workflow`. The workflow has no denoising stage. Levels b2–b4 reuse one HRF selection per session, as GLMsingle does.

| Level | Boldtailor |
|---|---|
| b1 | canonical HRF, OLS single-trial (`fit_single_trials`) |
| b2 | `select_hrfs` + `fit_selected_hrfs` with `default_hrf_library()` |
| b2-lib20 | same, with `glmsingle_hrf_library()`, to separate the selection rule from the library |
| b3 | b2 + `select_denoising` (defaults, gate on), its components added to each run's confounds |
| b4 | b3 + per-grayordinate fractional-ridge CV (workflow defaults) |
| LSS-assume, LSS-fit | least-squares-separate with the canonical HRF and with the b2 HRFs (R3 only) |

### Settings

- **Task model:** `task`, `response_time`, `trial_type`, as the workflow detects them for NSD.
- **Confounds:** GLMsingle polynomial drift basis per run: powers 0..d of `linspace(-1, 1, n)`, orthonormalised, with `d = alt_round(n * tr / 60 / 2)` (3 for NSD runs). Data-driven noise regressors come from boldtailor's denoising stage at b3.
- **Fraction grid:** `(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)` (WorkflowSettings default).
- **Encoding mode:** `within_run`.
- **Denoising:** `select_denoising` defaults.

## 4. ROI and data decisions

### No GLMsingle runs

The comparator is NSD's released GLMsingle betas, converted by the user to CIFTI under `BIDS/derivatives/betas-fsLR`:
- `desc-assumehrf` (b1), `desc-fithrf` (b2) and `desc-fithrfGLMdenoiseRR` (b4). NSD released no GLMdenoise-only version, so b3 has no released counterpart.
- Files: `sub-NN/ses-nsdYY/func/sub-NN_ses-nsdYY_task-nsdcore_space-fsLR_den-91k_desc-<version>_stat-effect_statmap.dscalar.nii`. Each holds 750 maps named `trial-001`…`trial-750`, in presentation order.
- Units: percent signal change. NSD's ×300 integer scaling has already been undone.
- Grid: fsLR 91k. The cortical part is exactly the ppdata grayordinates (same names and vertices).
- Source: resampled from NSD's native-surface betas with adaptive barycentric resampling.
- Also present: subject-level NSD NCSNR maps per version (reference only, not our metric).

### Data

Boldtailor fits NSD's own preprocessed series (ppdata, `func1pt8mm`), resampled by the user to fsLR 32k on the layer-B2 surface.
- Files: `derivatives/ppdata/subjNN/func1pt8mm/timeseries/sub-NN_ses-nsdYY_task-nsdcore_run-ZZ_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`.
- Timing: TR 4/3 s, 226 volumes per run.
- Grid: cortex only, 59,412 grayordinates, the same vertices as the released betas' cortex.
- Why: this is the preprocessing the released betas (and the paper) used, so the comparison isolates the GLM pipeline.
- Onsets: BIDS onsets map onto the ppdata time base by a constant `onset_offset`, calibrated once in the pilot by maximising the alignment check over candidate offsets.

### Events

The existing BIDS `events.tsv` files.

### Cortex only

Every fit and every analysis uses only cortical surface grayordinates: the CIFTI brain-model structures `CIFTI_STRUCTURE_CORTEX_LEFT` and `CIFTI_STRUCTURE_CORTEX_RIGHT`.
- Subcortical volume grayordinates are dropped when inputs are loaded, before any fitting.
- Whole-cortex summaries mean exactly this set: 59,412 grayordinates.

### ROI

Each subject's own nsdgeneral, from NSD's native-surface labels (`derivatives/freesurfer-NSD/subjNN/label/{lh,rh}.nsdgeneral.mgz`, 0/1 on the native vertices).
- Resampling: the labels are resampled to fsLR 32k with exactly the Workbench route used for the released betas: `wb_command -metric-resample ... ADAP_BARY_AREA` with area metrics.
- Then: the result is thresholded at 0.5 and mapped onto the cortical grayordinates.
- Deviation: the paper used the same subject-specific ROI, but in volume space rather than on the surface.

## 5. Repeat counts

Sub-07 pilot data (ses-nsd10–ses-nsd19):
```
7500 trials
Repeat counts: {1: 2308, 2: 1345, 3: 834}
Out-of-sample images: 88
```

## 6. Margin δ, replication criteria, alignment floor

Set in Task 12 before the freeze.

## 7. Deviations log

(empty at protocol draft)

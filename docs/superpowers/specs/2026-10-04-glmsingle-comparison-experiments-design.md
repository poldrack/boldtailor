# GLMsingle comparison experiments

## Goal

Run a set of experiments on the Natural Scenes Dataset (NSD) that parallel
Prince et al. (2022), "Improving the accuracy of single-trial fMRI response
estimates using GLMsingle" (*eLife* 11:e77599), and show two things:

1. Boldtailor's single-trial betas are on par with GLMsingle's at each stage
   of the paper's beta-version ladder, under matched inputs.
2. Boldtailor's additional capabilities work on real data: tuning without
   repeated conditions, cross-validated HRF selection, task modulators, and the
   denoising significance gate.

The experiments live in a standalone package in this repository, outside
`src/`. The core package gains no dependency on GLMsingle and no knowledge of
anatomy.

## Decisions

- **Comparator.** We run GLMsingle (Python package) ourselves on exactly the
  same time series, events, and confounds as boldtailor. NSD's released
  GLMsingle betas are not used, because they are not in fsLR space.
- **Data.**
  - Primary: fMRIPrep CIFTI (`space-fsLR_den-91k`).
  - Replication: NSD's own preprocessed series (`ppdata`, `func1pt8mm`)
    resampled to fsLR 32k (`desc-layerB2`), found under
    `BIDS/derivatives/ppdata/subjNN/func1pt8mm/timeseries/`.
  - Events: the existing BIDS `events.tsv` files for both.
- **Staging.**
  - Pilot: sub-07, which already has full fMRIPrep derivatives.
  - Confirmatory: all eight subjects, once their derivatives exist.
  - The pilot sessions are outside the confirmatory window.
- **TR grid.** NSD onsets are every 4 s and the fMRIPrep TR is 1.6 s, so half
  the onsets fall between volumes. GLMsingle needs onsets on the volume grid.
  For the primary comparison, both packages therefore get identical series
  upsampled to 0.8 s. A secondary analysis runs boldtailor on the native
  1.6 s series.
- **ROI.** The primary ROI is NSD's `nsdgeneral`, transformed from fsaverage
  to fsLR 32k.
- **"On par".** Defined by an equivalence test (TOST). Its margin is fixed
  from pilot variability and recorded in a frozen protocol before any
  confirmatory fit.
- **Feature experiments.** All four are included:
  - tuning without repeats;
  - HRF selection quality;
  - task modulators;
  - the denoising significance gate.

## Package layout (`experiments/glmsingle_comparison/`)

The package follows the conventions of `examples/validation`:
- tests are opt-in and outside the default `pytest` run;
- every `__init__.py` is empty;
- functions are short and written test-first.

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclass: BIDS root, subjects, pilot/confirmatory session windows, output root, conditions, `n_jobs`; loaded from a TOML file |
| `inputs.py` | Load one session's runs (BOLD, events, confounds) for a given source (`fmriprep`, `ppdata`) into arrays |
| `resample.py` | Per-run temporal resampling of BOLD and continuous confounds to a target TR; onset-grid check |
| `confounds.py` | Matched-minimal (GLMsingle polynomial basis) and fMRIPrep confound sets |
| `roi.py` | nsdgeneral fsaverage → fsLR 32k → grayordinate mask for a given CIFTI axis |
| `fit_boldtailor.py` | Boldtailor ladder levels via the public API |
| `fit_glmsingle.py` | GLMsingle ladder levels, converted to the common beta format |
| `betas.py` | Common beta store: read/write, per-session z-scoring |
| `metrics.py` | NCSNR, noise ceiling, split-half reliability, RDM reliability, lagged trial correlation |
| `features.py` | Feature-experiment analyses (no-repeats, HRF, modulators, gate) |
| `stats.py` | Per-subject summaries, TOST, descriptive CIs |
| `figures.py` | Figures and the results report |
| `run.py` | CLI: `uv run python -m experiments.glmsingle_comparison.run <stage> --config ...` with stages `prepare`, `fit`, `metrics`, `figures`; fits are resumable, so existing outputs are skipped |

Dependencies go in a new `experiments` uv dependency group: `glmsingle`,
`neuromaps`, and whatever they require. Task 1 checks whether neuromaps' label
transform needs Connectome Workbench (`wb_command`). If it does, that is
documented as a system prerequisite.

## Inputs

### Sessions

- **Pilot:** sub-07, `ses-nsd11`–`ses-nsd20`. Debugging iterations may use two
  sessions; the pilot metrics use all ten.
- **Confirmatory:** subjects 01–08, `ses-nsd01`–`ses-nsd10`.

Every subject completed at least 30 sessions, so both windows exist for
everyone. The pilot reports how many images have two or three presentations
inside a ten-session window. If too few images are repeated for stable NCSNR,
the windows are widened in the protocol before the freeze.

### Events

- **Columns.** Onsets, durations, the NSD image identifier (73k ID), and the
  columns the boldtailor task model uses (`response_time`, `trial_type`).
- **Condition labels.** For GLMsingle, a condition is a 73k image ID within
  the session.
- **Repeats.** Images repeated across sessions are matched by 73k ID during
  metric computation.

### Resampling for the primary comparison

- **Before resampling.** Non-steady-state volumes are trimmed first, using the
  rules the workflow already applies.
- **Method.** BOLD is resampled per run with cubic interpolation, along time
  only, to TR 0.8 s on a grid whose origin is the first retained volume.
  Continuous confound columns are resampled the same way.
- **Onset-grid check.** `resample.py` checks that every onset, measured on the
  run's acquisition time base, lies on the 0.8 s grid within 1 ms.
  - If the origin is offset by a constant, the grid is shifted to absorb it.
    The shift is recorded.
  - Otherwise the session fails with an error naming the run and the onsets
    that don't fit.
- **ppdata.** No resampling: the series are already at 1.0 s. The same onset
  check runs at 1.0 s.
- **Storage.** Resampled series are cached under the output root, so both
  packages read the same bytes.

### Confound conditions

- **Matched-minimal.**
  - Contents: GLMsingle's polynomial drift basis for each run (degree from run
    duration, as GLMsingle computes it). Boldtailor receives it as its
    confounds; GLMsingle builds it itself.
  - Oracle: `confounds.py` is tested against GLMsingle's own polynomial
    function on the same run lengths.
- **fMRIPrep.**
  - Contents: the NSD example's confound set (24 motion columns, six aCompCor
    components, cosines; non-steady-state handled by trimming).
  - Delivery: given to boldtailor directly and to GLMsingle through its
    extra-regressor option.
  - Scope: primary data only.

### ROI

- **Construction.** NSD's `lh/rh.nsdgeneral` fsaverage labels are transformed
  to fsLR 32k by nearest-neighbour label resampling (neuromaps), then mapped
  onto each CIFTI's cortical grayordinates. Subcortical grayordinates are
  excluded.
- **Deviation from the paper.** The paper used subject-specific nsdgeneral in
  functional volume space. This is a group fsaverage ROI. The reason is that
  the comparison runs in fsLR, where only the group label can be transformed
  directly.
- **Whole cortex.** Summaries are reported as a secondary ROI.

## Comparison ladder

Each package fits each session separately, as NSD ran GLMsingle. Every level
is run under each confound condition.

| Level | GLMsingle | Boldtailor |
|---|---|---|
| b1 | assumed HRF, OLS | canonical HRF, OLS single-trial (`fit_single_trials`) |
| b2 | HRF from its 20-HRF library by in-sample R² | `select_hrfs` + `fit_selected_hrfs`, run twice: `default_hrf_library()` (**b2**) and `glmsingle_hrf_library()` (**b2-lib20**) |
| b3 | b2 + GLMdenoise | b2 + `select_denoising` / `with_denoising` (gate on, defaults) |
| b4 | b3 + fractional ridge (repeat-based CV) | b3 + per-grayordinate fractional ridge CV (encoding CV, workflow defaults) |

- **GLMsingle settings.** Package defaults, except the settings required by
  the data: stimulus duration 3 s, TR 0.8 s (1.0 s for ppdata), and outputs
  kept in memory.
  - b2–b4 come from one GLMsingle call: its type-B (FITHRF), type-C
    (FITHRF_GLMDENOISE) and type-D (FITHRF_GLMDENOISE_RR) outputs.
  - b1 is a second call with the HRF library, GLMdenoise and fractional ridge
    all turned off. That call's type-B output uses the assumed HRF, matching
    the paper's AssumeHRF version.
  - Type A (ON-OFF) is not a ladder level.
  - The exact option names are verified in Task 1 against the installed
    version and recorded in `fit_glmsingle.py`.
- **Boldtailor task model.** The one the workflow detects for NSD: `task`,
  `response_time`, `trial_type`.
- **HRF reuse.** HRF selection runs once per session and confound condition.
  Levels b2–b4 reuse it, as GLMsingle does.
- **Secondary.** Boldtailor b1–b4 on native 1.6 s data, under fMRIPrep
  confounds, for the confirmatory subjects.

### Common beta format

Each fit writes one directory:

```
<output_root>/fits/sub-XX/ses-YY/<package>_<level>_<confounds>_<data>/
    betas.npy        float32, trials × grayordinates
    trials.tsv       run, trial index in run, onset, 73k image ID
    metadata.json    package + version, settings, inputs digest, runtime, git commit
```

- **Converter test.** The GLMsingle converter is tested on a small synthetic
  session: trial order and image IDs must round-trip.
- **Storage estimate.**
  - Per fit: about 275 MB per session at 91k grayordinates.
  - Primary fits: 9 per session and confound condition (GLMsingle b1–b4;
    boldtailor b1, b2, b2-lib20, b3, b4). Across 2 conditions × 80 sessions,
    that is about 400 GB.
  - Other fits: secondary, feature and ppdata fits add roughly another
    150 GB.
  - Requirement: the output root needs about 600 GB.
  - Disposal: `run.py fit --discard-betas-after-metrics` may delete a fit's
    betas once its metrics are written, keeping `trials.tsv` and the metadata.

## Metrics

All metrics use betas z-scored per grayordinate within each session, as NSD
does. Each metric has a unit test against a hand-computed oracle.

- **NCSNR (primary).**
  - Computation: for each grayordinate, pooled within-image variance (ddof = 1)
    across all images with at least two presentations in the window. Noise SD
    is its square root; signal SD is `sqrt(max(0, 1 − noise variance))`; NCSNR
    is signal SD / noise SD.
  - Noise ceiling: `100 · ncsnr² / (ncsnr² + 1/3)`.
  - Summary: per-subject median within the ROI.
- **Split-half reliability.** Correlation, per grayordinate, between images'
  first-presentation betas and the mean of their later presentations.
- **Amount of data.** NCSNR and split-half reliability recomputed for windows
  of the first 1, 2, … 10 sessions.
- **RDM reliability.** Correlation-distance RDMs over images with three
  presentations within the ROI, one from first presentations and one from the
  mean of the rest. Score: Spearman correlation of their upper triangles.
- **Lagged trial correlation.** Mean ROI pattern correlation between betas of
  different-image trials in the same run, as a function of trial lag (1–10).
  This measures the temporal-adjacency artifact the paper reports.

## Statistics

- **Primary hypothesis.** At b4 under matched-minimal confounds on primary
  data, boldtailor's per-subject median ROI NCSNR is equivalent to GLMsingle's.
  - Test: paired TOST on the relative difference (boldtailor − GLMsingle) /
    GLMsingle across the eight subjects, α = 0.05, with margin ±δ.
- **Secondary.** The same test at b1–b3, under fMRIPrep confounds, and for the
  ppdata replication. These are reported with their TOST p-values, labelled
  secondary.
- **Everything else** is descriptive: per-subject points, the subject mean,
  and a 95% CI, with no further hypothesis tests.
- **Margin δ.** The pilot report gives session-to-session and split-to-split
  variability of median ROI NCSNR for both packages. The protocol fixes δ with
  a written justification, the smallest difference judged meaningful given
  that variability, before the freeze.

## Feature experiments

1. **Tuning without repeats.**
   - Setup: GLMsingle is given a unique condition label per trial, so it sees
     no repeats.
   - Comparison: whatever GLMsingle then produces, compared with boldtailor's
     unchanged b4, whose tuning never uses image identity. Both are scored on
     the true repeats with the metrics above. What GLMsingle does without
     repeats (which stages run or are skipped, and any warnings) is recorded
     from its outputs.
   - Data: primary data, matched-minimal confounds.
2. **HRF selection quality.**
   - Held-out prediction: for the same 20-HRF library, each package's chosen
     HRF per grayordinate (GLMsingle's selection; boldtailor b2-lib20) is
     scored by leave-one-run-out task-model prediction with one shared scorer
     built on boldtailor's public API.
   - Isolating the choice: both choices go into identical OLS single-trial
     fits (same confounds, no denoising, no ridge). The resulting NCSNR
     difference isolates the choice itself.
   - Library effect: b2 versus b2-lib20 measures the effect of the larger
     library.
   - Stability: agreement of chosen HRFs across session pairs, via
     `compare_hrfs` curve correlations.
3. **Task modulators.**
   - Setup: boldtailor b4 with the task model `task` only, versus `task` +
     `response_time` + `trial_type`. The task model is used in HRF selection,
     denoising scoring, and ridge encoding.
   - Outcomes: NCSNR and split-half reliability, and the trial-level
     correlation between ROI betas and RT that the workflow already computes.
   - Reporting: the events sidecar's description of `trial_type` is quoted in
     the report.
4. **Denoising gate.**
   - Null control: each run's events are circularly shifted by a random offset
     of at least 30 s, which keeps the design structure but breaks alignment
     with the BOLD. Seeds are recorded.
   - Runs: `select_denoising` with the gate on and off, on the real and
     null-shifted pilot and confirmatory sessions.
   - Outcomes: the number of PCs chosen and how often the gate rejects.
     Expected: on nulls, pcstop alone picks PCs and the gate rejects them.
   - GLMsingle comparison: GLMsingle's own PC counts on the same sessions are
     reported alongside.

## ppdata replication

- **Scope.** Levels b1–b4, both packages, matched-minimal confounds, all eight
  subjects, `ses-nsd01`–`ses-nsd10`, TR 1.0 s.
- **Onset check.** Same as above. ppdata timing differs from fMRIPrep's (NSD
  ppdata are slice-time corrected and upsampled to 1.0 s). If a constant
  offset relates BIDS onsets to the ppdata time base, Task 1 derives it once,
  documents it, and checks it on every run.
- **Grid.** The ppdata CIFTIs are 32k cortical surfaces, so the ROI mask is
  built for that grid.

## Protocol and freeze

- **`experiments/glmsingle_comparison/protocol.md` contains:**
  - session windows;
  - conditions;
  - every package setting;
  - the margin δ and its justification;
  - the primary hypothesis;
  - the list of secondary analyses;
  - the figure list.
- **Freeze.** The file is committed with the pilot report and tagged
  `glmsingle-comparison-protocol-v1` before any confirmatory fit.
- **After the freeze.** Any change to code or settings that affects
  confirmatory results is logged in `protocol.md` as a dated deviation with
  its reason, and its results are reported alongside the frozen version's.

## Deliverables

- **Pilot report:** `docs/validation/glmsingle-comparison-pilot-<date>.md`.
  - All metrics for sub-07, `ses-nsd11`–`ses-nsd20`.
  - The repeat counts.
  - The variability estimates behind δ.
  - Runtime per fit.
- **Final report:** `docs/validation/glmsingle-comparison-<date>.md`.
  - Primary and secondary results.
  - Feature experiments and the ppdata replication.
  - The deviations from the paper and from the protocol.
- **Figures paralleling the paper's:**
  - ROI NCSNR by ladder level for both packages;
  - fsLR surface maps of NCSNR differences;
  - reliability versus amount of data;
  - RDM reliability;
  - lagged trial correlation;
  - one figure per feature experiment.
- **`docs/glmsingle-comparison.md`.** Its "no matched comparison has been run"
  statements are updated to cite the final report.

## Errors

| Condition | Where | Behaviour |
|---|---|---|
| onsets off the resampling grid (beyond constant shift) | `resample.py` | error naming run and onsets |
| missing run file, events, or confounds for a configured session | `inputs.py` | error naming subject, session, run |
| confound names differ across runs | `confounds.py` | error (as the workflow) |
| ROI label has no overlap with the CIFTI axis | `roi.py` | error |
| GLMsingle output missing an expected version | `fit_glmsingle.py` | error naming the version; the fit is not written |
| a fit directory exists with a different inputs digest | `run.py` | error unless `--refit`; never silently reused |

## Testing

- **Default suite.** Unaffected. All tests here are opt-in, following the
  `examples` convention.
- **Unit tests with oracles:**
  - NCSNR, noise ceiling, split-half, RDM, and lagged correlation on small
    arrays with known values;
  - the polynomial basis against GLMsingle's function;
  - resampling (onset-grid check with on-grid, constant-shift, and off-grid
    cases);
  - z-scoring;
  - the beta-format round-trip;
  - the circular-shift null (preserves durations and counts, minimum offset).
- **End-to-end test.** A synthetic session (a few hundred features, repeated
  images, known HRFs) runs both packages' b1–b4 and the metrics. Checks:
  - fits exist with the correct trial order;
  - NCSNR rises from b1 to b2 for both packages on data simulated with a
    non-canonical HRF.
- **ROI test.** On a synthetic label, skipped when neuromaps or Workbench is
  unavailable.

## Task outline (for the implementation plan)

1. Install the `experiments` group. Verify against the installed packages:
   GLMsingle's API (accepted data shapes, option names for the extra
   regressors, version outputs, behaviour without repeats) and neuromaps'
   label-transform requirements. Check sub-07's events against the 0.8 s grid
   and derive the ppdata time-base relation. Record the findings in the
   package README.
2. Config, inputs, resampling, confound sets.
3. ROI construction.
4. Common beta format and boldtailor ladder driver.
5. GLMsingle ladder driver and converter.
6. Metrics.
7. Pilot run (sub-07), pilot report, protocol with δ, freeze tag.
8. Feature experiments 1–4.
9. Confirmatory run, statistics, figures, final report; ppdata replication.

Steps 7 and 9 are compute runs started by the user. The plan gives their
commands and expected runtimes rather than running them inside an
implementation task.

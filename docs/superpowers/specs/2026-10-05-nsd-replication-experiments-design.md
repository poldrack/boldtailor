# Replicating the GLMsingle paper's NSD results with boldtailor

## Goal

Replicate, with boldtailor, the patterns that Prince et al. (2022) report on
the Natural Scenes Dataset (NSD). The paper is "Improving the accuracy of
single-trial fMRI response estimates using GLMsingle," *eLife* 11:e77599.
The goal is to show that boldtailor's single-trial pipeline produces the same
qualitative gains the paper attributes to GLMsingle. Then:

- compare boldtailor directly with NSD's released GLMsingle betas, on matched
  preprocessing and grayordinates;
- demonstrate boldtailor's additional capabilities.

We do not run GLMsingle. The comparison uses the betas NSD distributes.

## The paper's NSD analyses (replication targets)

The paper analyses NSD subjects 01–04, `nsd01`–`nsd10`. Betas are z-scored
within session, and all analyses use the nsdgeneral ROI. The beta versions
are:
- b1: assumed HRF;
- b2: fitted HRF;
- b3: b2 + GLMdenoise;
- b4: b3 + fractional ridge.

| ID | Paper | Analysis | Reported pattern |
|---|---|---|---|
| R1 | Fig. 2 | Voxel test-retest reliability across the three presentations of each image, summarised across voxel-reliability thresholds (r = −0.2 to 0.6) | Each stage b1→b4 adds reliability; gains grow with voxel reliability |
| R2 | Fig. 2 suppl. | HRF choice per voxel | Smooth spatial gradient across visual cortex; choices consistent across sessions |
| R3 | Fig. 3 | b1–b4 versus least-squares-separate (LSS) with assumed and fitted HRF; repeated with voxel selection and evaluation on separate data partitions | Ridge is more reliable than LSS, most clearly in reliable voxels; unchanged out of sample |
| R4 | Fig. 5 | Pattern correlation between trial pairs in temporal order, averaged over sessions, at voxel-reliability thresholds 0 and 0.3 | b1: up to r ≈ 0.5 between consecutive trials, decaying to 0 over about 100 s; b4 decays much faster |
| R5 | Fig. 6 | Correlation between subjects' RDMs over shared images, built from repetition-averaged betas | b4 strengthens agreement between subjects |
| R6 | Fig. 7 | Linear SVM decoding of image identity: train on n − 1 presentations, test on the held-out one | Mean accuracy roughly triples from b1 to b4 |

The paper's comparisons with BOLD5000 and StudyForrest, and its cross-dataset
RSA, are out of scope. The animacy MDS illustration (Fig. 7B) is optional and
not part of the protocol.

Task 1 reads the paper's Methods and records each metric's exact definition
in `protocol.md` before any code computes it. That covers:
- how voxel reliability is computed from the three presentations;
- which version's reliability defines the threshold masks;
- how the out-of-sample partitions are formed;
- the decoding image set and its folds.

Where the paper is ambiguous, the protocol records the choice made and why.

## Decisions

- **No GLMsingle runs.** The comparator is NSD's released GLMsingle betas:
  `betas_assumehrf` (b1), `betas_fithrf` (b2), and
  `betas_fithrf_GLMdenoise_RR` (b4). NSD released no GLMdenoise-only version,
  so b3 has no released counterpart.
- **Primary data.** fMRIPrep CIFTI (`space-fsLR_den-91k`) at the native TR of
  1.6 s, with exact onsets. No temporal resampling.
- **Matched data.** NSD's own preprocessed series (`ppdata`, `func1pt8mm`)
  resampled to fsLR 32k on the layer-B2 surface:
  `BIDS/derivatives/ppdata/subjNN/func1pt8mm/timeseries/sub-NN_ses-nsdYY_task-nsdcore_run-ZZ_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`.
  The user converts the released func1pt8mm betas to the same fsLR 32k
  layer-B2 CIFTI space with the same pipeline. The spec treats the
  conversion as an external input step and records it in the protocol.
- **Events.** The existing BIDS `events.tsv` files, for both data sources.
- **Staging.**
  - Pilot: sub-07, `nsd01`–`nsd10`. Sub-07 is not one of the paper's
    subjects, and its fMRIPrep derivatives already exist.
  - Primary replication: subjects 01–04, `nsd01`–`nsd10`, as in the paper.
  - Extension: subjects 05, 06 and 08, same window. Sub-07 is reported as
    the pilot.
- **Order of runs.** The fMRIPrep stages run when each subject's derivatives
  are available. The ppdata comparison can start once the ppdata series and
  converted betas are on disk.
- **ROI.** nsdgeneral, from NSD's fsaverage labels transformed to fsLR 32k
  (neuromaps nearest-neighbour). The 91k CIFTIs use its cortical part.
  - Deviation: the paper used subject-specific nsdgeneral in volume space.
  - If the user also converts each subject's func1pt8mm nsdgeneral with the
    layer-B2 pipeline, that subject-specific ROI is used for the ppdata
    comparison instead, and the group ROI becomes the fallback.
- **"On par"** is an equivalence test (TOST) on the matched ppdata
  comparison. Its margin is fixed from pilot variability before the freeze.
- **Feature experiments:**
  - HRF selection quality;
  - task modulators;
  - the denoising significance gate.

  Tuning without repeats is a property of every boldtailor result here, not
  a separate experiment. No boldtailor stage uses image identity, while the
  released b4 used repeats to tune GLMdenoise and ridge. The parity
  comparison is the evidence.

## Boldtailor ladder

All fits are per session and use the public API, not `run_workflow`. The
workflow has no denoising stage and fixes its confounds. Levels b2–b4 reuse
one HRF selection per session and data source, as GLMsingle does.

| Level | Boldtailor |
|---|---|
| b1 | canonical HRF, OLS single-trial (`fit_single_trials`) |
| b2 | `select_hrfs` + `fit_selected_hrfs` with `default_hrf_library()` |
| b2-lib20 | same, with `glmsingle_hrf_library()`, to separate the selection rule from the library |
| b3 | b2 + `select_denoising` / `with_denoising` (defaults, gate on) |
| b4 | b3 + per-grayordinate fractional-ridge CV (workflow defaults) |
| LSS-assume, LSS-fit | least-squares-separate with the canonical HRF and with the b2 HRFs (R3 only) |

- **Task model.** `task`, `response_time`, `trial_type`, as the workflow
  detects them for NSD.
- **Confounds for fMRIPrep data.** The NSD example's set: 24 motion columns,
  six aCompCor components, cosines, and trimmed non-steady-state volumes.
- **Confounds for ppdata.** The polynomial drift basis GLMsingle uses: degree
  `round(run duration in minutes / 2)`, per run. This matches what produced
  the released betas.
  - The released fits also included GLMdenoise PCs, which boldtailor's b3
    stage supplies.
  - The basis is tested against GLMsingle's `constructpolynomialmatrix` when
    `glmsingle` is installed. That is an opt-in test; `glmsingle` is a
    test-only dependency.
- **LSS** is implemented in the experiment package, not the core. Each trial
  gets its own model with that trial and an "all other trials" regressor, plus
  the same confounds. Design columns are built with Nilearn. It is tested
  against Nilearn `FirstLevelModel` fits on a small example.

## Package layout (`experiments/nsd_replication/`)

The package follows `examples/validation`:
- tests are opt-in and outside the default `pytest` run;
- every `__init__.py` is empty;
- functions are short and written test-first.

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclass loaded from TOML: BIDS root, subjects, session window, data sources, output root, `n_jobs` |
| `inputs.py` | Load one session's runs for a source (`fmriprep`, `ppdata`): BOLD, events, confounds; ppdata onset/time-base check |
| `confounds.py` | fMRIPrep set; GLMsingle polynomial basis |
| `roi.py` | nsdgeneral fsaverage → fsLR 32k → grayordinate mask for a CIFTI axis; subject-specific converted ROI when present |
| `released.py` | Read the user-converted released betas (b1, b2, b4) and align their trial order to the BIDS events |
| `fit.py` | Boldtailor ladder levels and LSS via the public API; writes the common beta format |
| `betas.py` | Common beta store; per-session z-scoring |
| `metrics.py` | R1–R6 metrics as defined in the protocol |
| `features.py` | HRF, modulator and gate experiments |
| `stats.py` | Replication criteria, TOST, descriptive CIs |
| `figures.py` | Figures and reports |
| `run.py` | CLI: `uv run python -m experiments.nsd_replication.run <stage> --config ...`; stages `fit`, `metrics`, `features`, `figures`; fits are resumable and existing outputs are skipped |

A new `experiments` uv dependency group holds `neuromaps` and the optional
`glmsingle` oracle. Task 1 checks whether neuromaps' label transform needs
Connectome Workbench (`wb_command`). If it does, that is documented as a
system prerequisite.

## Inputs

- **Images and repeats.**
  - Image identity is the 73k ID in the events.
  - Repeats are matched by ID across sessions.
  - The paper's analyses use images presented three times within the window,
    and so do these. The pilot report gives the counts.
- **Shared images for R5.** The images that subjects 01–04 each saw three
  times within the window.
- **Released betas.**
  - Format: one CIFTI per session and version, trials in presentation order.
  - Alignment: `released.py` matches trial order to the BIDS events by run
    and onset order, and checks the trial count per session (750).
  - Mismatches are errors.
  - The released betas' scaling (percent signal change, possibly stored as
    scaled integers) is undone before z-scoring. Task 1 records the scaling
    used.
- **ppdata timing.**
  - The TR is read from the files.
  - Task 1 derives how BIDS onsets relate to the ppdata time base, once. The
    relation must be a constant offset; it is documented and checked on
    every run.
  - Boldtailor needs no onset grid, so no resampling is involved.

## Common beta format

Each fit writes one directory:

```
<output_root>/fits/<source>/sub-XX/ses-YY/<level>/
    betas.npy       float32, trials × grayordinates
    trials.tsv      run, trial index in run, onset, 73k image ID
    metadata.json   level, settings, inputs digest, runtime, boldtailor version, git commit
```

Converted released betas are indexed in the same layout under
`<source>=released`, so metric code treats every version alike.

**Storage.** Roughly 250 GB:
- fMRIPrep: 7 fits × 80 sessions × about 275 MB;
- ppdata: 5 fits × 80 sessions × about 190 MB.

`run.py fit --discard-betas-after-metrics` may delete a fit's betas once every
metric for that subject is written. It keeps `trials.tsv` and the metadata.

## Comparison with released GLMsingle betas (ppdata)

- **Data.** For all eight subjects, `nsd01`–`nsd10`: boldtailor b1, b2, b3
  and b4 on ppdata, plus released b1, b2 and b4, all on the same grayordinates
  with the same trials.
- **Outputs:**
  - every R1–R6 metric for every version;
  - per-grayordinate fsLR maps of the R1 reliability difference
    (boldtailor − released) for b1, b2 and b4;
  - per-subject ROI summaries.
- **Pipeline check.** The b1 pair (canonical HRF, OLS, polynomial drift on
  the same data) should be nearly identical. Low agreement at b1 points to an
  input or alignment error, not a method difference. The pilot reports the
  correlation of b1 betas per grayordinate and sets a sanity floor in the
  protocol.

## Statistics

- **Replication criteria.** Each R1–R6 pattern is written in the protocol as
  a directional, per-subject criterion before the primary run. For example,
  R1: median ROI reliability at the r = 0.2 threshold is higher for b4 than
  for b1.
  - A pattern is replicated if its criterion holds in at least three of the
    four primary subjects.
  - Effect sizes are reported next to the paper's.
  - Subjects 05, 06 and 08 are reported as an extension with the same
    criteria.
- **Parity (primary hypothesis).** On ppdata at b4, boldtailor's per-subject
  median ROI reliability is equivalent to the released b4's.
  - Test: paired TOST on the relative difference across the eight subjects,
    α = 0.05, margin ±δ.
  - Secondary: the same test at b1 and b2, and for the R4–R6 summary values.
- **Margin δ.** The pilot report gives sub-07's session-to-session and
  split-to-split variability of the primary statistic. The protocol fixes δ
  with a written justification before the freeze.
- **Everything else** is descriptive: per-subject points, the subject mean,
  and a 95% CI.

## Feature experiments

1. **HRF selection quality.**
   - Data: ppdata, using the 20-HRF library.
   - Isolating the choice: put boldtailor's CV choice (b2-lib20) and the
     released GLMsingle HRF index into identical OLS single-trial fits, and
     compare R1 reliability. This applies if the user also converts NSD's
     released HRF index maps; otherwise this part is skipped and recorded as
     such.
   - Library effect: b2 versus b2-lib20.
   - Stability: session-to-session agreement of chosen HRFs (`compare_hrfs`),
     matching R2.
2. **Task modulators.**
   - Setup: boldtailor b4 with the task model `task` only, versus `task` +
     `response_time` + `trial_type`. The task model is used in HRF selection,
     denoising scoring and ridge encoding.
   - Outcomes: R1 reliability and the trial-level correlation between ROI
     betas and RT.
   - Data: fMRIPrep, primary subjects.
   - Reporting: the events sidecar's description of `trial_type` is quoted in
     the report.
3. **Denoising gate.**
   - Null control: each run's events are circularly shifted by a random
     offset of at least 30 s, with seeds recorded.
   - Runs: `select_denoising` with the gate on and off, on real and
     null-shifted sessions.
   - Outcomes: PC counts and gate rejection rates. Expected: on nulls, pcstop
     alone adds PCs and the gate rejects them.
   - Data: fMRIPrep, pilot and primary subjects.

## Protocol and freeze

- **`experiments/nsd_replication/protocol.md` contains:**
  - the metric definitions taken from the paper;
  - session windows and image sets;
  - every boldtailor setting;
  - the ROI;
  - the replication criteria;
  - δ and its justification;
  - the primary hypothesis;
  - the secondary list;
  - the figure list;
  - the record of the external beta and ROI conversion.
- **Freeze.** The file is committed with the pilot report and tagged
  `nsd-replication-protocol-v1` before any primary-subject fit.
- **After the freeze.** Changes that affect results are logged as dated
  deviations with their reasons, and reported next to the frozen version's
  results.

## Deliverables

- **Pilot report:** `docs/validation/nsd-replication-pilot-<date>.md`. Covers
  sub-07's R1–R6, repeat counts, the b1 agreement with released betas, the
  variability behind δ, and runtimes.
- **Final report:** `docs/validation/nsd-replication-<date>.md`. Covers the
  replication table (the paper's pattern, the boldtailor result, criterion
  met per subject), the parity results, the feature experiments, and the
  deviations.
- **Figures paralleling Figs. 2, 2-suppl., 3, 5, 6 and 7:**
  - for the boldtailor ladder;
  - with released versions overlaid on the ppdata comparison;
  - with fsLR maps of reliability and of the boldtailor − released
    difference.
- **`docs/glmsingle-comparison.md`.** Its "no matched comparison has been run"
  statements are updated to cite the final report.

## Errors

| Condition | Where | Behaviour |
|---|---|---|
| missing run file, events, or confounds for a configured session | `inputs.py` | error naming subject, session, run |
| ppdata onsets not related to BIDS onsets by the recorded constant offset | `inputs.py` | error naming run |
| released betas: missing version/session, trial count ≠ events, grayordinate axis ≠ ppdata | `released.py` | error naming subject, session, version |
| confound names differ across runs | `confounds.py` | error (as the workflow) |
| ROI has no overlap with the CIFTI axis | `roi.py` | error |
| a fit directory exists with a different inputs digest | `run.py` | error unless `--refit`; never silently reused |

## Testing

- **Default suite.** Unaffected. All tests here are opt-in.
- **Unit tests with hand-computed oracles:**
  - each R1–R6 metric on small arrays;
  - z-scoring;
  - beta-format round-trip;
  - released-beta trial alignment, including shuffled and short inputs;
  - the polynomial basis;
  - LSS against Nilearn;
  - the circular-shift null (preserves counts and durations; minimum offset).
- **End-to-end test.** A synthetic set of sessions (a few hundred features,
  images repeated three times, non-canonical HRFs, shared noise) runs the
  ladder and the metrics. Checks:
  - R1 reliability rises from b1 to b2 to b4;
  - b1 lag-1 pattern correlation exceeds b4's.
- **ROI test.** On a synthetic label, skipped when neuromaps or Workbench is
  unavailable.

## Task outline (for the implementation plan)

1. **Protocol draft.** Extract the paper's metric definitions. Install the
   `experiments` group and check the neuromaps requirements. Check sub-07's
   events and fMRIPrep inputs. Write `protocol.md` without δ.
2. Config, inputs, confound sets.
3. ROI construction.
4. Common beta format; boldtailor ladder and LSS driver.
5. Metrics R1–R6.
6. **Pilot.** The user runs sub-07. Write the pilot report and fix δ, the
   criteria and the sanity floor. Tag the freeze.
7. Released-beta reader and ppdata inputs. This waits on the user's
   converted files.
8. Feature experiments.
9. **Primary and extension runs.** Run them (user-started), then the
   statistics, figures, final report and documentation update.

Steps 6 and 9 are compute runs the user starts. The plan gives their commands
and expected runtimes rather than running them inside an implementation task.

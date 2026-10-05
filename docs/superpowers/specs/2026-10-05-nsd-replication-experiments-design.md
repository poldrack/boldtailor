# Replicating the GLMsingle paper's NSD results with boldtailor

## Goal

Replicate, with boldtailor, the patterns that Prince et al. (2022) report on
the Natural Scenes Dataset (NSD). The paper is "Improving the accuracy of
single-trial fMRI response estimates using GLMsingle," *eLife* 11:e77599.
The goal is to show that boldtailor's single-trial pipeline produces the same
qualitative gains the paper attributes to GLMsingle. Then:

- compare boldtailor directly with NSD's released GLMsingle betas on
  identical cortical grayordinates and trials;
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

- **No GLMsingle runs.** The comparator is NSD's released GLMsingle betas,
  converted by the user to CIFTI under `BIDS/derivatives/betas-fsLR`:
  - `desc-assumehrf` (b1), `desc-fithrf` (b2) and `desc-fithrfGLMdenoiseRR`
    (b4). NSD released no GLMdenoise-only version, so b3 has no released
    counterpart.
  - Files: `sub-NN/ses-nsdYY/func/sub-NN_ses-nsdYY_task-nsdcore_space-fsLR_den-91k_desc-<version>_stat-effect_statmap.dscalar.nii`.
    Each holds 750 maps named `trial-001`…`trial-750`, in presentation order.
  - Units: percent signal change. NSD's ×300 integer scaling has already been
    undone ("ScalingApplied": "divided by 300").
  - Grid: fsLR 91k. The cortical part is exactly the ppdata grayordinates
    (same names and vertices). Cortical values are finite; subcortex is NaN
    placeholders.
  - Source: resampled from NSD's native-surface betas with adaptive
    barycentric resampling via each subject's sphere registration. That
    conversion is external to this package and recorded in the protocol.
  - Also present: subject-level NSD NCSNR maps per version
    (`sub-NN/func/*_stat-ncsnr*`), computed by NSD over all of its sessions.
    They are a reference only, not our metric.
- **Data.** Boldtailor fits NSD's own preprocessed series (ppdata,
  `func1pt8mm`), resampled by the user to fsLR 32k on the layer-B2 surface.
  - Files: `derivatives/ppdata/subjNN/func1pt8mm/timeseries/sub-NN_ses-nsdYY_task-nsdcore_run-ZZ_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`.
  - Timing: TR 4/3 s, 226 volumes per run.
  - Grid: cortex only, 59,412 grayordinates, the same vertices as the released
    betas' cortex.
  - Why: this is the preprocessing the released betas (and the paper) used, so
    the comparison isolates the GLM pipeline. fMRIPrep derivatives are not used.
  - Onsets: BIDS onsets map onto the ppdata time base by a constant
    `onset_offset`, calibrated once in the pilot by maximising the alignment
    check over candidate offsets.
- **Events.** The existing BIDS `events.tsv` files.
- **Cortex only.** Every fit and every analysis uses only cortical surface
  grayordinates: the CIFTI brain-model structures `CIFTI_STRUCTURE_CORTEX_LEFT`
  and `CIFTI_STRUCTURE_CORTEX_RIGHT`.
  - What is excluded: subcortical volume grayordinates are dropped when inputs
    are loaded, before any fitting, so they never reach boldtailor, the betas,
    the metrics or the maps.
  - Where it happens: the selection is an input-step concern in the
    experiment package (`inputs.py`). Boldtailor's core is unchanged and
    still sees a plain features × timepoints matrix.
  - Whole-cortex summaries mean exactly this set: 59,412 grayordinates.
  - Released betas: the same structure filter applies. Files without both
    cortical structures, or whose cortex differs from the ppdata
    grayordinates, are an error.
- **Staging.**
  - Pilot: sub-07, `nsd10`–`nsd19`. ppdata and released betas exist for
    all 40 sub-07 sessions. Sub-07 is not one of the paper's subjects, so the
    pilot window does not need to match the primary one.
  - Primary replication: subjects 01–04, `nsd01`–`nsd10`, as in the paper.
  - Extension: subjects 05, 06 and 08, same window. Sub-07 is reported as
    the pilot.
- **Order of runs.** Each subject's stages run when its ppdata and converted
  released betas are available.
- **ROI.** Each subject's own nsdgeneral, from NSD's native-surface labels
  (`derivatives/freesurfer-NSD/subjNN/label/{lh,rh}.nsdgeneral.mgz`, 0/1 on the
  native vertices).
  - Resampling: the labels are resampled to fsLR 32k with exactly the Workbench
    route used for the released betas: `wb_command -metric-resample ...
    ADAP_BARY_AREA` with area metrics. The spheres and area metrics are read
    from the betas' JSON sidecars.
  - Then: the result is thresholded at 0.5 and mapped onto the cortical
    grayordinates.
  - Deviation: the paper used the same subject-specific ROI, but in volume
    space rather than on the surface.
- **"On par"** is an equivalence test (TOST) between boldtailor on ppdata
  and the released betas. Its margin is fixed from pilot variability before
  the freeze.
  - Caveat, stated in every report: both arms start from NSD's preprocessing,
    but they reach the surface differently. ppdata were sampled at layer B2;
    the released betas were resampled from NSD's native-surface betas.
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
workflow has no denoising stage. Levels b2–b4 reuse one HRF selection per
session, as GLMsingle does.

| Level | Boldtailor |
|---|---|
| b1 | canonical HRF, OLS single-trial (`fit_single_trials`) |
| b2 | `select_hrfs` + `fit_selected_hrfs` with `default_hrf_library()` |
| b2-lib20 | same, with `glmsingle_hrf_library()`, to separate the selection rule from the library |
| b3 | b2 + `select_denoising` (defaults, gate on), its components added to each run's confounds |
| b4 | b3 + per-grayordinate fractional-ridge CV (workflow defaults) |
| LSS-assume, LSS-fit | least-squares-separate with the canonical HRF and with the b2 HRFs (R3 only) |

- **Task model.** `task`, `response_time`, `trial_type`, as the workflow
  detects them for NSD.
- **Confounds.** GLMsingle's polynomial drift basis per run, as NSD's
  GLMsingle used: powers 0..d of `linspace(-1, 1, n)`, orthonormalised, with
  `d = alt_round(n * tr / 60 / 2)` (3 for NSD runs). Data-driven noise
  regressors come from boldtailor's denoising stage at b3.
- **LSS** is implemented in the experiment package, not the core. Each trial
  gets its own model with that trial and an "all other trials" regressor, plus
  the same confounds. It is tested against Nilearn `run_glm` fits on a small
  example.

## Package layout (`experiments/nsd_replication/`)

The package follows `examples/validation`:
- tests are opt-in and outside the default `pytest` run;
- every `__init__.py` is empty;
- functions are short and written test-first.

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclass loaded from TOML: BIDS root, released-betas root, subjects, session window, output root, `n_jobs` |
| `trials.py` | Standard trial tables, presentation order, repetition arrays |
| `inputs.py` | Load one session's ppdata runs (cortical surface grayordinates), BIDS events, and polynomial confounds |
| `confounds.py` | GLMsingle polynomial drift basis |
| `roi.py` | Subject nsdgeneral: native surface → fsLR 32k (Workbench, same route as the betas) → cortical grayordinate mask |
| `released.py` | Read the released betas (b1, b2, b4), check them against the ppdata grayordinates and the BIDS trials, and index them in the common format |
| `ladder.py`, `lss.py` | Boldtailor ladder levels and LSS via the public API |
| `betas.py` | Common beta store; per-session z-scoring |
| `metrics.py`, `hrf_maps.py` | R1–R6 metrics as defined in the protocol |
| `features.py` | HRF, modulator and gate experiments |
| `stats.py` | Replication criteria, TOST, descriptive CIs |
| `figures.py` | Figures and reports |
| `run.py` | CLI: `uv run python -m experiments.nsd_replication.run <stage> --config ...`; stages `fit`, `metrics`, `features`, `figures`; fits are resumable and existing outputs are skipped |

No new Python dependencies are needed. Connectome Workbench (`wb_command`)
is a documented system prerequisite for the ROI resampling.

## Inputs

- **Images and repeats.**
  - Image identity is the `73k_id` column of the events.
  - Repeats are matched by ID across sessions.
  - The paper's analyses use images presented three times within the window,
    and so do these. The pilot report gives the counts.
- **Shared images for R5.** The images that subjects 01–04 each saw three
  times within the window.
- **Released betas.**
  - Rows are taken in file order, `trial-001`…`trial-750`, and paired with the
    BIDS trial table in run, then onset order.
  - The row count must equal the session's BIDS trial count. The pilot's b1
    agreement check (below) confirms the order.
  - Mismatches are errors.
  - No rescaling: the values are already percent signal change, and z-scoring
    removes units anyway.

## Common beta format

Each fit writes one directory:

```
<output_root>/fits/<source>/sub-XX/ses-YY/<level>/
    betas.npy       float32, trials × cortical grayordinates
    trials.tsv      session, run, trial index in run, onset, 73k image ID
    metadata.json   level, settings, inputs digest, runtime, boldtailor version, git commit
```

`<source>` is `ppdata` for boldtailor fits and `released` for the indexed
released betas, so metric code treats every version alike.

**Storage.** Roughly 130 GB, at about 180 MB per fit at 59,412 cortical
grayordinates:
- boldtailor: 7 fits × 80 sessions;
- released: 3 versions × 80 sessions.

`run.py fit --discard-betas-after-metrics` may delete a fit's betas once every
metric for that subject is written. It keeps `trials.tsv` and the metadata.

## Comparison with released GLMsingle betas

- **Data.** For each subject, `nsd01`–`nsd10` (`nsd10`–`nsd19` for the
  pilot): boldtailor b1, b2, b3 and b4 on ppdata, plus released b1, b2 and
  b4, all on the same cortical grayordinates with the same trials.
- **Outputs:**
  - every R1–R6 metric for every version;
  - per-grayordinate fsLR maps of the R1 reliability difference
    (boldtailor − released) for b1, b2 and b4;
  - per-subject ROI summaries.
  - NSD's NCSNR maps are shown alongside as a reference.
- **Alignment check.** Boldtailor b1 and released b1 are both canonical-HRF
  OLS fits of NSD-preprocessed data, so their per-grayordinate correlation
  across trials should be high in the ROI. The same check calibrates
  `onset_offset` in the pilot.
  - Null: correlate after shifting the released trial order by one run
    (62–63 trials).
  - The pilot reports the true and null medians, and the protocol sets a
    sanity floor between them. A session below the floor stops the
    comparison for that subject, because trials or grayordinates are
    misaligned.

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
- **Parity (primary hypothesis).** At b4, boldtailor's per-subject median
  ROI reliability is equivalent to the released b4's.
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
   - Library effect: b2 versus b2-lib20.
   - Stability: session-to-session agreement of chosen HRFs (`compare_hrfs`),
     matching R2.
   - Isolating the choice: if the user also converts NSD's released HRF index
     maps to fsLR 91k, boldtailor's b2-lib20 choice and the GLMsingle index
     go into identical OLS single-trial fits on the same data, and R1
     reliability is compared. Without those maps this part is skipped and
     recorded as such.
2. **Task modulators.**
   - Setup: boldtailor b4 with the task model `task` only, versus `task` +
     `response_time` + `trial_type`. The task model is used in HRF selection,
     denoising scoring and ridge encoding.
   - Outcomes: R1 reliability and the trial-level correlation between ROI
     betas and RT.
   - Data: primary subjects.
   - Reporting: the events sidecar's description of `trial_type` is quoted in
     the report.
3. **Denoising gate.**
   - Null control: each run's events are circularly shifted by a random
     offset of at least 30 s, with seeds recorded.
   - Runs: `select_denoising` with the gate on and off, on real and
     null-shifted sessions.
   - Outcomes: PC counts and gate rejection rates. Expected: on nulls, pcstop
     alone adds PCs and the gate rejects them.
   - Data: pilot and primary subjects.

## Protocol and freeze

- **`experiments/nsd_replication/protocol.md` contains:**
  - the metric definitions taken from the paper;
  - session windows and image sets;
  - every boldtailor setting;
  - the ROI;
  - the replication criteria;
  - δ and its justification;
  - the alignment-check floor;
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

- **Pilot report:** `docs/validation/nsd-replication-pilot-<date>.md`. Covers:
  - sub-07's R1–R6 for boldtailor and released versions;
  - repeat counts;
  - the alignment check;
  - the variability behind δ;
  - runtimes.
- **Final report:** `docs/validation/nsd-replication-<date>.md`. Covers:
  - the replication table (the paper's pattern, the boldtailor result,
    criterion met per subject);
  - the parity results, with the surface-sampling caveat;
  - the feature experiments;
  - the deviations.
- **Figures paralleling Figs. 2, 2-suppl., 3, 5, 6 and 7:**
  - the boldtailor ladder with released versions overlaid;
  - fsLR maps of reliability and of the boldtailor − released difference.
- **`docs/glmsingle-comparison.md`.** Its "no matched comparison has been run"
  statements are updated to cite the final report.

## Errors

| Condition | Where | Behaviour |
|---|---|---|
| missing run file, events, or confounds for a configured session | `inputs.py` | error naming subject, session, run |
| a CIFTI lacks either cortical surface structure | `inputs.py`, `released.py` | error naming file |
| released betas: missing version/session, trial count ≠ events, cortex ≠ ppdata grayordinates | `released.py` | error naming subject, session, version |
| ppdata onsets fall outside the run's time base after `onset_offset` | `inputs.py` | error naming subject, session, run |
| released b1 alignment below the protocol floor | `run.py` | error naming subject, session |
| ROI has no overlap with the CIFTI axis | `roi.py` | error |
| a fit directory exists with a different inputs digest | `run.py` | error unless `--refit`; never silently reused |

## Testing

- **Default suite.** Unaffected. All tests here are opt-in.
- **Unit tests with hand-computed oracles:**
  - each R1–R6 metric on small arrays;
  - z-scoring;
  - beta-format round-trip;
  - the cortex filter;
  - released-beta reading, including wrong trial counts and axes;
  - the alignment check;
  - LSS against Nilearn;
  - the circular-shift null (preserves counts and durations; minimum offset).
- **End-to-end test.** A synthetic session (tens of features, images
  repeated three times, a non-canonical HRF, shared noise) runs the ladder
  and the metrics. Checks: R1 reliability rises from b1 to b2.
- **ROI tests.** Threshold and command construction on synthetic inputs. A
  data-gated sub-07 test runs the real resampling and is skipped without
  Workbench.
- **Data-gated tests.** Smoke tests that skip when
  `/Volumes/extdata1/NSD/BIDS` is not mounted: one ppdata session and one
  released file of sub-07.

## Task outline (for the implementation plan)

1. Scaffold, dependencies, configuration.
2. Trial tables; repeat counts; protocol draft with metric definitions.
3. Cortical inputs.
4. ROI construction.
5. Common beta store.
6. Boldtailor ladder.
7. LSS.
8. Metrics R1–R6.
9. CLI, resumable fits, subject-level metrics.
10. Statistics.
11. Released-beta reader and alignment check.
12. **Pilot.** The user runs sub-07. Write the pilot report and fix δ, the
    criteria and the alignment floor. Tag the freeze.
13. Feature experiments.
14. **Primary and extension runs.** Run them (user-started), then the
    statistics, figures, final report and documentation update.

Steps 12 and 14 are compute runs the user starts. The plan gives their
commands and expected runtimes rather than running them inside an
implementation task.

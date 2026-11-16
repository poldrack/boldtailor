# RT and HRF recovery simulations

Implements the amplitude/duration comparison in [plan.md](plan.md), adapting
the four task models in Mumford et al. (2024). All commands run from the
repository root. No additional dependencies are required.

## Transfer only the simulation inputs

On the computer holding the complete NSD tree, export all discovered subjects,
sessions and runs into one compressed HDF5 file:

```bash
uv sync
uv run python -m experiments.hrf_simulations.bundle \
  --bids-root /path/to/NSD/BIDS \
  --output /path/to/nsd_simulation_inputs.h5
```

The default export contains **three bilateral anatomical regions**, derived from
sub-07's FreeSurfer `aparc` labels and **applied to every subject**:

| ROI selector | Vertices |
| --- | ---: |
| `fusiform` | 1,750 |
| `lateraloccipital` | 2,404 |
| `inferiortemporal` | 1,674 |

Their union has **5,828 vertices**, stored once per run, with separate ROI labels.
The fusiform selection exactly matches the previous mask. These are common
surface selections, not subject-specific anatomical ROIs. The exporter verifies matching CIFTI
brainordinate ordering across runs and refuses incompatible inputs. Override
`--mask` with an NPZ containing `feature_ids` and optional arrays named after the
three ROIs to choose another shared selection.
Optional `--subjects sub-01 sub-07` and `--sessions 10 11 19` restrict extraction;
without these options it includes every discovered layerB2 run, including
sessions outside 10–19. It does not create missing CIFTIs.

The HDF5 file contains losslessly compressed selected BOLD time series, exact
frame times, original events and NSD motion TSVs, vertex identities, and source
paths/hashes. It contains everything required to recompute calibration and the
80-second HPF nuisance design. No CIFTI files, Workbench installation, or mask
file are needed on the receiving machine for the bundle workflow. Only one run
is processed at a time during export; source CIFTIs are also read sequentially
for hashing. This step can take substantial disk I/O time but does not fit models.

Missing events or confounds cause an error by default. If some sessions are
incomplete, explicitly use `--skip-incomplete` to omit those runs; every omission
and its reason are recorded in the embedded manifest. Invalid motion, incompatible
vertex ordering, and corrupt data are always errors. Calibration still requires
at least two runs per requested session. Extraction writes a temporary file and
publishes the destination only on success; it refuses to overwrite an existing
bundle. An interrupted export must be restarted (there is no export resume).

Copy **only the `.h5` file** to the other computer, then run:

```bash
uv sync
uv run make -f experiments/hrf_simulations/Makefile workflow \
  STAGE=expanded BUNDLE=/path/to/nsd_simulation_inputs.h5 N_JOBS=15 \
  CALIBRATION=experiments/hrf_simulations/calibration/bundle-sub07-v1 \
  OUTPUT=experiments/hrf_simulations/outputs/bundle-sub07-expanded-v1 \
  REPORT=experiments/hrf_simulations/reports/bundle-sub07-expanded-v1
```

Select a region using `ROI=fusiform`, `ROI=lateraloccipital`, or
`ROI=inferiortemporal`. The default remains fusiform, so simply exporting more
vertices does not change the primary analysis. To run all three sequentially:

```bash
for roi in fusiform lateraloccipital inferiortemporal; do
  uv run make -f experiments/hrf_simulations/Makefile workflow \
    STAGE=expanded BUNDLE=/path/to/nsd_simulation_inputs.h5 ROI="$roi" N_JOBS=15 \
    CALIBRATION="experiments/hrf_simulations/calibration/sub07-$roi-v1" \
    OUTPUT="experiments/hrf_simulations/outputs/sub07-$roi-expanded-v1" \
    REPORT="experiments/hrf_simulations/reports/sub07-$roi-expanded-v1" \
    NOTEBOOK="experiments/hrf_simulations/$roi-results.ipynb" || break
done
```

Each region receives its own noise/SNR calibration. The workflow rejects reuse
of another region's calibration. Older single-region bundles remain readable;
to add the new regions, rerun export into a **new** HDF5 filename. Copying the new
code cannot add BOLD data absent from an existing file.

The tracked `shared_regions.npz` and its JSON sidecar contain mask indices and
anatomical provenance. To reproduce them from the full dataset:

```bash
uv run python -m experiments.hrf_simulations.rois \
  --freesurfer /path/to/NSD/BIDS/derivatives/freesurfer-NSD/subj07 \
  --cifti /path/to/NSD/BIDS/derivatives/nsdpreproc/sub-07/ses-nsd11/func/sub-07_ses-nsd11_task-nsdcore_run-01_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii
```

`BUNDLE` replaces BIDS/mask access in both calibration and simulation. The current
experimental grid remains sessions 10–19 (expanded core: 10/11/19); exporting all
sessions does not silently broaden it. To analyze another subject, set
`SUBJECT=sub-01` and use distinct calibration/output/report directories. Donors
remain within that subject; the CLI rejects cross-subject calibration reuse.
Archive contents can be listed without NSD:

```bash
uv run python -c 'from experiments.hrf_simulations.bundle import inventory; print(inventory("/path/to/nsd_simulation_inputs.h5"))'
```

## Expanded study on another computer

Install uv, clone this repository, check out `main`, and run `uv sync` from the
repository root. The destination needs the NSD preprocessed BIDS data, events,
and NSD motion confounds for sub-07 sessions 10–19, plus Connectome Workbench
(`wb_command` on PATH). The fusiform vertex mask is tracked in the repository.
Calibration and simulation data are intentionally excluded from Git.

```bash
uv sync
uv run make -f experiments/hrf_simulations/Makefile plan STAGE=expanded
uv run make -f experiments/hrf_simulations/Makefile workflow \
  STAGE=expanded BIDS_ROOT=/path/to/NSD/BIDS N_JOBS=15
```

This regenerates calibration on the destination, then executes **12,020 datasets**
with 20 paired replicates per condition. Alternatively, copy the complete
`calibration/nsd-v1` directory (including COMPLETE.json); its hashes are checked
before reuse. Keep the complete calibration together, and retain its source
provenance. Use a new output directory for this code, not the earlier pilot.
Default expanded outputs are `outputs/expanded-v1` and `reports/expanded-v1`.
Run the same command again to resume interrupted work. No expanded analyses
were launched as part of preparing this code.

| Block | Datasets | Coverage |
| --- | ---: | --- |
| Core | 5,880 | Sessions 10/11/19; seven HRF truths; AR1 and spectral noise; median SNR |
| White control | 700 | Session 10; five original truths; median SNR |
| SNR | 3,360 | Three sessions; canonical/offgrid; q25/q75; AR1/spectral |
| RT strength | 640 | Session 10; canonical/offgrid; gamma .25/1; amplitude/additive duration |
| Unequal contrast | 560 | Session 10; condition heights 1/1.25 |
| Matched RT | 320 | Session 10; RT component RMS = half baseline RMS |
| Run variation | 560 | Session 10; independent mean-one lognormal coefficient multipliers, log SD .25 |

Controls use canonical/offgrid truths, AR1/spectral noise and median SNR unless
specified otherwise. All seven mechanism/convention combinations appear except
RT-strength/matched-RT controls, which apply only to amplitude and additive
RT-duration mechanisms. Additional off-grid fast/slow truths appear in the core.
With `SESSIONS` overridden, controls use the first requested session. These
sessions provide actual timing/RT variation; they are not independent subjects.

The expansion avoids the 315,000-dataset full factorial. Increase replication
only for selected conditions after reviewing Monte Carlo errors: generate
`plan --stage expanded --replicates 100 --list`, then run selected `--case ID`
with those same stage/replicate settings. IDs for existing replicates stay fixed.
Use the CLI for individual cases, or Make's `CASE` and `REPLICATES` variables.

Run variation uses separate random streams, preserving paired noise.
The old RMS-scaling block is no longer generated: it changed the random-effect
prior used by the former simulation-specific REML model. The current package
uses native unit-peak trial responses. Passing `regressor_scaling="unit_rms"`
now raises an error rather than silently running a different model.

Every selection model is evaluated with all four task models, including
**select without RT, evaluate with RT**. `crossed_summary.tsv` reports contrast
bias and Monte Carlo error. The reference is the noiseless task's projection
onto the known-HRF evaluation model, so comparisons remain meaningful under
model misspecification. This is not necessarily the underlying neural contrast.
`known_hrf_contrast_error` measures the noisy known-HRF benchmark's error;
`selection_contrast_error` isolates the difference caused by HRF selection.
`excess_recovery.tsv` subtracts the best attainable library shape RMSE; this is
a shape-error floor, not the likelihood-optimal candidate.

The earlier pilot's median noise ACF RMSE was .0775 (AR1), .0655 (white), and
.0130 (spectral); median spectral relative power error was 3.84e-16. These are
descriptive finite-run diagnostics, not proof that AR1 captures all NSD noise.
Inspect the generated stratified tables, failures, noise diagnostics and Monte
Carlo errors before increasing replication. The notebook includes the new
crossed evaluation and library-floor summaries when present.

## Makefile workflow

```bash
make -f experiments/hrf_simulations/Makefile help
make -f experiments/hrf_simulations/Makefile plan
make -f experiments/hrf_simulations/Makefile workflow
```

`workflow` (also `all`) runs calibration → simulations → report → executed
notebook, sequentially, using **15 simulation workers** and one BLAS thread
per worker. Each component also has its own target: `calibrate`, `run`,
`report`, `notebook`, `smoke`, and `test`. Bare `make -f ...` prints help.

Use `workflow LIMIT=1 REPLICATES=1` for an initial timing run. Other overrides
include `N_JOBS`, `STAGE`, `SESSIONS`, `CALIBRATION`, `OUTPUT`, `REPORT`, and
`SNR`. For example:

```bash
make -f experiments/hrf_simulations/Makefile run N_JOBS=15 LIMIT=15
make -f experiments/hrf_simulations/Makefile workflow STAGE=sensitivity
```

Completed calibration is verified and reused. `run` resumes matching completed
cases; use new output directories when changing code or settings. `LIMIT`
limits simulation datasets, not calibration coverage. Calibration remains a
sequential preprocessing step; `N_JOBS` controls simulation-case parallelism.
No target runs the full 10-session simulation stage unless `STAGE=full` is set.

## Quick synthetic verification

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m experiments.hrf_simulations.cli smoke \
  --output experiments/hrf_simulations/outputs/smoke

uv run python -m pytest experiments/hrf_simulations -q
```

The smoke command uses synthetic events, a small three-HRF candidate library,
and four runs. It exercises fitting, checkpoint verification, and reporting;
it is **not an NSD-calibrated scientific result**.

## Calibrate from NSD

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m experiments.hrf_simulations.cli calibrate \
  --bids-root /Volumes/extdata1/NSD/BIDS \
  --sessions 10 11 12 13 14 15 16 17 18 19 \
  --output experiments/hrf_simulations/calibration/nsd-v1
```

Uses sub-07 and the existing 1,750-vertex fusiform mask by default. Reference
fits include motion-24, an intercept, 80-second cosine terms, and fixed
fast/canonical/slow task and junk regressors covering both RT mechanisms.
Artifacts retain all vertex profiles; target-specific selection takes 64
vertices spanning variance and lag-one correlation **after excluding the
target session**. Per-vertex calibration rho comes from
`boldtailor.ar1.estimate_rho`; a vertex is invalid (NaN) when its residual
lag-one denominator is at most `64 · eps · ||y||²`, a floor relative to the
vertex's own energy (the earlier experiment-local code used an absolute
`eps` floor). Each replicate draws one vertex profile. Its AR(1) rho is
shrunk across calibration vertices using repeated-run uncertainty, and remains
constant across the replicate's runs. Independent innovations start each run
in its stationary distribution.

Noise alternatives are white Gaussian, AR(1), and independent random-phase
surrogates of compatible residual series. Surrogates preserve each selected
donor's periodogram, including any effects of reference-model projection;
they do not reconstruct physiological noise removed by that model.

SNR calibration uses the RMS of task predictions from coefficients averaged
over the other runs in the donor session, after projecting out target-run
nuisance, divided by an AR(1)-corrected residual scale. The variance correction
uses `RSS / trace(M R)` with a robust run-level pooled rho, where `M` is the
reference-design residual projection and `R` the AR(1) correlation matrix.
This accounts for colored-noise projection loss; it remains approximate when
the true residual process is not AR(1). Spectral surrogates instead retain the
actual residual spectrum and variance. SNR calibration is a
descriptive signal-scale calibration, not evidence of task predictability:
noisy predictions can have positive RMS. The q25/q50/q75 values are calculated
from eligible non-target calibration profiles. All values and donor IDs are
saved. `--snr` allows explicitly labeled fixed-SNR controls.

Source BOLD/events/confounds/mask hashes are stored in calibration provenance.
Calibration files contain residual data and stay local and git-ignored. The
calibration command refuses to overwrite a completed calibration directory.

## Inspect and run the pilot

```bash
uv run python -m experiments.hrf_simulations.cli plan --stage pilot
uv run python -m experiments.hrf_simulations.cli plan --stage pilot --replicates 1 --list

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m experiments.hrf_simulations.cli run \
  --calibration experiments/hrf_simulations/calibration/nsd-v1 \
  --output experiments/hrf_simulations/outputs/pilot-v1 \
  --stage pilot --limit 1 --n-jobs 15
```

Start with `--limit 1` to measure runtime; remove it to run the selected grid.
Default pilot: session 11, two truths, 20 paired replicates, three noise
families, median SNR, and seven unique mechanism/duration combinations:
**840 datasets**. Pure RT-duration data are generated once and fitted with both
fixed-duration conventions. Each dataset gets full-session and complementary
half-session fits. Committed empirical split definitions are reused when run IDs match;
otherwise a deterministic seed-20261110 partition is saved in the manifest.

The default worker count is **15**. `--n-jobs N` parallelizes independent cases;
each worker caches signal-independent event/nuisance inputs and AR(1)
reference bases. Candidate convolution and selection are handled by the package.
Set the thread environment variables shown above to avoid BLAS oversubscription,
including with one worker.
The code does not automatically launch a larger stage.

Options:

- `--case ID`: run one case from `plan --list` (same stage/replicate settings).
- `--replicates N`, `--sessions ...`: explicit reduced or expanded coverage.
- `--snr inf`: noiseless control; `--snr 0.5`: fixed RMS ratio control.
- `--zero-nuisance`: projection-invariance control.
- `--no-splits`: full-session fits only, explicitly recorded.
- `--stage full`: ten sessions, five HRFs, three SNR quantiles, 100 replicates
  by default: 315,000 datasets. Profile the pilot before considering this.
- `--stage sensitivity`: session 11 by default, with gamma 0.25/0.5/1,
  unequal condition heights, and matched RT/base component RMS controls.

Use separate output directories for changed code or controls. Case manifests
include the resolved SNR, seeds, donor arrays' fingerprint, calibration source
hashes, actual events/nuisance/timing fingerprint, candidate library, truth
curve, settings, and exact split IDs. Completion records hash every result.
An exact rerun uses completed artifacts; mismatched or corrupt artifacts fail
loudly. Interrupted cases can be rerun. A file lock prevents duplicate writers.

## What is being fitted

HRF identification calls `boldtailor.hrf_selection.select_hrfs` for both
`method="cv"` and `method="reml"`. Each simulated dataset is fitted with CV,
iid REML, and REML using a training-estimated AR(1) correlation. AR(1)-generated
datasets also include REML with the known generating rho.

| Identification model | Event duration | TaskModel columns |
| --- | --- | --- |
| ConsDurNoRT | Fixed duration | Task plus reference-coded trial type |
| RTDur | Observed RT | Task plus reference-coded trial type |
| ConsDurRTMod | Fixed duration | Task, reference-coded trial type, and raw RT |

Fixed duration is either 0.1 s or the observed stimulus duration (3 s in NSD).
Identification uses valid trials only. Missing and incorrect trials enter
separate, fixed canonical-HRF nuisance columns with their stimulus durations.
These nuisance columns are shared by CV and REML. RTs are not centered or
orthogonalized. The package adds its own intercept.

`ConsDurRTDur` remains an evaluation model but is not an identification model:
it requires fixed-duration and RT-duration responses simultaneously, which
the current package TaskModel does not represent. Explicit attempts to select
with it raise an error. All four original models remain available for crossed
contrast evaluation.

CV selects by leave-one-run-out task prediction and requires two training
runs. REML uses the package's canonical-reference fixed mean (including the
requested task columns) and one random-effect column per valid trial. It
requires one estimable run. Variances are shared across runs per vertex.
This differs from the former simulation-specific REML model, whose random
effects were aggregated task and junk coefficients. A mean-only signal may
now have no supported REML winner. Such results remain failures/undefined
assignments; they are never replaced by a canonical HRF.

**Normalization:** identification uses the package's unit-peak event responses.
Generation and contrast evaluation retain physical unit-height neural boxcars
convolved with unit-peak HRFs. Generation uses fractional-bin activity and FFT
convolution at 5 ms; evaluation uses cumulative integrals at exact event
boundaries. Longer durations retain their larger area in those two stages.
Thus RT-duration identification is not an exact physical-unit matched model.
The simulation tests the package as implemented; it does not change the
package's normalization to improve recovery.

Each score row records `selection_method`. `estimator=cv` uses CV;
`iid`, `ar1`, and `ar1_oracle` denote the REML noise variants.
`fits.json` stores package selection provenance and method-specific scores.
CV scores are R²; REML criteria and gaps are likelihood comparisons.
Winning and any-candidate search flags are retained, including unresolved
losing candidates. CV has no REML provisional-search flag.
Use a new output directory: manifests reject results from the older implementation.

AR(1) estimation uses a fixed reference containing both duration conventions
and RT mechanisms, shared by every fitted task model. Rho is estimated only
from the simulated training runs with `boldtailor.ar1.estimate_rho`, pooled
within each simulated vertex, with its projection-bias correction and
[-0.8, 0.8] clipping. Replicates are never pooled for estimation. The
signal-independent part of each run's reference (`boldtailor.ar1.prepare_reference`)
is cached per run, so repeated fits pay one projection per run. AR(1)-generated cases additionally include a
known-rho oracle diagnostic; calibration rho is otherwise hidden from fitting.

## Reporting

```bash
uv run python -m experiments.hrf_simulations.cli report \
  --input experiments/hrf_simulations/outputs/pilot-v1 \
  --output experiments/hrf_simulations/reports/pilot-v1
```

Outputs include conditional recovery means and Monte Carlo standard errors,
unconditional failure rates, paired REML-minus-CV and AR(1)-minus-iid errors, split-half shape
correlations, actual donor-versus-generated spectral diagnostics, and a recovery overview figure. Candidate
curves and best attainable library errors are retained per case. The overview
pools conditions descriptively; use the stratified tables for conclusions.

Amplitude coefficients are refit by OLS on evaluation runs. Consequently,
held-out scores are **HRF-only transfer**, not fully held-out prediction.
Task reconstruction is also compared to the known noiseless task component.
Recovery failures stay in denominators; shape-error averages condition on
successful fits. Within-participant Monte Carlo replicates are not independent
human participants or a group-level significance test.

The Python notebook source `results.py` loads a report without the NSD drive.
Rendered notebooks remain local. Select the report through `HRF_SIM_REPORT`.


## Inspect the combined data file

Open `validate_data.ipynb` (local rendered notebook), or its tracked Jupytext
source `validate_data.py`. It defaults to `/Volumes/extdata1/NSD/nsd_three_rois.h5`;
edit `bundle_path` or set `HRF_SIM_BUNDLE` to use another location. Run all cells
for subject/session coverage, ROI counts, export omissions, structural/event/motion
checks, and an example session load with BOLD and RT plots.

By default the notebook checks three BOLD columns in **every run**. Set
`full_signal_scan = True` to check every stored BOLD value. Both modes check
all runs' events and motion data. It does not compare against the original source
files or validate anatomical registration. `no_long_text_outputs = True` keeps
large tables compact. No data are modified and no model fitting is launched.

To generate the local notebook from a fresh checkout:

```bash
uv run python -m experiments.hrf_simulations.render \
  --source experiments/hrf_simulations/validate_data.py \
  --output experiments/hrf_simulations/validate_data.ipynb \
  --report experiments/hrf_simulations/reports
```

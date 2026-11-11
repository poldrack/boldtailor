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

The default selection is the same 1,750 fsLR fusiform columns used for sub-07,
**applied to every subject**, as requested. This is a common surface selection,
not a subject-specific anatomical ROI. The exporter verifies matching CIFTI
brainordinate ordering across runs and refuses incompatible inputs. Override
`--mask` with an NPZ containing `feature_ids` to choose another shared selection.
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

This regenerates calibration on the destination, then executes **12,580 datasets**
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
| Regressor scaling | 560 | Session 10; nuisance-projected training-design RMS normalization |

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

Run variation uses separate random streams, preserving paired noise. Scaling
changes the random-effect prior: each candidate's task/junk columns are divided
by their RMS after nuisance projection, pooled over training runs only, before
whitening. Native scaling remains the primary analysis; OLS evaluation always
uses physical units. Winner scales and run coefficients are saved.

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
target session**. Each replicate draws one vertex profile. Its AR(1) rho is
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
each worker keeps a 65,536-entry candidate-design cache, large enough for all
533 candidates × 12 runs × four models × two duration conventions. This can
use several GB per worker. Set the thread environment
variables shown above to avoid BLAS oversubscription, including with one worker.
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

| Task model | Condition columns | Extra common RT column |
| --- | --- | --- |
| ConsDurNoRT | Fixed-duration | None |
| RTDur | RT-duration | None |
| ConsDurRTMod | Fixed-duration | Fixed-duration, height = raw RT |
| ConsDurRTDur | Fixed-duration | RT-duration, height = 1 |

Fixed duration is either 0.1 s or the observed stimulus duration (3 s in NSD).
Missing and incorrect trials have separate stimulus-duration junk regressors.
Observed RTs, including long RTs, are preserved. RTs are neither centered nor
orthogonalized. Each candidate supplies one HRF for task and junk across runs.

**Scaling distinction from earlier empirical analyses:** their trial-regressor
helper makes each individual trial response peak at one. Here neural boxcars
have unit height and are convolved with a unit-peak HRF, with the integration
time step included. Fitting evaluates cumulative HRF integrals at exact onset
and offset lags; generation independently uses fractional-bin neural activity
and FFT convolution at 5 ms. This avoids rounding short events to NSD TR/50
bins. Thus longer neural durations retain their greater area.
This is essential for the Mumford comparison. The original iid scalar REML
and bounded AR(1) scalar REML numerical solvers are reused unchanged. Solver
parity on the original empirical designs is separately tested. Differences
between the two estimators include their existing variance-search conventions.

AR(1) estimation uses a fixed reference containing both duration conventions
and RT mechanisms, shared by every fitted task model. Rho is estimated only
from the simulated training runs, pooled within each simulated vertex, with
the existing projection-bias correction and [-0.8, 0.8] clipping. Replicates
are never pooled for estimation. AR(1)-generated cases additionally include a
known-rho oracle diagnostic; calibration rho is otherwise hidden from fitting.

## Reporting

```bash
uv run python -m experiments.hrf_simulations.cli report \
  --input experiments/hrf_simulations/outputs/pilot-v1 \
  --output experiments/hrf_simulations/reports/pilot-v1
```

Outputs include conditional recovery means and Monte Carlo standard errors,
unconditional failure rates, paired AR(1)-minus-iid errors, split-half shape
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

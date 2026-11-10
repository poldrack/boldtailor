# RT and HRF recovery simulations

Implements the amplitude/duration comparison in [plan.md](plan.md), adapting
the four task models in Mumford et al. (2024). All commands run from the
repository root. No additional dependencies are required.

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
  --stage pilot --limit 1 --n-jobs 1
```

Start with `--limit 1` to measure runtime; remove it to run the selected grid.
Default pilot: session 11, two truths, 20 paired replicates, three noise
families, median SNR, and seven unique mechanism/duration combinations:
**840 datasets**. Pure RT-duration data are generated once and fitted with both
fixed-duration conventions. Each dataset gets full-session and complementary
half-session fits. Existing empirical split manifests are reused when present;
otherwise a deterministic seed-20261110 partition is saved in the manifest.

The default worker count is **1**. `--n-jobs N` parallelizes independent cases;
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

# NSD HRF simulations implementation plan

Goal: implement the approved `plan.md` as a reproducible calibration and
simulation CLI, without launching the large study.

Architecture: local task designs and independent signal synthesis feed the
existing scalar REML numerical solvers. Calibration artifacts separate private
NSD data from portable simulation reports. Seeded cases are atomic checkpoints.
Stack: existing NumPy/SciPy/pandas/nilearn/joblib/pytest; uv, no new dependencies.

Constraints: actual events unchanged; motion-24 and 80 s cosines; four task
models; two estimators; held-out calibration sessions; no code in __init__.py;
write and commit failing tests before implementation.

## Steps

- [ ] Task 1: tests for physical boxcars, four designs, independent synthesis,
  RT preservation, noise spectra/variance, and invalid inputs. Run RED and
  commit tests. Implement `design.py`, `simulation.py`, `noise.py`; run GREEN.
- [ ] Task 2: tests for fixed-reference rho, original-solver parity, noiseless
  recovery, calibration exclusion, seeds and checkpoint integrity. Run RED,
  commit. Implement `fitting.py`, `calibration.py`, `metrics.py`, `runner.py`
  and `io.py`; run GREEN.
- [ ] Task 3: tests for case grid, CLI smoke/resume, complete reporting and
  portable report generation. Run RED and commit. Implement CLI and reports,
  document commands. Run complete new tests plus existing REML regression tests.
- [ ] Final: independent review, fix substantive findings test-first, verify
  synthetic end-to-end execution, commit implementation. No automatic full run.

Review focus: empty junk categories; long RT crossing scan end; incompatible
noise donor grids; corrupted/stale caches; failed estimates counted in reports.
Tests must pin each behavior. Calibration and simulation seeds have distinct
namespaces; all paired fits share identical generated data.

## Implementation rulings / progress

- Existing `trial_regressors` normalizes individual event responses to unit
  peak. That would erase much of the duration effect. Simulation designs must
  instead use unit-height neural boxcars and peak-normalized HRF kernels with
  a quadrature dt factor. Reuse numerical REML solvers, not event normalization.
  Report this convention explicitly; native-design solver parity is tested
  separately from the new physical-design comparison.
- Work on branch `hrf-simulations`; leave existing analysis files unchanged.

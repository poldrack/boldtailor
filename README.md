# boldtailor

Boldtailor fits first-level fMRI models to voxel or grayordinate time series.
You can estimate condition contrasts, measure variance explained by a task,
fit a separate response for every trial, and select an HRF for each brain
location using prediction across runs.

The Python API works with NumPy arrays and pandas tables. The repository also
includes examples that read fMRIPrep outputs and save NIfTI or CIFTI maps.

## What can I do with it?

| Task | Where to start |
| --- | --- |
| Fit condition effects and t contrasts with OLS or AR(1) noise | [Conventional GLMs](docs/user-guide.md#condition-effects-and-contrasts) |
| Fit conventional GLMs with an optimized HRF per voxel or grayordinate | [Voxelwise HRFs](docs/user-guide.md#voxelwise-hrfs-in-conventional-glms) |
| Fit a design matrix prepared by another tool | [Prepared designs](docs/user-guide.md#using-your-own-design-matrix) |
| Compare full-model and confound-only R² | [Variance explained](docs/user-guide.md#measuring-task-related-variance) |
| Estimate one beta per stimulus presentation, with OLS or fixed ridge | [Beta series](docs/user-guide.md#estimating-a-beta-for-every-trial) |
| Choose a ridge fraction at each grayordinate using held-out trial prediction | [Fractional ridge](docs/user-guide.md#fractional-ridge-at-each-grayordinate) |
| Select HRFs from continuous parameter samples or a grid using run-wise cross-validation | [HRF selection](docs/user-guide.md#selecting-an-hrf-for-each-location) |
| Compare HRFs across NSD sessions against canonical SPM | [Session reliability notebook](examples/NSD/nsd_session_hrf_reliability.ipynb) |
| Compare HRFs selected from separate sets of runs | [HRF reliability](docs/user-guide.md#comparing-hrfs-between-sets-of-runs) |
| Analyze NSD CIFTIs in notebooks, including RT checks and parallel fitting | [NSD example](examples/NSD/README.md) |
| Compare canonical and optimized HRFs in task/RT/trial-type GLMs | [Full NSD workflow notebook](examples/NSD/nsd_workflow.ipynb) |
| Fit and view whole-brain NIfTI contrast and R² maps | [Whole-brain stop-signal notebook](examples/stop_signal_demo.ipynb) |
| Run the whole NSD-style CIFTI analysis from the shell | [Command line](#command-line) |
| Save analysis records and results together | [Saving results](docs/user-guide.md#saving-results-and-analysis-records) |

## Install

Use Python 3.12 or later. From a checkout of this repository:

```bash
uv sync --group dev
```

This installs Boldtailor and the packages used by the examples. Run scripts
with `uv run python your_script.py`. To add a local checkout to another uv
project, use `uv add /path/to/boldtailor` from that project.

## A first model

This example runs without a dataset. The random signals demonstrate the API;
replace them with your BOLD data for an analysis.

```python
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec

signals = 100 + np.random.default_rng(7).normal(size=(100, 3))
events = pd.DataFrame({
    "onset": [10.0, 40.0, 70.0, 100.0],
    "duration": [2.0, 2.0, 2.0, 2.0],
    "trial_type": ["face", "house", "face", "house"],
})

data = from_arrays(signals, events, tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
    hrf_model="spm",
    noise_model="ar1",
)
result = fit(data, model)

effects = result.effect("face_gt_house")  # one value per feature
z_scores = result.z_score("face_gt_house")
r_squared = result.r2
```

Signals have shape **time points × features**. A feature can be a voxel, a
surface vertex, or a CIFTI grayordinate. For multiple runs, pass lists of
signal arrays and event tables. Each run is fitted separately and contrasts
are combined across runs. Feature order must match between runs.

To estimate a separate beta for each event in the same data:

```python
from boldtailor.single_trial import fit_single_trials

trials = fit_single_trials(data, ridge_alpha=0.1)
betas = trials.run_betas[0]  # trials × features for the first run
trial_metadata = trials.trial_table
```

This call uses a fixed penalty; `ridge_alpha=0` gives OLS. Repeated images
remain separate trials. To choose the penalty from data, use
[encoding-guided fractional ridge CV](docs/user-guide.md#fractional-ridge-at-each-grayordinate),
which predicts beta series from trial variables such as trial type and RT.

## Command line

`boldtailor run` fits the full workflow (canonical and optimized-HRF GLMs, HRF
reliability, trial-wise beta series, and summaries) for one subject and session
of a BIDS dataset with fMRIPrep CIFTI outputs, and writes a BIDS derivative
plus an HTML report:

```bash
uv run boldtailor run --bids-dir /data/nsd --subject sub-01 --session ses-nsd01 --task nsdcore
```

Results go to `<bids-dir>/derivatives/boldtailor_hrf-<library>_ridge-<mode>`
unless `--output-dir` is given. Add `--dry-run` to see the resolved plan first.
The [user guide](docs/user-guide.md#running-the-full-workflow) lists every flag,
the output files, and the exit codes.

## Working with images

The [NSD guide](examples/NSD/README.md) covers the NSD notebooks: conventional
and single-trial CIFTI models, optimized HRFs, odd/even HRF parameter maps, and
their outputs. It explains the notebook settings (`examples/NSD/nsd_settings.py`),
your own data paths, and running several workers. The single-session analysis
is also available as the [`boldtailor run`](#command-line) command.

The [full NSD workflow notebook](examples/NSD/nsd_workflow.ipynb) fits matched
GLMs with `task`, `response_time`, and `trial_type`, first with the canonical
SPM HRF and then with an optimized HRF per grayordinate. It also demonstrates
HRF reliability, single-trial beta series, nested ridge selection, held-out
trial encoding, RT checks, and CIFTI export.

The [whole-brain stop-signal notebook](examples/stop_signal_demo.ipynb) combines
multiple sessions in a common brain mask, fits contrasts, displays maps, and
optionally saves NIfTI results. Set `BOLDTAILOR_BIDS_ROOT` to your dataset and
edit the notebook's subject, session, and preprocessing settings.

These are dataset-specific examples. For a different dataset, load aligned
signals with your usual imaging tools and use the array API, or adapt an example.

## How does it compare with GLMsingle?

Both packages estimate single-trial responses with an HRF selected at each
brain location. Boldtailor also supports conventional contrasts and custom
design matrices. Its single-trial workflow uses supplied confounds, fixed or
encoding-guided ridge penalties, and HRF selection by task-model prediction
across runs.

The published GLMsingle workflow selects HRFs by in-sample fit, then uses
repeated conditions to tune data-derived denoising and fractional ridge.
Boldtailor's HRF selection needs no repeated images, but assumes that the
task-model response (the mean response under the default `TaskModel()`)
transfers between runs. Every event's predicted response is scaled to a peak of one (kernels are
also stored at unit peak). A beta is therefore the peak BOLD response to that
presentation in signal units, independent of TR, oversampling, and event
duration, and comparable across grayordinates with different selected HRFs.
This matches GLMsingle's convention. Nilearn derivative and FIR bases are passed to Nilearn unchanged (sum-to-one for the canonical bases); user-supplied kernels are used exactly as given. Its ridge CV predicts trial
beta series from variables such as trial type and RT. Fractional CV scores
fixed OLS targets; shared-alpha CV scores candidate-regularized targets. Both
assume that the predictor relationships transfer across runs. See the
[GLMsingle comparison](docs/glmsingle-comparison.md) for the methods, assumptions,
and differences from the locally developed GLMsingle API.

## Documentation

- [User guide](docs/user-guide.md): model choices, timing, confounds, HRFs, and interpretation.
- [API reference](docs/api.md): entry points, options, and returned values.
- [Result migration](docs/result-migration.md): current result classes and design access.
- [Documentation index](docs/README.md): current guides and dated development records.
- [GLMsingle comparison](docs/glmsingle-comparison.md): HRFs, denoising, regularization, and validation.
- [NSD guide](examples/NSD/README.md): notebook settings and CIFTI output reference.
- [NSD validation](docs/validation/nsd-session.md): recorded numerical checks and benchmarks.
- [Developer guide](docs/development.md): testing, architecture, provenance, and file-writing conventions.

Boldtailor is under active development. It currently supports t contrasts,
OLS/AR(1) conventional GLMs, and OLS, fixed-ridge, or encoding-tuned single-trial
fits. Automatic GLMdenoise and a general BIDS analysis command are not yet
available. HRF and encoding-guided ridge selection do not require repeated stimuli.

## License

Boldtailor is released under the [MIT License](LICENSE).

It bundles one third-party file: GLMsingle's 20-HRF library
(`src/boldtailor/_resources/glmsingle_hrf_library.tsv`, Copyright (c) 2021,
Kendrick Kay), redistributed under the BSD 3-Clause License reproduced in
`src/boldtailor/_resources/GLMsingle-LICENSE.txt`. That license covers only
this data file.

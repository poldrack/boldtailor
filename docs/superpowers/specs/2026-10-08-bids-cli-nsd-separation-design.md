# Separating the BIDS CLI from the NSD experiments

## Goal

Boldtailor's code serves two purposes. This change keeps them apart:

1. **The `boldtailor run` CLI** (`src/boldtailor/cli.py`, `src/boldtailor/workflow/`)
   analyses a BIDS dataset. It reads events from the raw BIDS directory and
   CIFTI time series and confounds from fMRIPrep derivatives. It stays
   purely BIDS + fMRIPrep and contains nothing NSD-specific.
2. **The NSD experiments** (`experiments/nsd_replication/`) are research code
   that is not shipped with the package. They read **only** the NSD team's
   preprocessed CIFTI files in `derivatives/nsdpreproc`, never fMRIPrep CIFTI 
   outputs. They use only boldtailor's public core API, never
   `boldtailor.workflow` or `boldtailor.cli`.

## Current state (2026-10-08)

- The CLI is already BIDS-only. `src/` contains no NSD logic. The only NSD
  traces are test-fixture names (`sub-07`, `ses-nsd10`, `nsdcore`), which stay.
- `experiments/nsd_replication/` reads the original non-BIDS ppdata layout,
  `derivatives/ppdata/subjNN/func1pt8mm/timeseries/`, through a hand-built
  path template (`ppdata_dir`).
- The experiments import workflow internals:
  - `boldtailor.workflow.inputs.detect_task_model`, in `inputs.py` and `conftest.py`;
  - `boldtailor.workflow.beta_series.trial_predictors`, in `ladder.py`, through a
    `SimpleNamespace` workaround;
  - `boldtailor.workflow.files.bids_label`, in `config.py`.
- `examples/NSD/` notebooks run the BIDS workflow on NSD's fMRIPrep CIFTI files.
  They demonstrate the CLI path, but their name presents them as the NSD analysis.

## Data facts

These were checked against `/Volumes/extdata1/NSD/BIDS`.

- `derivatives/nsdpreproc` is a BIDS version of ppdata. For sub-07 ses-nsd10
  run-01 the two time series are identical: shape (226, 59412), max absolute
  difference 0.0.
- nsdpreproc runs:
  - path: `sub-NN/ses-nsdYY/func/sub-NN_ses-nsdYY_task-nsdcore_run-ZZ_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`;
  - grayordinates: cortex only, 59,412;
  - TR 1.3333334 s, time axis starting at 0.
- Each run also has `desc-confounds_timeseries.tsv`, with six motion parameters,
  and `desc-design_timeseries.tsv`, with the TR-wise `73k_id`. **Neither is read.**
- Raw BIDS events exist for sub-07 ses-nsd10–19 only. nsdpreproc covers
  ses-nsd01–40. The experiments analyse only sessions that have raw events.
  sub-07 ses-nsd10 has 12 runs.

## Decisions

| Topic | Decision |
|---|---|
| NSD events | Raw BIDS `events.tsv` (`onset`, `duration`, `73k_id`, `trial_type`, `response_time`) |
| NSD time base | `onset_offset = 0.667` s stays; it was calibrated on ppdata, which is identical to nsdpreproc |
| NSD confounds | GLMsingle polynomial drift basis only (unchanged); nsdpreproc motion TSV not used |
| NSD CIFTI | nsdpreproc only; nothing under `experiments/` reads `derivatives/fmriprep*` or `derivatives/ppdata` |
| Coupling | Generic helpers move into public core modules; experiments import only core `boldtailor` |
| Examples | `examples/NSD/` becomes `examples/bids_workflow/`, relabeled as a BIDS + fMRIPrep CLI demo |

## (a) Core API moves (`src/boldtailor/`)

### `boldtailor/task_detection.py` (new public module)

These functions move here from `workflow/inputs.py`, with behaviour unchanged
except for the exception type:

- `detect_task_model(events_tables, modulators=None, labels=None) -> TaskModel`
- `observed_levels(tables, column, labels=None) -> set[str]`
- `task_model_notes(events_tables, modulators=None) -> list[str]`
- `TRIAL_TYPE_NOTE`, plus the private helpers `_table_levels`, `_everywhere`,
  `_automatic` and `_resolved`

Errors are raised as plain `ValueError` with the same messages as today. The
module imports nothing from `boldtailor.workflow`.

`workflow/inputs.py` imports these functions from core. Wherever the workflow
calls them, it catches `ValueError` and re-raises it as `InputError` with the
same message. That way the CLI keeps exiting with code 2 on bad events. The
workflow's own names stay in place: `InputError`, `selection_task_model`,
`validate_glm_events` and the session loading.

### `trial_predictors` moves into `boldtailor/trial_encoding.py`

- New signature: `trial_predictors(events_tables, task_model, labels=None) -> list[pd.DataFrame]`.
  It takes event tables, not workflow run objects. `labels` name the runs
  in error messages and default to `run 1`, `run 2`, ….
- `_indicator_frame`, `_predictor_frame` and `_censor_response_time` move with it.
  Behaviour is unchanged:
  - one 0/1 column per non-reference categorical level;
  - NaN for missing values;
  - non-finite or non-positive `response_time` censored to NaN;
  - a missing modulator column raises `ValueError` naming the run and the column.
- `workflow/beta_series.py` calls
  `trial_predictors([r.events for r in runs], task_model, [r.label for r in runs])`.
- `experiments/nsd_replication/ladder.py` calls it with `session.events` and
  `session.labels`, and the `SimpleNamespace` workaround is deleted.

### No compatibility shims

There are no re-exports from the old locations. Every caller in the repo is
updated: `src/`, `tests/`, `examples/` and `experiments/`. `docs/api.md`
documents both public entry points.

## (b) nsdpreproc loader (`experiments/nsd_replication/`)

### Configuration (`config.py`, `configs/*.toml`)

- `ppdata_dir` is removed. In its place, `nsdpreproc_dir: Path | None = None`
  resolves to `bids_dir / "derivatives" / "nsdpreproc"` when unset.
- A TOML that still has `ppdata_dir` fails with a `ValueError` naming
  `nsdpreproc_dir` as the replacement. This is a specific message, not the
  generic unknown-keys error.
- Labels are checked by a local helper in `config.py`, which takes over from
  `workflow.files.bids_label`. It requires `sub-<alnum>` and `ses-<alnum>`,
  distinct and non-empty, as today.
- In `pilot.toml`, `primary.toml` and `extension.toml`, the `ppdata_dir` line
  becomes `nsdpreproc_dir = "/Volumes/extdata1/NSD/BIDS/derivatives/nsdpreproc"`.

### Loading (`inputs.py`)

- `nsdpreproc_path(config, subject, session, run)` returns
  `nsdpreproc_dir/{sub}/{ses}/func/{sub}_{ses}_task-{task}_{run}_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`.
- Runs are listed from the raw BIDS events (`bids_dir/{sub}/{ses}/func/*_events.tsv`).
  If the nsdpreproc file for a run is missing, `FileNotFoundError` names
  that run and its path.
- `load_ppdata` becomes `load_nsdpreproc`, and `ppdata_brain` becomes
  `nsdpreproc_brain`. The following stay as they are:
  - selection of cortical grayordinates and the same-grayordinates check across runs;
  - `glmsingle_polynomials` confounds;
  - `onset_offset` added to the CIFTI time axis;
  - the onset-range check;
  - task-model detection, now through `boldtailor.task_detection.detect_task_model`.

### Source label rename

The source label `"ppdata"` becomes `"nsdpreproc"` in every place it appears:

- `run.py` `SOURCES`, the `--source` choices and their default;
- output and metric directory names (`fits/nsdpreproc/…`, `metrics/nsdpreproc/…`);
- version keys (`nsdpreproc:b1`, …) in `comparison.py`, `figure_stage.py` and `run.py`;
- the `pilot_report.py` choices;
- `Session.source`;
- the module docstrings and the comments in `confounds.py` and `calibrate_offset.py`;
- `README.md` and `protocol.md`.

`"released"` and `"comparison"` keep their names. No experiment outputs exist
yet, so no migration is needed.

## (c) Relabel the examples and docs

- `git mv examples/NSD examples/bids_workflow`, with file names unchanged.
- `examples/bids_workflow/README.md` opens by saying:
  - these notebooks show the BIDS + fMRIPrep `boldtailor run` path, using
    NSD's fMRIPrep outputs as the example dataset;
  - the nsdpreproc-based NSD replication lives in `experiments/nsd_replication/`.
- The `NSD_BIDS_ROOT`, `NSD_FMRIPREP_ROOT` and `NSD_OUTPUT_ROOT` environment
  variables keep their names.
- These living docs are updated for the new path and module locations:
  `README.md`, `docs/README.md`, `docs/user-guide.md`, `docs/api.md`, `docs/development.md`.
- `.github/workflows/tests.yml` is updated: `black --check src tests examples/bids_workflow`
  and `pytest … --run-notebooks examples/bids_workflow examples/validation`.
- `README.md` and `docs/development.md` gain a short "Two kinds of code" note:
  - the CLI uses BIDS raw events, fMRIPrep CIFTI files and fMRIPrep confounds;
  - `experiments/nsd_replication/` is research code that is not shipped. It
    uses nsdpreproc CIFTI files, raw BIDS events and GLMsingle polynomial
    terms, and only the public core API.
- Dated records stay unchanged, with their original paths: `docs/validation/*`,
  `docs/review-*`, `docs/test-suite-review-*`, and the earlier `docs/superpowers/` specs and plans.

## (d) Testing

RED-GREEN-Refactor. All new and moved tests are committed failing before any implementation.

### Default suite (`uv run pytest`, `tests/`)

- `tests/test_task_detection.py` holds the detection tests moved from
  `tests/workflow/test_inputs.py`, now importing from `boldtailor.task_detection`.
  They assert:
  - errors are `ValueError`;
  - `boldtailor.task_detection` does not import `boldtailor.workflow`
    (checked in a subprocess, through `sys.modules`).
- `tests/test_trial_encoding.py` gains the `trial_predictors` tests moved from
  `tests/workflow/test_beta_series.py`, rewritten for the event-table signature.
  They cover:
  - categorical indicator columns and their names;
  - NaN for missing categorical values;
  - RT censoring;
  - the missing-column error, naming the label;
  - default labels.
- In `tests/workflow/`, the existing tests that bad events produce `InputError`
  and CLI exit code 2 keep their assertions; only their import paths change.

### Experiment suite (opt-in: `uv run pytest experiments/nsd_replication`)

- `test_config.py`:
  - `nsdpreproc_dir` defaults to `bids_dir/derivatives/nsdpreproc`;
  - an explicit value is honoured;
  - `ppdata_dir` is rejected with a message naming `nsdpreproc_dir`;
  - label validation works without `boldtailor.workflow`.
- `test_inputs.py`:
  - `nsdpreproc_path` gives the BIDS layout;
  - a missing run raises `FileNotFoundError` naming the run;
  - synthetic loads report `source == "nsdpreproc"`.
- `test_run.py` / `test_figures.py` / `test_pilot_report.py`: the source label
  `nsdpreproc` appears in the output paths and in the CLI choices; `ppdata`
  appears nowhere.
- `test_boundary.py` (new): parses every `experiments/**/*.py` with `ast`. It fails on:
  - any import of `boldtailor.workflow` or `boldtailor.cli`;
  - any string literal containing the path fragments `derivatives/fmriprep`,
    `derivatives/ppdata` or `func1pt8mm`, or the key `fmriprep_dir`.

  `test_boundary.py` itself is excluded from the scan. Prose mentions of
  "ppdata" in docstrings are allowed, for example to say that nsdpreproc is
  its BIDS form.
- `test_nsdpreproc_data.py` (new, skipped when `/Volumes/extdata1/NSD/BIDS` is
  absent): loads sub-07 ses-nsd10 through `load_nsdpreproc` and checks:
  - 12 runs, each 226 frames by 59,412 cortical grayordinates;
  - `source == "nsdpreproc"`.

### Examples

These stay opt-in behind `--run-notebooks`, at `examples/bids_workflow`.

## Out of scope

- Using the nsdpreproc `desc-design_timeseries.tsv` or motion confounds.
- Analysing sessions that have no raw BIDS events.
- Recalibrating `onset_offset`.
- Any change to CLI behaviour, flags or outputs.

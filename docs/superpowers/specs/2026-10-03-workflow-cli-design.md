# Workflow package and command-line interface

Date: 2026-10-03. Status: approved design, awaiting implementation plan.

## Purpose

Run the full boldtailor analysis on one BIDS subject and session from the
command line, with outputs that are self-describing and reproducible, and move
the pipeline that today lives in `examples/NSD` into the package so it is
tested by default and usable for tasks other than NSD.

Success means: `boldtailor run --bids-dir D --subject S --session E --task T`
writes a complete BIDS derivative under `D/derivatives/boldtailor_<name>`
with an HTML report of every stage; the
NSD notebooks produce the same maps using the package; the default test suite
covers every stage; no code in the package imports from `examples`.

## Decisions already made

- Generic BIDS task on fMRIPrep CIFTI output. `--space` is a flag so other
  spaces can be added later; `fsLR-91k` is the only value accepted now.
- All four stages run by default: GLMs, split-half HRF reliability,
  single-trial beta series, RT and activation summaries.
- The default output directory name carries the HRF library and the ridge
  mode. `--output-dir` overrides it.
- The pipeline is ported into `boldtailor.workflow` as a consolidating port:
  modules move with their tests, legacy scripts are retired, the three
  ridge-mode output paths become one, the settings dictionary becomes a
  dataclass, result reuse is dropped, output descriptors are renamed.
- Modulators default to `response_time` (missing indicator) and `trial_type`
  when those event columns exist; `--modulator` replaces the whole set. No
  contrast or confound-selection flags.
- Regressor centering is removed from the package. Every modulator is
  uncentered. This leaves R², ΔR² and the HRF selection scores unchanged and
  changes only the meaning of the `task` contrast (response at modulator value
  zero rather than at the mean).
- Existing result directories need not be recognised; everything is rerun.

## Package layout

```
src/boldtailor/workflow/
    __init__.py      (empty)
    settings.py      WorkflowSettings, output_name, validation
    files.py         BIDS/fMRIPrep discovery per task and space
    inputs.py        loading, confound preset, trimming, modulator detection, ModelSpec
    analysis.py      block-parallel HRF selection, GLM and beta fits
    beta_series.py   nested-CV beta series; BetaModel for every ridge mode
    outputs.py       the writer for all stages
    artifacts.py     shared artifact helpers (tables, JSON, figures, scalars)
    plots.py         figures
    surfaces.py      surface rendering
    report.py        self-contained HTML report of every stage
    run.py           run_workflow(settings) -> WorkflowPaths
src/boldtailor/cli.py   argparse front end; console script `boldtailor`
```

Sources in `examples/NSD`: `workflow_files` → `files`; `workflow_inputs` and
`settings` → `inputs` and `settings`; `workflow_analysis` and
`parallel_blocks` → `analysis`; `ridge_workflow` and `ridge_provenance` →
`beta_series`; `workflow_outputs`, `ridge_outputs`, `fractional_outputs` →
`outputs`; `workflow_artifacts` → `artifacts`; `workflow_plots`,
`rt_diagnostics` → `plots`; `workflow_surfaces` → `surfaces`.

Retired with their tests: `nsd_hrf.py`, `nsd_single_trial.py`,
`nsd_cifti.py`, `hrf_artifacts.py`, `single_trial_artifacts.py`,
`workflow_reuse.py`, `ridge_reuse.py`, `notebook_paths.py`.

Remaining in `examples/NSD`, importing from the package: the three notebooks,
`session_hrf*.py`, `multisession_*.py`, and their opt-in tests. The
multisession helpers read saved maps through their own readers and must be
updated for the renamed descriptors.

## Settings

`WorkflowSettings` is a frozen dataclass with keyword-only fields:

| Field | Default | Notes |
| --- | --- | --- |
| `bids_dir` | required | expanded, must exist |
| `subject`, `session`, `task` | required | BIDS labels with prefixes (`sub-07`, `ses-nsd10`, `nsdcore`) |
| `fmriprep_dir` | the unique `derivatives/fmriprep*` under `bids_dir` | ambiguity or absence is an error listing candidates |
| `output_dir` | `bids_dir/derivatives/<output_name>` | |
| `space` | `"fsLR-91k"` | validated against `SUPPORTED_SPACES` |
| `modulators` | `None` | `None` means detect from events; otherwise a tuple of `Modulator` |
| `hrf_library` | `"default"` | `default`, `sobol`, `expanded`, `canonical` |
| `hrf_n_samples`, `hrf_seed` | 512, 0 | used by `default` and `sobol` |
| `hrf_selection_rt` | `True` | `False` drops `response_time` from the selection task model |
| `ridge_mode` | `"fractional_cv"` | `fractional_cv`, `cv`, `fixed`, `off` |
| `ridge_fractions` | `(0.1, …, 1.0)` | fractional_cv |
| `ridge_alphas` | `(0, 0.001, 0.01, 0.1, 1, 10, 100)` | cv |
| `ridge_percentile` | 90.0 | fractional_cv and cv |
| `ridge_alpha` | 0.1 | fixed |
| `encoding_mode` | `"within_run"` | or `absolute` |
| `stages` | all four | subset of `glms`, `reliability`, `betas`, `summaries`; `betas` requires `glms`, `summaries` requires `betas` |
| `surface_maps`, `surface_meshes` | `True`, `None` | meshes found under `fmriprep_dir` when `None` |
| `n_jobs`, `block_size`, `max_grayordinates` | 4, 4096, `None` | |
| `existing_results` | `"error"` | or `overwrite` |

`output_name()` returns `boldtailor_hrf-<library>_ridge-<mode>` where the
library token is `default512s0`, `sobol512s0`, `expanded`, or `canonical`
and the ridge token is `fractionalcv`, `cv`, `fixed0.1`, or `off`.
`to_dict()` is JSON-native and is what the settings file and the provenance
metadata record; `WorkflowSettings.from_dict()` reads it back.

Validation happens in `__post_init__` and raises `ValueError` with one
sentence naming the field.

## Input layer

Discovery (`files.py`): events at
`<bids>/<sub>/<ses>/func/<sub>_<ses>_task-<task>_run-*_events.tsv`, each
matched to `<fmriprep>/<sub>/<ses>/func/<stem>_space-fsLR_den-91k_bold.dtseries.nii`
plus the BOLD JSON, confounds TSV and JSON. Run sets must match exactly. The
space flag selects a loader table; only the CIFTI loader exists.

Loading (`inputs.py`) keeps the current behaviour: fMRIPrep confound preset
(24 motion parameters, six retained combined-mask aCompCor components, cosine
regressors, non-steady-state spikes), trimming of leading non-steady-state
volumes, event validation, sidecar-corrected frame times, and a check that all
runs share one grayordinate axis.

Modulator detection: with `modulators=None`, `response_time` present →
`Modulator("response_time", missing="indicator")`; `trial_type` present →
`Modulator("trial_type")`; neither → task regressor alone. Columns named in
`--modulator` must exist in every run's events; a missing column is an input
error naming the run.

The GLM `ModelSpec` has one contrast per task regressor, OLS noise, no extra
drift, canonical SPM for the canonical model, and the detected or given task
model.

## Package change: no centering

`Modulator` loses its `center` field and `to_dict` key; the task-design code
drops the centering branch; tests that exercised centering are removed or
rewritten to assert the uncentered design. Provenance metadata gains one
sentence in the task-model description stating that modulators are uncentered
and that the `task` contrast is the response at modulator value zero.

## Stage pipeline

`run_workflow(settings)` resolves paths, checks `existing_results`, loads
runs, builds the library, then runs enabled stages in order. Each stage is a
function of the runs, the settings, and earlier results.

1. `glms`: canonical GLM; HRF selection on all runs; optimized GLM; contrast,
   R², ΔR², selected-HRF and peak-time maps; design TSVs; library table.
2. `reliability`: odd and even selections, parameter agreement and curve
   correlation tables and figures. If fewer than two odd or two even runs, the
   stage records the reason in the settings file's `skipped` list and the
   run continues.
3. `betas`: canonical-HRF and optimized-HRF single-trial betas under
   `ridge_mode`, returning one `BetaModel` per HRF choice.
4. `summaries`: RT correlation maps, mean-beta activation maps, comparison
   figures.

`BetaModel` (frozen dataclass): `name`, `hrf` (`canonical` or `optimized`),
`mode`, per-run betas and trial tables, `tuning` (selected fraction or alpha
map and tuning curve, or `None` for `fixed` and `off`), `encoding_scores`
(held-out task ΔR² where cross-validated, else `None`), boundary flags, and
provenance. One writer in `outputs.py` serialises it for every mode.

## HTML report

After the stages finish, `report.py` writes one self-contained file,
`<output_dir>/<sub>_<ses>_task-<task>_report.html`, at the derivative root
(fMRIPrep's convention). It is built with the standard library only; figures
are embedded as base64 PNG so the file opens anywhere without the output tree.

Sections, in stage order, each rendered from the stage's results rather than
by re-reading files: settings and the equivalent command line; inputs (run
summary table, confound names, detected task model, library fingerprint);
design matrix figure; HRF library figure and table head; GLM comparison table
with the cortical R² and ΔR² surface figures; selected-HRF and peak-time
summaries; reliability tables and figures; beta-series tuning tables and
figures, encoding scores, and boundary summaries; RT and activation figures;
skipped stages with their reasons; and a file manifest linking every written
artifact and its provenance JSON by relative path. A skipped or disabled stage
renders a one-line note in its section. The report is written last and is
listed in the settings file, so an incomplete run has no report.

## Outputs

Layout: `<output_dir>/dataset_description.json`;
`<output_dir>/<sub>/<ses>/func/` holding `<stem>_space-fsLR_den-91k_desc-<D>_stat-<S>.dscalar.nii`
maps, design and table TSVs, PNG figures, per-activity provenance JSON, and
`<stem>_desc-boldtailor_settings.json` with the resolved settings, the
discovered runs, the detected task model, the library fingerprint, and any
skipped stages.

Descriptors: `canonicalGLM`, `optimizedGLM`, `HRF`, `HRFOdd`, `HRFEven`,
`canonicalBetas`, `optimizedBetas`, `RT`, `activation`. No `notebook` prefix.

`existing_results="error"` stops before loading data if the subject/session
output directory contains any boldtailor file; `overwrite` replaces the
subject/session outputs in one publication transaction.

## Command line

```
boldtailor run --bids-dir DIR --subject sub-07 --session ses-nsd10 --task nsdcore
    [--fmriprep-dir DIR] [--output-dir DIR] [--space fsLR-91k]
    [--modulator COLUMN[:indicator] ...]
    [--hrf-library default|sobol|expanded|canonical] [--hrf-n-samples 512] [--hrf-seed 0]
    [--no-rt-in-hrf-selection]
    [--ridge-mode fractional_cv|cv|fixed|off] [--ridge-alpha X]
    [--ridge-fractions F ...] [--ridge-alphas A ...] [--ridge-percentile 90]
    [--encoding-mode within_run|absolute]
    [--skip-stage {reliability,betas,summaries} ...]
    [--no-surface-maps] [--surface-mesh left=PATH right=PATH]
    [--n-jobs 4] [--block-size 4096] [--max-grayordinates N]
    [--existing-results error|overwrite] [--dry-run]
```

`--dry-run` prints the resolved settings, discovered runs, detected task
model and output directory, then exits 0 without fitting. Exit codes: 0
success; 1 usage or settings error (one line on stderr); 2 input discovery or
loading error listing what was missing. The console script is declared in
`pyproject.toml` as `boldtailor = "boldtailor.cli:main"`; `main(argv=None)`
returns the exit code so tests call it directly.

## Testing

- Example tests move with their modules into `tests/workflow/` and run by
  default. The synthetic BIDS/fMRIPrep fixture moves to `tests/conftest.py`.
- New tests: settings validation and `output_name`; `from_dict` round trip;
  fMRIPrep directory resolution (unique, ambiguous, absent); modulator
  detection and `--modulator` parsing; `BetaModel` shape across all four ridge
  modes; `run_workflow` end to end on the fixture with each stage toggled and
  with too few runs for reliability; CLI parsing, `--dry-run`, exit codes;
  outputs readable back with the renamed descriptors.
- Report: the HTML is produced on the fixture run, parses as well-formed
  HTML, contains one section per stage, embeds every figure, lists every
  written file, and notes skipped stages.
- Opt-in `examples/NSD` tests cover the notebooks and the session and
  multisession helpers after re-pointing.
- Package tests for the removed centering are replaced by tests asserting the
  uncentered design and the unchanged R².

## Out of scope

Volumetric or other CIFTI spaces, contrast specification, confound selection,
result reuse, multi-session orchestration from the CLI, and any change to the
fitting numerics.

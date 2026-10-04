# Categorical modulators

## Goal

Model discrete event columns, above all BIDS `trial_type`, as a set of
indicator regressors instead of dropping them. A `trial_type` with levels
`face`, `house`, `scrambled` yields the task regressor plus one indicator per
non-reference level, so condition differences are estimated alongside the
existing task response.

This replaces the workflow's interim rule (ruling R12 of the workflow/CLI
plan) that auto-detected `trial_type` only when it was numeric 0/1 and
otherwise left it out with a note.

## Decisions

- Reference (treatment) coding. The always-present `task` regressor is kept;
  a categorical column with k levels adds k − 1 indicators. `task` is the
  response to the reference level; each indicator is that level minus the
  reference. Modulators stay uncentered.
- Reference level: the first level in sorted order (numeric order when every
  level is numeric, otherwise lexical), overridable per modulator.
- Every non-reference level must occur in every run; otherwise the run is
  rejected before fitting with an error naming the run and the level.
- A `trial_type` with more than one distinct value is categorical whether its
  values are strings or numbers. Continuous parametric values belong in their
  own column as numeric modulators.
- Categorical expansion is a general modulator kind. `trial_type` is detected
  as categorical automatically; any other column opts in explicitly.
- Indicator names are uniform: `<column>[<level>]`, including binary 0/1
  columns (`trial_type[1]`). The NSD readers and tests move to the new name.
- Missing values (`n/a`, empty, NaN) are an error by default; with
  `missing="indicator"` a single `missing_<column>` regressor is added and
  those trials are 0 on every level indicator.

## Model (`src/boldtailor/model.py`)

`Modulator` gains three fields:

```python
Modulator(column, missing="error", kind="numeric", levels=None, reference=None)
```

- `kind` is `"numeric"` (current behaviour) or `"categorical"`.
- For `"categorical"`, `levels` is a tuple of at least two distinct non-empty
  strings and `reference` is one of them. `__post_init__` normalises levels to
  strings (an integral float such as `1.0` becomes `"1"`), stores them in
  canonical sorted order (numeric when all parse as numbers, else lexical),
  and defaults `reference` to the first level. Duplicates after
  normalisation are an error.
- For `"numeric"`, `levels` and `reference` must be `None`.
- `regressor_names(modulator)` is `(column,)` for numeric and
  `tuple(f"{column}[{level}]" for level in levels if level != reference)` for
  categorical. `TaskModel.regressor_names` is `("task", *names of each
  modulator in order)`; names must be unique across the model.
- `to_dict()` adds `kind`, and for categorical `levels` and `reference`.
  `TaskModel.to_dict()["regressors"]` uses the expanded names. The fingerprint
  therefore changes for every model (the `kind` key), which is acceptable:
  result reuse was retired.
- `is_subset_of` compares whole modulators, so a categorical modulator with
  different levels or reference is not a subset.

## Design expansion (`src/boldtailor/_task_design.py`)

`expand_events` builds, for a categorical modulator:

- one 0/1 amplitude per non-reference level, named as above;
- with `missing="indicator"` and at least one missing value in the run, a
  `missing_<column>` amplitude; with `missing="error"`, any missing value is
  an error naming the run and column.

Values are matched to levels after the same string normalisation. A value
outside `levels` is an error naming the run, column, and value. A run in which
a non-reference level has no trials hits the existing "no nonzero amplitude"
rule; its message names the run and the level. A run in which the reference
level has no trials is also rejected, because the indicators would then sum to
the task regressor.

The expanded table still feeds Nilearn's `make_first_level_design_matrix`
through the `modulation` column; bracketed condition names pass through
Nilearn unchanged (checked against Nilearn 0.14), and contrasts are weight
dictionaries, not formula strings.

## Workflow (`src/boldtailor/workflow/`)

### Detection (`inputs.py`)

- `response_time` detection is unchanged.
- `trial_type` present in every run: its levels are the union of non-missing
  values across runs. With two or more levels it becomes
  `Modulator("trial_type", kind="categorical", levels=..., missing="error")`;
  with fewer it is left out and `task_model_notes` records
  "trial_type has fewer than two levels; not used as a modulator".
- An explicit categorical modulator without `levels` gets them from the union
  across runs in the same way (an explicit reference must be among them).
- The binary-only checks are removed: `validate_glm_events`' 0/1 check,
  `TRIAL_TYPE_NOTE`, `_binary`, and `beta_series._check_trial_type`. Their
  replacement is the per-run level check above, raised as `InputError`
  (exit 2) before any fitting and naming the BIDS run label.
- `run_summary` replaces `type_0`/`type_1` with one `n_<column>_<level>`
  trial count per level of each categorical modulator.

### Beta-series encoding (`beta_series.py`)

`trial_predictors` emits the indicator columns of each categorical modulator
(named as the regressors) alongside numeric modulators, so the encoding
predictors are `task` plus every task-model regressor except `task`.

### Outputs and report

GLM effect, variance, t and z maps carry the expanded regressor names. The
metadata prose describes each categorical modulator: its levels, the
reference, and that each indicator is the level minus the reference. The
report's inputs section lists the levels and the reference.

## Command line (`src/boldtailor/cli.py`, `settings.parse_modulator`)

`--modulator COLUMN[:OPTION[,OPTION...]]` where options are `indicator`,
`categorical`, and `reference=LEVEL` (requires `categorical`). Examples:

```
--modulator response_time:indicator
--modulator trial_type:categorical,reference=face
--modulator condition:categorical,indicator
```

Unknown or repeated options are a settings error (exit 1). Settings store the
modulator without levels; levels are resolved from the events at load time.
`report.command_line` emits the same syntax and round-trips through the
parser.

## Examples and documentation

- `examples/NSD` readers and tests use `trial_type[1]` where they used
  `trial_type`; saved-output fixtures are regenerated accordingly.
- The workflow spec's modulator-detection paragraph, the user guide's
  modulator section, and the API reference describe categorical modulators,
  the reference rule, and the `--modulator` option syntax.

## Errors

| Condition | Where | Exit |
|---|---|---|
| categorical with < 2 levels, bad reference, levels on numeric | `Modulator` / settings | 1 |
| unknown/repeated `--modulator` option, `reference` without `categorical` | settings | 1 |
| run lacks a declared level (or the reference) | loading | 2 |
| value outside the declared levels | loading | 2 |
| missing value with `missing="error"` | loading | 2 |

## Testing

- `Modulator` validation, level normalisation and ordering, default and
  explicit reference, regressor names, `to_dict`, fingerprint differences.
- `expand_events` on a three-level string column against a hand-built oracle,
  and the design columns against Nilearn's `make_first_level_design_matrix`
  on the same expanded events.
- Missing level in a run, missing reference in a run, unknown value, and
  `n/a` under both missing policies.
- Detection: string levels, numeric multi-level, binary 0/1 (`trial_type[1]`),
  single-level (note), explicit categorical on another column.
- `parse_modulator` and CLI round trip for every option combination.
- `run_workflow` end to end on a fixture whose `trial_type` is
  `face`/`house`/`scrambled`: effect maps named `task`, `trial_type[house]`,
  `trial_type[scrambled]`; run summary counts; report and metadata text.
- The existing NSD-style fixtures pass with `trial_type[1]`.

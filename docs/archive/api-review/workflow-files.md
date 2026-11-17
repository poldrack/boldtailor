# `boldtailor.workflow.files`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/files.py)

BIDS/fMRIPrep discovery, CIFTI loading, nuisance selection, and sources.

## Definitions

- [`bids_label`](#definition-1)
- [`RunInputs`](#definition-2)
- [`RawRun`](#definition-3)
- [`input_paths`](#definition-4)
- [`_run_inputs`](#definition-5)
- [`discover_runs`](#definition-6)
- [`_acompcor`](#definition-7)
- [`_confound_names`](#definition-8)
- [`select_confounds`](#definition-9)
- [`_frame_times`](#definition-10)
- [`load_inputs`](#definition-11)
- [`_raw_run`](#definition-12)
- [`load_runs`](#definition-13)
- [`source_ref`](#definition-14)
- [`run_sources`](#definition-15)
- [`odd_even_parity`](#definition-16)
- [`reaction_times`](#definition-17)

<a id="definition-1"></a>

## `bids_label`

[Source, line 22](../../src/boldtailor/workflow/files.py#L22)

```python
bids_label(value, entity)
```

Whether value is a BIDS label such as ``sub-07`` for entity ``sub``.

<a id="definition-2"></a>

## `RunInputs`

[Source, line 28](../../src/boldtailor/workflow/files.py#L28)

```python
@dataclass(frozen=True)
class RunInputs
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
stem: str
bold: Path
events: Path
confounds: Path
confounds_json: Path
bold_json: Path
```

Usage in package/tests:

- [src/boldtailor/workflow/files.py:55](../../src/boldtailor/workflow/files.py#L55)

<a id="definition-3"></a>

## `RawRun`

[Source, line 38](../../src/boldtailor/workflow/files.py#L38)

```python
@dataclass(frozen=True)
class RawRun
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
inputs: RunInputs
image: nib.Cifti2Image
events: pd.DataFrame
confounds: pd.DataFrame
frame_times: np.ndarray
label: str
number: int
```

Usage in package/tests:

- [src/boldtailor/workflow/files.py:166](../../src/boldtailor/workflow/files.py#L166)

<a id="definition-4"></a>

## `input_paths`

[Source, line 48](../../src/boldtailor/workflow/files.py#L48)

```python
input_paths(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:63](../../src/boldtailor/workflow/files.py#L63)
- [src/boldtailor/workflow/outputs.py:1127](../../src/boldtailor/workflow/outputs.py#L1127)

<a id="definition-5"></a>

## `_run_inputs`

[Source, line 52](../../src/boldtailor/workflow/files.py#L52)

```python
_run_inputs(event, derivative, space_entity)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:79](../../src/boldtailor/workflow/files.py#L79)

<a id="definition-6"></a>

## `discover_runs`

[Source, line 69](../../src/boldtailor/workflow/files.py#L69)

```python
discover_runs(settings)
```

Require a unique CIFTI in the settings' space and complete metadata per run.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:163](../../src/boldtailor/workflow/inputs.py#L163)

<a id="definition-7"></a>

## `_acompcor`

[Source, line 86](../../src/boldtailor/workflow/files.py#L86)

```python
_acompcor(table, metadata)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:110](../../src/boldtailor/workflow/files.py#L110)

<a id="definition-8"></a>

## `_confound_names`

[Source, line 103](../../src/boldtailor/workflow/files.py#L103)

```python
_confound_names(table, metadata)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:119](../../src/boldtailor/workflow/files.py#L119)

<a id="definition-9"></a>

## `select_confounds`

[Source, line 117](../../src/boldtailor/workflow/files.py#L117)

```python
select_confounds(table: pd.DataFrame, metadata: dict) -> pd.DataFrame
```

Select motion24, top six combined-mask aCompCor, cosines, and NSS spikes.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:153](../../src/boldtailor/workflow/files.py#L153)

<a id="definition-10"></a>

## `_frame_times`

[Source, line 132](../../src/boldtailor/workflow/files.py#L132)

```python
_frame_times(inputs, image)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:152](../../src/boldtailor/workflow/files.py#L152)

<a id="definition-11"></a>

## `load_inputs`

[Source, line 147](../../src/boldtailor/workflow/files.py#L147)

```python
load_inputs(inputs: RunInputs)
```

Load image, raw events, selected nuisances, and sidecar-corrected frame times.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:163](../../src/boldtailor/workflow/files.py#L163)

<a id="definition-12"></a>

## `_raw_run`

[Source, line 162](../../src/boldtailor/workflow/files.py#L162)

```python
_raw_run(item)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:171](../../src/boldtailor/workflow/files.py#L171)

<a id="definition-13"></a>

## `load_runs`

[Source, line 169](../../src/boldtailor/workflow/files.py#L169)

```python
load_runs(inputs)
```

Load runs in BIDS run order; every run must share one BrainModel axis.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:163](../../src/boldtailor/workflow/inputs.py#L163)

<a id="definition-14"></a>

## `source_ref`

[Source, line 180](../../src/boldtailor/workflow/files.py#L180)

```python
source_ref(path, root, role, **annotations)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/files.py:197](../../src/boldtailor/workflow/files.py#L197)
- [src/boldtailor/workflow/files.py:204](../../src/boldtailor/workflow/files.py#L204)
- [src/boldtailor/workflow/files.py:205](../../src/boldtailor/workflow/files.py#L205)
- [src/boldtailor/workflow/files.py:202](../../src/boldtailor/workflow/files.py#L202)

<a id="definition-15"></a>

## `run_sources`

[Source, line 193](../../src/boldtailor/workflow/files.py#L193)

```python
run_sources(run, root, indices)
```

Source records for one run's BOLD (with grayordinates), events, confounds.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:188](../../src/boldtailor/workflow/inputs.py#L188)

<a id="definition-16"></a>

## `odd_even_parity`

[Source, line 214](../../src/boldtailor/workflow/files.py#L214)

```python
odd_even_parity(runs)
```

Run positions with odd and even BIDS run numbers, in input order.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:28](../../src/boldtailor/workflow/analysis.py#L28)
- [src/boldtailor/workflow/beta_series.py:77](../../src/boldtailor/workflow/beta_series.py#L77)
- [src/boldtailor/workflow/inputs.py:171](../../src/boldtailor/workflow/inputs.py#L171)
- [src/boldtailor/workflow/run.py:183](../../src/boldtailor/workflow/run.py#L183)

<a id="definition-17"></a>

## `reaction_times`

[Source, line 222](../../src/boldtailor/workflow/files.py#L222)

```python
reaction_times(runs, *, missing_ok=False)
```

Per-run RT arrays; unavailable (nonpositive or missing) RTs are NaN.

With ``missing_ok``, raw event tables may lack ``response_time`` or hold
non-numeric entries; those trials get NaN.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:262](../../src/boldtailor/workflow/analysis.py#L262)
- [src/boldtailor/workflow/plots.py:196](../../src/boldtailor/workflow/plots.py#L196)

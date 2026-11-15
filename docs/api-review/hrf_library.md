# `boldtailor.hrf_library`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/hrf_library.py)

Deterministic, peak-normalized HRFs; candidate 0 is Nilearn's SPM shape.

## Definitions

- [`_parameters`](#definition-1)
- [`_sampling`](#definition-2)
- [`_peak_normalized`](#definition-3)
- [`glmsingle_hrf_curves`](#definition-4)
- [`_glmsingle_index`](#definition-5)
- [`_kernel_samples`](#definition-6)
- [`_double_gamma_kernel`](#definition-7)
- [`_glmsingle_kernel`](#definition-8)
- [`HrfCandidate`](#definition-9)
- [`HrfCandidate.__post_init__`](#definition-10)
- [`HrfCandidate.kernel`](#definition-11)
- [`HrfLibrary`](#definition-12)
- [`HrfLibrary.__post_init__`](#definition-13)
- [`HrfLibrary.from_parameters`](#definition-14)
- [`HrfLibrary.from_table`](#definition-15)
- [`HrfLibrary.parameter_bounds`](#definition-16)
- [`HrfLibrary._custom_parameters`](#definition-17)
- [`HrfLibrary.informative_parameters`](#definition-18)
- [`HrfLibrary.candidate_bound_flags`](#definition-19)
- [`HrfLibrary.parameter_table`](#definition-20)
- [`HrfLibrary.timing_table`](#definition-21)
- [`_box_parameters`](#definition-22)
- [`_source_index`](#definition-23)
- [`_timing_row`](#definition-24)
- [`_glmsingle_realized`](#definition-25)
- [`_glmsingle_candidates`](#definition-26)
- [`_lobe_timing`](#definition-27)
- [`_lobe_parameters`](#definition-28)
- [`_lobe_height`](#definition-29)
- [`timing_parameters`](#definition-30)
- [`spm_parameters`](#definition-31)
- [`_dense_curve`](#definition-32)
- [`_subgrid_extremum`](#definition-33)
- [`_width_at`](#definition-34)
- [`realized_timing`](#definition-35)
- [`_realized_from_curve`](#definition-36)
- [`_realized_close`](#definition-37)
- [`_corrected_lobe`](#definition-38)
- [`_realized_error`](#definition-39)
- [`spm_parameters_from_realized`](#definition-40)
- [`expanded_hrf_library`](#definition-41)
- [`sobol_hrf_library`](#definition-42)
- [`_sobol_points`](#definition-43)
- [`_sobol_batches`](#definition-44)
- [`_doubling_batches`](#definition-45)
- [`_timing_bounds`](#definition-46)
- [`_scaled_timing`](#definition-47)
- [`_feasible_timing_rows`](#definition-48)
- [`timing_hrf_library`](#definition-49)
- [`glmsingle_hrf_library`](#definition-50)
- [`default_hrf_library`](#definition-51)

<a id="definition-1"></a>

## `_parameters`

[Source, line 69](../../src/boldtailor/hrf_library.py#L69)

```python
_parameters(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:392](../../src/boldtailor/hrf_library.py#L392)
- [src/boldtailor/hrf_library.py:431](../../src/boldtailor/hrf_library.py#L431)
- [src/boldtailor/hrf_library.py:493](../../src/boldtailor/hrf_library.py#L493)
- [src/boldtailor/hrf_library.py:167](../../src/boldtailor/hrf_library.py#L167)

<a id="definition-2"></a>

## `_sampling`

[Source, line 81](../../src/boldtailor/hrf_library.py#L81)

```python
_sampling(tr, oversampling)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:172](../../src/boldtailor/hrf_library.py#L172)

<a id="definition-3"></a>

## `_peak_normalized`

[Source, line 89](../../src/boldtailor/hrf_library.py#L89)

```python
_peak_normalized(values)
```

Scale a sampled kernel to unit peak so betas are peak BOLD responses.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:36](../../src/boldtailor/_hrf_design.py#L36)
- [src/boldtailor/hrf_library.py:177](../../src/boldtailor/hrf_library.py#L177)
- [src/boldtailor/hrf_library.py:174](../../src/boldtailor/hrf_library.py#L174)
- [src/boldtailor/hrf_library.py:176](../../src/boldtailor/hrf_library.py#L176)

<a id="definition-4"></a>

## `glmsingle_hrf_curves`

[Source, line 97](../../src/boldtailor/hrf_library.py#L97)

```python
@lru_cache(maxsize=1)
glmsingle_hrf_curves()
```

GLMsingle's 20 empirical HRFs, ``(20, 501)`` at 0.1 s, each at unit peak.

Copied unchanged from ``glmsingle/hrf/getcanonicalhrflibrary.tsv``
(cvnlab/GLMsingle, BSD-3; see ``_resources/glmsingle_hrf_library.md``)
and rescaled here so every candidate shares the package's peak-one
convention.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:141](../../src/boldtailor/hrf_library.py#L141)
- [src/boldtailor/hrf_library.py:349](../../src/boldtailor/hrf_library.py#L349)
- [tests/test_glmsingle_hrf.py:19](../../tests/test_glmsingle_hrf.py#L19)
- [tests/test_glmsingle_hrf.py:31](../../tests/test_glmsingle_hrf.py#L31)

<a id="definition-5"></a>

## `_glmsingle_index`

[Source, line 111](../../src/boldtailor/hrf_library.py#L111)

```python
_glmsingle_index(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:165](../../src/boldtailor/hrf_library.py#L165)

<a id="definition-6"></a>

## `_kernel_samples`

[Source, line 121](../../src/boldtailor/hrf_library.py#L121)

```python
_kernel_samples(duration, dt)
```

Number of samples covering ``duration`` at ``dt``; no extra tail sample
when the ratio is numerically an integer (TR inferred from frame
differences carries roundoff).

Usage in package/tests:

- [src/boldtailor/hrf_library.py:136](../../src/boldtailor/hrf_library.py#L136)

<a id="definition-7"></a>

## `_double_gamma_kernel`

[Source, line 134](../../src/boldtailor/hrf_library.py#L134)

```python
_double_gamma_kernel(parameters, dt)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:177](../../src/boldtailor/hrf_library.py#L177)

<a id="definition-8"></a>

## `_glmsingle_kernel`

[Source, line 140](../../src/boldtailor/hrf_library.py#L140)

```python
_glmsingle_kernel(index, dt)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:176](../../src/boldtailor/hrf_library.py#L176)

<a id="definition-9"></a>

## `HrfCandidate`

[Source, line 149](../../src/boldtailor/hrf_library.py#L149)

```python
@dataclass(frozen=True)
class HrfCandidate
```

Stable library index and identified kernel, sampled at TR/oversampling.

``parameters`` are the seven SPM double-gamma values for ``spm`` and
``double_gamma`` kinds and the one-based GLMsingle library index for
``glmsingle``. Every kernel is scaled to a peak of one.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
id: int
kind: str
parameters: tuple[float, ...]
```

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:31](../../src/boldtailor/_hrf_design.py#L31)
- [src/boldtailor/hrf_library.py:221](../../src/boldtailor/hrf_library.py#L221)
- [src/boldtailor/hrf_library.py:363](../../src/boldtailor/hrf_library.py#L363)
- [src/boldtailor/hrf_library.py:223](../../src/boldtailor/hrf_library.py#L223)

<a id="definition-10"></a>

## `HrfCandidate.__post_init__`

[Source, line 161](../../src/boldtailor/hrf_library.py#L161)

```python
HrfCandidate.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-11"></a>

## `HrfCandidate.kernel`

[Source, line 171](../../src/boldtailor/hrf_library.py#L171)

```python
HrfCandidate.kernel(self, tr, oversampling=OVERSAMPLING)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `HrfLibrary`

[Source, line 181](../../src/boldtailor/hrf_library.py#L181)

```python
@dataclass(frozen=True)
class HrfLibrary
```

Owned candidate order and 0.1-second curves; table access returns a copy.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
candidates: tuple[HrfCandidate, ...]
curves: np.ndarray = field(init=False, repr=False, compare=False)
times: np.ndarray = field(init=False, repr=False, compare=False)
fingerprint: str = field(init=False)
origin: Mapping[str, object] = field(default_factory=dict, compare=False)
```

<a id="definition-13"></a>

## `HrfLibrary.__post_init__`

[Source, line 190](../../src/boldtailor/hrf_library.py#L190)

```python
HrfLibrary.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `HrfLibrary.from_parameters`

[Source, line 215](../../src/boldtailor/hrf_library.py#L215)

```python
@classmethod
HrfLibrary.from_parameters(cls, parameters, origin=None, *, include_glmsingle=False)
```

Canonical SPM, the sorted double-gamma rows, then (optionally) the
20 GLMsingle kernels, with stable consecutive IDs.

<a id="definition-15"></a>

## `HrfLibrary.from_table`

[Source, line 236](../../src/boldtailor/hrf_library.py#L236)

```python
@classmethod
HrfLibrary.from_table(cls, table, origin=None)
```

Rebuild a library from its :attr:`parameter_table`.

<a id="definition-16"></a>

## `HrfLibrary.parameter_bounds`

[Source, line 246](../../src/boldtailor/hrf_library.py#L246)

```python
@property
HrfLibrary.parameter_bounds(self)
```

Per-parameter ``low``/``high`` over the six sampled box parameters.

Computed over custom candidates; canonical SPM is used only when there
are none. A parameter shared by all candidates has zero width.

<a id="definition-17"></a>

## `HrfLibrary._custom_parameters`

[Source, line 258](../../src/boldtailor/hrf_library.py#L258)

```python
HrfLibrary._custom_parameters(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `HrfLibrary.informative_parameters`

[Source, line 263](../../src/boldtailor/hrf_library.py#L263)

```python
@property
HrfLibrary.informative_parameters(self) -> tuple[str, ...]
```

Sampled parameters with at least three distinct custom values.

Constant and two-level grid parameters put every pick at an edge, so
they are excluded from the scalar ``at_parameter_bound`` diagnostic.

<a id="definition-19"></a>

## `HrfLibrary.candidate_bound_flags`

[Source, line 276](../../src/boldtailor/hrf_library.py#L276)

```python
HrfLibrary.candidate_bound_flags(self, margin=0.02)
```

Per-candidate ``(n_candidates, 6, 2)`` flags for ``[low, high]`` edges.

True where a custom candidate's parameter lies within ``margin`` of the
box width of that edge; canonical SPM rows are False.

<a id="definition-20"></a>

## `HrfLibrary.parameter_table`

[Source, line 291](../../src/boldtailor/hrf_library.py#L291)

```python
@property
HrfLibrary.parameter_table(self)
```

One row per candidate: SPM parameters (NaN for empirical kernels),
the GLMsingle ``source_index`` (NaN otherwise), and the 0.1 s peak time.

<a id="definition-21"></a>

## `HrfLibrary.timing_table`

[Source, line 310](../../src/boldtailor/hrf_library.py#L310)

```python
@property
HrfLibrary.timing_table(self)
```

Both parameterizations per candidate: SPM gamma values, lobe timing,
and realized timing. Gamma columns are NaN for empirical kernels; the
realized trough columns are NaN when a kernel has no undershoot.

<a id="definition-22"></a>

## `_box_parameters`

[Source, line 317](../../src/boldtailor/hrf_library.py#L317)

```python
_box_parameters(candidate)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:329](../../src/boldtailor/hrf_library.py#L329)
- [src/boldtailor/hrf_library.py:285](../../src/boldtailor/hrf_library.py#L285)
- [src/boldtailor/hrf_library.py:301](../../src/boldtailor/hrf_library.py#L301)

<a id="definition-23"></a>

## `_source_index`

[Source, line 323](../../src/boldtailor/hrf_library.py#L323)

```python
_source_index(candidate)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:303](../../src/boldtailor/hrf_library.py#L303)

<a id="definition-24"></a>

## `_timing_row`

[Source, line 327](../../src/boldtailor/hrf_library.py#L327)

```python
_timing_row(candidate)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:314](../../src/boldtailor/hrf_library.py#L314)

<a id="definition-25"></a>

## `_glmsingle_realized`

[Source, line 346](../../src/boldtailor/hrf_library.py#L346)

```python
_glmsingle_realized(index)
```

Realized timing of a GLMsingle kernel on the 0.01 s grid; the trough
columns are NaN when the kernel has no undershoot.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:332](../../src/boldtailor/hrf_library.py#L332)

<a id="definition-26"></a>

## `_glmsingle_candidates`

[Source, line 361](../../src/boldtailor/hrf_library.py#L361)

```python
_glmsingle_candidates(first_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:227](../../src/boldtailor/hrf_library.py#L227)

<a id="definition-27"></a>

## `_lobe_timing`

[Source, line 368](../../src/boldtailor/hrf_library.py#L368)

```python
_lobe_timing(delay, dispersion, onset)
```

Mode and standard deviation of a gamma lobe with shape delay/dispersion.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:393](../../src/boldtailor/hrf_library.py#L393)
- [src/boldtailor/hrf_library.py:394](../../src/boldtailor/hrf_library.py#L394)

<a id="definition-28"></a>

## `_lobe_parameters`

[Source, line 373](../../src/boldtailor/hrf_library.py#L373)

```python
_lobe_parameters(peak, sd, onset)
```

Invert :func:`_lobe_timing`: delay - dispersion = peak - onset, product = sd**2.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:428](../../src/boldtailor/hrf_library.py#L428)
- [src/boldtailor/hrf_library.py:429](../../src/boldtailor/hrf_library.py#L429)

<a id="definition-29"></a>

## `_lobe_height`

[Source, line 380](../../src/boldtailor/hrf_library.py#L380)

```python
_lobe_height(delay, dispersion)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:395](../../src/boldtailor/hrf_library.py#L395)
- [src/boldtailor/hrf_library.py:430](../../src/boldtailor/hrf_library.py#L430)

<a id="definition-30"></a>

## `timing_parameters`

[Source, line 384](../../src/boldtailor/hrf_library.py#L384)

```python
timing_parameters(parameters)
```

SPM gamma parameters to lobe timing: peak, SD, undershoot peak/SD, depth, onset.

Peak and SD are the mode and standard deviation of each gamma lobe; depth is
the undershoot lobe's height relative to the response lobe's height. The
response peak predicts the sampled kernel's maximum to about 0.1 s; the
undershoot values describe the lobe, not the trough of the combined curve.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:337](../../src/boldtailor/hrf_library.py#L337)
- [tests/test_hrf_timing.py:33](../../tests/test_hrf_timing.py#L33)
- [tests/test_hrf_timing.py:39](../../tests/test_hrf_timing.py#L39)
- [tests/test_hrf_timing.py:55](../../tests/test_hrf_timing.py#L55)

<a id="definition-31"></a>

## `spm_parameters`

[Source, line 407](../../src/boldtailor/hrf_library.py#L407)

```python
spm_parameters(timing)
```

Lobe timing (see :func:`timing_parameters`) back to SPM gamma parameters.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:574](../../src/boldtailor/hrf_library.py#L574)
- [tests/test_hrf_timing.py:35](../../tests/test_hrf_timing.py#L35)

<a id="definition-32"></a>

## `_dense_curve`

[Source, line 434](../../src/boldtailor/hrf_library.py#L434)

```python
_dense_curve(parameters)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:493](../../src/boldtailor/hrf_library.py#L493)

<a id="definition-33"></a>

## `_subgrid_extremum`

[Source, line 444](../../src/boldtailor/hrf_library.py#L444)

```python
_subgrid_extremum(times, curve, index)
```

Parabolic refinement of a grid extremum: (time, value).

Usage in package/tests:

- [src/boldtailor/hrf_library.py:501](../../src/boldtailor/hrf_library.py#L501)
- [src/boldtailor/hrf_library.py:502](../../src/boldtailor/hrf_library.py#L502)
- [src/boldtailor/hrf_library.py:357](../../src/boldtailor/hrf_library.py#L357)

<a id="definition-34"></a>

## `_width_at`

[Source, line 456](../../src/boldtailor/hrf_library.py#L456)

```python
_width_at(times, curve, index, level)
```

Width of the contiguous region around ``index`` where the curve is at least ``level``.

``level`` is compared with the same sign as the curve at ``index``; the
crossing points are linearly interpolated between grid samples.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:505](../../src/boldtailor/hrf_library.py#L505)
- [src/boldtailor/hrf_library.py:507](../../src/boldtailor/hrf_library.py#L507)
- [src/boldtailor/hrf_library.py:358](../../src/boldtailor/hrf_library.py#L358)

<a id="definition-35"></a>

## `realized_timing`

[Source, line 482](../../src/boldtailor/hrf_library.py#L482)

```python
realized_timing(parameters)
```

Peak time, response FWHM, trough time, undershoot FWHM, and trough depth.

All five are measured on the combined curve sampled at 0.01 s (extrema are
refined parabolically, half-maximum crossings are interpolated). Depth is
the trough amplitude relative to the unit peak and the undershoot FWHM is
the width at half that depth. These are the quantities
:func:`timing_hrf_library` samples; :func:`timing_parameters` describes the
gamma lobes instead. Raises ``ValueError`` when the curve has no trough
after its peak deeper than one part in a million of the peak.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:341](../../src/boldtailor/hrf_library.py#L341)
- [src/boldtailor/hrf_library.py:575](../../src/boldtailor/hrf_library.py#L575)
- [tests/test_hrf_timing.py:60](../../tests/test_hrf_timing.py#L60)
- [tests/test_hrf_timing.py:86](../../tests/test_hrf_timing.py#L86)

<a id="definition-36"></a>

## `_realized_from_curve`

[Source, line 497](../../src/boldtailor/hrf_library.py#L497)

```python
_realized_from_curve(times, curve)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:494](../../src/boldtailor/hrf_library.py#L494)
- [src/boldtailor/hrf_library.py:354](../../src/boldtailor/hrf_library.py#L354)

<a id="definition-37"></a>

## `_realized_close`

[Source, line 515](../../src/boldtailor/hrf_library.py#L515)

```python
_realized_close(realized, target, depth_tolerance)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-38"></a>

## `_corrected_lobe`

[Source, line 525](../../src/boldtailor/hrf_library.py#L525)

```python
_corrected_lobe(lobe, realized, target, damping)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:587](../../src/boldtailor/hrf_library.py#L587)

<a id="definition-39"></a>

## `_realized_error`

[Source, line 539](../../src/boldtailor/hrf_library.py#L539)

```python
_realized_error(realized, target, depth_tolerance)
```

Largest tolerance-scaled deviation between realized and target quantities.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:580](../../src/boldtailor/hrf_library.py#L580)

<a id="definition-40"></a>

## `spm_parameters_from_realized`

[Source, line 551](../../src/boldtailor/hrf_library.py#L551)

```python
spm_parameters_from_realized(timing, *, depth_tolerance=0.001, max_iterations=200, stall_iterations=15)
```

SPM gamma parameters whose combined curve realizes the requested timing.

``timing`` follows :data:`REALIZED_NAMES`: realized peak time, response
FWHM, trough time, undershoot FWHM, trough depth, onset, duration. The
gamma-lobe closed form (with SD = FWHM / 2.355) is the starting point and
the lobe peak, SD, trough, undershoot SD, and depth are corrected by damped
fixed-point iteration until the measured curve matches within 0.01 s
(times), 0.02 s (widths), and ``depth_tolerance`` (relative). The five
realized quantities are not independent for a double gamma; targets no
double gamma can realize raise ``ValueError`` once the iteration stalls for
``stall_iterations`` steps or exhausts ``max_iterations``.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:685](../../src/boldtailor/hrf_library.py#L685)
- [tests/test_hrf_timing.py:85](../../tests/test_hrf_timing.py#L85)
- [tests/test_hrf_timing.py:98](../../tests/test_hrf_timing.py#L98)
- [tests/test_hrf_timing.py:100](../../tests/test_hrf_timing.py#L100)

<a id="definition-41"></a>

## `expanded_hrf_library`

[Source, line 591](../../src/boldtailor/hrf_library.py#L591)

```python
expanded_hrf_library()
```

The original 648 double-gamma grid plus peak-scaled canonical SPM.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:341](../../src/boldtailor/workflow/settings.py#L341)

<a id="definition-42"></a>

## `sobol_hrf_library`

[Source, line 607](../../src/boldtailor/hrf_library.py#L607)

```python
sobol_hrf_library(n_samples=512, *, seed=0)
```

Sample continuous HRF parameters, plus peak-scaled canonical SPM at ID zero.

``n_samples`` must be a positive power of two; ``seed`` must be a
nonnegative integer. Scrambled Sobol points cover the six-dimensional
parameter box of :func:`expanded_hrf_library`, with duration fixed at
36 seconds. This balances parameter coverage, not waveform distances.
Custom candidates are sorted by parameters, not Sobol sequence order.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:339](../../src/boldtailor/workflow/settings.py#L339)
- [tests/test_hrf_timing.py:50](../../tests/test_hrf_timing.py#L50)

<a id="definition-43"></a>

## `_sobol_points`

[Source, line 623](../../src/boldtailor/hrf_library.py#L623)

```python
_sobol_points(n_samples, seed, *, dimensions)
```

The first ``n_samples`` scrambled Sobol points (``n_samples`` a power of two).

Usage in package/tests:

- [src/boldtailor/hrf_library.py:616](../../src/boldtailor/hrf_library.py#L616)

<a id="definition-44"></a>

## `_sobol_batches`

[Source, line 628](../../src/boldtailor/hrf_library.py#L628)

```python
_sobol_batches(n_samples, seed, *, dimensions)
```

Scrambled Sobol batches of ``n_samples``, ``n_samples``, ``2 n``, ``4 n``, ...

The batch sizes keep the total drawn a power of two, so every prefix is a
balanced Sobol set and the first batch equals :func:`_sobol_points`.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:682](../../src/boldtailor/hrf_library.py#L682)
- [src/boldtailor/hrf_library.py:625](../../src/boldtailor/hrf_library.py#L625)

<a id="definition-45"></a>

## `_doubling_batches`

[Source, line 646](../../src/boldtailor/hrf_library.py#L646)

```python
_doubling_batches(sampler, size)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:643](../../src/boldtailor/hrf_library.py#L643)

<a id="definition-46"></a>

## `_timing_bounds`

[Source, line 653](../../src/boldtailor/hrf_library.py#L653)

```python
_timing_bounds(bounds)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:712](../../src/boldtailor/hrf_library.py#L712)

<a id="definition-47"></a>

## `_scaled_timing`

[Source, line 667](../../src/boldtailor/hrf_library.py#L667)

```python
_scaled_timing(unit_points, limits)
```

Map unit-cube Sobol points onto the timing box, log-uniformly where listed.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:683](../../src/boldtailor/hrf_library.py#L683)

<a id="definition-48"></a>

## `_feasible_timing_rows`

[Source, line 679](../../src/boldtailor/hrf_library.py#L679)

```python
_feasible_timing_rows(n_samples, seed, limits, onset, duration)
```

Accept the first ``n_samples`` realizable Sobol points; count the rest.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:714](../../src/boldtailor/hrf_library.py#L714)

<a id="definition-49"></a>

## `timing_hrf_library`

[Source, line 694](../../src/boldtailor/hrf_library.py#L694)

```python
timing_hrf_library(n_samples=512, *, seed=0, bounds=None, onset=0.0, duration=36.0)
```

Sample realized HRF timing with scrambled Sobol, plus canonical SPM at ID zero.

The five sampled quantities follow :data:`REALIZED_NAMES` and are all
measured on the combined curve: peak time, response FWHM, trough time,
undershoot FWHM (width at half the trough depth), and trough depth relative
to the peak. Each point is
converted to SPM gamma parameters by :func:`spm_parameters_from_realized`;
points no double gamma can realize are skipped deterministically, and the
count skipped is recorded in ``origin["rejected"]``. ``onset`` is fixed
because onset and response delay trade off into the same peak time, which
is what makes the gamma-parameter box oversample near-identical waveforms.
``bounds`` overrides entries of :data:`TIMING_BOUNDS`. Quantities listed in
:data:`LOG_SCALED_TIMING` (trough depth) are sampled log-uniformly over
their bounds, the rest uniformly; ``origin["scales"]`` records which.
Candidates are stored with their SPM parameters, so downstream modeling is
unchanged.

Usage in package/tests:

- [src/boldtailor/hrf_library.py:744](../../src/boldtailor/hrf_library.py#L744)
- [tests/test_glmsingle_hrf.py:93](../../tests/test_glmsingle_hrf.py#L93)
- [tests/test_hrf_timing.py:111](../../tests/test_hrf_timing.py#L111)
- [tests/test_hrf_timing.py:132](../../tests/test_hrf_timing.py#L132)

<a id="definition-50"></a>

## `glmsingle_hrf_library`

[Source, line 728](../../src/boldtailor/hrf_library.py#L728)

```python
glmsingle_hrf_library()
```

Canonical SPM plus the 20 GLMsingle empirical HRFs.

Usage in package/tests:

- [tests/test_glmsingle_hrf.py:50](../../tests/test_glmsingle_hrf.py#L50)
- [tests/test_glmsingle_hrf.py:65](../../tests/test_glmsingle_hrf.py#L65)
- [tests/test_glmsingle_hrf.py:108](../../tests/test_glmsingle_hrf.py#L108)

<a id="definition-51"></a>

## `default_hrf_library`

[Source, line 738](../../src/boldtailor/hrf_library.py#L738)

```python
@lru_cache(maxsize=4)
default_hrf_library(n_samples=512, *, seed=0)
```

The library HRF selection uses when none is given: canonical SPM,
``n_samples`` timing-space candidates, then the 20 GLMsingle HRFs.

Built once per ``(n_samples, seed)`` and cached; libraries are immutable.

Usage in package/tests:

- [src/boldtailor/denoising.py:463](../../src/boldtailor/denoising.py#L463)
- [src/boldtailor/hrf_selection.py:143](../../src/boldtailor/hrf_selection.py#L143)
- [src/boldtailor/hrf_selection.py:224](../../src/boldtailor/hrf_selection.py#L224)
- [src/boldtailor/workflow/settings.py:337](../../src/boldtailor/workflow/settings.py#L337)

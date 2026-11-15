# `boldtailor._fit_lifecycle`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_fit_lifecycle.py)

Execution bookkeeping around explicit scientific fit operations.

## Definitions

- [`FitOperation`](#definition-1)
- [`FitOperation.provenance`](#definition-2)
- [`fit_operation`](#definition-3)

<a id="definition-1"></a>

## `FitOperation`

[Source, line 21](../../src/boldtailor/_fit_lifecycle.py#L21)

```python
@dataclass
class FitOperation
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
name: str
parent: ProvenanceRecord
execution_id: str = field(default_factory=lambda: str(uuid4()))
analysis_id: str | None = None
history: tuple[Mapping[str, object], ...] = ()
completed: Mapping[str, object] | None = None
```

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:57](../../src/boldtailor/_fit_lifecycle.py#L57)

<a id="definition-2"></a>

## `FitOperation.provenance`

[Source, line 29](../../src/boldtailor/_fit_lifecycle.py#L29)

```python
FitOperation.provenance(self, activity: Mapping[str, object], *, analysis_id: str | None, warnings: Sequence[Mapping[str, object]]=()) -> ProvenanceRecord
```

Prepare final provenance; completion is logged on successful exit.

<a id="definition-3"></a>

## `fit_operation`

[Source, line 56](../../src/boldtailor/_fit_lifecycle.py#L56)

```python
@contextmanager
fit_operation(name: str, parent: ProvenanceRecord) -> Iterator[FitOperation]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:155](../../src/boldtailor/_hrf_glm.py#L155)
- [src/boldtailor/_hrf_glm.py:194](../../src/boldtailor/_hrf_glm.py#L194)
- [src/boldtailor/_selected_hrf_fit.py:152](../../src/boldtailor/_selected_hrf_fit.py#L152)
- [src/boldtailor/denoising.py:469](../../src/boldtailor/denoising.py#L469)

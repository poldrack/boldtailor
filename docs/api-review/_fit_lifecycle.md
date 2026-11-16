# `boldtailor._fit_lifecycle`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_fit_lifecycle.py)

Updated 2026-11-16: independent logging scopes and provenance construction.

## Definitions

- [`FitOperation`](#definition-1)
- [`build_fit_provenance`](#definition-2)
- [`fit_operation`](#definition-3)

<a id="definition-1"></a>

## `FitOperation`

```python
@dataclass
class FitOperation:
    name: str
    parent: ProvenanceRecord
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    analysis_id: str | None = None
    history: tuple[Mapping[str, object], ...] = ()
```

Execution metadata. Callers may set `analysis_id` when the model identity is
known; no method call or completion flag is required for successful scope exit.

<a id="definition-2"></a>

## `build_fit_provenance`

```python
def build_fit_provenance(
    operation: FitOperation,
    activity: Mapping[str, object],
    *,
    warnings: Sequence[Mapping[str, object]] = (),
) -> ProvenanceRecord:
    ...
```

Builds provenance from the parent record, operation identity/history, activity,
software environment, and warnings. It does not mutate the operation or emit
log messages. Its completion snapshot has an independent timestamp and sequence
number from the completion log generated at scope exit. Repeated calls and calls
after scope exit are allowed. The builder uses the operation's identity rather
than inheriting unrelated ambient logging context.

<a id="definition-3"></a>

## `fit_operation`

```python
@contextmanager
def fit_operation(name: str, parent: ProvenanceRecord) -> Iterator[FitOperation]:
    ...
```

Logs start, then completion on normal exit or failure on an exception. Exceptions
are re-raised unchanged, with only the error category logged. Nested scopes get
fresh execution IDs and restore their enclosing logging context on exit.
Provenance construction is optional. Fitting callers keep result construction
inside the scope so constructor failures do not log successful completion.

See [lifecycle tests](../../tests/test_fit_lifecycle.py) and the
[entry-point coverage](../../tests/test_lifecycle.py).

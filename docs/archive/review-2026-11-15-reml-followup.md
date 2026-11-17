# Assessment of the REML review response

Date: 2026-11-15
Reviewed HEAD: `87affe6`

**Verdict: substantial improvement, but not adequate to close the review in full.**
R1 and the original R2 counterexample are addressed. R4 and R5 have reasonable
implementation fixes. R3 still has a reproducible false `no_signal` failure in
both REML and AR(1). R2's documentation also promises more than the fallback
path enforces. R6 and the upper-bound selection concern remain explicitly open.

## Findings requiring follow-up

### F1 (medium): the conditioning-weighted floor rejects exactly computed residuals

Location: `src/boldtailor/_numerics.py:54–69`, consumed by `reml._live` and
`_ar1_moments.basis_lag_ratio`.

Let `e0, e1, ...` be coordinate vectors in 30 dimensions. Use
`C = [1e-12 e0, e1]`, `Z = [e2]`, and `y = 2 e2 + e3`. Compare `y` with
`y + 100 e0`. The added vector lies in the fixed span. Both fixed columns are
retained, and both computed residuals are exactly `2 e2 + e3`, with energy 5.
Nevertheless:

| Result | Original signal | After fixed-span shift |
| --- | --- | --- |
| REML status | `ok` | `no_signal` |
| REML ratio | 107 | 0 |
| AR(1) raw lag ratio | 0.4 | NaN |

The shifted signal's floor is approximately 28.399, above the residual energy.
The `1e12` amplification reflects unequal column units, even though these
columns are orthogonal and their projection is exact. Replacing the first
fixed column with `e0`, without changing its span or the shifted signal,
restores `ok` and the lag ratio 0.4; the floor becomes about `2.84e-23`.

Thus this is neither loss of stored residual information nor a dropped-rank
case. It also represents a regression for these inputs: the old
`64 eps * raw_energy` floor is about `1.42e-10`, below 5.

The error analysis provides a conservative uncertainty scale, but being below
that bound does not establish that a particular computed residual is numerical
noise. Calibration only against exact-span signals establishes protection
against false live signals; it does not establish protection against false
dead signals. The new positive-residual tests use well-scaled designs and miss
this case. The response's claim that the floor restores invariance over the
resolvable range, and the corresponding claim in `docs/development.md`, are
too strong.

Required follow-up: add positive-residual tests with unequal fixed-column
scales and fixed-span shifts, alongside the existing exact-span tests. Improve
the projection/error policy so column units alone do not erase supported
residuals (column equilibration is one approach to investigate, with explicit
care for rank semantics). If a residual cannot be certified, distinguish that
uncertainty from a claim of `no_signal`. Merely lowering the multiplier is not
a sufficient general solution.

Saved executable evidence:
`docs/reml-review-2026-11-15-reproducers/test_response_projection.py`.
Both tests fail for the intended assertions on the reviewed HEAD. They are
outside the default suite; no implementation was changed.

### F2 (medium): the documented stationary-point guarantee is not enforced

Location: `src/boldtailor/_reml_search.py::_best_positive` and
`docs/development.md` under REML conventions.

The documentation says that an interior optimum is a stationary point to near
machine precision. The code can still return `log_grid[index]` when no refined
root is accepted, without verifying stationarity or exposing a fallback flag.
The response explicitly acknowledges the absence of an unresolved-search
status and adaptive subdivision. Finding no such case in the test collection
does not establish the advertised guarantee.

The all-crossings search and span-dependent grid are worthwhile fixes, and
the original counterexample now passes. A certified global solver is not
required to accept a clearly documented practical optimizer. However, either
enforce/check stationarity and expose unresolved outcomes, or remove the
unconditional stationary-point guarantee and explicitly document that an
unrefined grid fallback can receive `ok`. This finding is based on the control
flow and contract mismatch, not a new public-API numerical counterexample.

## Assessment by original finding

| Item | Assessment |
| --- | --- |
| R1 | Adequate for exact confounding at the retained-spectrum tolerance. Pooled degrees of freedom account for implicit zero directions; common versus differing group scales, NaN component estimates, and exclusion from selection are covered. Near-confounding remains a documented limitation. |
| R2 | Original failure fixed; stronger practical search and oracle. Full acceptance remains conditional on resolving F2. Missing two stationary points inside one cell is explicitly documented, not hidden. |
| R3 | Not closed: F1 reproduces the same class of false degeneracy in both public APIs. |
| R4 | Structural fix is adequate: the smaller Gram matrix removes unnecessary dependence on the number of columns for wide designs. Tests cover norm agreement and matrix shape. The requested representative benchmark and full wide-design oracle comparison remain absent. |
| R5 | Adequate for the narrowed per-search-array contract at the default budget, not a total-memory guarantee. Feature and crossing batching address the original oversized row; deterministic tests cover evaluation shapes and batch invariance. Peak-memory benchmarking remains absent. |
| R6 | Reasonable incremental consolidation, accurately described as partial. Other callers remain deferred. |
| Upper-bound selection | Still open. A flag on the selected fit is useful but does not reveal a truncated losing candidate whose true optimum might change the winner. No new numerical counterexample was established here. |

The response's correction of the original R2 mechanism is reasonable: the
failure was failure to refine a bracketed competing basin, not proof that the
minimum had no bracket anywhere on the grid. The commit history also supports
the claimed tests-before-implementation ordering for the fixes.

## Verification

- Targeted REML, AR(1), and numerics run, including the original three saved
  reproducers: **627 passed, 9 skipped, 2 xfailed**.
- New follow-up probes: **2 failed**, demonstrating F1 independently through
  the public REML and AR(1) APIs.
- Full default suite: **2312 passed, 1 failed, 9 skipped, 2 xfailed**. The
  failure is the reported `test_notebook_kernel_is_not_a_runtime_requirement`
  (`ipykernel` appears in installed runtime requirements). Neither
  `pyproject.toml` nor that test changed between the original reviewed commit
  and this HEAD.
- No implementation changes were made during this assessment.

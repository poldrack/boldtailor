# Pre-FitLins Remediation Design

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

**Date:** 2026-08-12

**Status:** Approved design

**Scope:** Targeted correctness, performance, privacy, documentation, and
versioning fixes before conventional FitLins integration

## Context

The architecture review in `docs/architecture-review-2026-08-12.md` identified
several concrete defects alongside broader architectural concerns. Boldtailor's
next product milestone remains conventional FitLins integration through the
prepared-design boundary. This remediation phase fixes the defects that would
otherwise weaken that boundary without pulling forward the larger imaging,
lifecycle, publication, or adaptive-model refactors.

The stop-signal notebook remains a temporary executable walkthrough for
understanding Boldtailor's API and intended data flow. It is not a stable
presentation contract or a substitute for the eventual FitLins adapter.

## Goals

1. Remove quadratic design-matrix copying from internal prepared fitting while
   preserving defensive copies at public boundaries.
2. Make task delta R-squared available for anonymous in-memory analyses without
   claiming deterministic provenance identity.
3. Apply one privacy-safe failure-logging policy to event-level and prepared
   fitting.
4. Make notebook configuration portable, explicit, and consistently
   five-session.
5. Resolve package versions consistently without breaking source-checkout
   imports.
6. Remove small verified dead code and brittle notebook presentation contracts.

## Non-goals

This phase does not:

- add or modify FitLins code;
- promote the example's BIDS discovery, masking, reconstruction, or derivative
  logic into Boldtailor;
- consolidate the event-level and prepared fit lifecycles;
- reorganize shared normalization helpers;
- redesign transactional publication or its threat model;
- implement optimized HRFs, GLMdenoise, fractional ridge, or GLMsingle; or
- make the notebook a permanent public application interface.

## Design-Matrix Ownership and Internal Access

Public `design_matrices` and `nuisance_design_matrices` properties continue to
return defensive deep copies. Callers may mutate returned DataFrames without
changing normalized analyses or results.

Internal fitting and provenance code must not retrieve one run by repeatedly
calling a property that copies every run. The owning containers will expose a
private trusted accessor or private trusted tuple for package-internal use.
These internal references must never be returned through a public API and must
not be mutated by package code.

Prepared fitting, task delta R-squared, and diagnostic construction will use
the trusted internal access path. The number of `DataFrame.copy(deep=True)`
operations during fitting must grow linearly with the number of runs, not
quadratically. Public defensive-copy behavior remains unchanged.

## Ephemeral Analysis Lineage

Every normalized `AnalysisData` and `PreparedDesignAnalysis` receives a fresh
private lineage token. Results created from an analysis retain the same token
internally.

The token exists only to establish in-process ownership: a full result can be
used for task delta R-squared only when its token matches the supplied
analysis. This permits anonymous array analyses to use the diagnostic while
still rejecting results from another analysis that happen to have compatible
dimensions.

Lineage tokens:

- are private implementation details;
- are never exposed by public properties;
- are never logged or serialized;
- do not contribute to provenance records or fingerprints;
- do not survive a serialization round trip; and
- do not imply reproducibility or stable cross-process identity.

Complete source metadata continues to determine deterministic data and
analysis fingerprints. Anonymous inputs continue to have no deterministic
metadata or analysis fingerprint and retain their provenance-quality warning.

## Failure Logging and Privacy

Event-level and prepared fitting use the same shared failure-record policy.
Structured failure events contain:

- the lifecycle event name;
- stage and severity;
- exception type; and
- one stable category: `validation_error`, `numerical_error`, or
  `internal_error`.

They do not contain exception messages, tracebacks, paths, environment values,
object representations, memory addresses, signals, tables, designs, estimates,
or statistics. The original exception is re-raised unchanged so callers retain
the detailed interactive diagnostic.

The full-environment `_redact_environment_values` scan is removed. Privacy no
longer depends on guessing which substrings in arbitrary exception text are
sensitive because arbitrary exception text never enters structured logs.
Provenance event history retains the same categorical failure fields as the
emitted log record.

Normalization failures are within the same privacy guarantee. Shared helpers
may distinguish normalization and fitting event names, but must apply the same
categorical payload rule whenever caller-derived exception text could otherwise
enter logs.

## Notebook Configuration

The notebook requires `BOLDTAILOR_BIDS_ROOT` for the machine-specific dataset
location. When it is absent, the first configuration cell raises an actionable
error showing how to set it. No personal absolute path is retained as a
fallback.

Analysis selection remains visible and ephemeral inside the notebook:

```python
SESSIONS = ("ses-02", "ses-04", "ses-06", "ses-08", "ses-10")
```

`BOLDTAILOR_SESSIONS` and `_configured_sessions()` are removed. Users edit the
ordinary notebook variable when defining a different analysis.

The generated BIDS fixture contains the complete five-session default, and
notebook tests execute that unmodified configuration. Bounded real-data
verification also executes all five default sessions.

The notebook remains an explicit API walkthrough. Its tests preserve semantic
contracts such as successful execution, the prepared-design boundary, compact
non-raw-data output, meaningful result maps, provenance, and temporary
publication. Exact display counts, literal titles, colormap names, and fixed
slice coordinates are not stable API contracts and must not be pinned merely
as presentation choices. Tests may still verify relationships that affect
scientific interpretation, such as all displayed maps sharing the same chosen
slice coordinates.

## Version Resolution

One internal helper resolves installed distribution versions with
`importlib.metadata.version()`. If metadata is genuinely unavailable in an
uninstalled source checkout, it returns `0+unknown` rather than failing import.

The helper is used for:

- Boldtailor version fields in stable BIDS and BEP028 provenance;
- Boldtailor, NumPy, and pandas versions in prepared-design provenance; and
- Nilearn numerical-backend metadata.

Version values are resolved lazily or through a safe module-level wrapper so
importing source modules cannot fail solely because Boldtailor is not installed
as a distribution. The hardcoded `BOLDTAILOR_VERSION = "0.1.0"` is removed.

## Small Correctness and Maintenance Fixes

The redundant reinsertion of `drift_order` into the model fingerprint is
removed because the activity already contains that exact key and value.

Machine-specific privacy assertions are replaced with fixture-supplied sentinel
paths, usernames, environment values, and hostnames. Assertions must prove that
provided sensitive values are absent rather than checking strings that never
entered the fixture.

Generated local artifacts are ignored through repository-wide patterns for
`__pycache__/`, `*.egg-info/`, and `.coverage`. Existing generated artifacts are
not committed.

The non-negotiable empty-`__init__.py` repository contract remains. Dependency
and packaging contracts may continue to protect intentional project policy,
but notebook presentation details are not treated as package API.

## Compatibility

Public fitting signatures, result accessors, deterministic fingerprint rules,
and defensive-copy guarantees remain compatible. The deliberate behavior
changes are:

- anonymous analyses may now compute task delta R-squared when the full result
  comes from the same in-memory analysis;
- structured failure logs no longer include exception messages;
- the notebook requires `BOLDTAILOR_BIDS_ROOT`;
- notebook session selection is an ordinary five-session variable; and
- presentation-only notebook details are no longer pinned by tests.

No new dependency is added. Every `__init__.py` remains completely empty.

## Testing

Implementation follows strict RED-GREEN-Refactor with test commits preceding
production commits.

Focused tests cover:

- linear rather than quadratic deep-copy counts across multiple run counts;
- unchanged public defensive-copy behavior;
- successful anonymous delta R-squared for matching lineage;
- rejection of dimension-compatible results from another lineage;
- absence of lineage tokens from logs and serialized provenance;
- identical categorical failure payloads for event-level and prepared paths;
- preservation of detailed exceptions for callers;
- proof that provided path, environment, username, hostname, traceback, repr,
  and address sentinels do not enter logs;
- required `BOLDTAILOR_BIDS_ROOT` with an actionable error;
- an unmodified five-session notebook configuration and fixture;
- semantic, output-tolerant notebook execution without cosmetic pinning;
- installed-version resolution and `0+unknown` fallback; and
- current BIDS provenance version projection.

Final verification runs:

- the focused warning-strict suites;
- the complete warning-strict pytest suite;
- Black;
- `uv lock --check`;
- diff and empty-initializer checks; and
- a five-session in-memory notebook execution against the local real dataset,
  with temporary-only derivative publication and no notebook write-back.

## Relationship to FitLins

This milestone stabilizes the array/DataFrame estimator boundary that FitLins
will consume. It does not move FitLins-owned responsibilities into Boldtailor.
After these fixes, conventional FitLins integration proceeds through
`PreparedDesignAnalysis` and `fit_prepared()` with cross-project fixtures.

The integration will then replace the notebook's role as the only end-to-end
application demonstration. At that point, application-level notebook tests can
move to FitLins, duplicate example helpers can be retired or minimized, and
the notebook can remain only if it is still useful as a smaller API tutorial.
Lifecycle consolidation should occur before adaptive HRF and denoising modes
would otherwise multiply the current duplication. Publication simplification
remains a separate decision based on its standalone use case and an explicit
threat and failure model.

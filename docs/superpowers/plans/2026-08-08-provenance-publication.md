# Derivative Logging and Provenance Implementation Plan

> Execution uses strict pytest RED-GREEN-Refactor cycles. Every task's tests
> must be committed and observed failing before its production implementation
> is written or committed.

**Goal:** Add immutable end-to-end provenance, correlated structured logging,
BIDS/BEP028 provenance projection, and a transactional publication core that
all future derivative writers must use.

**Architecture:** Provenance is a versioned, representation-neutral record
owned by `AnalysisData` and extended by `AnalysisResult`. Logging emits bounded,
structured lifecycle events through the `boldtailor` Python logger without
configuring application logging. Explicit derivative publication projects the
canonical record into BIDS metadata and the pinned BEP028 layout, then publishes
the whole artifact set under a lock with staging, rollback, and failure records.

**References:**

- BIDS software-development guidance:
  https://bids.neuroimaging.io/software_development/index.html
- BIDS BEP028 provenance draft:
  https://bids-specification.readthedocs.io/en/bep028/modality-agnostic-files/provenance.html

## Global constraints

- Write clean, modular code and prefer short functions.
- Use `uv` for package management and `uv run` for every local command.
- Every package `__init__.py` remains completely empty.
- Use function-style pytest tests and fixtures for persistent objects.
- Tests are committed before implementation. Each RED commit is run and shown
  to fail for the intended missing behavior before any associated production
  code is written. Tests are changed later only for a requirement change or a
  demonstrated test error.
- `fit()` performs no file I/O.
- The stable internal schema is canonical. BIDS and BEP028 files are exports of
  that record, not a second source of truth.
- Persist only dataset-relative POSIX paths or valid BIDS URIs. Reject absolute
  paths and path-like metadata that could expose a home directory, current
  working directory, username, or hostname.
- Artifact identity is metadata-only: source URI, role, media type, byte size,
  and modification time. Do not compute or emit content digests. A deterministic
  fingerprint hashes canonical metadata, not artifact bytes.
- Capture only curated runtime information: Boldtailor, Python, implementation,
  platform, Nilearn, NumPy, and pandas versions, with optional caller-supplied
  code URL, git revision, and container identifier. Never capture raw argv,
  environment variables, working directory, username, or hostname.
- Every normalize, fit, and publish attempt gets a fresh UUID execution ID. A
  deterministic data/analysis ID exists only when all required stable metadata
  exists; otherwise it is `None` and provenance records a quality warning.
- BEP028 export is enabled by default, records the pinned tested draft version,
  and has an explicit opt-out.
- Publication keeps a redacted failure JSONL log by default. It retains staged
  incomplete artifacts and their provenance only when explicitly requested.
- This delivery includes the provenance/logging foundation and internal writer
  core. It does not add BIDS discovery, NIfTI/CIFTI reconstruction, or concrete
  spatial derivative writers.

## Task 1: Canonical provenance schema and immutable source records

**Files:**

- Create: `src/boldtailor/provenance.py`
- Create: `tests/test_provenance.py`

**RED:** Add and commit tests proving the following observable contract:

- `SourceRef`, `RunSources`, and `ProvenanceRecord` own immutable copies of all
  nested inputs and expose no mutable mappings or sequences.
- `SourceRef` accepts a nonempty role/label, a dataset-relative POSIX URI or BIDS
  URI, optional media type, non-negative byte size, UTC ISO-8601 modification
  time, and JSON-safe annotations. It rejects absolute paths, traversal, invalid
  numeric fields, non-JSON values, NaN/infinity, and path-like annotation values.
- `RunSources` contains exactly one signal and events source and an optional
  confounds source.
- `ProvenanceRecord` uses schema identifier `boldtailor.provenance/1`, validates
  UUID execution IDs, preserves ordered immutable activities/events/warnings,
  and round-trips via `to_dict()` and `from_dict()` without sharing state.
- Canonical JSON is stable across mapping insertion order and produces a
  lowercase SHA-256 metadata fingerprint when every required source has URI,
  size, and modification time. The fingerprint is `None` and a provenance-
  quality warning is present when sources are anonymous or incomplete.
- Schema loading rejects unsupported major versions while accepting additive
  fields from version 1.
- No serialized record contains `Digest`, `digest`, absolute home/workspace
  prefixes, a username, or a hostname.

Run the focused tests and confirm they fail because the module/API is missing.

**GREEN:** Implement small validation, freezing, canonicalization, and
serialization helpers in `provenance.py`. Use standard-library dataclasses,
`MappingProxyType`, `json`, `hashlib`, `pathlib`, and `uuid`; do not add a schema
framework dependency. Do not put imports or exports in `__init__.py`.

Verify focused tests, the complete suite, and warnings-as-errors. Refactor only
while green, then commit production code separately from the RED commit.

## Task 2: Data lifecycle provenance and correlated structured logging

**Files:**

- Modify: `src/boldtailor/data.py`
- Create: `src/boldtailor/logging.py`
- Create: `tests/test_logging.py`
- Modify: `tests/test_data.py`

**RED:** Add and commit tests proving:

- `from_arrays(..., sources=None, provenance_metadata=None)` remains backward
  compatible and returns `AnalysisData.provenance`.
- `sources` accepts one `RunSources` per run and rejects count mismatches.
  Omitting sources creates anonymous in-memory source descriptors and a clear
  quality warning without inspecting or hashing array bytes.
- Caller metadata is JSON-safe, immutable, and rejects/redacts sensitive path-
  like values under the same policy as Task 1.
- Each call gets a fresh normalization UUID; stable source metadata yields the
  same data fingerprint across calls while anonymous input yields no fingerprint.
- The `boldtailor` logger emits structured normalization start/completion/failure
  records with timestamp, monotonic sequence, level, event, stage, execution ID,
  optional data/analysis ID, and optional run index.
- Logging uses `logging.getLogger("boldtailor")`, does not configure or mutate the
  root logger, does not duplicate events, and never logs signal values, event or
  confound table contents, argv, environment values, hostname, username, working
  directory, or absolute paths.
- A `contextvars`-backed binding keeps identifiers correlated across nested calls
  and resets them after success and failure.
- The canonical record carries a bounded ordered event history; no per-feature
  events are emitted.

**GREEN:** Add short logging/context helpers and extend `AnalysisData` with an
immutable provenance property. Construct a normalization activity and source
descriptors in `from_arrays`; emit failure logging before reraising validation
errors. Keep all previous array ownership and validation behavior unchanged.

Verify focused tests, full tests, and warnings-as-errors. Refactor while green
and commit production code separately.

## Task 3: Fit-stage provenance and model/design diagnostics

**Files:**

- Modify: `src/boldtailor/fit.py`
- Modify: `src/boldtailor/results.py`
- Modify: `src/boldtailor/provenance.py`
- Modify: `tests/test_fit.py`
- Modify: `tests/test_multirun.py`
- Modify: `tests/test_logging.py`

**RED:** Add and commit tests proving:

- `fit()` extends the input provenance rather than mutating it, gives every fit
  attempt a fresh UUID, and exposes the result through
  `AnalysisResult.provenance`.
- When a data fingerprint exists, the analysis fingerprint is deterministic over
  the canonical data fingerprint and complete model specification; changing a
  modeled option changes it. Anonymous input leaves it unavailable with a
  quality warning.
- Model serialization records named semantic contrasts or weight mappings,
  selected confounds, HRF, drift, high-pass, oversampling, min-onset, and noise
  model. A top-level importable callable is represented by module and qualified
  name; a lambda or local callable marks reproducibility partial and adds a
  warning without serializing source code or memory addresses.
- Each run records timing source, scan/feature counts, design column names,
  matrix rank, residual degrees of freedom, excluded-event count, min-onset
  cutoff, and rank-deficiency warnings. It does not serialize signal values,
  design values, estimates, statistics, or feature-level diagnostics.
- Fit start/completion/failure logging is correlated with normalization IDs and
  the fit execution ID. `fit()` still performs no filesystem writes.
- Existing `design_provenance` behavior and all Nilearn parity behavior remain
  unchanged.

**GREEN:** Add canonical model/callable and run-diagnostic builders, extend the
record immutably, and thread it through `make_result`. Keep numerical fitting
functions independent of provenance assembly where practical. Emit bounded
stage events through the shared logger and reraise original errors.

Verify focused tests, full tests, warnings-as-errors, and a test that patches
filesystem write primitives to prove `fit()` does not write. Refactor while
green and commit production separately.

## Task 4: Stable BIDS metadata and pinned BEP028 projection

**Files:**

- Create: `src/boldtailor/bids_provenance.py`
- Create: `tests/fixtures/bep028/` fixture files required for the pinned contract
- Create: `tests/test_bids_provenance.py`

**RED:** Add and commit exact projection tests against hand-authored fixtures:

- The projection returns an immutable mapping of dataset-relative POSIX paths to
  UTF-8 bytes; it performs no I/O.
- `dataset_description.json` contains `DatasetType: "derivative"`, Boldtailor in
  `GeneratedBy` with version and optional code/container fields, and normalized
  `SourceDatasets`/`DatasetLinks` when supplied.
- Default output contains `provenance.tsv`, its required JSON sidecar, and
  `prov/prov-<label>_act.json`, `_ent.json`, `_env.json`, and `_soft.json` records
  following the pinned BEP028 draft. Derivative-sidecar relationships point to
  the activity and source entities with dataset-relative paths or BIDS URIs.
- The export records the tested BEP028 draft identifier/version. Setting
  `export_bids_prov=False` omits only draft files; stable BIDS metadata and the
  canonical provenance JSON remain.
- Canonical JSON and correlated JSONL event output live under `logs/`, use
  deterministic ordering/newlines, and contain no content digest or sensitive
  path/runtime fields.
- Labels and paths are sanitized against traversal, separators, collisions, and
  invalid BIDS-style labels.

The fixtures pin the exact draft subset Boldtailor supports so upstream draft
changes cannot silently alter output. Tests validate semantic relationships,
not source-code text.

**GREEN:** Implement pure projection functions and small JSON/TSV serializers.
Use the internal record as the sole input. Add no validator/runtime dependency.
Document the exact pinned BEP028 subset in module constants and generated
metadata rather than claiming final-standard conformance.

Verify fixture tests, full tests, and warnings-as-errors. Refactor while green
and commit production separately.

## Task 5: Locked transactional artifact publication

**Files:**

- Create: `src/boldtailor/publication.py`
- Create: `tests/test_publication.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**RED:** Add and commit filesystem tests using `tmp_path` and real files:

- The internal `Artifact` value owns bytes and validates a relative POSIX path.
  `publish_artifact_set()` is the sole publication primitive and accepts
  `overwrite=False`, `retain_incomplete=False`, and `lock_timeout=30.0`.
- Preflight rejects empty sets, exact and case-folded duplicate paths, traversal,
  absolute paths, symlink destinations/parents, source-output overlap, invalid
  JSON/JSONL/TSV metadata payloads, and all collisions before replacing files.
- Publication acquires a cross-process lock, stages on the destination's file
  system, fsyncs staged files, and promotes them. Existing files are refused by
  default.
- With overwrite enabled, injected failure at every promotion boundary restores
  every original file byte-for-byte and leaves no new derivative artifact. A
  successful transaction replaces the requested set and removes backups/staging.
- Concurrent writers serialize; lock timeout produces a contextual error and no
  partial artifacts.
- Failure keeps one redacted JSONL failure log by default. With
  `retain_incomplete=True`, staged files and canonical provenance are retained
  under `.boldtailor/failed/<execution-id>/` and marked failed/unpublished.
- Source files are never modified. Temporary files, backups, lock data, and
  retained bundles are excluded from the returned published artifact paths.

**GREEN:** Add `filelock` as the only new runtime dependency. Implement short
preflight, staging, validation, promotion, rollback, cleanup, and failure-record
helpers. Accept source paths explicitly for overlap checks. Never follow
destination symlinks. Use `os.replace` for promotion and best-effort fsync of
files/directories. Log sanitized exception type/message, not traceback, inputs,
or environment.

Verify focused tests including fault injection/concurrency, full tests,
warnings-as-errors, lock consistency, and build. Refactor while green and commit
production separately.

## Task 6: Documentation and repository enforcement

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-07-general-first-level-fmri-package-design.md`
- Modify: `tests/test_repository_contracts.py`

**RED:** Add and commit behavioral repository-contract tests proving all package
`__init__.py` files are zero bytes and that the required runtime dependency and
packaged provenance modules are present in an installed wheel. Do not test human
prose by grepping exact wording.

**GREEN/DOCS:** Document source descriptors, anonymous-source limitations,
metadata-only identity, deterministic versus execution IDs, logger integration,
privacy/redaction, default and retained failure behavior, the pinned BEP028
draft/opt-out, and the rule that future adapters cannot write directly and must
route complete artifact sets through the publication core. State explicitly that
no BIDS discovery or NIfTI/CIFTI writer is included yet.

Verify the repository-contract tests, complete suite, warnings-as-errors,
`uv run black --check src tests`, `uv lock --check`, `uv build`, initializer
zero-byte checks, and `git diff --check`. Commit docs/production separately from
the preceding RED commit.

## Final review and completion

Generate a whole-branch review package from the merge base. Review schema
stability, privacy leakage, lifecycle correlation, strict TDD history,
transaction failure safety, dependency hygiene, and whether every future write
path is forced through the publication core. Resolve all Critical/Important
findings, run the full verification matrix again, and then use the finishing-a-
development-branch workflow.

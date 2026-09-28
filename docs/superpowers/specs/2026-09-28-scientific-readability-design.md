# Scientific readability refactor

> **Development plan:** Includes proposed work as well as implemented changes.
> See the [documentation index](../../README.md) for current API and methods guidance.

Date: 2026-09-28
Status: approved by the user on 2026-09-28; foundation, numerical, ownership, results, and shared-diagnostic increments implemented; lifecycle/logging, provenance, publication, and imaging work remains.

## Purpose and scope

Make Boldtailor easier for scientists to read, understand, and modify. Prefer
short functions that express analysis steps, explicit arrays and tables, and
small data containers. Reduce indirection and bookkeeping as well as line count.
Do not replace the current machinery with a generic workflow framework.

The user authorizes documented public API and output-metadata changes when
they materially simplify the project. Update examples and documentation in
the same change; avoid compatibility layers unless an identified consumer
needs one. Preserve scientific calculations and interpretation.

The architectural and cleanup findings in
`docs/review-2026-09-28-full-project.md` guide this work. Research decisions
about ridge tuning, fixed-effects weighting, or HRF selection objectives
belong to separate scientific work and must not change silently here.

## Baseline and preserved work

The requested merge contains only committed history through `57acff5`.
Existing uncommitted work is preserved separately in stash
`c4cfbec5fccd5397ea650c550f5b485ea588c2cd` (named
`pre-architecture-refactor: preserve uncommitted work 2026-09-28`).
That working tree passed 861 tests before preservation. The review describes
that broader tree, so its claims must be checked against the committed
baseline before implementation. In particular, do not assume that the
uncommitted scientific remediation is already present in main.

The committed-only suite finished with 752 passed and three failed notebook
tests. Each failure concerns a `stop_vs_go` contrast referring to `go_success`
in a prepared design that lacks that column. Inspection during planning traced
this to the notebook test helper rewriting both go-event labels to `go`, while
the committed notebook still uses the original labels. The tests also expect
the uncommitted notebook's contrast names and interactive views. Treat this
as a baseline integration mismatch, not a numerical-estimator failure.
With explicit user approval,
main was fast-forwarded to `57acff5` despite these known failures, then
`refactor/scientific-readability` was created from that commit. Address the
baseline failures explicitly before claiming a passing refactor suite.

The review document is retained as a reference on the refactoring branch;
its presence does not restore the stashed implementation. If a refactor
depends on a stashed fix, identify that dependency explicitly before work.

## Approach

Use incremental simplification, with independently reviewable changes.
A wholesale rewrite would make scientific equivalence harder to establish.
Cosmetic cleanup alone would leave the duplication and hidden control flow
identified by the review. The proposed order is:

1. Make the full test suite the default and establish a clean-runner CI check.
   Fix packaging, version lookup, and generated-file hygiene alongside this.
2. Clarify trial estimation and cross-validation. Separate nuisance projection,
   design decomposition, coefficient solving, and scoring. Replace consumers
   that coordinate generators with explicit candidate evaluation. Reuse the
   decomposition for fixed OLS targets without storing all candidate betas.
3. Consolidate fit bookkeeping and simplify data/result ownership. Share
   repeated validation and lifecycle operations through ordinary functions or
   one small context manager. Keep the statistical steps visible in each fit
   entry point. Remove repeated full-run defensive copies.
4. Simplify provenance, logging, and output publication against a documented
   local scientific-workflow contract. Keep useful reproducibility metadata;
   remove redundant record construction and conflicting privacy rules.
5. Move reusable imaging operations into the package, make notebooks narrative
   clients, and replace notebook instrumentation with direct module tests and
   a small execution smoke test. Update migration documentation throughout.

## Numerical code and containers

Keep raw-basis fractional ridge and normalized ridge visibly distinct.
Share projection and decomposition only where their mathematical meaning is
the same. Preserve scale-aware identifiability checks; eliminating a second
SVD is not justification for changing rank decisions on differently scaled
columns. Document coefficient units and array dimensions near the operations.

Use a small explicit prepared-fit container if it makes repeated evaluation
clearer. It should expose evaluation at a requested regularization value and
reuse per-run/per-HRF-group state. It must not depend on candidate call order
or accumulate a full candidate-by-trial-by-feature tensor.

Prefer composition or a single explicit result schema over mixins and deep
inheritance. Name the regularization parameter and grid directly rather than
detecting fields with `hasattr`. Own input arrays at construction and mark
result arrays read-only using normal NumPy flags. Document this as protection
against accidental mutation, not a security boundary. Avoid cross-module
mutation of frozen instances.

## Metadata, errors, and file output

Keep run/feature ordering, model settings, HRF identity, coefficient basis,
CV splits, and software versions available for interpretation and reproduction.
Label metadata identity accurately; do not imply that it verifies BOLD content.
Use one logging-boundary policy based on approved structured fields and error
categories. Keep actionable exceptions for callers without placing raw paths,
environment values, or arbitrary exception text in exported logs.

Proposed publication scope: a local scientific workflow, with explicit
overwrite behavior, basic path containment, input protection, and serialized
cooperating writers. Stage output on the destination filesystem and replace
individual files with rollback for recoverable Python-level failures.
Individual replacements do not make an entire artifact set atomic. Do not
promise all-or-nothing visibility to concurrent readers or crash recovery
without implementing and testing a stronger directory/manifest protocol.
Document limitations and preserve useful failure diagnostics before deleting
the existing recovery machinery. Retain the draft BIDS projection initially;
remove it only through an explicit consumer and requirements decision.

## Validation and completion criteria

Follow the repository's test-first requirement: write and run meaningful
failing tests for each new contract, commit them, then implement and refactor.
Keep independent numerical oracles and leakage tests. Revise tests only for
documented requirement changes or demonstrated test errors, never merely to
make a refactor pass. Avoid replacing behavior tests with source-text checks.

Check scientific equivalence for coefficients, fitted signals, contrasts,
NaN masks, HRF choices, CV losses, regularization choices, and run/feature
ordering within the existing justified tolerances. Exercise arbitrary
candidate evaluation order and bounded memory in the new CV design. Test
publication overwrite and injected failure behavior against its stated scope.

Run the complete pytest suite with warnings treated as errors, relevant
formatting checks, and an installed-package smoke test. All Python commands
use `uv run`; every `__init__.py` stays empty. Document changed APIs with
before/after examples and remove stale notebook outputs and personal defaults.

Judge the result by whether a scientist can trace design -> fitting ->
selection -> results without navigating logging and identity internals.
Report removed duplication and indirection; do not impose an arbitrary line
count target that encourages dense code or weakened scientific validation.

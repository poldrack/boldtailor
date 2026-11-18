# Documentation

Functions and classes can be imported from their defining modules, and the
top-level `boldtailor` package also re-exports the primary entry points. The
`boldtailor.workflow` initializer is empty.

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Installation and a runnable first model |
| [Practical guide](practical-guide.md) | Introduction to each analysis component, with runnable examples |
| [User guide](user-guide.md) | Scientific methods, interpretation, and [`boldtailor run`](user-guide.md#running-the-full-workflow) |
| [API reference](api.md) | Signatures, defaults, result fields, and provenance records |
| [Code-review guide](code-review.md) | Module organization, review routes, and runnable examples |
| [Developer guide](development.md) | Architecture, numerical conventions, logging, and file publication |
| [Testing guide](testing.md) | Module-to-test mappings, integration checks, and opt-in suites |
| [GLMsingle comparison](glmsingle-comparison.md) | Method differences and assumptions |
| [BIDS workflow examples (NSD data)](../examples/bids_workflow/README.md) | Dataset commands, notebooks, and saved outputs |

## Overview of methods

- **Conventional GLMs**: OLS or AR(1) contrast inference with equal-weight
  fixed-effects combination across runs, plus nested-OLS task-versus-nuisance
  $\Delta R^2$ diagnostics.
- **HRF selection**: Per-location selection from a candidate library by
  leave-one-run-out task prediction (`method="cv"`) or restricted maximum
  likelihood (`method="reml"`).
- **Single-trial estimation**: Per-trial beta estimation with OLS, normalized
  ridge (`ridge_alpha`), or per-location fractional ridge (`ridge_fraction`)
  tuned against fixed OLS validation targets using within-run centered trial
  encoding.
- **Task-guided denoising**: Optional GLMdenoise-style noise-pool PCA and
  held-out count selection with an AR(1)-prewhitened F-test significance gate.

## Historical records

Dated design specs, implementation plans, migration notes, code-review
inventories, and validation reports are archived under [`docs/archive/`](archive/).
They record historical decisions and experiments and may reference superseded
interfaces or defaults.

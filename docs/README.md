# Documentation

The guides below describe the current code on this branch. Import functions
from their named modules; package initializers are empty.

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Installation and a runnable first model |
| [User guide](user-guide.md) | Scientific methods, examples, and interpretation |
| [API reference](api.md) | Signatures, defaults, result fields, and source records |
| [Developer guide](development.md) | Implementation, testing, ownership, logging, and publication |
| [Publication migration](publication-migration.md) | Local writer guarantees and retained rollback backups |
| [Lifecycle migration](lifecycle-migration.md) | Shared fit events and categorical errors |
| [Result migration](result-migration.md) | Unified candidate scores and composed trial results |
| [Ownership migration](ownership-migration.md) | Ordinary read-only NumPy arrays and editable copies |
| [GLMsingle comparison](glmsingle-comparison.md) | Method differences and assumptions |
| [NSD guide](../examples/NSD/README.md) | Dataset-specific commands, notebooks, and saved artifacts |
| [Stop-signal notebook](../examples/stop_signal_demo.ipynb) | Prepared-design NIfTI workflow |

## Current methods

Conventional GLMs support OLS or AR(1) inference and equal-run contrast
combination. Task-versus-nuisance diagnostics refit nested OLS models.
Single-trial normalized ridge uses a fixed alpha on unit-norm projected
columns. Fractional ridge uses raw projected coefficient norms and fixed OLS
validation targets. Both encoding scorers default to within-run slopes and
centered held-out scoring; shared-alpha CV retains candidate-regularized
targets. The [user guide](user-guide.md) explains the assumptions and NaNs.

## Development and validation records

Files under `superpowers/plans`, `superpowers/specs`, and the validation
directories are dated records, not a substitute for the current API reference.
They may show proposed interfaces, old defaults, earlier working trees, and
test counts from other revisions. Reproduction commands describe those recorded
experiments; rerunning them with current defaults may produce different results.
Keep historical measurements and decisions intact when updating guidance.

The active [refactor roadmap](superpowers/plans/2026-09-28-scientific-readability-roadmap.md)
records the implemented stages, including the revised decision to keep imaging
and dataset adapters in examples, and the final architecture review. The original
[project review](review-2026-09-28-full-project.md) examined a broader working
tree that included separately preserved uncommitted work. It does not describe
the current branch verbatim.

- [Current refactor verification and final architecture review](validation/scientific-readability-notebooks-2026-09-28.md)
- [Publication verification](validation/scientific-readability-publication-2026-09-28.md)
- [Documentation audit](validation/documentation-audit-2026-09-28.md)
- [Fractional-default adoption record](validation/fractional-defaults-2026-09-28.md)
- [Historical fractional ablation](validation/fractional-ridge-ablation-2026-09-28.md)
- [Historical NSD measurements](validation/nsd-session.md)
- [Method recovery validation, 2026-10](validation/recovery-2026-10.md)

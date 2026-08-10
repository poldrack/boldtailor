# SDD ledger — plan: docs/superpowers/plans/2026-08-09-task-delta-r2.md

## Baseline

- Branch: `feature/task-delta-r2`
- Base commit: `6fc4cc7`
- Verification: `uv run pytest -q -W error` — 219 passed in 42.90s

## Tasks

- [x] Task 1: Nuisance-only design compilation
- [x] Task 2: Delta-R-squared result, fit, logging, and provenance
- [ ] Task 3: Deterministic delta-R-squared image
- [ ] Task 4: Notebook display and shareable provenance
- [ ] Task 5: Real-data and whole-branch verification

Task 1: complete (commits 6fc4cc7..d844742, review clean)
Task 2: minor (deferred): add direct multi-run pooled-R² and task-level zero-SST regression coverage
Task 2: fix round 1/5 (2 addressed, 1 open — successful completion missing from returned provenance history; commits 3550849..8525350)
Task 2: fix round 2/5 (1 addressed, 0 open — completion retained in returned provenance; commits 8525350..9659c7f)
Task 2: complete (commits 1a8b6f4..9659c7f, review clean; 1 deferred minor)

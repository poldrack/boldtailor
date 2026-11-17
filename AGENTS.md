# Development Instructions

## Code style

- Write clean, modular code.
- Prefer short functions over long functions.

## Package management

- Use uv for package management.
- Use `uv run` for all local Python and test commands.

## Package structure

- Export primary public symbols cleanly in top-level `__init__.py`.
- Keep module boundaries clean and avoid unnecessary micro-module fragmentation.

## Testing

- Use pytest and RED-GREEN-Refactor.
- Write and commit failing tests before implementation.
- Prefer test functions and pytest fixtures.
- Never weaken a test or simplify a requirement to make an incomplete implementation pass.

## Test selection

- During edits, run the affected module tests and tests of their direct consumers.
- Before committing, run those tests plus `uv run pytest -m integration -q -W error`.
- Run the full default suite before merging, for shared changes with unclear impact,
  and after changing test-selection rules: `uv run pytest -q -W error --durations=20`.
- Do not rerun the full suite after every local edit when targeted checks suffice.
- Use [docs/testing.md](docs/testing.md) for module mappings and opt-in scientific,
  experiment, and notebook checks. State which suites actually ran.

- Give every test function a short, one- or two-sentence purpose docstring.
- Review redundant tests by the behavior and parameter cases they protect; runtime
  alone is not a reason to delete scientific coverage.

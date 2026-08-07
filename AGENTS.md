# Development Instructions

## Code style

- Write clean, modular code.
- Prefer short functions over long functions.

## Package management

- Use uv for package management.
- Use `uv run` for all local Python and test commands.

## Package structure

- Every `__init__.py` must remain completely empty.

## Testing

- Use pytest and RED-GREEN-Refactor.
- Write and commit failing tests before implementation.
- Prefer test functions and pytest fixtures.
- Never weaken a test or simplify a requirement to make an incomplete implementation pass.

# Final review symlink-safety fix

## RED

- Commit: `8707480 test: reject symlinked persistent destinations`
- Changed only `tests/test_stop_signal_demo.py`.
- Added a parameterized behavioral regression using real filesystem symlinks:
  `<dataset>/derivatives` and `<dataset>/derivatives/boldtailor` each point to
  an external temporary directory.
- Command: `uv run pytest tests/test_stop_signal_demo.py -q`
- Output: `2 failed, 24 passed in 4.24s`.  Both new cases failed with
  `Failed: DID NOT RAISE ValueError`, demonstrating that persistent output
  could follow either symlink.

## GREEN

- Commit: `5177024 fix: reject symlinked report destinations`
- Changed only `examples/stop_signal_demo.py`.
- `publication_destination` now builds the persistent destination without
  resolving it, rejects a symlinked `derivatives` component or `boldtailor`
  leaf, and resolves only after those checks.  Temporary publication and the
  requested persistent-destination equality contract are unchanged.
- Commands and output:

  - `uv run pytest tests/test_stop_signal_demo.py -q` — `26 passed in 4.09s`.
  - `uv run pytest -q -W error` — `213 passed in 6.01s`.
  - `uv run git diff --check` — exit 0 with no output.

## Self-review

- The guard checks exactly the two persistent path components identified in
  the final review before any resolution can erase their symlink identity.
- The tests use external targets under `tmp_path`; no external BIDS dataset was
  changed.
- No `__init__.py` files were changed.  The implementation and its regression
  test are the only code/test changes in this fix wave; this report is the only
  documentation artifact.

## Concerns

None identified.  The guard intentionally does not alter the existing
temporary-output behavior or broaden the requested persistent destination
contract.

# Test-suite maintenance review — 2026-11-16

The review retired 12 tests that pinned documentation wording, retained the
scientific and behavioral regressions, and added short purpose docstrings to
all 1,600 remaining test functions across package, experiment and example files.
Function counts differ from pytest case counts because of parametrization.

## Scope and method

The default suite was inventoried across all 86 original test files, including
workflow tests. Static checks examined test bodies, assertion presence (including
assertion helpers and subprocess failures), identical bodies, source/prose
inspection, skips, expected failures and existing purpose documentation. Suspect
cases were inspected manually, including their parameter grids and regression
history. The full suite was timed and rerun after the maintenance changes.

Experiment and example test functions received purpose docstrings too. Their
executable bodies were compared with the committed versions using parsed Python
syntax trees with docstrings removed; they were unchanged. This was not a new
scientific validation of every optional experiment, nor a claim that all possible
redundancy can be established statically. No further obsolete behavioral test
was confirmed in this review.

## Retired checks

| Location | Removed checks | Reason |
| --- | --- | --- |
| `tests/test_docs_mention_cli.py` | Five tests requiring CLI/API phrases, particular headings, and terminology | They checked the presence or absence of words rather than command behavior or executable examples. |
| `tests/test_denoising_validation.py` | Seven tests requiring wording about the denoising contract, significance gate, prewhitening, attribution of the gate, interpretation of its statistic, pool deviations, and feature terminology | Word matching could pass with incorrect explanations and reject equivalent accurate wording. Review these scientific explanations as prose instead. |

The now-unused `denoising_doc_sections` helper was also removed. Documentation
itself was retained. The executable user-guide denoising example, actual CLI
checks, denoising numerical validation, resource attribution and licensing checks
remain. Removing prose tests primarily reduces maintenance friction; it is not
expected to materially shorten runtime.

## Investigated and retained

| Candidate | Why it remains useful |
| --- | --- |
| Six-run synthetic BIDS subprocess smoke test | Its history identifies a SIGBUS regression. A subprocess detects a native crash without terminating the pytest process. |
| Import-cycle smoke test | `subprocess.run(check=True)` is its assertion; it detects import-order failures in a fresh interpreter. |
| Prepared-model and canonical-HRF tests without inline `assert` statements | They call numerical assertion helpers; they are not assertion-free tests. |
| Two denoising validation tests with identical bodies | Their option grids cover general selection settings and significance-gate settings separately. |
| Two categorical-level tests with identical bodies | Their parameter grids cover different scalar and numeric-string conversions. |
| Two strict MixedLM expected failures | They document cases where the external optimizer misses the optimum; the independent dense REML oracle covers the same cases. Unexpected passes still fail. |
| Slow default-library denoising and HRF-library checks | They exercise the real default-library path, which reduced-library fixtures cannot replace. |
| Shared-helper, public-export, empty-initializer and package tests | These protect explicit architecture, public API and distribution requirements, even where assertions inspect structure. |
| License/attribution checks | These verify deliberate distribution requirements rather than general prose style. |

## Runtime findings and testing workflow

On this local environment, before the prose-only removals:

- Full default suite: 2,520 passed, 9 skipped, 2 expected failures in 209.54 s.
- REML module group (`tests/test_reml*.py`): 608 passed, 9 skipped,
  2 expected failures in 19.06 s.
- Representative integration suite: 21 passed in 7.86 s.

These are observed runs, not controlled performance benchmarks. The largest
individual call costs in the full run were:

| Test | Call time |
| --- | ---: |
| Default-library denoising rejects independent noise | 14.67 s |
| HRF selection uses the default library | 7.71 s |
| Default library combines timing samples, GLMsingle and canonical HRFs | 6.92 s |
| Workflow CLI reports missing inputs | 6.17 s |
| Notebook walkthrough preserves map/trial alignment | 4.15 s |

The primary time saving comes from affected-module tests plus the integration
suite during development, rather than deleting valuable coverage. The full
suite remains the pre-merge/CI check. CI now prints its 20 slowest phases so
future optimization can target measured bottlenecks. Fixture/library reuse may
be worth a separate investigation, provided it preserves isolation and testing
of the default settings.

## Final verification

After retiring the 12 prose checks and adding the documentation check, the
default suite passed: **2,509 passed, 9 skipped, 2 expected failures** in
203.67 s. Formatting and whitespace checks also passed. Optional experiment
and example suites were not rerun for docstring-only edits; parsed executable
bodies were verified unchanged.

## Ongoing maintenance

`tests/test_test_documentation.py` requires a purpose docstring on every package,
experiment and example test function. Descriptions identify the behavior being
checked; longer derivations belong in comments or oracle helpers. The integration
selection check protects representative public boundaries without claiming that
unmarked tests are exclusively unit tests. See [testing.md](testing.md) for the
commands and module-to-test mapping.

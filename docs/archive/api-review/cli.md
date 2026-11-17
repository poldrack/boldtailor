# `boldtailor.cli`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/cli.py)

Command line: `boldtailor run` executes the workflow on one BIDS subject's sessions.

## Definitions

- [`_OneLineParser`](#definition-1)
- [`_OneLineParser.error`](#definition-2)
- [`_mesh`](#definition-3)
- [`_choice`](#definition-4)
- [`_add_inputs`](#definition-5)
- [`_add_hrf`](#definition-6)
- [`_add_ridge`](#definition-7)
- [`_add_stages`](#definition-8)
- [`_add_execution`](#definition-9)
- [`build_parser`](#definition-10)
- [`_stages`](#definition-11)
- [`_modulators`](#definition-12)
- [`sessions_from_args`](#definition-13)
- [`_only_session`](#definition-14)
- [`settings_from_args`](#definition-15)
- [`_configure_logging`](#definition-16)
- [`_plan`](#definition-17)
- [`_print_plans`](#definition-18)
- [`_run_session`](#definition-19)
- [`_summary`](#definition-20)
- [`_run`](#definition-21)
- [`_plan_status`](#definition-22)
- [`_fail`](#definition-23)
- [`_execute`](#definition-24)
- [`main`](#definition-25)

<a id="definition-1"></a>

## `_OneLineParser`

[Source, line 25](../../src/boldtailor/cli.py#L25)

```python
class _OneLineParser(argparse.ArgumentParser)
```

Usage errors are one stderr line and exit status 1.

Usage in package/tests:

- [src/boldtailor/cli.py:126](../../src/boldtailor/cli.py#L126)

<a id="definition-2"></a>

## `_OneLineParser.error`

[Source, line 28](../../src/boldtailor/cli.py#L28)

```python
_OneLineParser.error(self, message)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `_mesh`

[Source, line 33](../../src/boldtailor/cli.py#L33)

```python
_mesh(text)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `_choice`

[Source, line 40](../../src/boldtailor/cli.py#L40)

```python
_choice(group, flag, choices, default, **kwargs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:64](../../src/boldtailor/cli.py#L64)
- [src/boldtailor/cli.py:84](../../src/boldtailor/cli.py#L84)
- [src/boldtailor/cli.py:92](../../src/boldtailor/cli.py#L92)
- [src/boldtailor/cli.py:99](../../src/boldtailor/cli.py#L99)

<a id="definition-5"></a>

## `_add_inputs`

[Source, line 50](../../src/boldtailor/cli.py#L50)

```python
_add_inputs(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `_add_hrf`

[Source, line 82](../../src/boldtailor/cli.py#L82)

```python
_add_hrf(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `_add_ridge`

[Source, line 90](../../src/boldtailor/cli.py#L90)

```python
_add_ridge(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `_add_stages`

[Source, line 102](../../src/boldtailor/cli.py#L102)

```python
_add_stages(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `_add_execution`

[Source, line 111](../../src/boldtailor/cli.py#L111)

```python
_add_execution(run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `build_parser`

[Source, line 125](../../src/boldtailor/cli.py#L125)

```python
build_parser()
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:287](../../src/boldtailor/cli.py#L287)

<a id="definition-11"></a>

## `_stages`

[Source, line 138](../../src/boldtailor/cli.py#L138)

```python
_stages(skipped)
```

Enabled stages; skipping a stage also skips every stage that requires it.

Usage in package/tests:

- [src/boldtailor/cli.py:189](../../src/boldtailor/cli.py#L189)

<a id="definition-12"></a>

## `_modulators`

[Source, line 146](../../src/boldtailor/cli.py#L146)

```python
_modulators(args)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:178](../../src/boldtailor/cli.py#L178)

<a id="definition-13"></a>

## `sessions_from_args`

[Source, line 152](../../src/boldtailor/cli.py#L152)

```python
sessions_from_args(args)
```

The requested sessions in run (sorted) order; a repeat is an error.

Usage in package/tests:

- [src/boldtailor/cli.py:161](../../src/boldtailor/cli.py#L161)
- [src/boldtailor/cli.py:274](../../src/boldtailor/cli.py#L274)

<a id="definition-14"></a>

## `_only_session`

[Source, line 160](../../src/boldtailor/cli.py#L160)

```python
_only_session(args)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:173](../../src/boldtailor/cli.py#L173)

<a id="definition-15"></a>

## `settings_from_args`

[Source, line 167](../../src/boldtailor/cli.py#L167)

```python
settings_from_args(args, session=None)
```

Settings for one session: ``session``, or the only one requested.

Usage in package/tests:

- [src/boldtailor/cli.py:274](../../src/boldtailor/cli.py#L274)

<a id="definition-16"></a>

## `_configure_logging`

[Source, line 199](../../src/boldtailor/cli.py#L199)

```python
_configure_logging(quiet)
```

Workflow progress to stderr; package events stay at warnings only.

Usage in package/tests:

- [src/boldtailor/cli.py:240](../../src/boldtailor/cli.py#L240)

<a id="definition-17"></a>

## `_plan`

[Source, line 208](../../src/boldtailor/cli.py#L208)

```python
_plan(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:241](../../src/boldtailor/cli.py#L241)

<a id="definition-18"></a>

## `_print_plans`

[Source, line 218](../../src/boldtailor/cli.py#L218)

```python
_print_plans(plans)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:243](../../src/boldtailor/cli.py#L243)

<a id="definition-19"></a>

## `_run_session`

[Source, line 222](../../src/boldtailor/cli.py#L222)

```python
_run_session(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:248](../../src/boldtailor/cli.py#L248)

<a id="definition-20"></a>

## `_summary`

[Source, line 232](../../src/boldtailor/cli.py#L232)

```python
_summary(sessions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:249](../../src/boldtailor/cli.py#L249)

<a id="definition-21"></a>

## `_run`

[Source, line 238](../../src/boldtailor/cli.py#L238)

```python
_run(args, sessions)
```

Check every session before fitting any, then run them one at a time.

Usage in package/tests:

- [src/boldtailor/cli.py:278](../../src/boldtailor/cli.py#L278)

<a id="definition-22"></a>

## `_plan_status`

[Source, line 253](../../src/boldtailor/cli.py#L253)

```python
_plan_status(plans)
```

Report the plans' problems and exit as the real run would (in run order).

Usage in package/tests:

- [src/boldtailor/cli.py:244](../../src/boldtailor/cli.py#L244)

<a id="definition-23"></a>

## `_fail`

[Source, line 264](../../src/boldtailor/cli.py#L264)

```python
_fail(message, code)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/cli.py:276](../../src/boldtailor/cli.py#L276)
- [src/boldtailor/cli.py:280](../../src/boldtailor/cli.py#L280)
- [src/boldtailor/cli.py:282](../../src/boldtailor/cli.py#L282)

<a id="definition-24"></a>

## `_execute`

[Source, line 269](../../src/boldtailor/cli.py#L269)

```python
_execute(args)
```

Settings and existing outputs exit 1, inputs exit 2; fitting errors propagate.

Usage in package/tests:

- [src/boldtailor/cli.py:290](../../src/boldtailor/cli.py#L290)

<a id="definition-25"></a>

## `main`

[Source, line 285](../../src/boldtailor/cli.py#L285)

```python
main(argv=None)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

# `boldtailor.bids_provenance`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/bids_provenance.py)

Project analysis records into BIDS metadata and provenance artifacts.

## Definitions

- [`project_bids_provenance`](#definition-1)
- [`_stable_artifacts`](#definition-2)
- [`_draft_artifacts`](#definition-3)
- [`_draft_graph`](#definition-4)
- [`_software`](#definition-5)
- [`_environment`](#definition-6)
- [`_activities`](#definition-7)
- [`_activity_command`](#definition-8)
- [`_activity_times`](#definition-9)
- [`_entities`](#definition-10)
- [`_file_entity`](#definition-11)
- [`_anonymous_entity`](#definition-12)
- [`_relationship_sidecars`](#definition-13)
- [`_source_identifier`](#definition-14)
- [`_generated_by`](#definition-15)
- [`_normalize_container`](#definition-16)
- [`_normalize_source_datasets`](#definition-17)
- [`_normalize_source_dataset`](#definition-18)
- [`_normalize_dataset_links`](#definition-19)
- [`_validate_link_location`](#definition-20)
- [`_normalize_sidecars`](#definition-21)
- [`_normalize_relationship_sources`](#definition-22)
- [`_record_source_uris`](#definition-23)
- [`_validate_sidecar_path`](#definition-24)
- [`_validate_label`](#definition-25)
- [`_slug`](#definition-26)
- [`_validate_text`](#definition-27)
- [`_validate_uri`](#definition-28)
- [`_require_record`](#definition-29)
- [`_extend_artifacts`](#definition-30)
- [`_provenance_tsv`](#definition-31)
- [`_provenance_sidecar`](#definition-32)
- [`_freeze_artifacts`](#definition-33)
- [`_pretty_json`](#definition-34)
- [`_compact_json`](#definition-35)
- [`_json_lines`](#definition-36)
- [`_canonical_text`](#definition-37)

<a id="definition-1"></a>

## `project_bids_provenance`

[Source, line 41](../../src/boldtailor/bids_provenance.py#L41)

```python
project_bids_provenance(record: ProvenanceRecord, *, dataset_name: str='Boldtailor derivatives', label: str='boldtailor', code_url: str | None=None, container: Mapping[str, str] | None=None, source_datasets: Sequence[Mapping[str, str]]=(), dataset_links: Mapping[str, str] | None=None, derivative_sidecars: Mapping[str, Sequence[str]] | None=None, export_bids_prov: bool=True) -> Mapping[str, bytes]
```

Project an internal record into deterministic in-memory BIDS artifacts.

Stable metadata follows BIDS 1.11.1. Draft files implement only the pinned
BEP028 subset named by ``BEP028_SUPPORTED_SUBSET``; they do not claim
conformance to a final BIDS provenance standard.

Usage in package/tests:

- [tests/test_bids_provenance.py:115](../../tests/test_bids_provenance.py#L115)
- [tests/test_bids_provenance.py:137](../../tests/test_bids_provenance.py#L137)
- [tests/test_bids_provenance.py:156](../../tests/test_bids_provenance.py#L156)
- [tests/test_bids_provenance.py:172](../../tests/test_bids_provenance.py#L172)

<a id="definition-2"></a>

## `_stable_artifacts`

[Source, line 83](../../src/boldtailor/bids_provenance.py#L83)

```python
_stable_artifacts(record: ProvenanceRecord, *, dataset_name: str, generated_by: Mapping[str, object], source_datasets: tuple[Mapping[str, str], ...], dataset_links: Mapping[str, str]) -> dict[str, bytes]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:63](../../src/boldtailor/bids_provenance.py#L63)

<a id="definition-3"></a>

## `_draft_artifacts`

[Source, line 109](../../src/boldtailor/bids_provenance.py#L109)

```python
_draft_artifacts(record: ProvenanceRecord, *, label: str, code_url: str | None, sidecars: Mapping[str, tuple[str, ...]]) -> dict[str, bytes]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:73](../../src/boldtailor/bids_provenance.py#L73)

<a id="definition-4"></a>

## `_draft_graph`

[Source, line 130](../../src/boldtailor/bids_provenance.py#L130)

```python
_draft_graph(record: ProvenanceRecord, *, code_url: str | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:116](../../src/boldtailor/bids_provenance.py#L116)

<a id="definition-5"></a>

## `_software`

[Source, line 155](../../src/boldtailor/bids_provenance.py#L155)

```python
_software(software_id: str, code_url: str | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:151](../../src/boldtailor/bids_provenance.py#L151)

<a id="definition-6"></a>

## `_environment`

[Source, line 166](../../src/boldtailor/bids_provenance.py#L166)

```python
_environment(record: ProvenanceRecord, identifier: str) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:150](../../src/boldtailor/bids_provenance.py#L150)

<a id="definition-7"></a>

## `_activities`

[Source, line 185](../../src/boldtailor/bids_provenance.py#L185)

```python
_activities(record: ProvenanceRecord, *, token: str, software_id: str, used: tuple[str, ...]) -> tuple[list[dict[str, object]], tuple[str, ...]]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:139](../../src/boldtailor/bids_provenance.py#L139)

<a id="definition-8"></a>

## `_activity_command`

[Source, line 209](../../src/boldtailor/bids_provenance.py#L209)

```python
_activity_command(name: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:200](../../src/boldtailor/bids_provenance.py#L200)

<a id="definition-9"></a>

## `_activity_times`

[Source, line 215](../../src/boldtailor/bids_provenance.py#L215)

```python
_activity_times(record: ProvenanceRecord) -> dict[str, str]
```

Bounds of the lifecycle events of the execution that wrote the last activity.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:205](../../src/boldtailor/bids_provenance.py#L205)

<a id="definition-10"></a>

## `_entities`

[Source, line 228](../../src/boldtailor/bids_provenance.py#L228)

```python
_entities(record: ProvenanceRecord, token: str) -> tuple[dict[str, list[dict[str, object]]], tuple[str, ...]]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:138](../../src/boldtailor/bids_provenance.py#L138)

<a id="definition-11"></a>

## `_file_entity`

[Source, line 254](../../src/boldtailor/bids_provenance.py#L254)

```python
_file_entity(uri: str, sha256: str | None=None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:241](../../src/boldtailor/bids_provenance.py#L241)

<a id="definition-12"></a>

## `_anonymous_entity`

[Source, line 274](../../src/boldtailor/bids_provenance.py#L274)

```python
_anonymous_entity(role: str, run_index: int, token: str) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:248](../../src/boldtailor/bids_provenance.py#L248)
- [src/boldtailor/bids_provenance.py:239](../../src/boldtailor/bids_provenance.py#L239)

<a id="definition-13"></a>

## `_relationship_sidecars`

[Source, line 282](../../src/boldtailor/bids_provenance.py#L282)

```python
_relationship_sidecars(sidecars: Mapping[str, tuple[str, ...]], graph: Mapping[str, object]) -> dict[str, bytes]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:126](../../src/boldtailor/bids_provenance.py#L126)

<a id="definition-14"></a>

## `_source_identifier`

[Source, line 300](../../src/boldtailor/bids_provenance.py#L300)

```python
_source_identifier(value: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:287](../../src/boldtailor/bids_provenance.py#L287)
- [src/boldtailor/bids_provenance.py:292](../../src/boldtailor/bids_provenance.py#L292)

<a id="definition-15"></a>

## `_generated_by`

[Source, line 304](../../src/boldtailor/bids_provenance.py#L304)

```python
_generated_by(code_url: str | None, container: Mapping[str, str] | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:62](../../src/boldtailor/bids_provenance.py#L62)

<a id="definition-16"></a>

## `_normalize_container`

[Source, line 320](../../src/boldtailor/bids_provenance.py#L320)

```python
_normalize_container(container: Mapping[str, str] | None) -> dict[str, str]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:314](../../src/boldtailor/bids_provenance.py#L314)

<a id="definition-17"></a>

## `_normalize_source_datasets`

[Source, line 339](../../src/boldtailor/bids_provenance.py#L339)

```python
_normalize_source_datasets(values: Sequence[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:67](../../src/boldtailor/bids_provenance.py#L67)

<a id="definition-18"></a>

## `_normalize_source_dataset`

[Source, line 349](../../src/boldtailor/bids_provenance.py#L349)

```python
_normalize_source_dataset(value: Mapping[str, str]) -> dict[str, str]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:344](../../src/boldtailor/bids_provenance.py#L344)

<a id="definition-19"></a>

## `_normalize_dataset_links`

[Source, line 365](../../src/boldtailor/bids_provenance.py#L365)

```python
_normalize_dataset_links(values: Mapping[str, str] | None) -> Mapping[str, str]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:68](../../src/boldtailor/bids_provenance.py#L68)

<a id="definition-20"></a>

## `_validate_link_location`

[Source, line 380](../../src/boldtailor/bids_provenance.py#L380)

```python
_validate_link_location(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:375](../../src/boldtailor/bids_provenance.py#L375)

<a id="definition-21"></a>

## `_normalize_sidecars`

[Source, line 387](../../src/boldtailor/bids_provenance.py#L387)

```python
_normalize_sidecars(record: ProvenanceRecord, values: Mapping[str, Sequence[str]] | None) -> Mapping[str, tuple[str, ...]]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:61](../../src/boldtailor/bids_provenance.py#L61)

<a id="definition-22"></a>

## `_normalize_relationship_sources`

[Source, line 406](../../src/boldtailor/bids_provenance.py#L406)

```python
_normalize_relationship_sources(values: Sequence[str], record_sources: frozenset[str]) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:399](../../src/boldtailor/bids_provenance.py#L399)

<a id="definition-23"></a>

## `_record_source_uris`

[Source, line 418](../../src/boldtailor/bids_provenance.py#L418)

```python
_record_source_uris(record: ProvenanceRecord) -> frozenset[str]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:395](../../src/boldtailor/bids_provenance.py#L395)

<a id="definition-24"></a>

## `_validate_sidecar_path`

[Source, line 427](../../src/boldtailor/bids_provenance.py#L427)

```python
_validate_sidecar_path(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:398](../../src/boldtailor/bids_provenance.py#L398)

<a id="definition-25"></a>

## `_validate_label`

[Source, line 434](../../src/boldtailor/bids_provenance.py#L434)

```python
_validate_label(value: object, name: str='label') -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:60](../../src/boldtailor/bids_provenance.py#L60)
- [src/boldtailor/bids_provenance.py:374](../../src/boldtailor/bids_provenance.py#L374)

<a id="definition-26"></a>

## `_slug`

[Source, line 440](../../src/boldtailor/bids_provenance.py#L440)

```python
_slug(value: object, fallback: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:275](../../src/boldtailor/bids_provenance.py#L275)
- [src/boldtailor/bids_provenance.py:195](../../src/boldtailor/bids_provenance.py#L195)

<a id="definition-27"></a>

## `_validate_text`

[Source, line 451](../../src/boldtailor/bids_provenance.py#L451)

```python
_validate_text(value: object, name: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:381](../../src/boldtailor/bids_provenance.py#L381)
- [src/boldtailor/bids_provenance.py:460](../../src/boldtailor/bids_provenance.py#L460)
- [src/boldtailor/bids_provenance.py:331](../../src/boldtailor/bids_provenance.py#L331)
- [src/boldtailor/bids_provenance.py:356](../../src/boldtailor/bids_provenance.py#L356)

<a id="definition-28"></a>

## `_validate_uri`

[Source, line 459](../../src/boldtailor/bids_provenance.py#L459)

```python
_validate_uri(value: object, name: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:313](../../src/boldtailor/bids_provenance.py#L313)
- [src/boldtailor/bids_provenance.py:335](../../src/boldtailor/bids_provenance.py#L335)
- [src/boldtailor/bids_provenance.py:383](../../src/boldtailor/bids_provenance.py#L383)
- [src/boldtailor/bids_provenance.py:361](../../src/boldtailor/bids_provenance.py#L361)

<a id="definition-29"></a>

## `_require_record`

[Source, line 470](../../src/boldtailor/bids_provenance.py#L470)

```python
_require_record(record: object) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:59](../../src/boldtailor/bids_provenance.py#L59)

<a id="definition-30"></a>

## `_extend_artifacts`

[Source, line 475](../../src/boldtailor/bids_provenance.py#L475)

```python
_extend_artifacts(artifacts: dict[str, bytes], additions: Mapping[str, bytes]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:126](../../src/boldtailor/bids_provenance.py#L126)
- [src/boldtailor/bids_provenance.py:71](../../src/boldtailor/bids_provenance.py#L71)

<a id="definition-31"></a>

## `_provenance_tsv`

[Source, line 485](../../src/boldtailor/bids_provenance.py#L485)

```python
_provenance_tsv(label: str) -> bytes
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:119](../../src/boldtailor/bids_provenance.py#L119)

<a id="definition-32"></a>

## `_provenance_sidecar`

[Source, line 494](../../src/boldtailor/bids_provenance.py#L494)

```python
_provenance_sidecar() -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:120](../../src/boldtailor/bids_provenance.py#L120)

<a id="definition-33"></a>

## `_freeze_artifacts`

[Source, line 516](../../src/boldtailor/bids_provenance.py#L516)

```python
_freeze_artifacts(artifacts: Mapping[str, bytes]) -> Mapping[str, bytes]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:80](../../src/boldtailor/bids_provenance.py#L80)

<a id="definition-34"></a>

## `_pretty_json`

[Source, line 527](../../src/boldtailor/bids_provenance.py#L527)

```python
_pretty_json(value: object) -> bytes
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:103](../../src/boldtailor/bids_provenance.py#L103)
- [src/boldtailor/bids_provenance.py:120](../../src/boldtailor/bids_provenance.py#L120)
- [src/boldtailor/bids_provenance.py:121](../../src/boldtailor/bids_provenance.py#L121)
- [src/boldtailor/bids_provenance.py:122](../../src/boldtailor/bids_provenance.py#L122)

<a id="definition-35"></a>

## `_compact_json`

[Source, line 532](../../src/boldtailor/bids_provenance.py#L532)

```python
_compact_json(value: object) -> bytes
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:105](../../src/boldtailor/bids_provenance.py#L105)

<a id="definition-36"></a>

## `_json_lines`

[Source, line 536](../../src/boldtailor/bids_provenance.py#L536)

```python
_json_lines(values: Sequence[object]) -> bytes
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:104](../../src/boldtailor/bids_provenance.py#L104)

<a id="definition-37"></a>

## `_canonical_text`

[Source, line 541](../../src/boldtailor/bids_provenance.py#L541)

```python
_canonical_text(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:345](../../src/boldtailor/bids_provenance.py#L345)
- [src/boldtailor/bids_provenance.py:537](../../src/boldtailor/bids_provenance.py#L537)
- [src/boldtailor/bids_provenance.py:533](../../src/boldtailor/bids_provenance.py#L533)

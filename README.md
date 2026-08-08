# boldtailor

Boldtailor is a composable first-level fMRI modeling package.

Phase 1 provides an array-based conventional GLM backed by Nilearn. BIDS,
NIfTI, CIFTI, adaptive HRFs, and the GLMsingle recipe are delivered in later
phases described by the package design.

## Array-based conventional GLM

```python
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec

events = pd.DataFrame(
    {
        "onset": [0.0, 8.0, 16.0, 24.0],
        "duration": [1.0, 1.0, 1.0, 1.0],
        "trial_type": ["face", "house", "face", "house"],
    }
)
signals = np.load("run_signals.npy")  # shape: time x features
data = from_arrays(signals, events, tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
    noise_model="ar1",
)

result = fit(data, model)
contrast_effect = result.effect("face_gt_house")
directional_p = result.one_sided_p_value("face_gt_house")
fit_quality = result.r2
```

`fit()` returns arrays only and never writes files. BIDS, NIfTI, and CIFTI
adapters are added in later phases. Contrast p-values are directional and
one-sided, matching Nilearn. For nonuniform or nonzero acquisition times, pass
one frame-time array per run instead of `tr`.

Features with zero centered sum of squares, including all-zero and constant
features, are accepted. Their per-run and aggregate R-squared values are NaN;
varying features in the same fit retain their ordinary estimates and
inference. Every run must have positive residual degrees of freedom for
contrast inference, otherwise `fit()` raises a run-specific `ValueError`.

## Provenance and lifecycle logging

`from_arrays()` accepts one `RunSources` descriptor per run. Each descriptor
identifies the signal and events inputs, plus confounds when present, with a
dataset-relative POSIX URI or BIDS URI and optional media type. Byte size and a
UTC modification time complete the stable source metadata:

```python
from boldtailor.provenance import RunSources, SourceRef

sources = (
    RunSources(
        signal=SourceRef(
            role="signal",
            uri="sub-01/func/sub-01_task-localizer_bold.npy",
            media_type="application/x-npy",
            byte_size=240128,
            modified_at="2026-08-08T12:00:00Z",
        ),
        events=SourceRef(
            role="events",
            uri="sub-01/func/sub-01_task-localizer_events.tsv",
            media_type="text/tab-separated-values",
            byte_size=384,
            modified_at="2026-08-08T12:01:00Z",
        ),
    ),
)
data = from_arrays(signals, events, tr=2.0, sources=sources)
```

Source identity is metadata-only: Boldtailor does not read source files or hash
signal, event, or confound contents. Complete source metadata produces a
deterministic `metadata_fingerprint`; `fit()` combines it with the complete,
reproducible `ModelSpec` to produce a deterministic `analysis_fingerprint`.
Execution IDs are different: every normalization and fit attempt receives a
fresh UUID used to correlate lifecycle events. Omitting `sources`, supplying an
incomplete descriptor, or using a local or lambda HRF leaves the corresponding
deterministic fingerprint unavailable and records a provenance-quality or
partial-reproducibility warning. Anonymous array analysis remains supported,
but it cannot establish stable data or analysis identity.

Normalization and fitting emit structured JSON records through the standard
`boldtailor` Python logger. The package does not install handlers or change
application logging configuration. Provenance retains a bounded event history;
records contain lifecycle metadata and correlation IDs, not raw signals,
tables, design values, estimates, statistics, command-line arguments,
environment variables, working directories, usernames, or hostnames. Absolute
and path-like caller metadata is rejected. Callers remain responsible for
de-identifying otherwise valid relative source URIs and annotations.

## BIDS provenance projection and artifact publication

`project_bids_provenance()` is a pure, in-memory projection from the canonical
Boldtailor provenance record to stable BIDS 1.11.1 derivative metadata. Draft
export is enabled by default and is pinned to
`BEP028@02172700aac8d1bdd67b45191f43533f426848dc`. The supported draft subset is
Activities, Files, Environments, Software, the provenance label table, and file
`GeneratedBy`/`Sources` relationships; this is not a claim of conformance to a
final BIDS provenance standard. `export_bids_prov=False` omits only those draft
files, leaving stable BIDS metadata and canonical provenance output intact.

`publish_artifact_set()` is the sole filesystem publication primitive. It
accepts and validates one complete in-memory `Artifact` set, locks the
destination, stages and fsyncs files, and rolls back the set on failure. By
default, failure leaves only a sanitized
`.boldtailor/publication_failures.jsonl` record. If rollback cannot restore an
original, Boldtailor automatically preserves the last recoverable copy under
`.boldtailor/failed/<execution-id>/recovery/`; this safety copy is independent
of `retain_incomplete`. Setting `retain_incomplete=True` additionally keeps the
requested failed artifact set under the same execution directory and marks
retained canonical provenance as failed and unpublished.

Future derivative adapters must first project every primary artifact, sidecar,
and provenance file into one complete artifact set, then route that set through
`publish_artifact_set()`; adapters must not write files directly. The current
release provides this publication core but no BIDS discovery and no NIfTI or
CIFTI writer. `fit()` performs zero filesystem I/O.

"""Content identities have one name each: ``data_fingerprint`` and ``analysis_fingerprint``."""

import json
import logging
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

import boldtailor
from boldtailor.data import from_arrays
from boldtailor.provenance import ProvenanceRecord, analysis_fingerprint
from boldtailor.single_trial import fit_single_trials

RETIRED = (
    "metadata_fingerprint",
    "sources_fingerprint",
    "data_id",
    "analysis_id",
    "parent_id",
    "comparison_id",
)


@pytest.fixture
def sourced(complete_sources):
    events = pd.DataFrame({"onset": [4.0, 20.0, 36.0], "duration": 1.0})
    signals = np.random.default_rng(1).normal(size=(30, 2))
    return from_arrays(signals, events, tr=2.0, sources=complete_sources(1))


def test_no_retired_identity_name_remains_in_the_package():
    """Random identifiers end in ``_id``; content hashes are fingerprints."""
    pattern = re.compile(r"(?<![A-Za-z0-9])_?[a-z_]*(" + "|".join(RETIRED) + r")\b")
    offenders = sorted(
        f"{path.name}:{number}: {match.group(0)}"
        for path in Path(boldtailor.__file__).parent.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        for match in pattern.finditer(line)
    )
    assert not offenders, "\n".join(offenders)


def test_a_record_exposes_both_fingerprints_as_fields(sourced):
    """The analysis fingerprint is a real field, not an entry in a catch-all mapping."""
    record = sourced.provenance
    assert re.fullmatch(r"[0-9a-f]{64}", record.data_fingerprint)
    assert record.analysis_fingerprint is None
    for retired in ("metadata_fingerprint", "sources_fingerprint"):
        assert not hasattr(record, retired)
    with pytest.raises(AttributeError):
        record.data_fingerprint = "x"
    fingerprint = analysis_fingerprint(record.data_fingerprint, {"name": "fit"})
    replaced = record.replace(analysis_fingerprint=fingerprint)
    assert replaced.analysis_fingerprint == fingerprint
    assert replaced.data_fingerprint == record.data_fingerprint
    assert replaced != record


def test_serialized_records_use_the_same_two_names(sourced):
    """Saved provenance carries the fingerprints under their attribute names."""
    fitted = fit_single_trials(sourced).provenance
    saved = fitted.to_dict()
    assert saved["data_fingerprint"] == fitted.data_fingerprint
    assert saved["analysis_fingerprint"] == fitted.analysis_fingerprint is not None
    assert not set(RETIRED) & set(saved)
    restored = ProvenanceRecord.from_dict(json.loads(json.dumps(saved)))
    assert restored == fitted
    assert restored.analysis_fingerprint == fitted.analysis_fingerprint
    bare = ProvenanceRecord(execution_id=str(uuid4())).to_dict()
    assert bare["data_fingerprint"] is None and bare["analysis_fingerprint"] is None


def test_log_events_use_the_same_two_names(sourced, caplog):
    """Lifecycle events name the fingerprints as provenance does."""
    caplog.set_level(logging.INFO, logger="boldtailor")
    fitted = fit_single_trials(sourced)
    events = [json.loads(r.getMessage()) for r in caplog.records]
    completed = [e for e in events if e["event"] == "single_trial_completed"][-1]
    assert completed["data_fingerprint"] == fitted.provenance.data_fingerprint
    assert completed["analysis_fingerprint"] == fitted.provenance.analysis_fingerprint
    assert not any(set(RETIRED) & set(event) for event in events)

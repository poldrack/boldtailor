"""Shared NSD artifacts: derivative descriptions."""

from importlib.metadata import version
import json

from examples.NSD.workflow_artifacts import dataset_description


def test_dataset_description_is_a_boldtailor_derivative():
    artifact = dataset_description("NSD single-trial models")
    assert artifact.path == "dataset_description.json"
    assert json.loads(artifact.payload) == {
        "Name": "NSD single-trial models",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "boldtailor", "Version": version("boldtailor")}],
    }

from importlib import metadata
from pathlib import Path


def test_notebook_kernel_is_not_a_runtime_requirement():
    """Verify notebook kernel is not a runtime requirement."""
    requirements = metadata.requires("boldtailor") or []
    assert not any(req.lower().startswith("ipykernel") for req in requirements)

from importlib import metadata
from pathlib import Path


def test_notebook_kernel_is_not_a_runtime_requirement():
    requirements = metadata.requires("boldtailor") or []
    assert not any(req.lower().startswith("ipykernel") for req in requirements)


def test_all_package_initializers_are_empty():
    root = Path(__file__).parents[1] / "src" / "boldtailor"
    initializers = list(root.rglob("__init__.py"))
    assert initializers
    assert all(path.read_bytes() == b"" for path in initializers)

import importlib
from importlib import metadata
import subprocess
import sys
import textwrap

import pytest


def test_source_checkout_version_is_explicitly_unknown(monkeypatch):
    software = importlib.import_module("boldtailor._software")

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(software.metadata, "version", missing)
    assert software.package_version("boldtailor") == "unknown"
    with pytest.raises(metadata.PackageNotFoundError):
        software.package_version("numpy")


def test_source_import_does_not_require_own_distribution_metadata():
    code = textwrap.dedent("""
        from importlib import metadata
        installed_version = metadata.version
        def version(name):
            if name == "boldtailor":
                raise metadata.PackageNotFoundError(name)
            return installed_version(name)
        metadata.version = version
        import boldtailor.prepared
        import boldtailor.prepared_fit
        import boldtailor.bids_provenance
        """)
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


def test_imports_do_not_query_versions():
    code = textwrap.dedent("""
        import importlib
        from importlib import metadata
        names = (
            "boldtailor.prepared", "boldtailor.prepared_fit",
            "boldtailor.bids_provenance",
        )
        modules = [importlib.import_module(name) for name in names]
        def unexpected(name):
            raise AssertionError(f"import-time version lookup: {name}")
        metadata.version = unexpected
        for module in modules:
            importlib.reload(module)
        """)
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr

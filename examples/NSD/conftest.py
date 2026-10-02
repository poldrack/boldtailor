import pytest


def pytest_addoption(parser):
    try:  # tests/conftest.py registers it too when both trees are collected
        parser.addoption(
            "--run-notebooks",
            action="store_true",
            default=False,
            help="execute notebook kernels (slow); off by default",
        )
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "notebook: executes a notebook kernel; needs --run-notebooks"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-notebooks"):
        return
    skip = pytest.mark.skip(reason="notebook execution needs --run-notebooks")
    for item in items:
        if "notebook" in item.keywords:
            item.add_marker(skip)

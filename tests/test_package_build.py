"""The package builds with uv's native backend and ships its data and licenses."""

import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

RESOURCES = "boldtailor/_resources/"
LICENSES = "boldtailor-0.1.0.dist-info/licenses/"


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        check=True,
        capture_output=True,
    )
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def test_build_backend_is_uv_build():
    config = tomllib.loads(Path("pyproject.toml").read_text())
    assert config["build-system"]["build-backend"] == "uv_build"
    assert config["build-system"]["requires"][0].startswith("uv_build")
    assert "setuptools" not in config.get("tool", {})


def test_wheel_ships_hrf_data_and_its_notice(wheel_names):
    for name in (
        "glmsingle_hrf_library.tsv",
        "glmsingle_hrf_library.md",
        "GLMsingle-LICENSE.txt",
    ):
        assert RESOURCES + name in wheel_names, name


def test_wheel_records_both_licenses(wheel_names):
    assert LICENSES + "LICENSE" in wheel_names
    assert any(
        n.startswith(LICENSES) and n.endswith("GLMsingle-LICENSE.txt")
        for n in wheel_names
    )


def test_wheel_ships_every_source_module(wheel_names):
    sources = Path("src").rglob("*.py")
    expected = {str(path.relative_to("src")) for path in sources}
    assert expected <= wheel_names

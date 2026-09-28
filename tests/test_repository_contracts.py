from pathlib import Path
import tomllib

ROOT = Path(__file__).parents[1]


def test_project_uses_uv_managed_src_layout():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert config["project"]["name"] == "boldtailor"
    assert config["project"]["requires-python"] == ">=3.12"
    assert "nilearn>=0.14.0,<0.15" in config["project"]["dependencies"]
    assert "nibabel" in config["dependency-groups"]["dev"]
    assert config["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert config["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert (ROOT / "uv.lock").is_file()

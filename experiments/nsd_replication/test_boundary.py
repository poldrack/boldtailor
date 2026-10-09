"""NSD experiments use only core boldtailor and never fMRIPrep CIFTI files."""

from pathlib import Path

import pytest

from experiments.nsd_replication.boundary import violations

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "source",
    [
        "from boldtailor.workflow.inputs import load_session\n",
        "import boldtailor.workflow.inputs as w\n",
        "import boldtailor.cli\n",
        "from boldtailor import workflow\n",
        "from boldtailor import cli, model\n",
        "from boldtailor.workflow import inputs\n",
        'P = "{s}_space-fsLR_den-91k_bold.dtseries.nii"\n',
        'P = "derivatives/ppdata/subj07"\n',
        'P = "subj{nn}/func1pt8mm/timeseries"\n',
    ],
)
def test_each_violation_form_is_caught(source):
    assert violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "from boldtailor.model import TaskModel\n",
        "from boldtailor import model\n",
        "import boldtailor.task_detection\n",
        '"""nsdpreproc is the BIDS form of NSD\'s ppdata."""\n',
        'P = "derivatives/fmriprep/sub-07/anat/x_desc-confounds_timeseries.tsv"\n',
        'P = "fmriprep_dir"\n',
        'P = "{s}_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii"\n',
        'P = "{s}_space-fsLR_den-91k_desc-assumehrf_stat-effect_statmap.dscalar.nii"\n',
    ],
)
def test_allowed_sources_pass(source):
    assert violations(source) == []


def _checked(path):
    """Test modules may name forbidden files (e.g. decoys) but never import them."""
    return violations(path.read_text(), check_strings=not path.name.startswith("test_"))


def test_experiments_respect_the_boundary():
    found = {
        str(path.relative_to(ROOT)): problems
        for path in sorted(ROOT.rglob("*.py"))
        if path.name not in {"test_boundary.py", "boundary.py"}
        and (problems := _checked(path))
    }
    assert found == {}


def test_test_modules_are_still_checked_for_imports(tmp_path):
    path = tmp_path / "test_decoy.py"
    path.write_text('import boldtailor.workflow\nP = "x_den-91k_bold.dtseries.nii"\n')
    assert _checked(path) == ["line 1: imports boldtailor.workflow"]

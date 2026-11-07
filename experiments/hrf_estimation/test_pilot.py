"""Exercise a complete tiny comparison and cortical map round trip."""

import importlib
import json

import nibabel as nib
import numpy as np

from experiments.hrf_estimation.test_library_cv import planted_runs


def test_pilot_records_refit_scores_and_preserves_unprocessed_vertices(
    planted_runs, tmp_path
):
    runs, library = planted_runs
    brain = nib.cifti2.BrainModelAxis.from_surface(np.arange(10), 10, "CortexLeft")
    fn = importlib.import_module("experiments.hrf_estimation.pilot").run_comparison
    table = fn(runs, library, brain, tmp_path, seed=9)
    assert set(table.method) == {"canonical", "librarycv"}
    assert set(table.evaluation) == {"insample", "transfer01", "transfer02"}
    assert len(table) == 12  # two vertices, two methods, three evaluations
    best = table[(table.method == "librarycv") & (table.evaluation != "insample")]
    np.testing.assert_allclose(best.r2_full, 1, atol=1e-10)
    files = list(tmp_path.glob("*desc-librarycvtransfer01_stat-r2task.dscalar.nii"))
    assert len(files) == 1
    image = nib.load(files[0])
    values = np.asarray(image.dataobj)[0]
    assert image.header.get_axis(1) == brain
    assert np.isnan(values[[0, 1, 2, 3, 5, 6, 8, 9]]).all()
    np.testing.assert_allclose(values[[4, 7]], 1, atol=1e-10)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["metric"] == "HRF-transfer R2 with refitted amplitudes"
    assert manifest["cutoff_seconds"] == 80
    assert len(manifest["splits"]) == 2
    assert (tmp_path / "COMPLETE.json").exists()


def test_comparison_supports_experimental_estimators(planted_runs,tmp_path):
    runs,library=planted_runs
    brain=nib.cifti2.BrainModelAxis.from_surface(np.arange(10),10,'CortexLeft')
    fn=importlib.import_module('experiments.hrf_estimation.pilot').run_comparison
    table=fn(runs,library,brain,tmp_path,methods=('reml','continuouslbfgsb','continuouspowell'),
             settings={'lambda':0,'max_starts':1,'maxiter':5,'maxfun':50})
    assert set(table.method)=={'reml','continuouslbfgsb','continuouspowell'}
    assert len(table)==18
    assert table.r2_full.notna().all()

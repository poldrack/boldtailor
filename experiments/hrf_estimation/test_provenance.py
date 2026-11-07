import importlib


def test_package_scientific_changes_invalidate_implementation_identity(tmp_path):
    experiment=tmp_path/'experiment';package=tmp_path/'package'
    experiment.mkdir();package.mkdir()
    (experiment/'design.py').write_text('experiment = 1\n')
    scientific=package/'_hrf_design.py';scientific.write_text('normalization = 1\n')
    fingerprint=importlib.import_module('experiments.hrf_estimation.provenance').implementation_fingerprint
    first=fingerprint(experiment,package)
    scientific.write_text('normalization = 2\n')
    assert fingerprint(experiment,package)!=first
    # Resources used by the scientific package also affect reuse.
    second=fingerprint(experiment,package)
    (package/'kernel.tsv').write_text('0\t1\n')
    assert fingerprint(experiment,package)!=second

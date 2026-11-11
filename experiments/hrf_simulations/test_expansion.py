import numpy as np
import pandas as pd
from experiments.hrf_simulations.test_runner import tiny_case


def test_balanced_expansion():
    from experiments.hrf_simulations.cases import simulation_cases
    cases = simulation_cases('expanded', replicates=1)
    assert {c['session'] for c in cases} == {10, 11, 19}
    assert len(cases) < 1000
    assert len({c['id'] for c in cases}) == len(cases)
    assert {c['snr_level'] for c in cases} == {'q25', 'q50', 'q75'}
    assert len({c['truth'] for c in cases}) == 7
    assert any(c['regressor_scaling'] == 'unit_rms' for c in cases)


def test_run_variation_preserves_noise(tiny_case):
    from experiments.hrf_simulations.dataset import generate
    runs, library, profile, case = tiny_case
    _, a, meta = generate(runs, library.candidates[0], profile, case)
    _, b, varied = generate(runs, library.candidates[0], profile, {**case, 'run_coefficient_sd': .25})
    assert meta['noise_hash'] == varied['noise_hash']
    assert not np.allclose(a, b)
    assert len(varied['run_coefficients']) == len(runs)


def test_crossed_evaluation_and_floor(tmp_path, tiny_case):
    from experiments.hrf_simulations.runner import run_case
    from experiments.hrf_simulations.reporting import build_report
    runs, library, profile, case = tiny_case
    folder = tmp_path / 'case-test'
    scores = run_case(runs, library, profile, {**case, 'snr': float('inf')}, folder, models=('ConsDurNoRT',), splits=False)
    crossed = pd.read_csv(folder / 'crossed_scores.tsv', sep='\t')
    assert len(crossed) == 12
    assert crossed.evaluation_model.nunique() == 4
    assert (scores.curve_rmse_excess >= 0).all()
    matched = crossed[crossed.evaluation_model == 'ConsDurRTMod']
    assert np.allclose(matched.known_hrf_contrast_error, 0, atol=1e-8)
    assert 'contrast_error' in crossed
    build_report(tmp_path, tmp_path / 'report')
    assert (tmp_path / 'report/crossed_summary.tsv').exists()


def test_scaling_uses_projected_training_design(tiny_case):
    from experiments.hrf_simulations.fitting import random_scales
    from experiments.hrf_simulations.design import task_design
    from experiments.hrf_estimation.scoring import residuals
    runs, library, _, _ = tiny_case
    parts = [task_design(r, library.candidates[0], 'ConsDurRTMod', 'brief') for r in runs]
    z = [residuals(p.nuisance, np.column_stack([p.task, p.junk]))[0] for p in parts]
    expected = np.sqrt(np.sum([np.sum(a*a, axis=0) for a in z], axis=0) / sum(len(a) for a in z))
    expected[expected < 1e-12] = 1
    assert np.allclose(random_scales(parts, 'unit_rms'), expected)
    assert np.all(random_scales(parts, 'native') == 1)

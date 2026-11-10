from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.hrf_estimation.sensitivity import synthetic_templates


def test_donor_selection_excludes_target_and_spans_features():
    from experiments.hrf_simulations.calibration import select_donors

    table = pd.DataFrame(
        [
            dict(session=s, feature_id=v, variance=v + 1.0, rho=v / 100.0, snr=0.4)
            for s in (10, 11, 12)
            for v in range(100)
        ]
    )
    selected = select_donors(table, 11, 64)
    assert set(selected.session) == {10, 12}
    assert selected.feature_id.nunique() == 64
    assert {0, 99}.issubset(set(selected.feature_id))
    bad = table[table.session == 11]
    with pytest.raises(ValueError, match="calibration"):
        select_donors(bad, 11)


def test_calibration_roundtrip_and_target_exclusion(tmp_path):
    from experiments.hrf_simulations.calibration import calibrate, draw_profile

    rng = np.random.default_rng(2)
    sessions = {}
    for s in (10, 11):
        sessions[s] = tuple(
            replace(
                r,
                run_id=f"ses-{s}_{r.run_id}",
                signals=rng.normal(size=r.signals.shape),
            )
            for r in synthetic_templates()[:3]
        )
    calibrate(sessions, tmp_path, provenance={"source": "synthetic test"})
    profile = draw_profile(tmp_path, 11, sessions[11], seed=4)
    assert profile["calibration_sessions"] == [10]
    assert len(profile["runs"]) == 3
    assert np.isfinite(profile["rho"]) and abs(profile["rho"]) <= 0.8
    assert profile["variance"] > 0
    assert profile["snr_quantiles"].shape == (3,)
    assert profile["runs"][0]["residual"].shape == sessions[11][0].signals[:, 0].shape
    incompatible = [replace(sessions[11][0], frame_times=np.arange(200.0))]
    with pytest.raises(ValueError, match="compatible"):
        draw_profile(tmp_path, 11, incompatible, seed=4)


def test_generated_dataset_reproducibility_pairing_and_events():
    from experiments.hrf_simulations.dataset import generate
    from experiments.hrf_simulations.truths import truth_library

    runs = synthetic_templates()[:2]
    profile = dict(
        rho=0.4,
        variance=1.0,
        feature_id=10,
        calibration_sessions=[10],
        runs=[
            dict(
                residual=np.sin(np.arange(len(r.frame_times))),
                nuisance=np.zeros(len(r.confounds.columns)),
            )
            for r in runs
        ],
    )
    case = dict(
        truth="canonical",
        mechanism="amplitude",
        convention="stimulus",
        noise="ar1",
        snr=0.5,
        seed=20,
        gamma=0.5,
        betas=[1.0, 1.0],
    )
    a, sa, meta = generate(runs, truth_library()["canonical"], profile, case)
    b, sb, _ = generate(runs, truth_library()["canonical"], profile, case)
    d, sd, other = generate(
        runs, truth_library()["canonical"], profile, {**case, "mechanism": "duration"}
    )
    for original, x, y in zip(runs, a, b):
        np.testing.assert_array_equal(x.signals, y.signals)
        pd.testing.assert_frame_equal(original.events, x.events)
    assert meta["noise_hash"] == other["noise_hash"]
    assert meta["realized_target_snr"] == pytest.approx(0.5)
    assert sa[0].shape == (len(runs[0].frame_times),)


def test_spectral_generation_saves_actual_donor_diagnostics():
    from experiments.hrf_simulations.dataset import generate
    from experiments.hrf_simulations.truths import truth_library
    run = synthetic_templates()[0]
    residual = np.random.default_rng(2).normal(size=len(run.frame_times))
    profile = dict(rho=.7,variance=10.,feature_id=0,
        runs=[dict(residual=residual,nuisance=np.zeros(len(run.confounds.columns)))])
    case = dict(mechanism='none',convention='brief',noise='spectral',snr=.5,seed=3)
    _,_,meta = generate([run],truth_library()['canonical'],profile,case)
    donor = meta['donor_diagnostics'][0]
    generated = meta['noise_diagnostics'][0]
    np.testing.assert_allclose(donor['periodogram'],generated['periodogram'],atol=1e-12)
    assert donor['variance'] == pytest.approx(np.var(residual))


def test_ar1_variance_calibration_corrects_projection_loss():
    from experiments.hrf_simulations.calibration import _reference_fit
    from experiments.hrf_simulations.noise import ar1
    run = synthetic_templates()[0]
    rng = np.random.default_rng(15)
    values = np.column_stack([ar1(len(run.frame_times),.7,2.,rng) for _ in range(128)])
    result = _reference_fit(replace(run,signals=values,feature_ids=np.arange(128)))
    assert result['variance'].mean() == pytest.approx(2.,rel=.12)
    assert result['variance_rho'] == pytest.approx(.7,abs=.1)

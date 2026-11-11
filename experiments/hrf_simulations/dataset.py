"""Paired single-vertex datasets, with one signal multiplier per case."""

from dataclasses import replace
from hashlib import sha256

import numpy as np

from experiments.hrf_estimation.scoring import residuals
from experiments.hrf_simulations.noise import ar1, phase_surrogate, diagnostics
from experiments.hrf_simulations.simulation import signal_components


def _projected_energy(runs, values):
    return sum(
        float(np.sum(residuals(r.confounds.to_numpy(float), v[:, None])[0] ** 2))
        for r, v in zip(runs, values, strict=True)
    )


def _signals(runs, truth, case):
    gamma = case.get("gamma", 0.5)
    if case.get("match_rt_rms", False) and case["mechanism"] in (
        "amplitude",
        "additive_duration",
    ):
        base = [
            signal_components(
                r,
                truth,
                "none",
                case["convention"],
                betas=case.get("betas", (1.0, 1.0)),
            )[0]
            for r in runs
        ]
        full = [
            signal_components(
                r,
                truth,
                case["mechanism"],
                case["convention"],
                gamma=1.0,
                betas=case.get("betas", (1.0, 1.0)),
            )[0]
            for r in runs
        ]
        delta = [a - b for a, b in zip(full, base, strict=True)]
        gamma = 0.5 * np.sqrt(
            _projected_energy(runs, base) / _projected_energy(runs, delta)
        )
    sd = case.get("run_coefficient_sd", 0.0)
    if sd < 0:
        raise ValueError("run_coefficient_sd must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([int(case["seed"]), 8102]))
    factors = rng.lognormal(-sd * sd / 2, sd, (len(runs), 3))
    coefficients = [
        dict(
            betas=(np.asarray(case.get("betas", (1.0, 1.0))) * f[:2]).tolist(),
            gamma=float(gamma * f[2]),
        )
        for f in factors
    ]
    components = [
        signal_components(
            r,
            truth,
            case["mechanism"],
            case["convention"],
            gamma=coefficient["gamma"],
            betas=coefficient["betas"],
        )
        for r, coefficient in zip(runs, coefficients, strict=True)
    ]
    return components, gamma, coefficients


def generate(runs, truth, profile, case):
    snr = case["snr"]
    if not snr > 0 or np.isnan(snr):
        raise ValueError("SNR must be positive; infinity denotes noiseless")
    if case["noise"] not in ("white", "ar1", "spectral"):
        raise ValueError("unknown noise family")
    streams = np.random.SeedSequence([int(case["seed"]), 1729]).spawn(len(runs))
    noises, nuisances = [], []
    for run, donor, stream in zip(runs, profile["runs"], streams, strict=True):
        rng = np.random.default_rng(stream)
        if len(donor["residual"]) != len(run.frame_times):
            raise ValueError("incompatible donor residual length")
        noise = (
            phase_surrogate(donor["residual"], rng)
            if case["noise"] == "spectral"
            else ar1(
                len(run.frame_times),
                profile["rho"] if case["noise"] == "ar1" else 0.0,
                profile["variance"],
                rng,
            )
        )
        noises.append(noise if np.isfinite(snr) else np.zeros_like(noise))
        nuisance = run.confounds.to_numpy(float) @ donor["nuisance"]
        nuisances.append(
            nuisance
            if not case.get("zero_nuisance", False)
            else np.zeros_like(nuisance)
        )
    components, gamma, coefficients = _signals(runs, truth, case)
    task = [p[0] for p in components]
    noise_energy = _projected_energy(runs, noises)
    signal_energy = _projected_energy(runs, task)
    if signal_energy <= 0 or (np.isfinite(snr) and noise_energy <= 0):
        raise ValueError("degenerate signal or noise after nuisance projection")
    multiplier = (
        snr * np.sqrt(noise_energy / signal_energy) if np.isfinite(snr) else 1.0
    )
    output, targets = [], []
    for run, (task, junk), noise, nuisance in zip(
        runs, components, noises, nuisances, strict=True
    ):
        targets.append(multiplier * task)
        output.append(
            replace(
                run,
                signals=(multiplier * (task + junk) + noise + nuisance)[:, None],
                feature_ids=np.array([profile["feature_id"]]),
                brain_axis=None,
            )
        )
    digest = sha256(b"".join(n.tobytes() for n in noises)).hexdigest()
    metadata = dict(
        noise_hash=digest,
        signal_multiplier=multiplier,
        effective_gamma=gamma,
        run_coefficients=coefficients,
        realized_target_snr=(
            np.sqrt(_projected_energy(runs, targets) / noise_energy)
            if noise_energy > 0
            else np.inf
        ),
        noise_diagnostics=[diagnostics(n) for n in noises],
        donor_diagnostics=[diagnostics(d["residual"]) for d in profile["runs"]],
    )
    return tuple(output), tuple(targets), metadata

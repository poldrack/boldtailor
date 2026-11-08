"""Run a paired two-session pilot of pooled, shrinkage, and surface AR(1)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import pandas as pd
from boldtailor.hrf_library import default_hrf_library
from experiments.hrf_estimation.inputs import discover_runs, load_run
from experiments.hrf_estimation.pilot import _save_fit, _atomic_text
from experiments.hrf_estimation.provenance import identities, completion_record
from experiments.hrf_estimation.regularized_noise import (
    run_rhos, shrink_rho, surface_weights, subset_run, fit_grouped, whitening_diagnostics,
)
from experiments.hrf_estimation.scoring import score_hrfs
from experiments.hrf_estimation.splits import make_splits

ROOT = Path(__file__).resolve().parent


def _json(path, value):
    _atomic_text(path, json.dumps(value, default=lambda x: x.tolist(), indent=2)+"\n")


def _noise_maps(samples, weights):
    pooled = float(np.clip(np.nanmedian(samples),-.8,.8))
    shrink = shrink_rho(samples, pooled)
    raw = np.nanmean(samples,axis=0)
    raw = np.where(np.isfinite(raw),raw,pooled)
    maps = dict(pooled=np.full(samples.shape[1],pooled), shrinkage=shrink["rho"],
                surface=np.clip(weights@raw,-.8,.8))
    return maps,shrink


def _evaluate(train, test, library, rho, folder, method, label, details):
    start = time.monotonic()
    fit = fit_grouped(train,library,rho,verbose=True)
    score = score_hrfs(test,fit)
    _save_fit(fit, folder/f"{method}_{label}_hrf.npz")
    _json(folder/f"{method}_{label}_diagnostics.json", dict(
        **fit.diagnostics, training_run_ids=fit.training_run_ids, **details))
    score.per_run.to_csv(folder/f"{method}_{label}_runmetrics.tsv",sep="\t",index=False)
    white = whitening_diagnostics(test,rho).assign(method=method,evaluation=label)
    seconds = time.monotonic()-start
    print(f"{method} {label}: {seconds:.1f}s",flush=True)
    return score.summary.reset_index().assign(method=method,evaluation=label,seconds=seconds),white


def run_pilot(runs, library, weights, indices, destination, bin_width=.02):
    folder = Path(destination)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"COMPLETE.json").unlink(missing_ok=True)
    splits = make_splits(tuple(r.run_id for r in runs),20261107,1)
    _json(folder/"manifest.json",dict(splits=[asdict(s) for s in splits],
        seed=20261107, bin_width=bin_width, fwhm_mm=5., variance_prior_df=4,
        feature_ids=runs[0].feature_ids[indices], noise_feature_ids=runs[0].feature_ids,
        library_fingerprint=library.fingerprint,
        implementation_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.glob("*.py") if not p.name.startswith("test_")},
        **identities(runs,runs[0].brain_axis)))
    scores,whitening,rho_rows = [],[],[]
    lookup = {r.run_id:r for r in runs}
    for number,split in enumerate(splits,1):
        label = f"transfer{number:02}"
        fulltrain = tuple(lookup[k] for k in split.train)
        train = tuple(subset_run(r,indices) for r in fulltrain)
        test = tuple(subset_run(lookup[k],indices) for k in split.test)
        samples = run_rhos(fulltrain)
        maps,shrink = _noise_maps(samples,weights)
        np.savez_compressed(folder/f"{label}_noise_maps.npz", feature_ids=runs[0].feature_ids,
                            samples=samples, **maps, weights=shrink["weight"], uncertainty=shrink["uncertainty"])
        for method,values in maps.items():
            selected = values[indices]
            rho = selected if method=="pooled" else np.clip(np.round(selected/bin_width)*bin_width,-.8,.8)
            for vertex,raw,used in zip(train[0].feature_ids,selected,rho):
                rho_rows.append(dict(feature_id=int(vertex),evaluation=label,method=method,rho=raw,rho_used=used))
            result,white = _evaluate(train,test,library,rho,folder,method,label,
                dict(tau2=shrink["tau2"], pooled_rho=float(maps["pooled"][0]), bin_width=bin_width))
            scores.append(result);whitening.append(white)
            pd.concat(scores).to_csv(folder/"scores.tsv",sep="\t",index=False)
            pd.concat(whitening).to_csv(folder/"whitening.tsv",sep="\t",index=False)
            pd.DataFrame(rho_rows).to_csv(folder/"rho.tsv",sep="\t",index=False)
    _json(folder/"COMPLETE.json",completion_record(folder,len(pd.concat(scores))))


def anatomical_weights(brain, destination, bids):
    weights = np.zeros((len(brain),len(brain)))
    sources = {}
    for hemisphere,structure in [("L","CIFTI_STRUCTURE_CORTEX_LEFT"),("R","CIFTI_STRUCTURE_CORTEX_RIGHT")]:
        path = bids/f"derivatives/fmriprep-25.2.5/sub-07/ses-nsdanat/anat/sub-07_ses-nsdanat_hemi-{hemisphere}_space-fsLR_den-32k_midthickness.surf.gii"
        xyz,faces = nib.load(path).agg_data()
        selected = np.flatnonzero(brain.name == structure)
        weights[np.ix_(selected,selected)] = surface_weights(xyz,faces,brain.vertex[selected],5.)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _json(destination/"surfaces.json",sources)
    np.savez_compressed(destination/"surface_weights.npz",weights=weights)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session",type=int,choices=[11,16],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    bids = Path("/Volumes/extdata1/NSD/BIDS")
    vertices = np.load(ROOT/"pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz")["feature_ids"]
    common = np.load(ROOT/"reports/core_hrf_shape_summary.npz")["feature_ids"]
    selected = np.sort(np.random.default_rng(20261108).choice(common,256,replace=False))
    indices = np.searchsorted(vertices,selected)
    assert np.array_equal(vertices[indices],selected)
    specs = discover_runs(bids,"sub-07",(args.session,))
    runs = tuple(load_run(s,vertices,80) for s in specs)
    assert len(runs)==12
    weights = anatomical_weights(runs[0].brain_axis,args.output,bids)
    print(f"Loaded session {args.session}: {len(vertices)} noise vertices, {len(indices)} HRF vertices",flush=True)
    run_pilot(runs,default_hrf_library(),weights,indices,args.output)


if __name__ == "__main__":
    main()

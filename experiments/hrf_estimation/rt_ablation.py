"""Matched original REML with and without an RT regressor."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from boldtailor.hrf_library import default_hrf_library
from experiments.hrf_estimation.inputs import discover_runs, load_run
from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_estimation.pilot import run_comparison
from experiments.hrf_estimation.scoring import score_hrfs
from experiments.hrf_estimation.provenance import file_hash, completion_record


def crossed_scores(runs, fits, label):
    tables = []
    for selection, estimate in fits.items():
        for evaluation in ("amplitude", "none"):
            score = score_hrfs(runs, estimate, evaluation).summary.reset_index()
            tables.append(score.assign(selection_rt=selection, evaluation_rt=evaluation, evaluation=label))
    return pd.concat(tables, ignore_index=True)


def load_fit(path, train, library):
    with np.load(path) as data:
        assert np.array_equal(data["feature_ids"], train[0].feature_ids)
        assert tuple(data["training_run_ids"]) == tuple(r.run_id for r in train)
        kernels = tuple(library.candidates[int(i)] if i >= 0 else None for i in data["candidate_ids"])
        return HrfEstimate(data["feature_ids"].copy(), kernels, tuple(data["status"]),
                           tuple(data["training_run_ids"]), {}, train[0].brain_axis)


def run_session(root, session):
    records = json.loads((root/"reports/focused_comparison_provenance.json").read_text())
    record = next(r for r in records if r["session"] == session and r["method"] == "Original REML")
    source = root/record["source"]
    previous = json.loads((source/"manifest.json").read_text())
    vertices = np.load(root/"pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz")["feature_ids"]
    specs = discover_runs("/Volumes/extdata1/NSD/BIDS", "sub-07", (session,))
    runs = tuple(load_run(s, vertices, 80) for s in specs)
    brain = nib.load(specs[0].bold_path).header.get_axis(1)
    library = default_hrf_library()
    folder = root/f"outputs/reml-rt-ablation/session{session}"
    run_comparison(runs, library, brain, folder, methods=("reml",), settings={"rt_mode":"none"})
    manifest = json.loads((folder/"manifest.json").read_text())
    for key in ("input_fingerprint", "library_fingerprint", "splits"):
        assert manifest[key] == previous[key], key
    lookup = {r.run_id:r for r in runs}
    evaluations = [("insample", runs, runs)] + [
        (f"transfer{i:02}", tuple(lookup[k] for k in s["train"]), tuple(lookup[k] for k in s["test"]))
        for i,s in enumerate(manifest["splits"],1)]
    tables = []
    hashes = {}
    for label, train, test in evaluations:
        old = next(source.glob(f"*desc-reml{label}_hrf.npz"))
        assert file_hash(old) == record["artifact_hashes"][str(old.relative_to(root))]
        new = next(folder.glob(f"*desc-reml{label}_hrf.npz"))
        fits = {"amplitude":load_fit(old,train,library), "none":load_fit(new,train,library)}
        tables.append(crossed_scores(test,fits,label))
        hashes[str(old.relative_to(root))] = file_hash(old)
    pd.concat(tables).to_csv(folder/"crossed_scores.tsv.gz",sep="\t",index=False)
    (folder/"reference.json").write_text(json.dumps(dict(source=record["source"],hashes=hashes),indent=2))
    (folder/"COMPLETE.json").write_text(json.dumps(completion_record(folder, len(pd.concat(tables))),indent=2))
    print(f"COMPLETE session {session}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--sessions",type=int,nargs="+",required=True)
    args = parser.parse_args()
    for session in args.sessions:
        run_session(args.root,session)


if __name__ == "__main__":
    main()

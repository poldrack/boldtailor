"""Build portable matched-vertex reports for the regularized AR1 pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/"outputs/regularized-ar1-pilot"
REPORT = ROOT/"reports/regularized_ar1"
METHODS = ["Library CV","Original REML","pooled","shrinkage","surface"]


def _curve(path):
    with np.load(path) as z:
        frame = pd.DataFrame(z["curves"],index=z["feature_ids"])
    valid = np.isfinite(frame).all(axis=1)&frame.std(axis=1).gt(1e-12)&frame.max(axis=1).gt(0)
    return frame[valid]


def _load():
    scores,white,rhos,curves,sources = [],[],[],{},[]
    for session in [11,16]:
        folder = OUTPUT/f"session{session}"
        assert (folder/"COMPLETE.json").exists()
        for name,target in [("scores.tsv",scores),("whitening.tsv",white),("rho.tsv",rhos)]:
            target.append(pd.read_csv(folder/name,sep="\t").assign(session=session))
        for method in ["pooled","shrinkage","surface"]:
            for label in ["transfer01","transfer02"]:
                path = folder/f"{method}_{label}_hrf.npz"
                curves[session,method,label] = _curve(path)
        sources.extend(p for p in folder.iterdir() if p.is_file())
    records = json.loads((ROOT/"reports/focused_comparison_provenance.json").read_text())
    baseline = pd.read_csv(ROOT/"reports/focused_comparison_scores.tsv.gz",sep="\t")
    baseline = baseline[baseline.session.isin([11,16])&baseline.method.isin(METHODS[:2])&baseline.evaluation.str.startswith("transfer")]
    scores.append(baseline)
    for record in records:
        if record["session"] not in [11,16] or record["method"] not in METHODS[:2]:
            continue
        token = "librarycv" if record["method"]=="Library CV" else "reml"
        for label in ["transfer01","transfer02"]:
            path = next((ROOT/record["source"]).glob(f"*desc-{token}{label}_hrf.npz"))
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record["artifact_hashes"][str(path.relative_to(ROOT))]
            curves[record["session"],record["method"],label] = _curve(path)
            sources.append(path)
    return pd.concat(scores),pd.concat(white),pd.concat(rhos),curves,sources


def build_report():
    REPORT.mkdir(parents=True,exist_ok=True)
    scores,white,rhos,curves,sources = _load()
    strength = []
    for session in [11,16]:
        for label in ["transfer01","transfer02"]:
            with np.load(OUTPUT/f"session{session}/{label}_noise_maps.npz") as z:
                strength.append(dict(session=session,evaluation=label,n_noise_vertices=len(z["feature_ids"]),
                    median_weight=float(np.median(z["weights"])),weight_q10=float(np.quantile(z["weights"],.1)),
                    weight_q90=float(np.quantile(z["weights"],.9)),
                    shrinkage_upper_fraction=float(np.mean(z["shrinkage"]==.8)),
                    surface_upper_fraction=float(np.mean(z["surface"]==.8))))
    pd.DataFrame(strength).to_csv(REPORT/"shrinkage_strength.tsv",sep="\t",index=False)
    common = set.intersection(*[set(frame.index) for frame in curves.values()])
    for _,frame in scores.groupby(["session","method","evaluation"]):
        valid = np.isfinite(frame[["sse_full","sse_nuisance","sst"]]).all(axis=1)&frame.sse_nuisance.gt(0)&frame.sst.gt(0)
        common &= set(frame.loc[valid,"feature_id"])
    ids = np.array(sorted(common))
    scores = scores[scores.feature_id.isin(ids)]
    summary,shape_rows,means = [],[],[]
    for session in [11,16]:
        for method in METHODS:
            data = scores[(scores.session==session)&(scores.method==method)]
            pair = [curves[session,method,label].loc[ids].to_numpy() for label in ["transfer01","transfer02"]]
            x,y = [v-v.mean(axis=1,keepdims=True) for v in pair]
            corr = (x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))
            corr = np.clip(corr,-1,1)
            means.append([np.mean(v/v.max(axis=1,keepdims=True),axis=0) for v in pair])
            summary.append(dict(session=session,method=method,n_common=len(ids),
                r2_task_partial=1-data.sse_full.sum()/data.sse_nuisance.sum(),
                r2_full=1-data.sse_full.sum()/data.sst.sum(),median_shape_r=np.median(corr)))
            shape_rows.extend(dict(session=session,method=method,feature_id=int(v),shape_r=c) for v,c in zip(ids,corr))
    pd.DataFrame(summary).to_csv(REPORT/"summary.tsv",sep="\t",index=False)
    pd.DataFrame(shape_rows).to_csv(REPORT/"shape_stability.tsv.gz",sep="\t",index=False)
    scores.to_csv(REPORT/"scores.tsv.gz",sep="\t",index=False)
    white[white.feature_id.isin(ids)].to_csv(REPORT/"whitening.tsv.gz",sep="\t",index=False)
    rhos.to_csv(REPORT/"rho.tsv.gz",sep="\t",index=False)
    np.savez_compressed(REPORT/"mean_shapes.npz",feature_ids=ids,times=np.arange(360)*.1,
                        curves=np.asarray(means).reshape(2,5,2,360),methods=np.array(METHODS),sessions=[11,16])
    provenance = dict(n_common=len(ids),candidate_vertices=256,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (REPORT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    build_report()

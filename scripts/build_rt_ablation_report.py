"""Create portable original-REML RT-ablation tables and shapes."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1] / "experiments/hrf_estimation"
OUT = ROOT / "outputs/reml-rt-ablation"
REPORT = ROOT / "reports/reml_rt_ablation"


def correlation(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return np.clip(
        (x * y).sum(axis=1) / np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1)), -1, 1
    )


def load():
    scores = []
    curves = {}
    sources = {}
    common = None
    for session in range(10, 20):
        folder = OUT / f"session{session}"
        completed = json.loads((folder / "COMPLETE.json").read_text())
        for name, digest in completed["artifact_hashes"].items():
            path = folder / name
            assert sha256(path.read_bytes()).hexdigest() == digest
            sources[str(path.relative_to(ROOT))] = digest
        data = pd.read_csv(folder / "crossed_scores.tsv.gz", sep="\t").assign(
            session=session
        )
        scores.append(data)
        reference = json.loads((folder / "reference.json").read_text())
        for rel, digest in reference["hashes"].items():
            assert sha256((ROOT / rel).read_bytes()).hexdigest() == digest
            sources[rel] = digest
        for mode, directory in [
            ("none", folder),
            ("amplitude", ROOT / reference["source"]),
        ]:
            for label in ["insample", "transfer01", "transfer02"]:
                path = next(directory.glob(f"*desc-reml{label}_hrf.npz"))
                with np.load(path) as z:
                    h = z["curves"].copy()
                    ids = z["feature_ids"].copy()
                valid = (
                    np.isfinite(h).all(axis=1)
                    & (h.max(axis=1) > 0)
                    & (h.std(axis=1) > 1e-12)
                )
                current = set(ids[valid])
                common = current if common is None else common & current
                curves[session, mode, label] = (ids, h)
    scores = pd.concat(scores, ignore_index=True)
    transfer = scores[scores.evaluation.str.startswith("transfer")]
    for _, g in transfer.groupby(
        ["session", "selection_rt", "evaluation_rt", "evaluation"]
    ):
        valid = (
            np.isfinite(g[["sse_full", "sse_nuisance", "sst"]]).all(axis=1)
            & g.sse_nuisance.gt(0)
            & g.sst.gt(0)
        )
        common &= set(g.loc[valid, "feature_id"])
    ids = np.array(sorted(common))
    aligned = {}
    for key, (original, h) in curves.items():
        lookup = {v: i for i, v in enumerate(original)}
        h = h[[lookup[v] for v in ids]]
        aligned[key] = h / h.max(axis=1, keepdims=True)
    return transfer[transfer.feature_id.isin(ids)], aligned, ids, sources


def score_summary(data, grouping):
    rows = []
    for key, g in data.groupby(grouping):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            dict(
                zip(grouping, key),
                n_common=g.feature_id.nunique(),
                r2_full=1 - g.sse_full.sum() / g.sst.sum(),
                r2_task_partial=1 - g.sse_full.sum() / g.sse_nuisance.sum(),
                delta_r2_task=(g.sse_nuisance.sum() - g.sse_full.sum()) / g.sst.sum(),
            )
        )
    return pd.DataFrame(rows)


def shape_summary(h, ids):
    shape = []
    between = []
    cross = []
    means = []
    for mode in ["amplitude", "none"]:
        means.append([h[s, mode, "insample"].mean(axis=0) for s in range(10, 20)])
        for s in range(10, 20):
            r = correlation(h[s, mode, "transfer01"], h[s, mode, "transfer02"])
            shape.append(
                dict(
                    session=s,
                    selection_rt=mode,
                    n_common=len(ids),
                    median_curve_r=np.median(r),
                )
            )
            for t in range(10, 20):
                r = correlation(h[s, mode, "insample"], h[t, mode, "insample"])
                cross.append(
                    dict(
                        session_a=s,
                        session_b=t,
                        selection_rt=mode,
                        n_common=len(ids),
                        median_curve_r=np.median(r),
                    )
                )
    for s in range(10, 20):
        r = correlation(h[s, "amplitude", "insample"], h[s, "none", "insample"])
        between.append(
            dict(
                session=s,
                n_common=len(ids),
                median_curve_r=np.median(r),
                q25=np.quantile(r, 0.25),
                q75=np.quantile(r, 0.75),
            )
        )
    return pd.DataFrame(shape), pd.DataFrame(between), pd.DataFrame(cross), means


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    scores, h, ids, sources = load()
    summary = score_summary(scores, ["session", "selection_rt", "evaluation_rt"])
    pooled = score_summary(scores, ["selection_rt", "evaluation_rt"])
    shape, between, cross, means = shape_summary(h, ids)
    for name, frame in [
        ("summary", summary),
        ("pooled", pooled),
        ("split_half_shapes", shape),
        ("between_models", between),
        ("cross_session_shapes", cross),
    ]:
        frame.to_csv(REPORT / f"{name}.tsv", sep="\t", index=False)
    scores.to_csv(REPORT / "scores.tsv.gz", sep="\t", index=False)
    np.savez_compressed(
        REPORT / "mean_shapes.npz",
        feature_ids=ids,
        times=np.arange(360) * 0.1,
        curves=np.array(means),
        sessions=np.arange(10, 20),
        modes=["amplitude", "none"],
    )
    (REPORT / "provenance.json").write_text(
        json.dumps(dict(n_common=len(ids), source_hashes=sources), indent=2) + "\n"
    )
    print(pooled.to_string(index=False))
    print(shape.groupby("selection_rt").median_curve_r.mean())


if __name__ == "__main__":
    main()

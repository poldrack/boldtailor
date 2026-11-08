"""Rebuild empirical-versus-parametric HRF selection summaries from saved fits."""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from boldtailor.hrf_library import HrfLibrary, default_hrf_library

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def load_selections():
    """Validate source hashes and candidate identities on the shared shape mask."""
    library = default_hrf_library()
    parametric = HrfLibrary.from_parameters(
        [c.parameters for c in library.candidates if c.kind == "double_gamma"]
    )
    libraries = {"Library CV": library, "Parametric library CV": parametric}
    ids = np.load(REPORTS / "core_hrf_shape_summary.npz")["feature_ids"]
    provenance = json.loads((REPORTS / "focused_comparison_provenance.json").read_text())
    rows, sources = [], []
    for record in provenance:
        if record["method"] not in libraries:
            continue
        folder = ROOT / record["source"]
        manifest = json.loads((folder / "manifest.json").read_text())
        candidates = libraries[record["method"]]
        assert manifest["library_fingerprint"] == candidates.fingerprint
        path = next(folder.glob("*desc-librarycvinsample_hrf.npz"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["artifact_hashes"][str(path.relative_to(ROOT))]
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=digest,
                            library_fingerprint=candidates.fingerprint))
        with np.load(path) as fit:
            lookup = dict(zip(fit["feature_ids"], fit["candidate_ids"]))
        for vertex in ids:
            candidate = candidates.candidates[int(lookup[vertex])]
            rows.append(dict(method=record["method"], session=record["session"],
                             feature_id=int(vertex), candidate_id=candidate.id,
                             empirical=candidate.kind == "glmsingle"))
    (REPORTS / "empirical_selection_provenance.json").write_text(
        json.dumps(dict(mask="core_hrf_shape_summary.npz:feature_ids",
                        evaluation="insample (all 12 runs)", sources=sources), indent=2) + "\n"
    )
    return pd.DataFrame(rows)


def session_rates(selections):
    rates = selections.groupby(["method", "session"]).empirical.agg(
        n_common="size", n_empirical="sum").reset_index()
    rates["n_parametric"] = rates.n_common - rates.n_empirical
    rates["empirical_percent"] = 100 * rates.n_empirical / rates.n_common
    return rates


def session_agreement(selections):
    full = selections.query("method == 'Library CV'").pivot(
        index="feature_id", columns="session", values="candidate_id")
    param = selections.query("method == 'Parametric library CV'").pivot(
        index="feature_id", columns="session", values="candidate_id")
    empirical = selections.query("method == 'Library CV'").pivot(
        index="feature_id", columns="session", values="empirical")
    rows = []
    for a, b in combinations(full.columns, 2):
        x, y = empirical[a], empirical[b]
        both, neither, union = x & y, ~x & ~y, x | y
        agreement = (x == y).mean()
        chance = x.mean()*y.mean() + (1-x.mean())*(1-y.mean())
        same = full[a] == full[b]
        rows.append(dict(session_a=a, session_b=b, n_common=len(full),
                         n_both_empirical=both.sum(), n_both_parametric=neither.sum(),
                         n_switched=(x != y).sum(), family_agreement=agreement,
                         family_kappa=(agreement-chance)/(1-chance) if chance < 1 else np.nan,
                         empirical_jaccard=both.sum()/union.sum() if union.any() else np.nan,
                         n_same_empirical_id=(same & both).sum(),
                         same_empirical_id_given_both=same[both].mean(),
                         full_exact_id_agreement=same.mean(),
                         parametric_exact_id_agreement=(param[a] == param[b]).mean()))
    return pd.DataFrame(rows), empirical.sum(axis=1).rename("n_empirical_sessions")


def main():
    selections = load_selections()
    rates = session_rates(selections)
    pairs, frequency = session_agreement(selections)
    selections.to_csv(REPORTS / "empirical_selections.tsv.gz", sep="\t", index=False)
    rates.to_csv(REPORTS / "empirical_selection_rates.tsv", sep="\t", index=False)
    pairs.to_csv(REPORTS / "empirical_selection_agreement.tsv", sep="\t", index=False)
    frequency.reset_index().to_csv(REPORTS / "empirical_selection_frequency.tsv", sep="\t", index=False)
    print(rates.to_string(index=False))
    print(pairs.mean(numeric_only=True).to_string())
    print(frequency.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

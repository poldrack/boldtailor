"""R2: HRF choice maps and their session-to-session consistency."""

import pandas as pd

from boldtailor.reliability import compare_hrfs


def session_consistency(library, hrf_indices, sessions):
    result = compare_hrfs(library, hrf_indices, sessions)
    pairwise, baseline = result["summary"][0], result["summary"][1]
    return pd.DataFrame(
        {"mean_pairwise_r": pairwise, "mean_canonical_baseline": baseline}
    )

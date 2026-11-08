# Empirical HRF selection across sessions 10–19

### Empirical versus parametric HRF selection

The **full library** contains the canonical HRF, 512 parametric double-gamma HRFs, and 20 empirical GLMsingle HRFs. The **parametric-only library** contains the same canonical and 512 parametric candidates; its empirical selection rate is necessarily zero. Here “parametric” includes the canonical candidate. Both families are library HRFs; the distinction is empirical versus parametric origin.

This report uses **all-run CV selections (12 runs per session)** on the same **1,487 matched vertices** as the shape comparisons. Candidate identities were verified against each saved library fingerprint and fit hash. These are selection frequencies, not evidence that a family significantly outperforms another. Rates are not adjusted for the unequal number of candidates. The six-run transfer fits are not pooled with these selections.

| Session | Empirical vertices | Parametric vertices | Empirical % |
|---|---:|---:|---:|
| 10 | 542 | 945 | 36.4 |
| 11 | 190 | 1297 | 12.8 |
| 12 | 66 | 1421 | 4.4 |
| 13 | 268 | 1219 | 18.0 |
| 14 | 274 | 1213 | 18.4 |
| 15 | 524 | 963 | 35.2 |
| 16 | 189 | 1298 | 12.7 |
| 17 | 132 | 1355 | 8.9 |
| 18 | 268 | 1219 | 18.0 |
| 19 | 424 | 1063 | 28.5 |

Parametric-only CV selects 0 empirical and 1,487 parametric HRFs in each session by construction.

#### Consistency of empirical selections across sessions

All 45 session pairs compare the **same vertices**. Family agreement counts both empirical→empirical and parametric→parametric selections. **Cohen’s κ** adjusts this agreement for the two sessions’ marginal selection rates. **Empirical Jaccard overlap** is the number selecting empirical in both sessions divided by the number selecting empirical in either; it excludes parametric→parametric agreements.

The conditional exact-ID measure asks: among vertices choosing empirical in both sessions, what fraction chose the **same one of the 20 empirical HRFs**? The table also compares exact candidate-ID agreement for the full and parametric-only methods. Exact-ID agreement is stricter than HRF shape correlation: nearby candidates can have very similar curves.

Empirical selection varies substantially by session (**4.4%–36.4%**). Across the 45 distinct session pairs, mean family agreement is **73.4%**, but mean κ is only **0.143** and mean empirical Jaccard overlap is **16.8%**. Among vertices selecting empirical in both sessions, the mean pairwise rate of selecting the exact same empirical HRF is **24.3%**.

Exact candidate agreement averages **5.5% for full-library CV** and **6.3% for parametric-only CV**. This low identity stability coexists with the high curve correlations above: candidate identity is much less stable than broad HRF shape. Of the 1,487 vertices, **430 never select empirical**, **1,057 select empirical at least once**, and **none select empirical in all ten sessions**. All averages here weight the 45 session pairs equally; they are descriptive summaries, not independent statistical replications.


See `empirical_selection_rates.tsv`, `empirical_selection_agreement.tsv`, `empirical_selection_frequency.tsv`, and `empirical_selections.tsv.gz` for the underlying records. Source hashes and library fingerprints are in `empirical_selection_provenance.json`. Rebuild with `uv run python experiments/hrf_estimation/build_empirical_report.py`. Figures are in `session_comparison.ipynb`.

# Portable inputs for the historical results notebook

These files reproduce `../../results.py` without ignored local fit directories.
Source SHA256 digests and original completion records are in `provenance.json`.
Source files were verified against their saved completion hashes before export.

Scores are gzip-compressed TSVs. Each HRF NPZ stores the median of individually
peak-normalized valid curves, its time axis, and valid-vertex count. These are
exactly the curves plotted by the historical notebook, not a new model fit.
Each directory's `COMPLETE.json` verifies the transformed snapshot files.
`progress_snapshot.json` preserves historical queue counts rather than claiming
that missing local directories mean unfinished analysis.

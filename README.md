# boldtailor

Boldtailor is a composable first-level fMRI modeling package.

Phase 1 provides an array-based conventional GLM backed by Nilearn. BIDS,
NIfTI, CIFTI, adaptive HRFs, and the GLMsingle recipe are delivered in later
phases described by the package design.

## Array-based conventional GLM

```python
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec

events = pd.DataFrame(
    {
        "onset": [0.0, 8.0, 16.0, 24.0],
        "duration": [1.0, 1.0, 1.0, 1.0],
        "trial_type": ["face", "house", "face", "house"],
    }
)
signals = np.load("run_signals.npy")  # shape: time x features
data = from_arrays(signals, events, tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
    noise_model="ar1",
)

result = fit(data, model)
contrast_effect = result.effect("face_gt_house")
directional_p = result.one_sided_p_value("face_gt_house")
fit_quality = result.r2
```

`fit()` returns arrays only and never writes files. BIDS, NIfTI, and CIFTI
adapters are added in later phases. Contrast p-values are directional and
one-sided, matching Nilearn. For nonuniform or nonzero acquisition times, pass
one frame-time array per run instead of `tr`.

Features with zero centered sum of squares, including all-zero and constant
features, are accepted. Their per-run and aggregate R-squared values are NaN;
varying features in the same fit retain their ordinary estimates and
inference. Every run must have positive residual degrees of freedom for
contrast inference, otherwise `fit()` raises a run-specific `ValueError`.

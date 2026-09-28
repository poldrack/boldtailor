"""Run explicitly in an isolated environment containing the built wheel."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import boldtailor
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared


def main():
    assert "site-packages" in Path(boldtailor.__file__).parts
    assert importlib.util.find_spec("ipykernel") is None
    rng = np.random.default_rng(721)
    x = rng.normal(size=60)
    design = pd.DataFrame({"task": x, "constant": np.ones(60)})
    y = (2 * x + 10 + rng.normal(scale=0.1, size=60))[:, None]
    prepared = PreparedDesignAnalysis.from_arrays(
        y,
        design,
        tr=1.0,
        column_roles={"task": "task", "constant": "intercept"},
    )
    result = fit_prepared(prepared, contrasts={"task": "task"}, noise_model="ols")
    expected = np.linalg.lstsq(design.to_numpy(), y, rcond=None)[0][0]
    np.testing.assert_allclose(result.effect("task"), expected, atol=1e-10)


if __name__ == "__main__":
    main()

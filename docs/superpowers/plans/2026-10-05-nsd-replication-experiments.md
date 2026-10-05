# NSD Replication Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/nsd_replication/`, a resumable experiment package. It runs boldtailor's b1–b4 single-trial ladder on NSD cortical CIFTI data, computes the Prince et al. (2022) NSD metrics (R1–R6), compares against NSD's released GLMsingle betas, and runs three feature experiments.

**Architecture:**
- A standalone namespace package outside `src/` with no `__init__.py` files, like `examples/validation`.
- Each module has one job: config, trial tables, inputs, confounds, ROI, beta store, ladder fits, LSS, metrics, stats, features, figures, CLI.
- It calls boldtailor's public API and reuses the workflow's fMRIPrep loaders. The boldtailor core is not modified.
- Fits are written per session and level to a common beta format. Metrics read only that format, so boldtailor and released betas are treated alike.

**Tech Stack:** Python ≥ 3.12, uv, boldtailor (this repo), nibabel, numpy, pandas, scipy, scikit-learn (LinearSVC), Nilearn (LSS oracle), and Connectome Workbench `wb_command` (ROI resampling).

**Spec:** `docs/superpowers/specs/2026-10-05-nsd-replication-experiments-design.md`

## Global Constraints

- Every analysis uses only the cortical surface grayordinates: the CIFTI structures `CIFTI_STRUCTURE_CORTEX_LEFT` and `CIFTI_STRUCTURE_CORTEX_RIGHT`. Subcortical volume grayordinates are dropped when inputs are loaded, before any fitting.
- The core package (`src/boldtailor/`) is not modified and gains no dependency. Anatomy (cortex filter, ROI) lives only in the experiment package.
- Every `__init__.py` stays empty. This plan creates none, because `experiments` and `experiments/nsd_replication` are namespace packages.
- Use `uv` for packages and `uv run` for every command. This plan adds no Python dependencies; if one becomes necessary, it goes in an `experiments` dependency group, never in the core dependencies.
- TDD: the failing test is committed first, in its own commit (`test: ... (RED)`), then the implementation.
- Tests are opt-in. They live in `experiments/nsd_replication/` and run with `uv run pytest experiments/nsd_replication -W error`. `pyproject.toml` keeps `testpaths = ["tests"]`, so the default suite is unaffected.
- Code is formatted with Black: `uv run black experiments/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Use these names and paths exactly:
  - BIDS root: `/Volumes/extdata1/NSD/BIDS`
  - fMRIPrep: `derivatives/fmriprep-25.2.5`
  - released betas: `derivatives/betas-fsLR/sub-NN/ses-nsdYY/func/sub-NN_ses-nsdYY_task-nsdcore_space-fsLR_den-91k_desc-<version>_stat-effect_statmap.dscalar.nii`, with 750 maps `trial-001`…`trial-750` in percent signal change. NSD's ×300 scaling is already undone, so no rescaling is applied.
  - image column: `73k_id`
  - task: `nsdcore`
- Sessions:
  - pilot: `sub-07`, `ses-nsd10`–`ses-nsd19`;
  - primary: `sub-01`–`sub-04`, `ses-nsd01`–`ses-nsd10`;
  - extension: `sub-05`, `sub-06`, `sub-08`, `ses-nsd01`–`ses-nsd10`.
- Released GLMsingle versions (`desc-` values): `assumehrf` (b1), `fithrf` (b2), `fithrfGLMdenoiseRR` (b4).
- The released betas share the fMRIPrep fsLR 91k grayordinate axis. ppdata is not used.
- Every comparison report states that the two arms differ in preprocessing (fMRIPrep versus NSD's pipeline) as well as in the GLM.
- Freeze tag: `nsd-replication-protocol-v1`.
- Reliability thresholds: r = −0.2 to 0.6 in steps of 0.05.
- The R1 threshold mask is the composite reliability: the mean over the beta versions being compared.

## Review Focus

1. **Images with more than 3 presentations, or any presented twice in one run.** Expected: the repetition array uses each image's first three presentations in time order and never crashes. Pinned in Task 2 by `test_repetition_array_uses_first_three_in_time_order`.
2. **Constant or all-NaN grayordinates** (medial wall, dropout). Expected: z-scoring and correlations give NaN for those features, never warnings or crashes, and every ROI summary ignores NaN. Pinned in Task 5 by `test_zscore_constant_feature_is_nan_without_warning` and in Task 8 by `test_reliability_constant_feature_is_nan`.
3. **A fit directory left half-written by a crash.** Expected: it is detected as incomplete and refit, not read as valid. `metadata.json` is written last, and its absence means incomplete. Pinned in Task 5 by `test_incomplete_fit_is_not_complete`.
4. **Released-beta trial counts or order that don't match the events.** For example, a session with one run missing gives 688 trials instead of 750. Expected: an error naming subject, session and version. Pinned in Task 11 by `test_released_trial_count_mismatch_names_session`.
5. **Resampled ROI labels are fractional** at the ROI edge after adaptive barycentric resampling. Expected: a vertex is inside when its value is ≥ 0.5, so the ROI neither grows nor shrinks systematically. Pinned in Task 4 by `test_roi_threshold_at_half`.

---

## File structure

| File | Responsibility |
|---|---|
| `experiments/nsd_replication/conftest.py` | Put the repo root on `sys.path`; shared synthetic fixtures |
| `experiments/nsd_replication/README.md` | Prerequisites (Workbench, label files, converted released betas), commands, verified API notes |
| `experiments/nsd_replication/protocol.md` | Frozen protocol |
| `experiments/nsd_replication/config.py` | `ExperimentConfig`, `load_config` |
| `experiments/nsd_replication/configs/pilot.toml`, `primary.toml`, `extension.toml` | Run configurations |
| `experiments/nsd_replication/trials.py` | Standard trial tables, presentation order, repetition arrays |
| `experiments/nsd_replication/inputs.py` | Cortex filter, `Session`, fMRIPrep loader, `analysis_data` |
| `experiments/nsd_replication/roi.py` | nsdgeneral fsaverage → fsLR 32k → cortical mask |
| `experiments/nsd_replication/betas.py` | Fit directories, read/write, completeness, z-scoring |
| `experiments/nsd_replication/ladder.py` | Boldtailor b1, b2, b2-lib20, b3, b4 per session |
| `experiments/nsd_replication/lss.py` | Least-squares-separate betas |
| `experiments/nsd_replication/metrics.py` | R1, R3B, R4, R5, R6 |
| `experiments/nsd_replication/hrf_maps.py` | R2: HRF index maps and session consistency |
| `experiments/nsd_replication/stats.py` | TOST, replication criteria, CIs |
| `experiments/nsd_replication/released.py` | Released-beta reader and alignment |
| `experiments/nsd_replication/features.py` | HRF-choice, modulator and gate experiments |
| `experiments/nsd_replication/figures.py` | Figures |
| `experiments/nsd_replication/run.py` | CLI stages `fit`, `metrics`, `features`, `figures` |
| `experiments/nsd_replication/test_*.py` | One test module per source module |

---

### Task 1: Scaffold, dependencies, configuration

**Files:**
- Create: `experiments/nsd_replication/conftest.py`, `config.py`, `test_config.py`, `configs/pilot.toml`, `README.md`

**Interfaces:**
- Produces:
  - `ExperimentConfig` (frozen dataclass) with fields:
    - required: `bids_dir: Path`, `output_dir: Path`, `freesurfer_dir: Path`, `subjects: tuple[str, ...]`, `sessions: tuple[str, ...]`;
    - optional: `fmriprep_dir: Path | None = None`, `released_dir: Path | None = None`, `alignment_floor: float | None = None`, `task: str = "nsdcore"`, `image_column: str = "73k_id"`, `n_jobs: int = 4`, `block_size: int = 4096`.
  - `load_config(path) -> ExperimentConfig`.

- [ ] **Step 1: Check prerequisites**

No new Python dependencies are needed: scikit-learn, Nilearn, nibabel and scipy are already project dependencies. Run `uv sync --group dev`, then run `which wb_command`; expected: a path. The machine has `/Applications/wb_view.app/Contents/usr/bin/wb_command`.

- [ ] **Step 2: Write `conftest.py`**

```python
import sys
from pathlib import Path

# Modules import siblings as experiments.nsd_replication.*; tests run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 3: Write the failing test `test_config.py`**

```python
from pathlib import Path

import pytest

from experiments.nsd_replication.config import ExperimentConfig, load_config

TOML = """
bids_dir = "/data/BIDS"
output_dir = "/data/out"
freesurfer_dir = "/data/freesurfer"
subjects = ["sub-07"]
sessions = ["ses-nsd10", "ses-nsd11"]
n_jobs = 2
"""


def test_load_config_converts_paths_and_tuples(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML)
    config = load_config(path)
    assert config.bids_dir == Path("/data/BIDS")
    assert config.subjects == ("sub-07",)
    assert config.sessions == ("ses-nsd10", "ses-nsd11")
    assert config.n_jobs == 2
    assert config.image_column == "73k_id"


@pytest.mark.parametrize(
    "field, value",
    [("subjects", ()), ("subjects", ("07",)), ("sessions", ("ses-1", "ses-1"))],
)
def test_bad_labels_rejected(field, value):
    values = dict(
        bids_dir=Path("/b"),
        output_dir=Path("/o"),
        freesurfer_dir=Path("/l"),
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )
    values[field] = value
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**values)


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML + 'colour = "red"\n')
    with pytest.raises(ValueError, match="colour"):
        load_config(path)
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest experiments/nsd_replication/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: ... config`.

- [ ] **Step 5: Commit RED**

```bash
git add experiments/nsd_replication/conftest.py experiments/nsd_replication/test_config.py
git commit -m "test: experiment configuration (RED)"
```

- [ ] **Step 6: Implement `config.py`**

```python
"""Experiment configuration loaded from a TOML file."""

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib

from boldtailor.workflow.files import bids_label


def _labels(name, values, entity):
    values = tuple(values)
    if (
        not values
        or len(set(values)) != len(values)
        or not all(bids_label(v, entity) for v in values)
    ):
        raise ValueError(f"{name} must be distinct {entity}-<label> values: {values}")
    return values


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    bids_dir: Path
    output_dir: Path
    freesurfer_dir: Path
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    fmriprep_dir: Path | None = None
    released_dir: Path | None = None
    alignment_floor: float | None = None
    task: str = "nsdcore"
    image_column: str = "73k_id"
    n_jobs: int = 4
    block_size: int = 4096

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "subjects", _labels("subjects", self.subjects, "sub"))
        set_(self, "sessions", _labels("sessions", self.sessions, "ses"))
        for name in ("n_jobs", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def load_config(path) -> ExperimentConfig:
    values = tomllib.loads(Path(path).read_text())
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    paths = {k: Path(v) for k, v in values.items() if k.endswith("_dir")}
    return ExperimentConfig(**{**values, **paths})
```

- [ ] **Step 7: Write `configs/pilot.toml` and `README.md`**

```toml
bids_dir = "/Volumes/extdata1/NSD/BIDS"
output_dir = "/Volumes/extdata1/NSD/BIDS/derivatives/nsd-replication"
freesurfer_dir = "/Volumes/extdata1/NSD/BIDS/derivatives/freesurfer-NSD"
released_dir = "/Volumes/extdata1/NSD/BIDS/derivatives/betas-fsLR"
subjects = ["sub-07"]
sessions = ["ses-nsd10", "ses-nsd11", "ses-nsd12", "ses-nsd13", "ses-nsd14",
            "ses-nsd15", "ses-nsd16", "ses-nsd17", "ses-nsd18", "ses-nsd19"]
n_jobs = 4
```

`README.md` contains these sections:

- **Purpose.** One paragraph, linking the spec.
- **Prerequisites:**
  - `uv sync --group dev`;
  - Connectome Workbench `wb_command` on `PATH`;
  - each subject's native-surface `lh.nsdgeneral.mgz` and `rh.nsdgeneral.mgz` under `freesurfer_dir/subjNN/label/`;
  - the released betas converted to CIFTI under `released_dir` (`derivatives/betas-fsLR`).
- **Tests:** `uv run pytest experiments/nsd_replication -W error`.
- **Verified API notes:**
  - ROI resampling: `wb_command -metric-resample <in> <source sphere> <target sphere> ADAP_BARY_AREA <out> -area-metrics <source area> <target area>`, with inputs read from the released betas' JSON sidecars, then thresholded at 0.5.

- [ ] **Step 8: Run the tests, format, commit**

Run: `uv run pytest experiments/nsd_replication -W error -v`
Expected: 5 passed.

```bash
uv run black experiments/
git add experiments/nsd_replication
git commit -m "feat: NSD replication experiment config and scaffold"
```

---

### Task 2: Trial tables, repetitions, protocol draft

**Files:**
- Create: `experiments/nsd_replication/trials.py`, `test_trials.py`, `protocol.md`

**Interfaces:**
- Produces:
  - `TRIAL_COLUMNS = ("session", "run", "trial", "onset", "image")`
  - `trial_table(events: Sequence[pd.DataFrame], labels: Sequence[str], session: str, image_column: str) -> pd.DataFrame`. One row per trial in run then onset order; `trial` is the 0-based index within the run; `image` is an int.
  - `presentation_order(trials: pd.DataFrame) -> pd.Series`. The 0-based presentation number of each row's image, ordered by (session, run, onset). Same index as `trials`.
  - `images_with(trials, n=3) -> np.ndarray`. Sorted image IDs with at least `n` presentations.
  - `repetition_array(betas: np.ndarray, trials: pd.DataFrame, images: np.ndarray, n=3) -> np.ndarray`. Shape `(n, len(images), features)`, using each image's first `n` presentations.
  - `out_of_sample_images(trials, n=3) -> np.ndarray`. Images whose first `n` presentations fall in `n` distinct sessions.
  - `repeat_counts(trials) -> dict[int, int]`. Maps a presentation count to the number of images with that count.

- [ ] **Step 1: Write the failing tests `test_trials.py`**

```python
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.trials import (
    images_with,
    out_of_sample_images,
    presentation_order,
    repeat_counts,
    repetition_array,
    trial_table,
)


def _events(onsets, images):
    return pd.DataFrame(
        {"onset": onsets, "duration": 3.0, "73k_id": images, "trial_type": 0}
    )


def test_trial_table_orders_runs_and_onsets():
    events = [_events([8.0, 4.0], [5, 6]), _events([4.0], [5])]
    table = trial_table(events, ["run-01", "run-02"], "ses-nsd10", "73k_id")
    assert list(table.columns) == ["session", "run", "trial", "onset", "image"]
    assert table["image"].tolist() == [6, 5, 5]
    assert table["trial"].tolist() == [0, 1, 0]
    assert table["run"].tolist() == ["run-01", "run-01", "run-02"]


def test_trial_table_missing_image_column_raises():
    with pytest.raises(ValueError, match="73k_id"):
        trial_table([_events([4.0], [1]).drop(columns="73k_id")], ["run-01"], "s", "73k_id")


def _trials(rows):
    return pd.DataFrame(rows, columns=["session", "run", "trial", "onset", "image"])


def test_repetition_array_uses_first_three_in_time_order():
    trials = _trials(
        [
            ("ses-b", "run-01", 0, 4.0, 9),  # 3rd presentation (later session)
            ("ses-a", "run-01", 0, 4.0, 9),  # 1st
            ("ses-a", "run-01", 1, 8.0, 9),  # 2nd (same run as the 1st)
            ("ses-c", "run-01", 0, 4.0, 9),  # 4th: ignored
        ]
    )
    betas = np.array([[3.0], [1.0], [2.0], [4.0]])
    assert presentation_order(trials).tolist() == [2, 0, 1, 3]
    reps = repetition_array(betas, trials, np.array([9]))
    assert reps.shape == (3, 1, 1)
    assert reps[:, 0, 0].tolist() == [1.0, 2.0, 3.0]


def test_images_with_and_counts():
    trials = _trials(
        [("s", "r", i, float(i), img) for i, img in enumerate([1, 1, 1, 2, 2, 3])]
    )
    assert images_with(trials, 3).tolist() == [1]
    assert images_with(trials, 2).tolist() == [1, 2]
    assert repeat_counts(trials) == {1: 1, 2: 1, 3: 1}


def test_out_of_sample_requires_distinct_sessions():
    trials = _trials(
        [
            ("s1", "r", 0, 0.0, 1), ("s2", "r", 0, 0.0, 1), ("s3", "r", 0, 0.0, 1),
            ("s1", "r", 1, 4.0, 2), ("s1", "r", 2, 8.0, 2), ("s2", "r", 1, 4.0, 2),
        ]
    )
    assert out_of_sample_images(trials).tolist() == [1]
```

- [ ] **Step 2: Run, verify failure, commit RED**

Run: `uv run pytest experiments/nsd_replication/test_trials.py -v`
Expected: FAIL with `ModuleNotFoundError`.

```bash
git add experiments/nsd_replication/test_trials.py
git commit -m "test: trial tables and repetition arrays (RED)"
```

- [ ] **Step 3: Implement `trials.py`**

```python
"""Standard trial tables and image-repetition bookkeeping."""

import numpy as np
import pandas as pd

TRIAL_COLUMNS = ("session", "run", "trial", "onset", "image")
_ORDER = ["session", "run", "onset"]


def _run_table(events, label, session, image_column):
    if image_column not in events:
        raise ValueError(f"{session} {label}: events lack the {image_column} column")
    ordered = events.sort_values("onset", kind="stable").reset_index(drop=True)
    return pd.DataFrame(
        {
            "session": session,
            "run": label,
            "trial": np.arange(len(ordered)),
            "onset": ordered["onset"].astype(float),
            "image": ordered[image_column].astype(int),
        }
    )


def trial_table(events, labels, session, image_column):
    tables = [
        _run_table(e, label, session, image_column)
        for e, label in zip(events, labels, strict=True)
    ]
    return pd.concat(tables, ignore_index=True)


def presentation_order(trials):
    ordered = trials.sort_values(_ORDER, kind="stable")
    return ordered.groupby("image").cumcount().reindex(trials.index)


def images_with(trials, n=3):
    counts = trials["image"].value_counts()
    return np.sort(counts.index[counts >= n].to_numpy())


def repeat_counts(trials):
    counts = trials["image"].value_counts().value_counts()
    return {int(k): int(v) for k, v in sorted(counts.items())}


def repetition_array(betas, trials, images, n=3):
    order = presentation_order(trials).to_numpy()
    position = pd.Series(np.arange(len(images)), index=images)
    out = np.full((n, len(images), betas.shape[1]), np.nan)
    keep = (order < n) & trials["image"].isin(images).to_numpy()
    rows = position[trials["image"].to_numpy()[keep]].to_numpy()
    out[order[keep], rows] = betas[keep]
    return out


def out_of_sample_images(trials, n=3):
    first = trials[presentation_order(trials) < n]
    sessions = first.groupby("image")["session"].nunique()
    counts = first.groupby("image").size()
    return np.sort(sessions.index[(sessions == n) & (counts == n)].to_numpy())
```

- [ ] **Step 4: Run, format, commit GREEN**

Run: `uv run pytest experiments/nsd_replication/test_trials.py -W error -v`
Expected: all PASS.

```bash
uv run black experiments/
git add experiments/nsd_replication/trials.py
git commit -m "feat: trial tables and repetition arrays"
```

- [ ] **Step 5: Count the pilot's repeats**

The script below needs only the BIDS events. Run it and paste its output into `protocol.md`:

```bash
uv run python - <<'EOF'
from pathlib import Path
import pandas as pd
from experiments.nsd_replication.trials import trial_table, repeat_counts, out_of_sample_images
root = Path("/Volumes/extdata1/NSD/BIDS/sub-07")
tables = []
for ses in sorted(p.name for p in root.glob("ses-nsd1*")):
    files = sorted((root / ses / "func").glob("*_events.tsv"))
    events = [pd.read_csv(f, sep="\t") for f in files]
    labels = [f.name.split("_")[3] for f in files]
    tables.append(trial_table(events, labels, ses, "73k_id"))
trials = pd.concat(tables, ignore_index=True)
print(len(trials), repeat_counts(trials), len(out_of_sample_images(trials)))
EOF
```

Expected output: about 7500 trials, plus the count dictionary and the number of out-of-sample images.

- [ ] **Step 6: Write the `protocol.md` draft (no δ yet) and commit**

The protocol has these sections:

1. **Windows and subjects.** Copy them from the Global Constraints.
2. **Metric definitions.** These come from the paper's Methods; quotations were checked on 2026-10-05 against PMC9708069.
   - **R1.** Betas are z-scored per grayordinate within each session. Use images with ≥ 3 presentations in the window, taking each image's first three.
     - Reliability is the mean over k ∈ {1, 2, 3} of the Pearson r, across images, between presentation k and the mean of the other two.
     - *Deviation:* the Methods text lists two of the three splits. We use all three, which is the "all possible unique split-halves" the paper describes.
     - Masks: composite reliability (the mean over the compared versions) ≥ t, for t = −0.2:0.05:0.6, within the ROI.
     - Plotted: for each version and t, the mean over masked grayordinates of (version reliability − composite).
     - Also reported: the median ROI reliability per version.
   - **R3B.** Restrict to images whose first three presentations fall in three distinct sessions. This follows the paper's rule of removing repeats inside one GLMsingle partition (one session).
   - **R4.** For each session, the trial × trial Pearson correlation of ROI patterns, restricted to grayordinates with composite reliability ≥ 0 and ≥ 0.3.
     - Summary: the mean correlation over same-run trial pairs at lag L = 1…15 trials, with L × 4 s on the x-axis, averaged over sessions.
     - *Deviation:* the paper shows full session RSMs. We summarise within runs, because between-run time gaps vary.
   - **R5.** For each subject, the RDM is 1 − Pearson r between repetition-averaged ROI patterns of the shared images. Score each subject pair by the Pearson r of the RDMs' upper triangles. Grayordinate masks: composite ≥ 0, 0.2 and 0.4.
     - *Deviation:* the paper used 241 images shared with BOLD5000. We use the NSD images that every primary subject saw at least three times in the window.
   - **R6.** Linear SVM (`sklearn.svm.LinearSVC`, default C, one-vs-rest) over images with ≥ 3 presentations. Three folds: train on two presentations, test on the third. Report accuracy and chance (1 / number of classes). Masks: composite ≥ 0, 0.1, 0.2, 0.3 and 0.4, skipped when fewer than 10 grayordinates remain.
   - **R2.** For each session, b2's HRF index map. Consistency is measured with `boldtailor.reliability.compare_hrfs` over the window's sessions.
3. **Ladder and settings.** Copy the spec's ladder table. Fraction grid: the `WorkflowSettings` default `(0.1, …, 1.0)`. Encoding mode `within_run`. Denoising: `select_denoising` defaults.
4. **ROI.** Copy from the spec.
5. **Repeat counts.** Paste the Step 5 output.
6. **Margin δ, replication criteria, alignment floor.** Write: "Set in Task 12 before the freeze."
7. **Deviations log.** Starts empty.

```bash
git add experiments/nsd_replication/protocol.md
git commit -m "docs: NSD replication protocol draft with metric definitions"
```

---

### Task 3: Cortical inputs

**Files:**
- Create: `experiments/nsd_replication/inputs.py`, `test_inputs.py`

**Interfaces:**
- Consumes:
  - `ExperimentConfig` (Task 1);
  - from the boldtailor workflow: `load_session`, `block_signals` and `detect_task_model` (`boldtailor.workflow.inputs`), `WorkflowSettings` and `resolve_fmriprep_dir` (`boldtailor.workflow.settings`);
  - `boldtailor.data.from_arrays`.
- Produces:
  - `CORTEX = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")`
  - `cortical_indices(brain: nib.cifti2.BrainModelAxis) -> np.ndarray`
  - `Session`, a frozen dataclass:
    - `subject: str`, `session: str`, `labels: tuple[str, ...]`;
    - `signals: tuple[np.ndarray, ...]`: float32, time × cortical features;
    - `events: tuple[pd.DataFrame, ...]`, `confounds: tuple[pd.DataFrame, ...]`, `frame_times: tuple[np.ndarray, ...]`;
    - `brain: nib.cifti2.BrainModelAxis`: cortical-only;
    - `task_model: TaskModel | None`;
    - `source: str`: always `"fmriprep"` here.
  - `analysis_data(session, indices, confounds=None) -> AnalysisData`
  - `load_fmriprep(config, subject, session) -> Session`
  - `fmriprep_brain(config, subject, session) -> nib.cifti2.BrainModelAxis`. The cortical axis of the session's first fMRIPrep run, read from the header only.

- [ ] **Step 1: Write the failing tests `test_inputs.py`**

```python
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.inputs import CORTEX, Session, analysis_data, cortical_indices

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain(with_cortex=True):
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, :] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, "CIFTI_STRUCTURE_THALAMUS_LEFT")
    return (left + thal + right) if with_cortex else (left + thal)


def test_cortical_indices_keep_surface_in_order():
    assert cortical_indices(_brain()).tolist() == [0, 1, 4]


def test_missing_cortical_structure_raises():
    with pytest.raises(ValueError, match="CORTEX_RIGHT"):
        cortical_indices(_brain(with_cortex=False))


def test_analysis_data_selects_columns_and_keeps_runs():
    signals = (np.arange(12, dtype=np.float32).reshape(4, 3),) * 2
    events = (pd.DataFrame({"onset": [0.0], "duration": [1.0]}),) * 2
    session = Session(
        subject="sub-07", session="ses-nsd10", labels=("run-01", "run-02"),
        signals=signals, events=events,
        confounds=(pd.DataFrame({"c": [0.0, 1, 0, 1]}),) * 2,
        frame_times=(np.arange(4) * 1.6,) * 2, brain=_brain()[[0, 1, 4]],
        task_model=None, source="fmriprep",
    )
    data = analysis_data(session, np.array([0, 2]))
    assert data.n_runs == 2
    np.testing.assert_array_equal(data.signals[0], signals[0][:, [0, 2]])


def _pilot_config():
    from experiments.nsd_replication.config import ExperimentConfig

    return ExperimentConfig(
        bids_dir=DATA, output_dir=DATA / "derivatives/nsd-replication",
        freesurfer_dir=DATA, subjects=("sub-07",), sessions=("ses-nsd10",),
    )


@pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")
def test_fmriprep_pilot_session_is_cortical_only():
    from experiments.nsd_replication.inputs import fmriprep_brain, load_fmriprep

    session = load_fmriprep(_pilot_config(), "sub-07", "ses-nsd10")
    assert len(session.labels) == 12
    assert session.signals[0].shape[1] == len(session.brain) == 59412
    assert set(session.brain.name) == set(CORTEX)
    assert "73k_id" in session.events[0]
    assert fmriprep_brain(_pilot_config(), "sub-07", "ses-nsd10") == session.brain
```

- [ ] **Step 2: Run, verify failure, commit RED**

Run: `uv run pytest experiments/nsd_replication/test_inputs.py -v`
Expected: FAIL with `ModuleNotFoundError`.

```bash
git add experiments/nsd_replication/test_inputs.py
git commit -m "test: cortex filter and cortical session loading (RED)"
```

- [ ] **Step 3: Implement `inputs.py`**

```python
"""Load one NSD fMRIPrep session as cortical-only arrays for boldtailor."""

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.model import TaskModel
from boldtailor.workflow.inputs import block_signals, detect_task_model, load_session
from boldtailor.workflow.settings import WorkflowSettings, resolve_fmriprep_dir

CORTEX = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")


@dataclass(frozen=True, kw_only=True)
class Session:
    subject: str
    session: str
    labels: tuple[str, ...]
    signals: tuple[np.ndarray, ...]
    events: tuple[pd.DataFrame, ...]
    confounds: tuple[pd.DataFrame, ...]
    frame_times: tuple[np.ndarray, ...]
    brain: nib.cifti2.BrainModelAxis
    task_model: TaskModel | None
    source: str


def cortical_indices(brain):
    present = {name for name, _, _ in brain.iter_structures()}
    missing = [name for name in CORTEX if name not in present]
    if missing:
        raise ValueError(f"CIFTI lacks cortical surface structures: {missing}")
    return np.flatnonzero(brain.surface_mask & np.isin(brain.name, CORTEX))


def analysis_data(session, indices, confounds=None):
    confounds = session.confounds if confounds is None else confounds
    return from_arrays(
        [np.asarray(y[:, indices], dtype=float) for y in session.signals],
        list(session.events),
        frame_times=list(session.frame_times),
        confounds=list(confounds),
    )


def _settings(config, subject, session):
    return WorkflowSettings(
        bids_dir=config.bids_dir,
        fmriprep_dir=config.fmriprep_dir,
        output_dir=config.output_dir / "workflow-unused",
        subject=subject,
        session=session,
        task=config.task,
    )


def load_fmriprep(config, subject, session):
    runs = load_session(_settings(config, subject, session), hrf_only=True)
    brain = runs[0].image.header.get_axis(1)
    cortex = cortical_indices(brain)
    labels = tuple(r.label for r in runs)
    return Session(
        subject=subject,
        session=session,
        labels=labels,
        signals=tuple(y.astype(np.float32) for y in block_signals(runs, cortex)),
        events=tuple(r.events for r in runs),
        confounds=tuple(r.confounds for r in runs),
        frame_times=tuple(r.frame_times for r in runs),
        brain=brain[cortex],
        task_model=detect_task_model([r.events for r in runs], labels=list(labels)),
        source="fmriprep",
    )


def fmriprep_brain(config, subject, session):
    root = config.fmriprep_dir or resolve_fmriprep_dir(config.bids_dir)
    func = root / subject / session / "func"
    pattern = f"{subject}_{session}_task-{config.task}_run-*_space-fsLR_den-91k_bold.dtseries.nii"
    files = sorted(func.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{subject} {session}: no fMRIPrep CIFTI in {func}")
    brain = nib.load(files[0]).header.get_axis(1)
    return brain[cortical_indices(brain)]
```

`fmriprep_dir=None` lets `WorkflowSettings` resolve the BIDS root's single `derivatives/fmriprep*`. If constructing `WorkflowSettings` rejects `output_dir`, rule on it: pass a path under `config.output_dir`, and record the ruling. If `resolve_fmriprep_dir`'s signature differs from `resolve_fmriprep_dir(bids_dir)` (`boldtailor/workflow/settings.py:29`), call it as defined there.

- [ ] **Step 4: Run, format, commit GREEN**

Run: `uv run pytest experiments/nsd_replication -W error -v`
Expected: all PASS. The data-gated test passes when `/Volumes/extdata1` is mounted.

```bash
uv run black experiments/
git add experiments/nsd_replication/inputs.py
git commit -m "feat: cortical-only NSD fMRIPrep session loading"
```

---

### Task 4: Subject nsdgeneral ROI on fsLR cortex

**Files:**
- Create: `experiments/nsd_replication/roi.py`, `test_roi.py`

**Background.** NSD gives each subject's nsdgeneral as native-surface labels:
- files: `<freesurfer_dir>/subjNN/label/{lh,rh}.nsdgeneral.mgz`;
- values: float 0/1, on the subject's native vertices, which are the same vertices as NSD's native-surface betas (sub-07: 198,770 left, 200,392 right).

The released betas were resampled to fsLR 32k with Workbench adaptive barycentric resampling. Each beta's JSON sidecar records the inputs used:
- `SourceSpheres` / `TargetSpheres`: lists `[left, right]`;
- `SourceAreaMetrics` / `TargetAreaMetrics`: lists `[left, right]`.

The ROI goes through the identical resampling, then is thresholded at 0.5. That way ROI and betas share one route.

**Interfaces:**
- Consumes:
  - `ExperimentConfig.freesurfer_dir` and `ExperimentConfig.released_dir` (Task 1);
  - `released_path` (Task 11) is not available yet, so this task builds the sidecar path itself with the same file naming.
- Produces:
  - `FSLR_VERTICES = 32492`
  - `label_paths(config, subject) -> dict[str, Path]`. Keys `"L"` and `"R"`, pointing at `freesurfer_dir/subj{NN}/label/{lh,rh}.nsdgeneral.mgz`.
  - `resampling_inputs(sidecar: dict) -> dict[str, dict[str, Path]]`. Maps `"L"`/`"R"` to `{"source_sphere", "target_sphere", "source_area", "target_area"}`.
  - `resample_command(metric_in, metric_out, inputs) -> list[str]`, which gives `["wb_command", "-metric-resample", metric_in, source_sphere, target_sphere, "ADAP_BARY_AREA", metric_out, "-area-metrics", source_area, target_area]`.
  - `fslr_labels(config, subject, sidecar) -> dict[str, np.ndarray]`. Float arrays of length 32492, cached as `output_dir/roi/{subject}_hemi-{L,R}_nsdgeneral.func.gii`.
  - `roi_mask(brain, labels, threshold=0.5) -> np.ndarray[bool]`. Per cortical grayordinate; True where the label value is ≥ threshold. Raises if the result is empty.
  - `load_roi(config, subject, brain) -> np.ndarray[bool]`. Reads the sidecar of the released `assumehrf` beta for `config.sessions[0]`, then calls `roi_mask(brain, fslr_labels(...))`.

- [ ] **Step 1: Write the failing tests `test_roi.py`**

```python
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from experiments.nsd_replication.roi import resample_command, resampling_inputs, roi_mask

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain():
    left = nib.cifti2.BrainModelAxis.from_surface(np.array([0, 3]), 32492, "CIFTI_STRUCTURE_CORTEX_LEFT")
    right = nib.cifti2.BrainModelAxis.from_surface(np.array([5]), 32492, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    return left + right


def _labels():
    return {"L": np.zeros(32492), "R": np.zeros(32492)}


def test_roi_threshold_at_half():
    labels = _labels()
    labels["L"][[0, 3]] = [0.6, 0.4]
    labels["R"][5] = 1.0
    assert roi_mask(_brain(), labels).tolist() == [True, False, True]


def test_empty_roi_raises():
    with pytest.raises(ValueError, match="empty"):
        roi_mask(_brain(), _labels())


def test_wrong_label_length_raises():
    labels = _labels()
    labels["L"] = np.zeros(10)
    with pytest.raises(ValueError, match="32492"):
        roi_mask(_brain(), labels)


def test_resampling_inputs_and_command():
    sidecar = {
        "SourceSpheres": ["/s/lh.sphere", "/s/rh.sphere"],
        "TargetSpheres": ["/t/L.sphere", "/t/R.sphere"],
        "SourceAreaMetrics": ["/s/lh.area", "/s/rh.area"],
        "TargetAreaMetrics": ["/t/L.area", "/t/R.area"],
    }
    inputs = resampling_inputs(sidecar)
    assert inputs["R"]["source_sphere"] == Path("/s/rh.sphere")
    command = resample_command("in.gii", "out.gii", inputs["L"])
    assert command == [
        "wb_command", "-metric-resample", "in.gii", "/s/lh.sphere", "/t/L.sphere",
        "ADAP_BARY_AREA", "out.gii", "-area-metrics", "/s/lh.area", "/t/L.area",
    ]


def test_resampling_inputs_missing_key_raises():
    with pytest.raises(ValueError, match="TargetAreaMetrics"):
        resampling_inputs({"SourceSpheres": [1, 2], "TargetSpheres": [1, 2], "SourceAreaMetrics": [1, 2]})


@pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")
def test_pilot_roi_on_fmriprep_cortex(tmp_path):
    import shutil

    if shutil.which("wb_command") is None:
        pytest.skip("Connectome Workbench not installed")
    from experiments.nsd_replication.config import ExperimentConfig
    from experiments.nsd_replication.inputs import fmriprep_brain
    from experiments.nsd_replication.roi import load_roi

    config = ExperimentConfig(
        bids_dir=DATA, output_dir=tmp_path,
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        released_dir=DATA / "derivatives/betas-fsLR",
        subjects=("sub-07",), sessions=("ses-nsd10",),
    )
    mask = load_roi(config, "sub-07", fmriprep_brain(config, "sub-07", "ses-nsd10"))
    assert mask.shape == (59412,)
    assert 0.05 < mask.mean() < 0.4
```

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_roi.py -v   # FAIL: ModuleNotFoundError
git add experiments/nsd_replication/test_roi.py
git commit -m "test: subject nsdgeneral ROI resampled like the released betas (RED)"
```

- [ ] **Step 3: Implement `roi.py`**

```python
"""Subject nsdgeneral, resampled to fsLR 32k exactly as the released betas were."""

import json
from pathlib import Path
import subprocess
import tempfile

import nibabel as nib
import numpy as np

FSLR_VERTICES = 32492
_HEMI = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}
_KEYS = {
    "source_sphere": "SourceSpheres",
    "target_sphere": "TargetSpheres",
    "source_area": "SourceAreaMetrics",
    "target_area": "TargetAreaMetrics",
}
_SIDECAR = (
    "{subject}/{session}/func/{subject}_{session}_task-{task}_space-fsLR_den-91k"
    "_desc-assumehrf_stat-effect_statmap.json"
)


def label_paths(config, subject):
    label = config.freesurfer_dir / f"subj{subject.removeprefix('sub-')}" / "label"
    return {"L": label / "lh.nsdgeneral.mgz", "R": label / "rh.nsdgeneral.mgz"}


def resampling_inputs(sidecar):
    missing = [key for key in _KEYS.values() if key not in sidecar]
    if missing:
        raise ValueError(f"beta sidecar lacks resampling inputs: {missing}")
    return {
        hemi: {name: Path(sidecar[key][i]) for name, key in _KEYS.items()}
        for i, hemi in enumerate(("L", "R"))
    }


def resample_command(metric_in, metric_out, inputs):
    return [
        "wb_command", "-metric-resample", str(metric_in),
        str(inputs["source_sphere"]), str(inputs["target_sphere"]),
        "ADAP_BARY_AREA", str(metric_out),
        "-area-metrics", str(inputs["source_area"]), str(inputs["target_area"]),
    ]


def _save_metric(values, path):
    array = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32))
    nib.save(nib.gifti.GiftiImage(darrays=[array]), path)


def _resample(source, target, inputs):
    with tempfile.TemporaryDirectory() as tmp:
        metric = Path(tmp) / "native.func.gii"
        _save_metric(np.asarray(nib.load(source).dataobj).ravel(), metric)
        subprocess.run(resample_command(metric, target, inputs), check=True)


def fslr_labels(config, subject, sidecar):
    inputs = resampling_inputs(sidecar)
    out = config.output_dir / "roi"
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for hemi, source in label_paths(config, subject).items():
        target = out / f"{subject}_hemi-{hemi}_nsdgeneral.func.gii"
        if not target.is_file():
            _resample(source, target, inputs[hemi])
        labels[hemi] = nib.load(target).agg_data().astype(float)
    return labels


def roi_mask(brain, labels, threshold=0.5):
    for hemi, values in labels.items():
        if len(values) != FSLR_VERTICES:
            raise ValueError(f"{hemi} labels must have {FSLR_VERTICES} vertices")
    hemis = np.array([_HEMI[name] for name in brain.name])
    mask = np.zeros(len(brain), bool)
    for hemi in ("L", "R"):
        rows = hemis == hemi
        mask[rows] = labels[hemi][brain.vertex[rows]] >= threshold
    if not mask.any():
        raise ValueError("nsdgeneral ROI is empty on these grayordinates")
    return mask


def load_roi(config, subject, brain):
    path = config.released_dir / _SIDECAR.format(
        subject=subject, session=config.sessions[0], task=config.task
    )
    sidecar = json.loads(path.read_text())
    return roi_mask(brain, fslr_labels(config, subject, sidecar))
```

- [ ] **Step 4: Run, format, commit GREEN**

```bash
uv run pytest experiments/nsd_replication/test_roi.py -W error -v   # PASS (data test passes when mounted)
uv run black experiments/
git add experiments/nsd_replication/roi.py
git commit -m "feat: subject nsdgeneral ROI resampled like the released betas"
```

---

### Task 5: Beta store and z-scoring

**Files:**
- Create: `experiments/nsd_replication/betas.py`, `test_betas.py`

**Interfaces:**
- Consumes: `TRIAL_COLUMNS` (Task 2).
- Produces:
  - `fit_dir(output_dir, source, subject, session, level) -> Path`, which is `output_dir/"fits"/source/subject/session/level`.
  - `write_fit(path, betas, trials, metadata, extras=None)`. Writes `betas.npy` (float32), `trials.tsv`, every `extras` array as `<name>.npy`, and `metadata.json` last.
  - `is_complete(path) -> bool`. True only when `metadata.json`, `betas.npy` and `trials.tsv` exist.
  - `read_fit(path) -> tuple[np.ndarray, pd.DataFrame, dict]`
  - `read_extra(path, name) -> np.ndarray`
  - `zscore(betas) -> np.ndarray`. Per column across rows, ddof 0. Columns with zero variance or any NaN give NaN, without warnings.
  - `inputs_digest(session) -> str`. SHA-256 over labels, frame times, events and confounds; signals are excluded for speed.

- [ ] **Step 1: Write the failing tests**

```python
import json

import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.betas import (
    fit_dir, is_complete, read_extra, read_fit, write_fit, zscore,
)


def _trials():
    return pd.DataFrame(
        {"session": "ses-a", "run": "run-01", "trial": [0, 1], "onset": [4.0, 8.0], "image": [5, 6]}
    )


def test_round_trip(tmp_path):
    path = fit_dir(tmp_path, "fmriprep", "sub-07", "ses-nsd10", "b1")
    betas = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_fit(path, betas, _trials(), {"level": "b1"}, extras={"hrf_indices": np.array([3, 4])})
    got, trials, meta = read_fit(path)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, betas)
    pd.testing.assert_frame_equal(trials, _trials())
    assert meta == {"level": "b1"}
    assert read_extra(path, "hrf_indices").tolist() == [3, 4]


def test_incomplete_fit_is_not_complete(tmp_path):
    path = tmp_path / "fit"
    path.mkdir()
    np.save(path / "betas.npy", np.zeros((1, 1)))
    assert not is_complete(path)


def test_row_count_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="rows"):
        write_fit(tmp_path / "f", np.zeros((3, 1)), _trials(), {})


def test_zscore_columns():
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    z = zscore(values)
    np.testing.assert_allclose(z[:, 0], [-1.224744871, 0, 1.224744871])


def test_zscore_constant_feature_is_nan_without_warning():
    values = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 1.0]])
    with np.errstate(all="raise"):
        z = zscore(values)
    assert np.isnan(z[:, 1]).all() and np.isnan(z[:, 2]).all()
```

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_betas.py -v   # FAIL
git add experiments/nsd_replication/test_betas.py
git commit -m "test: common beta store and z-scoring (RED)"
```

- [ ] **Step 3: Implement `betas.py`**

```python
"""Common on-disk beta format shared by boldtailor and released betas."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.nsd_replication.trials import TRIAL_COLUMNS

_REQUIRED = ("betas.npy", "trials.tsv", "metadata.json")


def fit_dir(output_dir, source, subject, session, level):
    return Path(output_dir) / "fits" / source / subject / session / level


def write_fit(path, betas, trials, metadata, extras=None):
    if len(betas) != len(trials):
        raise ValueError(f"betas rows ({len(betas)}) must match trials ({len(trials)})")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").unlink(missing_ok=True)
    np.save(path / "betas.npy", np.asarray(betas, dtype=np.float32))
    trials.loc[:, list(TRIAL_COLUMNS)].to_csv(path / "trials.tsv", sep="\t", index=False)
    for name, values in (extras or {}).items():
        np.save(path / f"{name}.npy", np.asarray(values))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def is_complete(path):
    return all((Path(path) / name).is_file() for name in _REQUIRED)


def read_fit(path):
    path = Path(path)
    if not is_complete(path):
        raise FileNotFoundError(f"incomplete fit: {path}")
    trials = pd.read_csv(path / "trials.tsv", sep="\t")
    metadata = json.loads((path / "metadata.json").read_text())
    return np.load(path / "betas.npy"), trials, metadata


def read_extra(path, name):
    return np.load(Path(path) / f"{name}.npy")


def zscore(betas):
    values = np.asarray(betas, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values).all(axis=0)
    sd = np.zeros(values.shape[1])
    sd[finite] = values[:, finite].std(axis=0)
    ok = finite & (sd > 0)
    out[:, ok] = (values[:, ok] - values[:, ok].mean(axis=0)) / sd[ok]
    return out


def inputs_digest(session):
    digest = hashlib.sha256()
    digest.update(json.dumps([session.source, list(session.labels)]).encode())
    for times, events, confounds in zip(
        session.frame_times, session.events, session.confounds, strict=True
    ):
        digest.update(np.asarray(times, dtype="<f8").tobytes())
        digest.update(events.to_csv(index=False).encode())
        digest.update(confounds.to_csv(index=False).encode())
    return digest.hexdigest()
```

- [ ] **Step 4: Run, format, commit GREEN**

```bash
uv run pytest experiments/nsd_replication/test_betas.py -W error -v   # PASS
uv run black experiments/
git add experiments/nsd_replication/betas.py
git commit -m "feat: common beta store and per-session z-scoring"
```

---

### Task 6: Boldtailor ladder

**Files:**
- Create: `experiments/nsd_replication/ladder.py`, `test_ladder.py`; add a synthetic-session fixture to `conftest.py`

**Interfaces:**
- Consumes:
  - `Session`, `analysis_data` (Task 3);
  - from boldtailor: `fit_single_trials`, `select_hrfs`, `fit_selected_hrfs`, `select_denoising`, `score_fraction_candidates`, `select_ridge_fractions`, `default_hrf_library`, `glmsingle_hrf_library`;
  - `boldtailor.workflow.beta_series.trial_predictors`;
  - `boldtailor.parallel.map_blocks`.
- Produces:
  - `LEVELS = ("b1", "b2", "b2-lib20", "b3", "b4")`
  - `FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)`
  - `LadderFit`, a frozen dataclass with fields `betas: np.ndarray` (trials × features), `extras: dict[str, np.ndarray]` and `record: dict`.
  - `fit_ladder(session, *, levels=LEVELS, task_model=None, block_size=4096, n_jobs=1, denoising=None) -> dict[str, LadderFit]`. `task_model=None` means `session.task_model`. Trial order is run order, then onset within run, matching `trial_table`.
  - `select_session_denoising(session, task_model, library, gate=True) -> DenoisingResult`. Uses all features at once.
  - `denoised_confounds(session, result) -> tuple[pd.DataFrame, ...]`. Adds the result's component columns to each run's baseline confounds, exactly as `with_denoising` does.

Design notes the implementer must follow:

- HRF selection runs once per block on the baseline data. Its `HrfSelectionResult` is reused by b3 and b4 (`fit_selected_hrfs` accepts a selection made on other data with the same features).
- Denoising selection needs every feature at once, because the noise pool spans features. `select_denoising` therefore runs on `analysis_data(session, all cortical indices)`.
- `with_denoising` checks data identity and can't be applied per block. Blocks instead get `denoised_confounds(session, result)`. Test that these confounds equal `with_denoising(full_data, result).confounds`.
- b4 scores fractions on the denoised block with `score_fraction_candidates(data, predictors, fractions=..., library=default_hrf_library(), run_labels=..., encoding_mode="within_run")`, selects with `select_ridge_fractions(scores)`, and fits `fit_selected_hrfs(denoised_block, hrf_selection=b2_selection, ridge_fraction=selection.ridge_fraction)`.
  - Pass the fraction grid in the same order and form that `boldtailor/workflow/beta_series.py:fit_cv_beta_series` passes to the scorer (lines 525–541, including `fraction_grid` and the `reversed` step).
  - Pass the final fraction array exactly as `_final_fit` does.
- Predictors: `trial_predictors(runs, task_model)` reads `run.events` and `run.label`. Pass `[SimpleNamespace(events=e, label=l) for e, l in zip(session.events, session.labels)]`.
- Betas are concatenated in run order with `np.vstack(result.run_betas)`. Each run's betas follow the order of `data.events[i]` sorted by onset. Assert in the tests that `result.trial_table` onsets match `trial_table(...)` onsets row for row.
- Extras saved:
  - b2 and b2-lib20: `hrf_indices`;
  - b3: `denoise_n_components`, `denoise_pcstop_count`, and the gate decision as `denoise_gate_rejected` (0/1);
  - b4: `ridge_fraction`.

- [ ] **Step 1: Add the synthetic-session fixture to `conftest.py`**

The fixture is called `synthetic_session`. It has:
- 6 runs × 120 volumes at TR 1.6, with 24 trials per run at 4 s spacing starting at 8 s, and `duration` 3;
- 40 images, each presented 3 times across the session, ordered at random with seed 0;
- 30 features. Features 0–19 respond to every trial with a per-image amplitude (image effect plus trial noise), convolved with a delayed double-gamma: SPM parameters with peak 6.5 s, outside the canonical 6 s;
- shared low-rank noise (2 components) plus AR(1) noise with ρ = 0.3;
- `response_time` uniform on [0.5, 1.5]; `trial_type` alternating 0/1; `73k_id` the image ID;
- confounds: a cosine column and an intercept-free linear trend;
- task model `detect_task_model(events)`;
- `brain` a surface axis of 30 left-cortex vertices; `source = "fmriprep"`.

Build it with numpy, and convolve with `boldtailor.hrf_library` kernels. The full fixture code goes in `conftest.py`, roughly 60 lines, written by the implementer to the parameters above. All randomness is seeded.

- [ ] **Step 2: Write the failing tests `test_ladder.py`**

```python
import numpy as np

from boldtailor.denoising import with_denoising
from boldtailor.hrf_library import default_hrf_library

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.ladder import (
    LEVELS, denoised_confounds, fit_ladder, select_session_denoising,
)
from experiments.nsd_replication.trials import trial_table


def test_ladder_levels_shapes_and_order(synthetic_session):
    fits = fit_ladder(synthetic_session, block_size=16)
    trials = trial_table(synthetic_session.events, synthetic_session.labels, "s", "73k_id")
    assert set(fits) == set(LEVELS)
    for level, fit in fits.items():
        assert fit.betas.shape == (len(trials), 30), level
        assert fit.record["level"] == level
    assert fits["b2"].extras["hrf_indices"].shape == (30,)
    assert fits["b4"].extras["ridge_fraction"].shape == (30,)


def test_trial_order_matches_trial_table(synthetic_session):
    fits = fit_ladder(synthetic_session, levels=("b1",), block_size=16)
    trials = trial_table(synthetic_session.events, synthetic_session.labels, "s", "73k_id")
    np.testing.assert_allclose(fits["b1"].record["onsets"], trials["onset"])


def test_blocks_do_not_change_results(synthetic_session):
    a = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=7)
    b = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=30)
    for level in ("b1", "b2"):
        np.testing.assert_allclose(a[level].betas, b[level].betas, rtol=1e-6, atol=1e-8)


def test_denoised_confounds_match_with_denoising(synthetic_session):
    library = default_hrf_library(32, seed=0)
    result = select_session_denoising(synthetic_session, synthetic_session.task_model, library)
    full = analysis_data(synthetic_session, np.arange(30))
    expected = with_denoising(full, result).confounds
    for got, want in zip(denoised_confounds(synthetic_session, result), expected, strict=True):
        assert list(got.columns) == list(want.columns)
        np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())


def test_fitted_hrf_improves_reliability_on_delayed_hrf(synthetic_session):
    from experiments.nsd_replication.metrics import voxel_reliability
    from experiments.nsd_replication.betas import zscore
    from experiments.nsd_replication.trials import images_with, repetition_array

    fits = fit_ladder(synthetic_session, levels=("b1", "b2"), block_size=30)
    trials = trial_table(synthetic_session.events, synthetic_session.labels, "s", "73k_id")
    images = images_with(trials, 3)
    rel = {
        level: np.nanmedian(
            voxel_reliability(repetition_array(zscore(fits[level].betas), trials, images))[:20]
        )
        for level in ("b1", "b2")
    }
    assert rel["b2"] > rel["b1"]
```

The last test imports `metrics.voxel_reliability` from Task 8, so it stays RED until Task 8 lands. Mark it `@pytest.mark.xfail(strict=True, reason="needs Task 8 metrics")` here, and remove the mark in Task 8.

`fit_ladder`'s record includes `"onsets"`, the onsets in output row order, so the order test can compare them.

- [ ] **Step 3: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_ladder.py -v   # FAIL: ModuleNotFoundError
git add experiments/nsd_replication/conftest.py experiments/nsd_replication/test_ladder.py
git commit -m "test: boldtailor b1-b4 ladder on a synthetic session (RED)"
```

- [ ] **Step 4: Implement `ladder.py`**

Short functions, one per level:

```python
"""Boldtailor's b1-b4 single-trial ladder for one cortical session."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from boldtailor.denoising import select_denoising
from boldtailor.fractional_ridge import score_fraction_candidates, select_ridge_fractions
from boldtailor.hrf_library import default_hrf_library, glmsingle_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.parallel import map_blocks
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor.workflow.beta_series import trial_predictors

from experiments.nsd_replication.inputs import analysis_data

LEVELS = ("b1", "b2", "b2-lib20", "b3", "b4")
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class LadderFit:
    betas: np.ndarray
    extras: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def _blocks(n_features, size):
    return [np.arange(s, min(s + size, n_features)) for s in range(0, n_features, size)]


def _stack(result):
    return np.vstack(result.run_betas)


def _onsets(result):
    return result.trial_table["onset"].to_numpy(dtype=float)


def select_session_denoising(session, task_model, library, gate=True):
    data = analysis_data(session, np.arange(session.signals[0].shape[1]))
    return select_denoising(
        data, task_model=task_model, library=library, significance_gate=gate,
        run_labels=list(session.labels),
    )


def denoised_confounds(session, result):
    names = list(result.component_names)
    return tuple(
        pd.concat([frame, pd.DataFrame(c, columns=names, index=frame.index)], axis=1)
        for frame, c in zip(session.confounds, result.run_components, strict=True)
    )


def _predictors(session, task_model):
    runs = [SimpleNamespace(events=e, label=l) for e, l in zip(session.events, session.labels)]
    return trial_predictors(runs, task_model)
```

Then write these block functions, each under 25 lines, so `map_blocks` can pickle them:
- `_b1_block(indices, session)` → `fit_single_trials(analysis_data(...), run_labels=...)`.
- `_selection_block(indices, session, library, task_model)` → `select_hrfs`.
- `_selected_block(indices, session, selection, confounds, fractions)` → `fit_selected_hrfs`.
- `_b4_block(indices, session, selection, confounds, task_model)` → score, select, fit.

`fit_ladder` assembles the levels as follows:
- every block result is placed by its feature indices into a full `(trials, features)` array;
- the record holds `level`, `n_features`, `block_size`, the library fingerprint, the denoising counts and `onsets`;
- denoising runs once (for b3 and b4) with `default_hrf_library()` and `session.task_model`, unless `denoising` is passed in, which lets feature experiments reuse it;
- the order of computation is b1; b2 and b2-lib20 selections; b3 denoising; b4 fractions.

The implementer writes the block functions to these signatures. The engineer must not change any core boldtailor file.

- [ ] **Step 5: Run, format, commit GREEN**

```bash
uv run pytest experiments/nsd_replication/test_ladder.py -W error -v
# Expected: 4 PASS, 1 XFAIL (needs Task 8)
uv run black experiments/
git add experiments/nsd_replication/ladder.py
git commit -m "feat: boldtailor b1-b4 ladder for cortical NSD sessions"
```

---

### Task 7: Least-squares-separate betas (R3)

**Files:**
- Create: `experiments/nsd_replication/lss.py`, `test_lss.py`

**Interfaces:**
- Consumes:
  - `SingleTrialResult.design`: `SharedTrialDesign.matrices` holds per-run DataFrames with the trial columns first, then the nuisance columns. `SelectedTrialDesign.matrix(run, hrf_id)` holds the same per HRF.
  - `fit_single_trials` and `fit_selected_hrfs` results from Task 6.
- Produces:
  - `lss_run(x: np.ndarray, nuisance: np.ndarray, y: np.ndarray) -> np.ndarray` (trials × features). For trial i, the design is `[x[:, i], x.sum(1) - x[:, i], nuisance]` and the beta is the first coefficient (OLS).
  - `lss_canonical(result, signals) -> np.ndarray`
  - `lss_selected(result, signals) -> np.ndarray`. Groups features by `result.design.hrf_indices` and calls `lss_run` with that HRF's matrix.
  - `n_trials(result, run) -> int`: the number of trial columns, from `result.trial_table` rows for that run.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from nilearn.glm.first_level import run_glm

from experiments.nsd_replication.lss import lss_run


def test_lss_run_matches_per_trial_ols():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 5))
    nuisance = np.column_stack([np.ones(80), np.linspace(-1, 1, 80)])
    y = rng.normal(size=(80, 3))
    got = lss_run(x, nuisance, y)
    for i in range(5):
        design = np.column_stack([x[:, i], x.sum(1) - x[:, i], nuisance])
        _, results = run_glm(y, design, noise_model="ols")
        want = next(iter(results.values())).theta[0]
        np.testing.assert_allclose(got[i], want, rtol=1e-8)


def test_lss_single_trial_equals_joint_ols():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 1))
    nuisance = np.ones((40, 1))
    y = rng.normal(size=(40, 2))
    joint = np.linalg.lstsq(np.column_stack([x, nuisance]), y, rcond=None)[0][0]
    np.testing.assert_allclose(lss_run(x, nuisance, y)[0], joint, rtol=1e-10)
```

Add `test_lss_canonical_shape(synthetic_session)`. It fits `fit_single_trials` on the fixture and asserts that `lss_canonical` returns `(n_trials, 30)` with finite values for features 0–19.

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_lss.py -v   # FAIL
git add experiments/nsd_replication/test_lss.py
git commit -m "test: least-squares-separate betas (RED)"
```

- [ ] **Step 3: Implement `lss.py`**

```python
"""Least-squares-separate single-trial betas from boldtailor trial designs."""

import numpy as np


def lss_run(x, nuisance, y):
    total = x.sum(axis=1)
    betas = np.empty((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        design = np.column_stack([x[:, i], total - x[:, i], nuisance])
        betas[i] = np.linalg.lstsq(design, y, rcond=None)[0][0]
    return betas


def n_trials(result, run):
    return int(np.sum(result.trial_table["run_index"] == run))


def lss_canonical(result, signals):
    out = []
    for run, (matrix, y) in enumerate(zip(result.design.matrices, signals, strict=True)):
        k = n_trials(result, run)
        values = matrix.to_numpy()
        out.append(lss_run(values[:, :k], values[:, k:], np.asarray(y, float)))
    return np.vstack(out)


def lss_selected(result, signals):
    ids = result.design.hrf_indices
    out = []
    for run, y in enumerate(signals):
        k = n_trials(result, run)
        betas = np.full((k, len(ids)), np.nan)
        for hrf in np.unique(ids[ids >= 0]):
            cols = np.flatnonzero(ids == hrf)
            matrix = result.design.matrix(run, int(hrf))
            betas[:, cols] = lss_run(matrix[:, :k], matrix[:, k:], np.asarray(y, float)[:, cols])
        out.append(betas)
    return np.vstack(out)
```

If the trial table's run column isn't named `run_index`, rule on it: read the actual column name from `SingleTrialResult.trial_table` and use it. That is the one name to verify against the result object.

- [ ] **Step 4: Wire LSS into the ladder and commit GREEN**

Add the levels `"lss-assume"` and `"lss-fit"` to `ladder.py` as an opt-in list, `LSS_LEVELS = ("lss-assume", "lss-fit")`. `fit_ladder(levels=...)` accepts them:
- `lss-assume` uses b1's `fit_single_trials` result per block;
- `lss-fit` uses b2's `fit_selected_hrfs` OLS result per block.

Extend `test_ladder_levels_shapes_and_order` to also request `LSS_LEVELS` and check their shapes.

```bash
uv run pytest experiments/nsd_replication -W error -v
uv run black experiments/
git add experiments/nsd_replication/lss.py experiments/nsd_replication/ladder.py experiments/nsd_replication/test_ladder.py
git commit -m "feat: least-squares-separate levels for R3"
```

---

### Task 8: Metrics R1, R3B, R4, R5, R6 and R2

**Files:**
- Create: `experiments/nsd_replication/metrics.py`, `hrf_maps.py`, `test_metrics.py`, `test_hrf_maps.py`
- Modify: `experiments/nsd_replication/test_ladder.py` (remove the xfail mark)

**Interfaces:**
- Consumes: `repetition_array`, `images_with`, `out_of_sample_images` (Task 2); `zscore` (Task 5).
- Produces:
  - `THRESHOLDS = np.round(np.arange(-0.2, 0.6001, 0.05), 2)`
  - `columnwise_corr(a, b) -> np.ndarray`: per-column Pearson r over rows; NaN when either column is constant or non-finite.
  - `voxel_reliability(reps: np.ndarray) -> np.ndarray`: reps is `(3, images, F)`; returns F values.
  - `threshold_curves(reliabilities: dict[str, np.ndarray], roi: np.ndarray, thresholds=THRESHOLDS) -> pd.DataFrame`, with columns `version, threshold, n_features, mean_difference`.
  - `lagged_correlation(betas, trials, mask, max_lag=15) -> pd.DataFrame`, with columns `lag, mean_r, n_pairs`. Same-run pairs only.
  - `rdm(patterns: np.ndarray) -> np.ndarray`: 1 − Pearson r between rows.
  - `rdm_agreement(rdms: dict[str, np.ndarray]) -> pd.DataFrame`, with columns `subject_a, subject_b, r`.
  - `decoding_accuracy(reps: np.ndarray, mask: np.ndarray) -> dict`, with keys `accuracy`, `chance`, `n_classes`, `n_features`.
  - `hrf_maps.session_consistency(library, hrf_indices: np.ndarray, sessions) -> pd.DataFrame`. Wraps `compare_hrfs`, with one row per feature giving the mean pairwise and mean baseline.

- [ ] **Step 1: Write the failing tests `test_metrics.py`**

```python
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.metrics import (
    columnwise_corr, decoding_accuracy, lagged_correlation, rdm, rdm_agreement,
    threshold_curves, voxel_reliability,
)


def test_reliability_hand_computed():
    reps = np.zeros((3, 4, 1))
    reps[:, :, 0] = [[1, 2, 3, 4], [1, 2, 3, 5], [2, 2, 3, 4]]
    expected = np.mean(
        [
            np.corrcoef(reps[k, :, 0], np.delete(reps, k, 0).mean(0)[:, 0])[0, 1]
            for k in range(3)
        ]
    )
    np.testing.assert_allclose(voxel_reliability(reps), [expected])


def test_reliability_constant_feature_is_nan():
    reps = np.ones((3, 4, 1))
    with np.errstate(all="raise"):
        assert np.isnan(voxel_reliability(reps)).all()


def test_threshold_curves_difference_from_composite():
    rel = {"b1": np.array([0.1, 0.3, 0.5]), "b4": np.array([0.3, 0.5, 0.7])}
    # Composite is [0.2, 0.4, 0.6]; thresholds sit between composite values.
    curves = threshold_curves(rel, np.array([True, True, False]), thresholds=np.array([0.15, 0.35]))
    row = curves.query("version == 'b4' and threshold == 0.15").iloc[0]
    assert row.n_features == 2 and row.mean_difference == pytest.approx(0.1)
    row = curves.query("version == 'b1' and threshold == 0.35").iloc[0]
    assert row.n_features == 1 and row.mean_difference == pytest.approx(-0.1)


def test_lagged_correlation_same_run_only():
    trials = pd.DataFrame(
        {"session": "s", "run": ["r1", "r1", "r1", "r2"], "trial": [0, 1, 2, 0],
         "onset": [0, 4, 8, 0], "image": [1, 2, 3, 4]}
    )
    betas = np.array([[1.0, 0, 0], [1, 0.1, 0], [0, 0, 1], [1, 0, 0]])
    table = lagged_correlation(betas, trials, np.ones(3, bool), max_lag=2)
    lag1 = table.query("lag == 1").iloc[0]
    assert lag1.n_pairs == 2
    expected = np.mean([np.corrcoef(betas[0], betas[1])[0, 1], np.corrcoef(betas[1], betas[2])[0, 1]])
    assert lag1.mean_r == pytest.approx(expected)


def test_rdm_and_agreement():
    patterns = np.array([[1.0, 2, 3], [3, 2, 1], [1, 2, 4]])
    d = rdm(patterns)
    assert d[0, 1] == pytest.approx(2.0) and np.allclose(np.diag(d), 0)
    table = rdm_agreement({"sub-01": d, "sub-02": d})
    assert table.iloc[0].r == pytest.approx(1.0)


def test_decoding_perfect_on_separable_images():
    rng = np.random.default_rng(0)
    prototypes = rng.normal(size=(5, 20)) * 5
    reps = prototypes[None] + rng.normal(scale=0.1, size=(3, 5, 20))
    result = decoding_accuracy(reps, np.ones(20, bool))
    assert result["accuracy"] == 1.0 and result["chance"] == pytest.approx(0.2)
    assert result["n_classes"] == 5


def test_decoding_too_few_features_returns_nan():
    reps = np.zeros((3, 5, 20))
    mask = np.zeros(20, bool)
    mask[:5] = True
    assert np.isnan(decoding_accuracy(reps, mask)["accuracy"])
```

`test_hrf_maps.py` builds a 2-session index array (`[[1, 2, 3], [1, 2, 4]]`) on `glmsingle_hrf_library()`. It asserts that the per-feature mean pairwise r equals 1.0 for features 0 and 1, and is below 1 for feature 2.

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_metrics.py experiments/nsd_replication/test_hrf_maps.py -v   # FAIL
git add experiments/nsd_replication/test_metrics.py experiments/nsd_replication/test_hrf_maps.py
git commit -m "test: R1-R6 metrics and HRF session consistency (RED)"
```

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Prince et al. (2022) NSD metrics on z-scored single-trial betas."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

THRESHOLDS = np.round(np.arange(-0.2, 0.6001, 0.05), 2)
MIN_DECODING_FEATURES = 10


def columnwise_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(a.shape[1], np.nan)
    ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    da, db = a[:, ok] - a[:, ok].mean(0), b[:, ok] - b[:, ok].mean(0)
    na, nb = np.sqrt((da**2).sum(0)), np.sqrt((db**2).sum(0))
    good = (na > 0) & (nb > 0)
    idx = np.flatnonzero(ok)[good]
    out[idx] = (da[:, good] * db[:, good]).sum(0) / (na[good] * nb[good])
    return out


def voxel_reliability(reps):
    scores = [
        columnwise_corr(reps[k], np.delete(reps, k, axis=0).mean(axis=0))
        for k in range(reps.shape[0])
    ]
    return np.mean(scores, axis=0)


def threshold_curves(reliabilities, roi, thresholds=THRESHOLDS):
    stack = np.vstack(list(reliabilities.values()))
    composite = stack.mean(axis=0)
    rows = []
    for t in thresholds:
        mask = roi & np.isfinite(composite) & (composite >= t)
        for version, values in reliabilities.items():
            diff = values[mask] - composite[mask]
            mean = float(np.mean(diff)) if mask.any() else np.nan
            rows.append((version, float(t), int(mask.sum()), mean))
    return pd.DataFrame(rows, columns=["version", "threshold", "n_features", "mean_difference"])


def _pattern_corr(patterns):
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = centered / norms[:, None]
    return unit @ unit.T


def lagged_correlation(betas, trials, mask, max_lag=15):
    sums, counts = np.zeros(max_lag + 1), np.zeros(max_lag + 1, int)
    for _, rows in trials.groupby(["session", "run"], sort=False):
        order = rows.sort_values("onset").index.to_numpy()
        corr = _pattern_corr(betas[np.ix_(order, np.flatnonzero(mask))])
        for lag in range(1, min(max_lag, len(order) - 1) + 1):
            values = np.diagonal(corr, offset=lag)
            values = values[np.isfinite(values)]
            sums[lag] += values.sum()
            counts[lag] += len(values)
    lags = np.arange(1, max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[1:] / counts[1:]
    return pd.DataFrame({"lag": lags, "mean_r": means, "n_pairs": counts[1:]})


def rdm(patterns):
    return 1 - _pattern_corr(np.asarray(patterns, float))


def rdm_agreement(rdms):
    rows = []
    for a, b in combinations(sorted(rdms), 2):
        iu = np.triu_indices_from(rdms[a], k=1)
        rows.append((a, b, float(np.corrcoef(rdms[a][iu], rdms[b][iu])[0, 1])))
    return pd.DataFrame(rows, columns=["subject_a", "subject_b", "r"])


def decoding_accuracy(reps, mask):
    n_reps, n_images, _ = reps.shape
    result = dict(chance=1 / n_images, n_classes=n_images, n_features=int(mask.sum()))
    if mask.sum() < MIN_DECODING_FEATURES:
        return {**result, "accuracy": np.nan}
    data = np.nan_to_num(reps[:, :, mask])
    labels = np.arange(n_images)
    correct = []
    for k in range(n_reps):
        train = np.delete(data, k, axis=0).reshape(-1, data.shape[2])
        model = LinearSVC().fit(train, np.tile(labels, n_reps - 1))
        correct.append(model.predict(data[k]) == labels)
    return {**result, "accuracy": float(np.mean(correct))}
```

`LinearSVC` may emit `ConvergenceWarning`, which `-W error` turns into a failure. Fit inside `warnings.catch_warnings()` filtering `sklearn.exceptions.ConvergenceWarning`, and record the count of non-converged folds in the result dict as `n_unconverged`.

- [ ] **Step 4: Implement `hrf_maps.py`**

```python
"""R2: HRF choice maps and their session-to-session consistency."""

import pandas as pd

from boldtailor.reliability import compare_hrfs


def session_consistency(library, hrf_indices, sessions):
    result = compare_hrfs(library, hrf_indices, sessions)
    pairwise, baseline = result["summary"][0], result["summary"][1]
    return pd.DataFrame({"mean_pairwise_r": pairwise, "mean_canonical_baseline": baseline})
```

- [ ] **Step 5: Remove the Task 6 xfail; run; commit GREEN**

```bash
uv run pytest experiments/nsd_replication -W error -v   # all PASS, including the b2 > b1 test
uv run black experiments/
git add experiments/nsd_replication
git commit -m "feat: Prince et al. NSD metrics R1-R6 and HRF consistency"
```

---

### Task 9: CLI, resumable fits, subject-level metrics

**Files:**
- Create: `experiments/nsd_replication/run.py`, `test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `main(argv: list[str] | None = None) -> int`. Command: `uv run python -m experiments.nsd_replication.run {fit,metrics} --config PATH [--source fmriprep|released] [--levels ...] [--refit] [--discard-betas-after-metrics]`.
  - `fit_session(config, source, subject, session, levels, refit=False) -> list[Path]`. Skips complete fits whose metadata `inputs_digest` matches. A complete fit with a different digest raises `ValueError` naming the fit, unless `refit`.
  - `subject_metrics(config, source, subject, levels) -> dict[str, pd.DataFrame]`. Loads every session's fits, z-scores each session, concatenates, and computes R1 curves, R3B curves, R4 tables at composite ≥ 0 and 0.3, and R6 decoding at the protocol thresholds. Writes TSVs to `output_dir/metrics/<source>/<subject>/`.
  - `group_rsa(config, source, subjects, level) -> pd.DataFrame`: R5 across subjects. Writes `output_dir/metrics/<source>/rsa_<level>.tsv`.

- [ ] **Step 1: Write the failing tests `test_run.py`**

These use a tmp output dir and monkeypatch the loader to return the `synthetic_session` fixture:

```python
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication import run
from experiments.nsd_replication.betas import fit_dir, is_complete


@pytest.fixture
def config(tmp_path):
    from experiments.nsd_replication.config import ExperimentConfig
    return ExperimentConfig(
        bids_dir=tmp_path, output_dir=tmp_path / "out", freesurfer_dir=tmp_path,
        subjects=("sub-07",), sessions=("ses-a", "ses-b"), block_size=16,
    )


@pytest.fixture
def patched(monkeypatch, synthetic_session):
    monkeypatch.setattr(run, "_load", lambda config, source, subject, session: synthetic_session)
    monkeypatch.setattr(run, "load_roi", lambda config, subject, brain: np.arange(30) < 20)


def test_fit_session_writes_and_skips(config, patched):
    paths = run.fit_session(config, "fmriprep", "sub-07", "ses-a", ("b1",))
    assert all(is_complete(p) for p in paths)
    mtime = (paths[0] / "betas.npy").stat().st_mtime_ns
    run.fit_session(config, "fmriprep", "sub-07", "ses-a", ("b1",))
    assert (paths[0] / "betas.npy").stat().st_mtime_ns == mtime


def test_digest_mismatch_raises_unless_refit(config, patched):
    (path,) = run.fit_session(config, "fmriprep", "sub-07", "ses-a", ("b1",))
    meta = path / "metadata.json"
    meta.write_text(meta.read_text().replace('"inputs_digest": "', '"inputs_digest": "x'))
    with pytest.raises(ValueError, match="inputs digest"):
        run.fit_session(config, "fmriprep", "sub-07", "ses-a", ("b1",))
    run.fit_session(config, "fmriprep", "sub-07", "ses-a", ("b1",), refit=True)


def test_subject_metrics_end_to_end(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "fmriprep", "sub-07", ses, ("b1", "b2", "b4"))
    tables = run.subject_metrics(config, "fmriprep", "sub-07", ("b1", "b2", "b4"))
    r1 = tables["r1"]
    assert set(r1["version"]) == {"b1", "b2", "b4"}
    assert (config.output_dir / "metrics/fmriprep/sub-07/r1.tsv").is_file()
    assert {"r1", "r1_median", "r3b", "r4_t0.0", "r4_t0.3", "r6"} <= set(tables)
```

Using the same synthetic session for both sessions makes each image's presentation count 6. That is fine, because `repetition_array` uses the first three.

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_run.py -v   # FAIL
git add experiments/nsd_replication/test_run.py
git commit -m "test: resumable fit stage and subject metrics (RED)"
```

- [ ] **Step 3: Implement `run.py`**

Functions, each short:
- `_load(config, source, subject, session)` calls `load_fmriprep` (the only fitted source). With `--source released`, the fit stage calls `released.index_released` instead (Task 11).
- `_metadata(level, session, fit)` combines `fit.record` with `inputs_digest`, `boldtailor.__version__` (or `importlib.metadata.version("boldtailor")`), the `git rev-parse HEAD` output and the source.
- `fit_session(...)`:
  1. compute the paths;
  2. check completeness and digests;
  3. load the session once, only if some level is missing;
  4. call `fit_ladder(session, levels=missing, block_size=config.block_size, n_jobs=config.n_jobs)`;
  5. call `write_fit` per level, with `trial_table(session.events, session.labels, session, config.image_column)`.
- `_subject_betas(config, source, subject, level)` reads each session's fit, z-scores it, and returns stacked betas plus the concatenated trials.
- `subject_metrics(...)` builds repetition arrays for `images_with(trials, 3)` and `out_of_sample_images(trials)`, then computes:
  - reliabilities;
  - `threshold_curves`;
  - medians within the ROI (`r1_median`: version, median);
  - `lagged_correlation` per session averaged over sessions, at composite ≥ 0 and ≥ 0.3;
  - `decoding_accuracy` at composite ≥ 0, 0.1, 0.2, 0.3, 0.4 within the ROI.

  The ROI comes from `load_roi(config, subject, brain)`, where `brain` is taken from a freshly loaded first session. To avoid loading BOLD, it is saved as `brain.npy` (vertex and structure arrays) next to the fits during `fit_session`.
- `group_rsa(...)` takes the shared images seen at least three times by every subject, repetition-averages them within the ROI using composite ≥ 0, builds an `rdm` per subject, and computes `rdm_agreement`.
- `--discard-betas-after-metrics` deletes `betas.npy` for the subject's fits only after every metric TSV for that subject exists.
- `main` parses the arguments with `argparse` and loops over subjects and sessions from the config.

- [ ] **Step 4: Run, format, commit GREEN**

```bash
uv run pytest experiments/nsd_replication -W error -v   # all PASS
uv run black experiments/
git add experiments/nsd_replication/run.py
git commit -m "feat: resumable NSD replication fit and metrics stages"
```

---

### Task 10: Statistics

**Files:**
- Create: `experiments/nsd_replication/stats.py`, `test_stats.py`

**Interfaces:**
- Produces:
  - `relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray`, which is `(a − b) / b` per subject.
  - `tost_paired(diff: np.ndarray, margin: float, alpha=0.05) -> dict`. Keys `mean`, `ci90`, `p_lower`, `p_upper`, `p`, `equivalent`. Two one-sided t-tests (`scipy.stats.ttest_1samp`, `alternative`); `p = max(p_lower, p_upper)`.
  - `criterion_met(values_by_subject: dict[str, tuple[float, float]], minimum: int) -> dict`. Each value is a (b4, b1) pair, and a subject meets the criterion when b4 > b1. Returns the per-subject booleans and `replicated = count >= minimum`.
  - `mean_ci(values, level=0.95) -> tuple[float, float, float]`: t-based.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from experiments.nsd_replication.stats import criterion_met, mean_ci, relative_difference, tost_paired


def test_tost_equivalent_when_tight():
    diff = np.array([0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.0])
    result = tost_paired(diff, margin=0.05)
    assert result["equivalent"] and result["p"] < 0.05


def test_tost_not_equivalent_when_shifted():
    diff = np.full(8, 0.1) + np.linspace(-0.01, 0.01, 8)
    assert not tost_paired(diff, margin=0.05)["equivalent"]


def test_tost_rejects_bad_margin():
    with pytest.raises(ValueError, match="margin"):
        tost_paired(np.zeros(4), margin=0)


def test_relative_difference():
    np.testing.assert_allclose(relative_difference(np.array([1.1, 0.9]), np.array([1.0, 1.0])), [0.1, -0.1])


def test_criterion_three_of_four():
    values = {"sub-01": (0.3, 0.2), "sub-02": (0.3, 0.2), "sub-03": (0.3, 0.2), "sub-04": (0.1, 0.2)}
    result = criterion_met(values, minimum=3)
    assert result["replicated"] and result["per_subject"]["sub-04"] is False


def test_mean_ci_contains_mean():
    mean, low, high = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert low < mean == 2.0 < high
```

- [ ] **Step 2: RED commit; Step 3: implement; Step 4: GREEN commit**

```python
"""Parity (TOST), replication criteria, and descriptive intervals."""

import numpy as np
from scipy import stats


def relative_difference(a, b):
    return (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def tost_paired(diff, margin, alpha=0.05):
    if not margin > 0:
        raise ValueError("margin must be positive")
    diff = np.asarray(diff, float)
    lower = stats.ttest_1samp(diff, -margin, alternative="greater").pvalue
    upper = stats.ttest_1samp(diff, margin, alternative="less").pvalue
    mean, sem = diff.mean(), stats.sem(diff)
    t = stats.t.ppf(1 - alpha, len(diff) - 1)
    p = max(lower, upper)
    return dict(mean=mean, ci90=(mean - t * sem, mean + t * sem),
                p_lower=lower, p_upper=upper, p=p, equivalent=bool(p < alpha))


def criterion_met(values_by_subject, minimum):
    per = {s: bool(b4 > b1) for s, (b4, b1) in values_by_subject.items()}
    return dict(per_subject=per, count=sum(per.values()),
                replicated=sum(per.values()) >= minimum)


def mean_ci(values, level=0.95):
    values = np.asarray(values, float)
    mean = values.mean()
    half = stats.t.ppf((1 + level) / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half
```

```bash
git add experiments/nsd_replication/test_stats.py && git commit -m "test: TOST and replication criteria (RED)"
uv run pytest experiments/nsd_replication/test_stats.py -W error -v
uv run black experiments/ && git add experiments/nsd_replication/stats.py && git commit -m "feat: TOST parity and replication criteria"
```

---

### Task 11: Released betas, alignment check, comparison metrics

**Files:**
- Create: `experiments/nsd_replication/released.py`, `test_released.py`
- Modify: `experiments/nsd_replication/run.py` (`fit --source released`; comparison metrics), `test_run.py`

**Interfaces:**
- Consumes:
  - `cortical_indices` and `fmriprep_brain` (Task 3);
  - `trial_table` (Task 2);
  - `fit_dir`, `write_fit` and `read_fit` (Task 5);
  - `columnwise_corr`, `voxel_reliability` and `threshold_curves` (Task 8).
- Produces:
  - `RELEASED = {"b1": "assumehrf", "b2": "fithrf", "b4": "fithrfGLMdenoiseRR"}`
  - `released_path(config, subject, session, level) -> Path`
  - `load_released(path, brain, n_trials, name) -> np.ndarray` (float32, trials × cortical features). Errors name the subject, session and version through `name`.
  - `bids_trials(config, subject, session) -> pd.DataFrame`. Built with `trial_table` from the raw BIDS events, in BIDS run order.
  - `index_released(config, subject, session) -> list[Path]`. Writes `fits/released/<subject>/<session>/<level>/` for each level in `RELEASED`. Metadata holds `level`, `version`, `file`, `file_size` and `file_mtime_ns`. A complete entry whose size and mtime match is skipped.
  - `alignment_check(ours_b1: np.ndarray, released_b1: np.ndarray, run_length: int, roi: np.ndarray) -> dict`. Keys `true_median` and `null_median`. The first is the median over ROI grayordinates of `columnwise_corr(ours, released)`. The second uses `np.roll(released, run_length, axis=0)`.
  - In `run.py`:
    - `comparison_metrics(config, subject) -> dict[str, pd.DataFrame]`. Takes boldtailor b1, b2 and b4 plus released b1, b2 and b4, all named `<source>:<level>`. It computes R1 curves over that combined set (composite = mean of all six) and ROI medians, and writes `metrics/comparison/<subject>/r1.tsv`, `r1_median.tsv` and `r1_difference_<level>.npy` (per-grayordinate boldtailor − released reliability for b1, b2 and b4).
    - Every session's `trials.tsv` for released and fMRIPrep fits must be identical. Otherwise raise `ValueError` naming the session.

- [ ] **Step 1: Write the failing tests `test_released.py`**

```python
import nibabel as nib
import numpy as np
import pytest

from experiments.nsd_replication.released import alignment_check, load_released


def _brain():
    left = nib.cifti2.BrainModelAxis.from_surface(np.array([0, 1]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT")
    right = nib.cifti2.BrainModelAxis.from_surface(np.array([0]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    mask = np.zeros((2, 1, 1), bool)
    mask[0, 0, 0] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, "CIFTI_STRUCTURE_THALAMUS_LEFT")
    return left + right + thal


def _write(path, values, names=None):
    names = names or [f"trial-{i:03d}" for i in range(1, len(values) + 1)]
    image = nib.Cifti2Image(values, header=(nib.cifti2.ScalarAxis(names), _brain()))
    nib.save(image, path)
    return path


def test_load_released_keeps_cortex_in_order(tmp_path):
    values = np.array([[1, 2, 3, np.nan], [4, 5, 6, np.nan]], dtype=np.float32)
    path = _write(tmp_path / "b.dscalar.nii", values)
    cortex = _brain()[[0, 1, 2]]
    got = load_released(path, cortex, 2, "sub-07 ses-nsd10 assumehrf")
    np.testing.assert_array_equal(got, values[:, :3])


def test_released_trial_count_mismatch_names_session(tmp_path):
    path = _write(tmp_path / "b.dscalar.nii", np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="sub-07 ses-nsd10 assumehrf.*3 trials.*750"):
        load_released(path, _brain()[[0, 1, 2]], 750, "sub-07 ses-nsd10 assumehrf")


def test_released_axis_mismatch_raises(tmp_path):
    path = _write(tmp_path / "b.dscalar.nii", np.zeros((2, 4), np.float32))
    with pytest.raises(ValueError, match="grayordinates"):
        load_released(path, _brain()[[0, 1]], 2, "x")


def test_released_map_order_checked(tmp_path):
    path = _write(tmp_path / "b.dscalar.nii", np.zeros((2, 4), np.float32), ["trial-002", "trial-001"])
    with pytest.raises(ValueError, match="order"):
        load_released(path, _brain()[[0, 1, 2]], 2, "x")


def test_missing_released_file_names_version(tmp_path):
    with pytest.raises(FileNotFoundError, match="fithrf"):
        load_released(tmp_path / "missing.dscalar.nii", _brain(), 2, "sub-07 ses-nsd10 fithrf")


def test_alignment_check_detects_shift():
    rng = np.random.default_rng(0)
    released = rng.normal(size=(120, 5))
    ours = released + rng.normal(scale=0.5, size=released.shape)
    result = alignment_check(ours, released, 60, np.ones(5, bool))
    assert result["true_median"] > 0.8 and abs(result["null_median"]) < 0.3
```

Add a data-gated test, `test_pilot_released_matches_fmriprep_axis`, skipped unless the volume is mounted. It calls `load_released` on `released_path(config, "sub-07", "ses-nsd10", "b4")` with `fmriprep_brain(...)` and `len(bids_trials(...))`, and asserts the shape `(750, 59412)`.

In `test_run.py`, add `test_comparison_metrics_combined_versions`. It writes synthetic released fits with `write_fit` under `source="released"`, using the same `trials.tsv` as the fMRIPrep fits. It asserts that `r1.tsv` has six versions and that `r1_difference_b4.npy` has length 30. A second test, `test_trial_tables_must_match`, perturbs one released `trials.tsv` and expects the `ValueError`.

- [ ] **Step 2: Run, verify failure, commit RED**

```bash
uv run pytest experiments/nsd_replication/test_released.py experiments/nsd_replication/test_run.py -v   # FAIL
git add experiments/nsd_replication/test_released.py experiments/nsd_replication/test_run.py
git commit -m "test: released GLMsingle betas, alignment check, comparison metrics (RED)"
```

- [ ] **Step 3: Implement `released.py`**

```python
"""NSD's released GLMsingle betas, converted to fsLR 91k CIFTI, in the common format."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from experiments.nsd_replication.betas import fit_dir, is_complete, write_fit
from experiments.nsd_replication.inputs import cortical_indices, fmriprep_brain
from experiments.nsd_replication.metrics import columnwise_corr
from experiments.nsd_replication.trials import trial_table

RELEASED = {"b1": "assumehrf", "b2": "fithrf", "b4": "fithrfGLMdenoiseRR"}
_NAME = (
    "{subject}_{session}_task-{task}_space-fsLR_den-91k_desc-{version}"
    "_stat-effect_statmap.dscalar.nii"
)


def released_path(config, subject, session, level):
    name = _NAME.format(
        subject=subject, session=session, task=config.task, version=RELEASED[level]
    )
    return Path(config.released_dir) / subject / session / "func" / name


def _check_maps(axis, n_trials, name):
    maps = list(axis.name)
    if len(maps) != n_trials:
        raise ValueError(f"{name}: {len(maps)} trials, but the events have {n_trials}")
    if maps != [f"trial-{i:03d}" for i in range(1, n_trials + 1)]:
        raise ValueError(f"{name}: maps are not trial-001... in presentation order")


def load_released(path, brain, n_trials, name):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name}: missing {path}")
    image = nib.load(path)
    axis = image.header.get_axis(1)
    cortex = cortical_indices(axis)
    if axis[cortex] != brain:
        raise ValueError(f"{name}: grayordinates differ from the fMRIPrep cortex")
    _check_maps(image.header.get_axis(0), n_trials, name)
    values = np.asarray(image.dataobj, dtype=np.float32)[:, cortex]
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite cortical betas")
    return values


def bids_trials(config, subject, session):
    func = config.bids_dir / subject / session / "func"
    files = sorted(func.glob(f"{subject}_{session}_task-{config.task}_run-*_events.tsv"))
    if not files:
        raise FileNotFoundError(f"{subject} {session}: no events in {func}")
    events = [pd.read_csv(f, sep="\t") for f in files]
    labels = [f.name.split("_")[3] for f in files]
    return trial_table(events, labels, session, config.image_column)


def _stat(path):
    info = Path(path).stat()
    return {"file": str(path), "file_size": info.st_size, "file_mtime_ns": info.st_mtime_ns}


def _current(target, stat):
    if not is_complete(target):
        return False
    import json

    meta = json.loads((target / "metadata.json").read_text())
    return all(meta.get(k) == v for k, v in stat.items())


def index_released(config, subject, session):
    brain = fmriprep_brain(config, subject, session)
    trials = bids_trials(config, subject, session)
    paths = []
    for level, version in RELEASED.items():
        source = released_path(config, subject, session, level)
        target = fit_dir(config.output_dir, "released", subject, session, level)
        stat = _stat(source) if source.is_file() else {}
        if not stat or not _current(target, stat):
            betas = load_released(source, brain, len(trials), f"{subject} {session} {version}")
            write_fit(target, betas, trials, {"level": level, "version": version, **stat})
        paths.append(target)
    return paths


def alignment_check(ours_b1, released_b1, run_length, roi):
    true = columnwise_corr(ours_b1[:, roi], released_b1[:, roi])
    null = columnwise_corr(ours_b1[:, roi], np.roll(released_b1[:, roi], run_length, axis=0))
    return {"true_median": float(np.nanmedian(true)), "null_median": float(np.nanmedian(null))}
```

Move the `json` import to the top of the module when implementing; it's shown inline above only to keep `_current` self-explanatory.

- [ ] **Step 4: Implement the `run.py` changes**

- `fit --source released` loops over the configured subjects and sessions and calls `index_released`.
- `metrics --source comparison` calls `comparison_metrics(config, subject)` for each subject. It also writes `metrics/comparison/<subject>/alignment.tsv`, with one row per session from `alignment_check` (with `run_length` = the trial count of that session's first run). It raises `ValueError` naming the session when `true_median` is below `config.alignment_floor`. `alignment_floor` (Task 1) defaults to `None`, which skips the check until the pilot sets the floor.

- [ ] **Step 5: Run, format, commit GREEN**

```bash
uv run pytest experiments/nsd_replication -W error -v   # all PASS
uv run black experiments/
git add experiments/nsd_replication
git commit -m "feat: released GLMsingle betas, alignment check, comparison metrics"
```

---

### Task 12: Pilot run, pilot report, freeze (compute run started by the user)

**Files:**
- Create: `docs/validation/nsd-replication-pilot-<date>.md`, `experiments/nsd_replication/pilot_report.py`, `test_pilot_report.py`
- Modify: `experiments/nsd_replication/protocol.md`, `configs/pilot.toml` (`alignment_floor`)

- [ ] **Step 1: User prerequisites**

1. Confirm that `freesurfer_dir/subjNN/label/{lh,rh}.nsdgeneral.mgz` exist for each configured subject. They exist for sub-07.
2. Confirm `wb_command` is on `PATH`. It is installed at `/Applications/wb_view.app/Contents/usr/bin/wb_command`.

- [ ] **Step 2: User runs the pilot** (estimate: several hours per session; the full workflow took 105 min per session)

```bash
C=experiments/nsd_replication/configs/pilot.toml
uv run python -m experiments.nsd_replication.run fit --config $C --source fmriprep --levels b1 b2 b2-lib20 b3 b4 lss-assume lss-fit
uv run python -m experiments.nsd_replication.run fit --config $C --source released
uv run python -m experiments.nsd_replication.run metrics --config $C --source fmriprep
uv run python -m experiments.nsd_replication.run metrics --config $C --source released
uv run python -m experiments.nsd_replication.run metrics --config $C --source comparison
```

- [ ] **Step 3: `pilot_report.py` (TDD)**

`pilot_report.py` holds `variability(config, source, levels) -> pd.DataFrame`. For each version, it computes the median ROI R1 reliability from leave-one-session-out subsets of the 10 sessions. It reports the SD across subsets and the split-to-split SD (odd versus even sessions).

`test_pilot_report.py` writes two synthetic sessions through the Task 9 fixture route, and asserts the column set (`version, loso_sd, split_sd, median`) and finite values. Commit RED, implement, commit GREEN.

- [ ] **Step 4: Write the pilot report; complete the protocol**

`docs/validation/nsd-replication-pilot-<date>.md` contains:
- sub-07 R1–R6 for boldtailor and released versions, with the paper's direction for each;
- the comparison curves and difference maps;
- repeat counts;
- the alignment table;
- the variability table;
- runtime per level, from the fit metadata;
- the preprocessing caveat.

Complete these `protocol.md` sections:
- **δ.** Propose δ = 2 × the larger of the two SDs as a fraction of the median, rounded up to 0.01, with a written justification. Present the proposal to the user for approval before tagging.
- **Replication criteria R1–R6.** Each is directional, per subject, and must hold in ≥ 3 of 4 subjects:
  - R1: median ROI reliability b4 > b1, and the b4 curve at t = 0.2 above b1;
  - R3: b4 above both LSS versions at t = 0.2;
  - R4: b4 lag-1 mean r below b1;
  - R5: mean pairwise RDM r b4 > b1;
  - R6: accuracy b4 > b1 at composite ≥ 0.2.
- **Alignment floor.** The midpoint between the pilot's lowest session `true_median` and its highest `null_median`. If the true medians don't clearly exceed the null medians, stop and report to the user: trials or grayordinates are misaligned. Set `alignment_floor` in every config.

- [ ] **Step 5: Freeze (after user approval of δ)**

```bash
git add docs/validation experiments/nsd_replication
git commit -m "docs: NSD replication pilot report and frozen protocol"
git tag nsd-replication-protocol-v1
```

---

### Task 13: Feature experiments

**Files:**
- Create: `experiments/nsd_replication/features.py`, `test_features.py`
- Modify: `run.py` (stage `features`)

**Interfaces:**
- Produces:
  - `circular_shift(events: pd.DataFrame, run_duration: float, rng, minimum=30.0) -> pd.DataFrame`. Shifts all onsets by one random offset in [minimum, run_duration − minimum], modulo `run_duration`, then re-sorts by onset. Durations and row count are preserved.
  - `gate_experiment(session, seed) -> pd.DataFrame`. Rows: real and null data, each with gate on and off. Columns `condition, gate, n_components, pcstop_count, decision`.
  - `modulator_experiment(session) -> dict[str, LadderFit]`. b4 with `TaskModel()` (task only) versus `session.task_model`.
  - `hrf_choice_experiment(session, released_hrf_indices: np.ndarray | None) -> dict`. OLS `fit_selected_hrfs` with boldtailor's b2-lib20 choice versus GLMsingle's released index, on the same `glmsingle_hrf_library()`. Returns `None` with a `skipped` reason when the released indices are absent.
  - `rt_correlation(betas, events, roi) -> float`. The Pearson r between trial RT and the mean ROI beta, over trials with finite RT.

- [ ] **Step 1: Failing tests:**
  - `test_circular_shift_preserves_counts_and_minimum`: over 100 seeds, every shift is at least 30 s modulo the duration, and row counts and durations are unchanged.
  - `test_gate_rejects_on_null(synthetic_session)`: on the null-shifted synthetic session with gate on, `n_components == 0`. With gate off, the `pcstop_count` is recorded.
  - `test_modulator_experiment_returns_two_fits(synthetic_session)`.
  - `test_hrf_choice_skips_without_released`.
  - `test_rt_correlation_hand_computed`.

Commit RED.

- [ ] **Step 2: Implement with short functions; GREEN commit.**

- [ ] **Step 3: User runs the features**

```bash
uv run python -m experiments.nsd_replication.run features --config experiments/nsd_replication/configs/pilot.toml --source fmriprep
```

---

### Task 14: Figures, primary and extension runs, final report, docs update

**Files:**
- Create: `experiments/nsd_replication/figures.py`, `test_figures.py`, `configs/primary.toml`, `configs/extension.toml`, `docs/validation/nsd-replication-<date>.md`
- Modify: `docs/glmsingle-comparison.md`, `experiments/nsd_replication/README.md`

- [ ] **Step 1: Failing smoke tests for `figures.py`**

Each figure function takes the metric TSVs and returns a `matplotlib.figure.Figure`. The functions are:
- `fig_r1_curves`
- `fig_r1_maps`, which uses `boldtailor.workflow.surfaces` only if the surface meshes are configured, and otherwise a flat per-hemisphere scatter
- `fig_r3_lss`
- `fig_r4_lag`
- `fig_r5_rsa`
- `fig_r6_decoding`
- `fig_r2_hrf`
- `fig_parity`
- `fig_gate`
- `fig_modulators`

Each test checks the axes count and labels on synthetic tables, with the `Agg` backend. Commit RED, implement, commit GREEN.

- [ ] **Step 2: Write the configs**

- `primary.toml`: subjects `sub-01`…`sub-04`, sessions `ses-nsd01`…`ses-nsd10`.
- `extension.toml`: subjects `sub-05`, `sub-06`, `sub-08`, same sessions.

- [ ] **Step 3: User runs the primary and extension fits and metrics** (same commands as Task 12 Step 2, including `fit --source released` and `metrics --source comparison`, with each config). Then:

```bash
uv run python -m experiments.nsd_replication.run metrics --config experiments/nsd_replication/configs/primary.toml --source fmriprep
uv run python -m experiments.nsd_replication.run figures --config experiments/nsd_replication/configs/primary.toml
```

- [ ] **Step 4: Final report**

`docs/validation/nsd-replication-<date>.md` contains:
- the replication table: the paper's pattern, the boldtailor result, and per-subject criteria for R1–R6;
- the parity TOST, boldtailor (fMRIPrep) versus released, at b4 (primary) and at b1, b2 and R4–R6 (secondary), with the preprocessing caveat;
- the feature experiments;
- the deviation log;
- figures.

- [ ] **Step 5: Update `docs/glmsingle-comparison.md`**

Replace "No matched comparison with GLMsingle's denoising has been run" and the equivalent sentences with a summary and a link to the final report. Commit:

```bash
git add docs experiments/nsd_replication
git commit -m "docs: NSD replication results and GLMsingle comparison update"
```

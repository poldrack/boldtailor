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

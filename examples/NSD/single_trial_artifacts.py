"""In-memory CIFTI and metadata artifacts for the NSD single-trial example."""

import json

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.cifti import scalar_artifact  # noqa: F401
from boldtailor.publication import Artifact

STATISTICS = (
    "fullrsquared",
    "confoundsrsquared",
    "deltarsquared",
    "rtcorrelation",
    "rtcount",
)


def model_paths(runs, subject, session, model, *, descriptor=None):
    directory = f"{subject}/{session}/func"
    stem = f"{directory}/{subject}_{session}_task-nsdcore"
    description = f"desc-{descriptor or ('singletrial' + model)}"
    base = f"{stem}_{description}"
    return {
        "betas": [
            f"{directory}/{r.inputs.stem}_space-fsLR_den-91k_{description}_betas.dscalar.nii"
            for r in runs
        ],
        "designs": (
            [f"{directory}/{r.inputs.stem}_{description}_design.tsv" for r in runs]
            if descriptor is None
            else []
        ),
        "trials": f"{base}_trials.tsv",
        "metadata": f"{base}_metadata.json",
        "provenance": f"{base}_provenance.json",
        **{
            s: f"{stem}_space-fsLR_den-91k_{description}_stat-{s}.dscalar.nii"
            for s in STATISTICS
        },
    }


def diagnostic_paths(subject, session):
    stem = (
        f"{subject}/{session}/func/{subject}_{session}_task-nsdcore_desc-singletrialOLS"
    )
    return f"{stem}_selectedvertices.tsv", f"{stem}_scatter.png"


def all_model_paths(paths):
    for value in paths.values():
        yield from value if isinstance(value, list) else [value]


def json_artifact(path, value):
    return Artifact(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def table_artifact(path, frame):
    return Artifact(path, frame.to_csv(sep="\t", index=False, na_rep="n/a").encode())


def single_trial_artifacts(runs, brain, result, rt, paths, metadata):
    artifacts = []
    table = result["trial_table"]
    for index, run in enumerate(runs):
        rows = table[table.run_index == index]
        artifacts.append(
            scalar_artifact(
                paths["betas"][index],
                brain,
                result["betas"][index],
                rows.trial_id.tolist(),
            )
        )
        if paths["designs"]:
            design = result["designs"][index].copy()
            design.insert(0, "frame_time", run.frame_times)
            artifacts.append(table_artifact(paths["designs"][index], design))
    artifacts.append(table_artifact(paths["trials"], table))
    for stat, values in zip(STATISTICS[:3], result["maps"], strict=True):
        artifacts.append(scalar_artifact(paths[stat], brain, values[None], [stat]))
    names = ["all", "odd", "even", *[r.label for r in runs]]
    for statistic, diagnostics in (("rtcorrelation", rt), ("rtcount", rt["counts"])):
        values = np.vstack(
            [diagnostics[name] for name in ("all", "odd", "even")]
            + [diagnostics["per_run"]]
        )
        artifacts.append(scalar_artifact(paths[statistic], brain, values, names))
    artifacts.append(json_artifact(paths["metadata"], metadata))
    artifacts.append(json_artifact(paths["provenance"], result["provenance"]))
    return tuple(artifacts)


def selected_vertex_table(vertices, brain, diagnostics):
    columns = ["grayordinate_index", "structure", "surface_vertex"]
    columns += [
        f"{model}_{split}_r" for model in diagnostics for split in ("odd", "even")
    ]
    rows = []
    for vertex in vertices:
        row = [vertex, brain.name[vertex], brain.vertex[vertex]]
        row += [
            diagnostics[model][split][vertex]
            for model in diagnostics
            for split in ("odd", "even")
        ]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

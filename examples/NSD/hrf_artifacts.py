"""CIFTI maps, exact grouped designs, and labeled HRF/RT diagnostics."""

from io import BytesIO

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from boldtailor._hrf_cv import prepare_runs
from boldtailor.data import from_arrays
from boldtailor.hrf_library import PARAMETER_NAMES
from boldtailor.publication import Artifact

if __package__:
    from .single_trial_artifacts import json_artifact, table_artifact, scalar_artifact
    from .rt_diagnostics import correlate_rt, scatter_artifact
else:
    from single_trial_artifacts import json_artifact, table_artifact, scalar_artifact
    from rt_diagnostics import correlate_rt, scatter_artifact


def selection_paths(runs, subject, session):
    directory = f"{subject}/{session}/func"
    stem = f"{directory}/{subject}_{session}_task-nsdcore_desc-hrfSelection"
    stats = (
        "hrfindex",
        "hrfparameters",
        "selectioncvr2",
        "canonicalcvr2",
        "deltacvr2",
        "oddhrfindex",
        "evenhrfindex",
        "oddhrfparameters",
        "evenhrfparameters",
        "testr2",
        "canonicaltestr2",
        "deltatestr2",
    )
    paths = {
        s: f"{directory}/{subject}_{session}_task-nsdcore_space-fsLR_den-91k_desc-hrfSelection_stat-{s}.dscalar.nii"
        for s in stats
    }
    for name, extension in (
        ("library", "tsv"),
        ("curves", "npz"),
        ("folds", "tsv"),
        ("eligibility", "tsv"),
        ("metadata", "json"),
        ("provenance", "json"),
        ("splitmetadata", "json"),
        ("splitprovenance", "json"),
        ("selectedvertices", "tsv"),
        ("scatter", "png"),
        ("selectedhrfs", "png"),
        ("rt_provenance", "json"),
    ):
        paths[name] = f"{stem}_{name}.{extension}"
    paths["library_plot"] = f"{stem}_library.png"
    paths["designs"] = [
        f"{directory}/{r.inputs.stem}_desc-hrfSelection_designs.npz" for r in runs
    ]
    return paths


def comparison_paths(full_r2_path):
    path = full_r2_path.replace("stat-fullrsquared", "stat-hrfdeltarsquared")
    return {
        "hrfdeltarsquared": path,
        "hrfcomparison_metadata": path.removesuffix(".dscalar.nii") + ".json",
    }


def comparison_artifacts(brain, optimized, canonical, paths, metadata):
    optimized, canonical = np.asarray(optimized), np.asarray(canonical)
    delta = np.full(len(brain), np.nan)
    np.subtract(
        optimized,
        canonical,
        out=delta,
        where=np.isfinite(optimized) & np.isfinite(canonical),
    )
    yield scalar_artifact(
        paths["hrfdeltarsquared"],
        brain,
        delta[None],
        ["optimized_full_r2_minus_canonical_full_r2"],
    )
    yield json_artifact(
        paths["hrfcomparison_metadata"],
        dict(
            metadata,
            Formula="optimized_full_r2 - canonical_full_r2",
            Interpretation="positive: optimized HRF fits better; negative: canonical HRF fits better",
            CanonicalHRF="Nilearn spm; oversampling 50; same trial model, confounds and ridge alpha",
            Undefined="NaN where either full R2 is undefined or canonical design is ineligible",
        ),
    )


def npz_artifact(path, **arrays):
    with BytesIO() as stream:
        np.savez_compressed(stream, **arrays)
        return Artifact(path, stream.getvalue())


def figure_artifact(path, figure):
    try:
        with BytesIO() as stream:
            figure.savefig(stream, format="png", dpi=130)
            return Artifact(path, stream.getvalue())
    finally:
        figure.clear()


def _library_plot(library, path):
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    for c, curve in zip(library.candidates, library.curves, strict=True):
        axis.plot(
            library.times,
            curve,
            color="tab:blue" if c.id else "black",
            alpha=0.06 if c.id else 1,
            lw=0.6 if c.id else 2,
            zorder=1 if c.id else 3,
            label="Canonical SPM" if c.id == 0 else None,
        )
    axis.set(
        xlabel="Time (s)",
        ylabel="Sample weight (0.1 s; sum = 1)",
        title=f"Expanded HRF library: {len(library.candidates)} candidates",
    )
    axis.axhline(0, color="gray", lw=0.5)
    axis.legend()
    return figure_artifact(path, figure)


def _fold_table(runs, train, test):
    rows = []
    for scope, indices in (
        ("all_run_selection", list(range(len(runs)))),
        ("odd_run_selection", train if len(train) >= 2 else []),
        ("even_run_selection", test if len(test) >= 2 else []),
    ):
        for held in indices:
            rows.append(
                dict(
                    scope=scope,
                    train="|".join(runs[i].label for i in indices if i != held),
                    test=runs[held].label,
                )
            )
    if len(train) >= 2 and test:
        rows.append(
            dict(
                scope="independent_evaluation",
                train="|".join(runs[i].label for i in train),
                test="|".join(runs[i].label for i in test),
            )
        )
    return pd.DataFrame(rows)


def _grouped_design_artifacts(runs, library, state, paths):
    # Dummy signals only construct timing/confound cache keys; no BOLD fit.
    data = from_arrays(
        [np.zeros((len(r.frame_times), 1)) for r in runs],
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
    )
    ids = np.unique(
        state["maps"]["hrfindex"][np.isfinite(state["maps"]["hrfindex"])]
    ).astype(int)
    for index, (run, design) in enumerate(
        zip(runs, prepare_runs(data, library), strict=True)
    ):
        arrays = {
            f"hrf_{cid}": np.asarray(design.trial_matrix(int(cid)), dtype=np.float64)
            for cid in ids
        }
        arrays.update(
            nuisance=np.asarray(design.nuisance, dtype=np.float64),
            frame_times=np.asarray(run.frame_times, dtype=np.float64),
            trial_columns=np.array(
                [f"{run.label}_trial-{i+1:04d}" for i in range(len(run.events))]
            ),
            nuisance_columns=np.array([*run.confounds.columns, "constant"]),
            hrf_ids=ids.astype(str),
            library_fingerprint=np.array(library.fingerprint),
        )
        yield npz_artifact(paths[index], **arrays)


def parameter_artifact(brain, library, ids, path):
    """Look up the exact selected library entry, preserving undefined features."""
    names = [*PARAMETER_NAMES, "peak_time"]
    parameters = np.full((len(names), len(brain)), np.nan)
    valid = np.isfinite(ids)
    parameters[:, valid] = (
        library.parameter_table[names].to_numpy()[ids[valid].astype(int)].T
    )
    return scalar_artifact(path, brain, parameters, names)


def split_selection_artifacts(runs, brain, library, state, paths, train, test):
    splits = {}
    for half, indices in (("odd", train), ("even", test)):
        splits[half] = dict(
            RunLabels=[runs[i].label for i in indices],
            Available=len(indices) >= 2,
            Reason=(
                "" if len(indices) >= 2 else "Requires at least two runs in this half"
            ),
        )
        yield parameter_artifact(
            brain,
            library,
            state["maps"][half + "hrfindex"],
            paths[half + "hrfparameters"],
        )
    yield json_artifact(
        paths["splitmetadata"],
        dict(
            LibraryFingerprint=library.fingerprint,
            CandidateCount=len(library.candidates),
            Splits=splits,
            Method="Separate task-model leave-one-run-out CV R² prediction within each half",
            Eligibility="Trial designs must be estimable across all runs; uses timing and confounds, never opposite-half BOLD",
            ParameterMaps=[*PARAMETER_NAMES, "peak_time"],
            PeakTime="Seconds at the full HRF maximum on the saved 0.1-second grid",
            HRFNormalization="discrete sum one; convolution at TR/50",
            Undefined="NaN where the half is unavailable or its signal has no variance outside nuisance span",
            Interpretation="Compare parameters or curves; HRF IDs are categorical library indices, not ordered measurements",
        ),
    )
    yield json_artifact(paths["splitprovenance"], state["split_provenance"])


def selection_artifacts(runs, brain, library, state, paths, metadata, train, test):
    for name, values in state["maps"].items():
        yield scalar_artifact(paths[name], brain, values[None], [name])
    yield parameter_artifact(
        brain, library, state["maps"]["hrfindex"], paths["hrfparameters"]
    )
    yield from split_selection_artifacts(
        runs, brain, library, state, paths, train, test
    )
    yield table_artifact(paths["library"], library.parameter_table)
    yield npz_artifact(
        paths["curves"],
        times=library.times,
        curves=library.curves,
        hrf_ids=np.arange(len(library.candidates)).astype(str),
    )
    yield _library_plot(library, paths["library_plot"])
    yield table_artifact(paths["folds"], _fold_table(runs, train, test))
    yield table_artifact(
        paths["eligibility"], pd.concat(state["eligibility"], ignore_index=True)
    )
    yield json_artifact(paths["metadata"], metadata)
    yield json_artifact(
        paths["provenance"],
        dict(selection=state["provenance"], evaluation=state["evaluation_provenance"]),
    )
    yield from _grouped_design_artifacts(runs, library, state, paths["designs"])


def _vertex_table(brain, state, vertices, diagnostics):
    table = pd.DataFrame(
        dict(
            grayordinate_index=vertices,
            structure=brain.name[vertices],
            surface_vertex=brain.vertex[vertices],
            canonical_odd_r=state["canonical_odd_rt"][vertices],
            odd_hrf_id=state["maps"]["oddhrfindex"][vertices],
        )
    )
    for name, diagnostic in diagnostics.items():
        table[name + "_even_r"] = diagnostic["even"]
    return table


def _selected_curves(library, state, vertices, path):
    figure = Figure(figsize=(9, 5), layout="constrained")
    axis = figure.subplots()
    for vertex in vertices:
        cid = state["maps"]["oddhrfindex"][vertex]
        if np.isfinite(cid):
            axis.plot(
                library.times,
                library.curves[int(cid)],
                label=f"Grayordinate {vertex}, HRF {int(cid)}",
            )
    axis.plot(
        library.times, library.curves[0], color="black", ls="--", label="Canonical SPM"
    )
    axis.set(
        xlabel="Time (s)",
        ylabel="Sample weight (0.1 s; sum = 1)",
        title="Odd-run HRFs at canonical odd-RT-selected vertices",
    )
    axis.legend(fontsize=8)
    return figure_artifact(path, figure)


def rt_artifacts(runs, brain, library, state, vertices, fits, evaluation, paths, test):
    even = [runs[i] for i in test]
    rt = [
        pd.to_numeric(
            r.events.get("response_time", pd.Series(np.nan, index=r.events.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        for r in even
    ]
    numbers = [r.number for r in even]
    diagnostics = {
        name: correlate_rt(fit.run_betas, rt, run_numbers=numbers)
        for name, fit in fits.items()
    }
    yield table_artifact(
        paths["selectedvertices"], _vertex_table(brain, state, vertices, diagnostics)
    )
    if fits:
        yield scatter_artifact(
            {name: fit.run_betas for name, fit in fits.items()},
            rt,
            numbers,
            np.arange(len(vertices)),
            paths["scatter"],
            vertex_labels=vertices,
        )
    else:
        figure = Figure(figsize=(8, 3), layout="constrained")
        axis = figure.subplots()
        axis.text(
            0.5,
            0.5,
            "Independent RT check unavailable: insufficient runs or eligible vertices",
            ha="center",
            va="center",
            wrap=True,
        )
        axis.set_axis_off()
        yield figure_artifact(paths["scatter"], figure)
    yield _selected_curves(library, state, vertices, paths["selectedhrfs"])
    yield json_artifact(
        paths["rt_provenance"],
        dict(
            vertices=vertices.tolist(),
            selection="canonical OLS odd-run RT only",
            hrfs="selected within odd runs",
            evaluation=None if evaluation is None else evaluation.provenance.to_dict(),
            fits={name: fit.provenance.to_dict() for name, fit in fits.items()},
        ),
    )

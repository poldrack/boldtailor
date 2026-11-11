"""Single-file, lossless VTC inputs for simulations without the NSD tree."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

from filelock import FileLock
import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from experiments.hrf_estimation.inputs import RunData, _spec, load_run
from experiments.hrf_estimation.timing import build_nuisance
from experiments.hrf_simulations.io import file_hash

SCHEMA = "boldtailor-simulation-inputs-v1"
DEFAULT_MASK = (
    Path(__file__).parents[1]
    / "hrf_estimation/pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz"
)


def _blob(group, name, value):
    group.create_dataset(
        name,
        data=np.frombuffer(value, dtype=np.uint8),
        compression="gzip",
        shuffle=True,
        fletcher32=True,
    )


def _json(group, name, value):
    _blob(group, name, json.dumps(value, sort_keys=True).encode())


def _read_json(group, name):
    return json.loads(group[name][:].tobytes())


def _check(handle):
    if handle.attrs.get("schema") != SCHEMA or not handle.attrs.get("complete", False):
        raise ValueError("unrecognized or incomplete simulation bundle")


def _brain_record(brain):
    if np.any(brain.vertex < 0):
        raise ValueError("the shared fsLR selection must contain surface vertices only")
    return dict(
        name=brain.name.tolist(),
        vertex=brain.vertex.tolist(),
        nvertices=brain.nvertices,
    )


def _write_run(handle, spec, vertices, root, reference):
    image = nib.load(spec.bold_path)
    brain = image.header.get_axis(1)
    if reference is not None and not np.array_equal(brain, reference):
        raise ValueError(f"incompatible fsLR brainordinate ordering: {spec.bold_path}")
    run = load_run(spec, vertices)
    subject, session = run.run_id.split("_")[:2]
    group = handle.require_group(f"subjects/{subject}/{session}").create_group(
        run.run_id
    )
    group.create_dataset(
        "signals",
        data=run.signals,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )
    group.create_dataset("frame_times", data=run.frame_times, fletcher32=True)
    for name, path in (("events", spec.events_path), ("motion", spec.confounds_path)):
        _blob(group, name, path.read_bytes())
    _json(
        group,
        "provenance",
        {
            "files": [
                {
                    "path": str(p.relative_to(root)),
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in (spec.bold_path, spec.events_path, spec.confounds_path)
            ]
        },
    )
    return brain, run.brain_axis


def _extract(root, temporary, vertices, subjects, sessions, skip_incomplete):
    paths = sorted(
        (root / "derivatives/nsdpreproc").glob(
            "sub-*/ses-nsd*/func/*_task-nsdcore_run-*_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii"
        )
    )
    skipped, count, reference = [], 0, None
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema"] = SCHEMA
        handle.attrs["complete"] = False
        handle.create_dataset("feature_ids", data=vertices)
        for path in paths:
            subject, session = path.name.split("_")[:2]
            if subjects and subject not in subjects:
                continue
            if sessions and int(session.removeprefix("ses-nsd")) not in sessions:
                continue
            try:
                spec = _spec(root, path)
            except FileNotFoundError as error:
                if not skip_incomplete:
                    raise
                skipped.append(dict(run=path.name, reason=str(error)))
                continue
            reference, selected = _write_run(handle, spec, vertices, root, reference)
            if count == 0:
                _json(handle, "brain", _brain_record(selected))
            count += 1
            print(f"EXPORTED {spec.run_id}: {len(vertices)} vertices", flush=True)
        if count == 0:
            raise ValueError("no complete matching layerB2 runs to export")
        _json(
            handle,
            "manifest",
            dict(
                runs=count,
                skipped=skipped,
                source_root=str(root),
                requested_subjects=subjects,
                requested_sessions=sessions,
                selection="same fsLR CIFTI columns for every subject; ordering verified",
            ),
        )
        handle.attrs["complete"] = True
    return count, skipped


def export_bundle(
    root, destination, vertices, subjects=None, sessions=None, skip_incomplete=False
):
    """Stream runs to a temporary file; never publish a partial export."""
    root, destination = Path(root), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            raise FileExistsError(destination)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".partial", dir=destination.parent
        )
        os.close(descriptor)
        try:
            count, skipped = _extract(
                root,
                temporary,
                np.asarray(vertices),
                subjects,
                sessions,
                skip_incomplete,
            )
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(
        f"Wrote {count} runs to {destination}; skipped {len(skipped)} incomplete runs",
        flush=True,
    )


def inventory(path):
    with h5py.File(path, "r") as handle:
        _check(handle)
        return {
            subject: sorted(int(s.removeprefix("ses-nsd")) for s in group)
            for subject, group in handle["subjects"].items()
        }


def bundle_provenance(path):
    with h5py.File(path, "r") as handle:
        _check(handle)
        manifest = _read_json(handle, "manifest")
    return dict(
        bundle=str(path),
        bundle_sha256=file_hash(path),
        schema=SCHEMA,
        manifest=manifest,
    )


def _load_run(group, vertices, brain, cutoff, first_vertex):
    times = group["frame_times"][:]
    events = pd.read_csv(BytesIO(group["events"][:].tobytes()), sep="\t")
    motion = pd.read_csv(BytesIO(group["motion"][:].tobytes()), sep="\t")
    columns = slice(0, 1) if first_vertex else slice(None)
    return RunData(
        run_id=group.name.rsplit("/", 1)[1],
        signals=group["signals"][:, columns],
        events=events,
        confounds=build_nuisance(motion, times, cutoff),
        frame_times=times,
        feature_ids=vertices[columns],
        brain_axis=brain[columns],
    )


def load_session(path, subject, session, cutoff=80.0, first_vertex=False):
    with h5py.File(path, "r") as handle:
        _check(handle)
        key = f"subjects/{subject}/ses-nsd{session:02}"
        if key not in handle:
            raise ValueError(f"bundle missing subject/session: {subject}/{session}")
        vertices = handle["feature_ids"][:]
        brain = nib.cifti2.BrainModelAxis(**_read_json(handle, "brain"))
        return tuple(
            _load_run(group, vertices, brain, cutoff, first_vertex)
            for _, group in sorted(handle[key].items())
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument(
        "--subjects", nargs="+", help="default: all discovered subjects"
    )
    parser.add_argument(
        "--sessions", nargs="+", type=int, help="default: all discovered sessions"
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="explicitly omit runs missing events or motion; omissions are recorded",
    )
    args = parser.parse_args(argv)
    with np.load(args.mask) as mask:
        vertices = mask["feature_ids"]
    export_bundle(
        args.bids_root,
        args.output,
        vertices,
        args.subjects,
        args.sessions,
        args.skip_incomplete,
    )


if __name__ == "__main__":
    main()

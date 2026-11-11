"""Reproducible shared anatomical fsLR regions from sub-07 aparc labels."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from experiments.hrf_estimation.roi import resample_mask, vtc_indices
from experiments.hrf_simulations.io import file_hash, write_json

REGIONS = ("fusiform", "lateraloccipital", "inferiortemporal")
PILOT_INPUTS = Path(__file__).parents[1] / "hrf_estimation/pilot_inputs"
DEFAULT_REGIONS = Path(__file__).with_name("shared_regions.npz")


def map_regions(brain, annotations, source_spheres, target_spheres):
    return {
        name: vtc_indices(
            brain,
            {
                side: resample_mask(
                    source_spheres[side],
                    target_spheres[side],
                    annotations[side] == name,
                )
                for side in ("L", "R")
            },
        )
        for name in REGIONS
    }


def build_regions(freesurfer, cifti, output):
    annotations, source, target, hashes = {}, {}, {}, {}
    for side, prefix in (("L", "lh"), ("R", "rh")):
        annot = Path(freesurfer) / "label" / f"{prefix}.aparc.annot"
        sphere = Path(freesurfer) / "surf" / f"{prefix}.sphere.reg"
        template = (
            PILOT_INPUTS
            / f"tpl-fsLR_space-fsaverage_hemi-{side}_den-32k_sphere.surf.gii"
        )
        labels, _, names = nib.freesurfer.read_annot(annot)
        decoded = np.asarray([n.decode() for n in names])
        annotations[side] = np.where(
            labels >= 0, decoded[np.maximum(labels, 0)], "unknown"
        )
        source[side] = nib.freesurfer.read_geometry(sphere)[0]
        target[side] = nib.load(template).agg_data("pointset")
        hashes[side] = {str(p): file_hash(p) for p in (annot, sphere, template)}
    brain = nib.load(cifti).header.get_axis(1)
    regions = map_regions(brain, annotations, source, target)
    vertices = np.unique(np.concatenate(list(regions.values())))
    np.savez_compressed(output, feature_ids=vertices, **regions)
    write_json(
        Path(output).with_suffix(".json"),
        dict(
            reference_subject="sub-07",
            regions={k: len(v) for k, v in regions.items()},
            union_vertices=len(vertices),
            source_hashes=hashes,
            cifti_reference=str(cifti),
            mapping="nearest native vertex on fsaverage-registered unit spheres",
            selection="sub-07 anatomical aparc labels applied at shared fsLR locations across subjects",
        ),
    )
    return regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freesurfer", type=Path, required=True, help="subj07 FreeSurfer directory"
    )
    parser.add_argument("--cifti", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGIONS)
    args = parser.parse_args(argv)
    regions = build_regions(args.freesurfer, args.cifti, args.output)
    print({name: len(ids) for name, ids in regions.items()})


if __name__ == "__main__":
    main()

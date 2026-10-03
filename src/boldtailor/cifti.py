"""CIFTI-2 dense scalar export, checked reloads, and grayordinate identity.

Grayordinate order always follows the input BrainModelAxis; undefined values
stay NaN at their original positions.
"""

from hashlib import sha256

import nibabel as nib
import numpy as np

from boldtailor.publication import Artifact

CORTEX_STRUCTURES = dict(
    left="CIFTI_STRUCTURE_CORTEX_LEFT", right="CIFTI_STRUCTURE_CORTEX_RIGHT"
)


def scalar_artifact(path, brain, values, names):
    """A float32 ``.dscalar.nii`` artifact with one named map per row."""
    axes = (nib.cifti2.ScalarAxis(names), brain)
    image = nib.Cifti2Image(
        np.asarray(values, dtype=np.float32), nib.Cifti2Header.from_axes(axes)
    )
    image.nifti_header.set_intent("ConnDenseScalar")
    return Artifact(path, image.to_bytes())


def read_scalar(path, brain, names=None):
    """Load saved maps after checking the grayordinate axis and, optionally, names."""
    image = nib.load(path)
    if image.header.get_axis(1) != brain:
        raise ValueError(
            f"Saved CIFTI grayordinate axis differs from current inputs: {path}"
        )
    if names is not None and list(image.header.get_axis(0).name) != list(names):
        raise ValueError(f"Unexpected map names in {path}")
    return image.get_fdata()


def cortical_values(values, brain):
    """Scatter a grayordinate vector onto each cortex, leaving absent vertices NaN."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(brain),):
        raise ValueError("values must contain one value per CIFTI grayordinate")
    result = {}
    for hemi, name in CORTEX_STRUCTURES.items():
        if name not in brain.nvertices:
            continue
        selected = brain.name == name
        cortex = np.full(brain.nvertices[name], np.nan)
        cortex[brain.vertex[selected]] = values[selected]
        cortex[~np.isfinite(cortex)] = np.nan
        result[hemi] = cortex
    return result


def spatial_signature(brain, indices):
    """SHA-256 of the ordered BrainModel axis XML and the selected indices."""
    header = nib.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(["spatial_identity"]), brain)
    )
    return sha256(
        header.to_xml() + np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()

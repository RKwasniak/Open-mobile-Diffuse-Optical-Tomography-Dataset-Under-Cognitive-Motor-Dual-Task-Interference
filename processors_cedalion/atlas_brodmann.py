"""Brodmann-area parcellation and ROIs (FPA).

The cedalion Brodmann atlas (volumetric MNI152, labels such as ``left_BA10``) is
mapped onto the cortical surface with
``TwoSurfaceHeadModel.assign_parcels_via_mni_coords`` (nearest labelled voxel).
The reconstructed image shares the head model's brain-vertex order, so the labels
are copied onto it by position (:func:`relabel_image_parcels`).

Atlas: Mai & Majtanik, *Human Brain in Standard MNI Space: A Comprehensive Pocket
Atlas* (2017), as bundled with cedalion; workflow from cedalion
``examples/head_models/52_mni_atlas_labels_aal3_brodmann.ipynb``.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# ROI definitions (Brodmann-area numbers); edit to match the montage
# ---------------------------------------------------------------------------
#   FPA  frontopolar area: BA10
BRODMANN_ROIS: dict[str, list[int]] = {
    "FPA": [10],
}

_BA_RE = re.compile(r"BA0*([0-9]+)([a-zA-Z]*)$")   # e.g. left_BA9, right_BA44a

# ---------------------------------------------------------------------------
# Atlas loading & assignment
# ---------------------------------------------------------------------------

def load_brodmann_atlas() -> tuple:
    """Return ``(nifti_path, num2label)`` for the bundled Brodmann atlas.

    ``num2label`` maps the integer voxel labels in the NIfTI volume to their
    string names (``{1: 'left_BA1', ...}``).
    """
    import cedalion.data

    nifti, labels_json = cedalion.data.get_atlas_files("brodmann")
    with open(labels_json, "r", encoding="utf-8") as fin:
        raw = json.load(fin)
    num2label = {entry["index"]: entry["name"] for entry in raw["labels"]}
    return nifti, num2label

def assign_brodmann_to_head(
    head_ijk,
    coordinate_label: str = "parcel_brodmann",
    mni_eps: float = 5.0,
):
    """Assign Brodmann labels to the brain vertices of a head model.

    Parameters
    ----------
    head_ijk : cedalion.dot.TwoSurfaceHeadModel
        Head model in voxel (ijk) space, e.g. from ``load_head_model("colin27")``.
    coordinate_label : str
        Name of the new per-vertex coordinate. Default ``"parcel_brodmann"``.
    mni_eps : float
        Search radius (mm) around each vertex in MNI152 space. Increase if some
        vertices remain ``Background``.

    Returns
    -------
    TwoSurfaceHeadModel
        A copy of the head model whose ``brain.vertices`` carry ``coordinate_label``.
    """
    nifti, num2label = load_brodmann_atlas()
    head_labeled = head_ijk.assign_parcels_via_mni_coords(
        coordinate_label=coordinate_label,
        label_mapping=num2label,
        voxel_label_niftii=nifti,
        voxel_label_crs="mni152",
        mni_eps=mni_eps,
    )
    labels = np.asarray(head_labeled.brain.vertices.coords[coordinate_label].values)
    n_bg = int((labels == "Background").sum())
    uniq = sorted(set(labels) - {"Background"})
    print(
        f"[atlas] Brodmann assigned to {len(labels)} vertices "
        f"({len(uniq)} areas, {n_bg} background). "
        f"Increase mni_eps if background is high."
    )
    return head_labeled

def brodmann_labels_for_head(head_labeled, coordinate_label="parcel_brodmann") -> np.ndarray:
    """Return the per-brain-vertex Brodmann label array (in vertex order)."""
    return np.asarray(head_labeled.brain.vertices.coords[coordinate_label].values)

# ---------------------------------------------------------------------------
# Transfer labels onto the reconstructed image
# ---------------------------------------------------------------------------

def _assign_parcel_coord(data: xr.DataArray, labels: np.ndarray, out_coord: str) -> xr.DataArray:
    """Attach ``labels`` as the ``out_coord`` coordinate on the ``vertex`` dim.

    Handles both full images (brain+scalp vertices, distinguished by ``is_brain``)
    and brain-only images. Brain vertices must be in head-model order.
    """
    if "vertex" not in data.dims:
        raise ValueError("Expected a 'vertex' dimension on the input DataArray.")

    n_vertex = data.sizes["vertex"]
    n_brain = len(labels)

    if "is_brain" in data.coords:
        is_brain = np.asarray(data.coords["is_brain"].values, dtype=bool)
        n_is_brain = int(is_brain.sum())
        if n_is_brain != n_brain:
            raise ValueError(
                f"Head model has {n_brain} brain vertices but the image has "
                f"{n_is_brain} (is_brain==True). Are head model and Adot consistent?"
            )
        full = np.array(["Background"] * n_vertex, dtype=object)
        full[is_brain] = labels
    else:
        if n_vertex != n_brain:
            raise ValueError(
                f"No 'is_brain' coord and vertex count ({n_vertex}) != number of "
                f"brain labels ({n_brain}); cannot align."
            )
        full = np.asarray(labels, dtype=object)

    return data.assign_coords({out_coord: ("vertex", full)})

def relabel_image_parcels(
    img: xr.DataArray,
    head_labeled,
    atlas_coord: str = "parcel_brodmann",
    out_coord: str = "parcel",
) -> xr.DataArray:
    """Replace an image's ``parcel`` coordinate with Brodmann labels.

    Works on the raw reconstruction (``chromo, vertex, trial_type, reltime`` with
    ``is_brain``) or on the baseline-corrected brain-only image. After this call,
    the ROI functions in :mod:`roi_analysis` select vertices by Brodmann area.

    Parameters
    ----------
    img : xr.DataArray
        Reconstructed image with a ``vertex`` dimension.
    head_labeled : TwoSurfaceHeadModel
        Head model returned by :func:`assign_brodmann_to_head`.
    atlas_coord : str
        Coordinate name on the head model brain vertices (``"parcel_brodmann"``).
    out_coord : str
        Coordinate name to write on the image (default ``"parcel"``, read by the
        ROI functions; use ``"parcel_brodmann"`` to keep both).
    """
    labels = brodmann_labels_for_head(head_labeled, atlas_coord)
    return _assign_parcel_coord(img, labels, out_coord)

# ---------------------------------------------------------------------------
# ROI selection helpers
# ---------------------------------------------------------------------------

def _ba_number(label: str) -> Optional[int]:
    """Extract the Brodmann-area number from a label like ``left_BA9`` -> 9."""
    if not isinstance(label, str):
        return None
    m = _BA_RE.search(label)
    return int(m.group(1)) if m else None

def _hemisphere(label: str) -> Optional[str]:
    lo = label.lower()
    if lo.startswith("left") or lo.startswith("l_") or lo.endswith("_lh") or lo.endswith("_l"):
        return "left"
    if lo.startswith("right") or lo.startswith("r_") or lo.endswith("_rh") or lo.endswith("_r"):
        return "right"
    return None

def select_roi_parcels(
    available_parcels: Iterable[str],
    roi: str | Iterable[int],
    hemisphere: str = "both",
) -> list[str]:
    """Return the Brodmann labels that make up an ROI, restricted to those present.

    Parameters
    ----------
    available_parcels : iterable of str
        Parcel labels actually present (e.g. sensitive parcels, or the unique
        labels on the reconstructed image).
    roi : str or iterable of int
        Either a key of :data:`BRODMANN_ROIS` (``"FPA"`` …) or an
        explicit list of Brodmann-area numbers, e.g. ``[10]``.
    hemisphere : {"both", "left", "right"}
        Restrict to one hemisphere.

    Returns
    -------
    list of str
        Sorted subset of ``available_parcels`` belonging to the ROI.
    """
    if isinstance(roi, str):
        if roi not in BRODMANN_ROIS:
            raise KeyError(f"Unknown ROI {roi!r}. Known: {list(BRODMANN_ROIS)} "
                           f"or pass a list of BA numbers.")
        ba_numbers = set(BRODMANN_ROIS[roi])
    else:
        ba_numbers = set(int(x) for x in roi)

    hemisphere = hemisphere.lower()
    out = []
    for label in available_parcels:
        n = _ba_number(label)
        if n is None or n not in ba_numbers:
            continue
        if hemisphere != "both":
            h = _hemisphere(label)
            if h is not None and h != hemisphere:
                continue
        out.append(label)
    return sorted(set(out))

def available_parcels_from_image(img: xr.DataArray, coord: str = "parcel") -> list[str]:
    """List the unique, non-background parcel labels on a (relabeled) image."""
    vals = np.asarray(img.coords[coord].values)
    return sorted(set(v for v in vals if isinstance(v, str) and v != "Background"))

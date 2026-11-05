"""Utilities for modifying .snirf files."""

from pathlib import Path

import h5py
import numpy as np


DEFAULT_LANDMARKS = {
    "LPA": (-86.076, -19.990, -47.986),
    "RPA": (85.794, -20.009, -48.031),
    "Nz": (0.0083, 86.811, -39.983),
    "Iz": (0.0, -118.570, -23.080),
}


def _find_nirs_probe(f: h5py.File) -> str:
    """Return the HDF5 path to the probe group."""
    for key in ["nirs/probe", "nirs1/probe"]:
        if key in f:
            return key
    raise KeyError("No nirs/probe or nirs1/probe group found in the SNIRF file.")


def _replace_dataset(group: h5py.Group, name: str, data, **kwargs):
    """Delete existing dataset if present, then create a new one."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def _read_existing_labels(probe: h5py.Group) -> list[str]:
    """Read existing landmarkLabels as a list of Python strings."""
    if "landmarkLabels" not in probe:
        return []
    raw = probe["landmarkLabels"][:]
    return [l.decode("ascii") if isinstance(l, bytes) else str(l) for l in raw]


def _read_existing_pos3d(probe: h5py.Group) -> np.ndarray:
    """Read existing landmarkPos3D as (N, 3) float64 array."""
    if "landmarkPos3D" not in probe:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(probe["landmarkPos3D"][:], dtype=np.float64)


def _read_existing_pos2d(probe: h5py.Group) -> np.ndarray:
    """Read existing landmarkPos2D as (N, 2) float64 array."""
    if "landmarkPos2D" not in probe:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(probe["landmarkPos2D"][:], dtype=np.float64)


def add_landmarks_to_snirf(
    snirf_path: str | Path,
    landmarks: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Append landmark labels and positions to a .snirf file.

    Preserves all existing landmarks. New landmarks are appended.
    If a landmark with the same name already exists, it is overwritten.
    Updates both landmarkPos3D and landmarkPos2D (x, y from 3D coords).

    Parameters
    ----------
    snirf_path : str or Path
        Path to the .snirf file to modify.
    landmarks : dict, optional
        Mapping of landmark name -> (x, y, z). Defaults to LPA, RPA, Nz, Iz.
    """
    if landmarks is None:
        landmarks = DEFAULT_LANDMARKS

    snirf_path = Path(snirf_path)
    if not snirf_path.exists():
        raise FileNotFoundError(f"SNIRF file not found: {snirf_path}")

    with h5py.File(snirf_path, "r+") as f:
        probe_path = _find_nirs_probe(f)
        probe = f[probe_path]

        # Read existing data
        existing_labels = _read_existing_labels(probe)
        existing_pos3d = _read_existing_pos3d(probe)
        existing_pos2d = _read_existing_pos2d(probe)

        # Build combined labels and positions, overwriting duplicates
        combined_labels = list(existing_labels)
        combined_pos3d = list(existing_pos3d)
        combined_pos2d = list(existing_pos2d)

        # Pad pos2d/pos3d lists if they are shorter than labels
        while len(combined_pos3d) < len(combined_labels):
            combined_pos3d.append(np.array([0.0, 0.0, 0.0]))
        while len(combined_pos2d) < len(combined_labels):
            combined_pos2d.append(np.array([0.0, 0.0]))

        for name, (x, y, z) in landmarks.items():
            if name in combined_labels:
                idx = combined_labels.index(name)
                combined_pos3d[idx] = np.array([x, y, z])
                combined_pos2d[idx] = np.array([x, y])
            else:
                combined_labels.append(name)
                combined_pos3d.append(np.array([x, y, z]))
                combined_pos2d.append(np.array([x, y]))

        # Encode labels as fixed-length ASCII byte strings
        max_len = max(len(l) for l in combined_labels)
        dtype_str = f"|S{max_len}"
        labels_encoded = np.array(
            [l.encode("ascii") for l in combined_labels], dtype=dtype_str
        )

        pos3d_array = np.array(combined_pos3d, dtype=np.float64)
        pos2d_array = np.array(combined_pos2d, dtype=np.float64)

        # Write back
        _replace_dataset(probe, "landmarkLabels", labels_encoded)
        _replace_dataset(probe, "landmarkPos3D", pos3d_array)
        _replace_dataset(probe, "landmarkPos2D", pos2d_array)

    print(f"Landmarks in {snirf_path.name}: {len(combined_labels)} total")
    for name, (x, y, z) in landmarks.items():
        print(f"  + {name}: ({x:.3f}, {y:.3f}, {z:.3f})")
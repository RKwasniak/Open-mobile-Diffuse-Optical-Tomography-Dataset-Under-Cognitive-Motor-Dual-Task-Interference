"""Import fNIRS data organised in the NIRS-BIDS structure for group analysis.

This module implements a *reader* for BIDS-organised fNIRS datasets, complementing
cedalion's :mod:`cedalion.io.bids` (which focuses on *writing* / converting SNIRF
collections into BIDS). It walks a BIDS tree, parses the entities encoded in the
file names, joins them with ``participants.tsv`` and returns a tidy manifest plus
lazily-loaded :class:`cedalion.dataclasses.Recording` objects.

The layout follows the NIRS-BIDS extension (BIDS >= 1.8), described in
Luke, Oostenveld, Cockx et al., *NIRS-BIDS: Brain Imaging Data Structure Extended
to Near-Infrared Spectroscopy*, Scientific Data 12:159 (2025),
https://doi.org/10.1038/s41597-024-04136-9 ::

    bids_root/
        dataset_description.json
        participants.tsv
        participants.json
        sub-<label>/
            [ses-<label>/]
                nirs/
                    sub-<label>[_ses-<label>][_task-<label>][_acq-<label>][_run-<index>]_nirs.snirf
                    ...[_channels.tsv, _optodes.tsv, _coordsystem.json, _events.tsv, _nirs.json]

Only the SNIRF files (``*_nirs.snirf``) carry the light-intensity data; the sidecar
files carry redundant, easily-parsable metadata. A BIDS-compliant SNIRF file holds a
single run.

Typical usage
-------------
>>> from processors_cedalion.bids_loader import build_bids_manifest, iter_bids_recordings
>>> manifest = build_bids_manifest("/data/my_bids_dataset", condition_from="task")
>>> for run_id, rec, amp, meta in iter_bids_recordings(manifest):
...     ...                                      # one Recording at a time
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# BIDS filename parsing
# ---------------------------------------------------------------------------

# Canonical order of BIDS entities relevant to NIRS. Additional key-value pairs
# encountered in a filename are still captured (kept generic on purpose).
_BIDS_ENTITY_ORDER = ("sub", "ses", "task", "acq", "run", "rec", "dir", "recording")

# One key-value token, e.g. "sub-01", "task-nback", "run-2".
_ENTITY_RE = re.compile(r"(?P<key>[a-zA-Z0-9]+)-(?P<value>[a-zA-Z0-9]+)")

def parse_bids_filename(path: str | Path) -> dict:
    """Parse the BIDS entities encoded in a file name.

    Parameters
    ----------
    path : str or Path
        Any BIDS file path (only the file name is inspected).

    Returns
    -------
    dict
        Entity key -> value for every ``key-value`` token found, plus
        ``suffix`` (the trailing token, e.g. ``"nirs"``) and ``extension``.
        ``run`` is converted to ``int`` when present. Missing entities are
        simply absent from the dict.

    Examples
    --------
    >>> parse_bids_filename("sub-07_ses-1_task-mc_run-2_nirs.snirf")
    {'sub': '07', 'ses': '1', 'task': 'mc', 'run': 2, 'suffix': 'nirs',
     'extension': '.snirf'}
    """
    p = Path(path)
    name = p.name
    # Split double extensions such as .snirf / .nii.gz cleanly.
    stem = name
    extension = ""
    for ext in (".snirf", ".nii.gz", ".json", ".tsv", ".nii"):
        if name.endswith(ext):
            stem = name[: -len(ext)]
            extension = ext
            break
    else:
        stem = p.stem
        extension = p.suffix

    tokens = stem.split("_")
    entities: dict = {}
    suffix = None
    for tok in tokens:
        m = _ENTITY_RE.fullmatch(tok)
        if m:
            entities[m.group("key")] = m.group("value")
        else:
            # A token without a dash is the suffix (last one wins, as per BIDS).
            suffix = tok

    if "run" in entities:
        try:
            entities["run"] = int(entities["run"])
        except (TypeError, ValueError):
            pass

    entities["suffix"] = suffix
    entities["extension"] = extension
    return entities

def find_nirs_files(bids_root: str | Path, suffix: str = "nirs") -> list[Path]:
    """Recursively collect all ``*_<suffix>.snirf`` files under a BIDS root.

    Files inside ``derivatives/`` and ``sourcedata/`` are ignored so that only
    the raw acquisition data is returned.
    """
    bids_root = Path(bids_root)
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    files = []
    for f in sorted(bids_root.rglob(f"*_{suffix}.snirf")):
        parts = set(f.relative_to(bids_root).parts)
        if {"derivatives", "sourcedata"} & parts:
            continue
        files.append(f)

    if not files:
        raise FileNotFoundError(
            f"No *_{suffix}.snirf files found under {bids_root}. "
            "Is this a NIRS-BIDS dataset?"
        )
    return files

# ---------------------------------------------------------------------------
# participants.tsv / sidecar handling
# ---------------------------------------------------------------------------

def read_participants(bids_root: str | Path) -> pd.DataFrame:
    """Read ``participants.tsv`` from the dataset root.

    Returns an empty DataFrame (with a ``participant_id`` column) when the file
    is absent, so the join in :func:`build_bids_manifest` degrades gracefully.
    """
    bids_root = Path(bids_root)
    ptsv = bids_root / "participants.tsv"
    if not ptsv.exists():
        return pd.DataFrame(columns=["participant_id"])

    df = pd.read_csv(ptsv, sep="\t", dtype=str)
    if "participant_id" not in df.columns:
        raise ValueError(
            f"{ptsv} is missing the required 'participant_id' column."
        )
    # normalise: keep bare label too (without the leading 'sub-')
    df["subject_label"] = df["participant_id"].str.replace(r"^sub-", "", regex=True)
    return df

def events_tsv_path(snirf_path: str | Path) -> Optional[Path]:
    """Return the path to the ``*_events.tsv`` sidecar, or ``None`` if absent."""
    snirf_path = Path(snirf_path)
    ev = snirf_path.with_name(snirf_path.name.replace("_nirs.snirf", "_events.tsv"))
    return ev if ev.exists() else None

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_bids_manifest(
    bids_root: str | Path,
    condition_from: str = "ses",
    condition_map: Optional[dict] = None,
    task_filter: Optional[str] = None,
    participants_extra_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a tidy, one-row-per-run manifest of a NIRS-BIDS dataset.

    Each row corresponds to one ``*_nirs.snirf`` file. Entities parsed from the
    file name (``sub``, ``ses``, ``task``, ``acq``, ``run``) are joined with the
    columns of ``participants.tsv`` on ``participant_id``.

    Parameters
    ----------
    bids_root : str or Path
        Path to the BIDS dataset root.
    condition_from : str, default "ses"
        Which entity/column defines the experimental *condition* used later to
        split quality metrics and responses (e.g. sitting vs walking). Common
        choices are ``"ses"``, ``"task"`` or ``"acq"``. Set to ``None`` to skip.
    condition_map : dict, optional
        Optional mapping applied to the ``condition_from`` values to produce a
        human-readable ``condition`` column, e.g.
        ``{"sit": "SITTING", "walk": "WALKING"}``. Values not in the mapping are
        passed through unchanged.
    task_filter : str, optional
        If given, keep only rows whose ``task`` equals this value.
    participants_extra_cols : list of str, optional
        Extra ``participants.tsv`` columns to keep (besides those auto-detected).

    Returns
    -------
    pandas.DataFrame
        Columns: ``run_id``, ``subject_label``, ``participant_id``, ``task``,
        ``session``, ``acq``, ``run``, ``condition`` (if requested), any
        participant columns, and ``snirf_path``. Sorted by (subject, session,
        task, run).
    """
    bids_root = Path(bids_root)
    files = find_nirs_files(bids_root)
    participants = read_participants(bids_root)

    rows = []
    for f in files:
        ent = parse_bids_filename(f)
        sub_label = ent.get("sub")
        rows.append(
            {
                "run_id": f.stem,  # unique per run within the dataset
                "subject_label": sub_label,
                "participant_id": f"sub-{sub_label}" if sub_label else None,
                "task": ent.get("task"),
                "session": ent.get("ses"),
                "acq": ent.get("acq"),
                "run": ent.get("run"),
                "snirf_path": f,
            }
        )
    files_df = pd.DataFrame(rows)

    if task_filter is not None:
        files_df = files_df[files_df["task"] == task_filter].copy()
        if files_df.empty:
            raise RuntimeError(f"No runs left after task_filter={task_filter!r}.")

    # --- condition column -------------------------------------------------
    if condition_from is not None:
        col = {"ses": "session"}.get(condition_from, condition_from)
        if col not in files_df.columns:
            raise ValueError(
                f"condition_from={condition_from!r} maps to column {col!r} which "
                f"is not available. Available: {list(files_df.columns)}"
            )
        cond = files_df[col].astype("object")
        if condition_map:
            cond = cond.map(lambda v: condition_map.get(v, v))
        files_df["condition"] = cond

    # --- join participants ------------------------------------------------
    if not participants.empty:
        keep = ["participant_id", "subject_label"]
        auto = [c for c in participants.columns if c not in ("participant_id", "subject_label")]
        if participants_extra_cols is not None:
            auto = [c for c in auto if c in participants_extra_cols]
        keep += auto
        manifest = files_df.merge(
            participants[keep].drop_duplicates("participant_id"),
            on=["participant_id", "subject_label"],
            how="left",
        )
        missing = sorted(set(files_df.participant_id.dropna()) - set(participants.participant_id))
        if missing:
            print(f"[bids] {len(missing)} subject(s) not in participants.tsv: {missing}")
    else:
        manifest = files_df
        print("[bids] no participants.tsv found — proceeding without demographics.")

    sort_cols = [c for c in ("subject_label", "session", "task", "run") if c in manifest.columns]
    manifest = manifest.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    n_sub = manifest["subject_label"].nunique()
    print(f"[bids] {len(manifest)} run(s) across {n_sub} subject(s) discovered in {bids_root}")
    if "condition" in manifest.columns:
        print("[bids] runs per condition:")
        for c, k in manifest["condition"].value_counts(dropna=False).items():
            print(f"        {c}: {k}")
    return manifest

# ---------------------------------------------------------------------------
# Loading recordings
# ---------------------------------------------------------------------------

def load_bids_recording(
    snirf_path: str | Path,
    add_events_from_tsv: bool = True,
    landmarks: Optional[dict] = None,
):
    """Load a single BIDS SNIRF file into a cedalion Recording.

    Parameters
    ----------
    snirf_path : str or Path
        Path to a ``*_nirs.snirf`` file.
    add_events_from_tsv : bool, default True
        If a ``*_events.tsv`` sidecar exists and the SNIRF file has no stimulus
        information, populate ``rec.stim`` from the TSV (via
        :func:`cedalion.io.read_events_from_tsv`).
    landmarks : dict, optional
        If given, inject these fiducials (LPA/RPA/Nz/Iz, ...) into the SNIRF file
        before loading (see :func:`processors_cedalion.snirf_utils.add_landmarks_to_snirf`).
        Use for datasets whose probes lack landmarks needed for registration.

    Returns
    -------
    (rec, amp) : tuple
        The cedalion Recording and its amplitude DataArray.
    """
    import cedalion.io  # local import: keep module importable without cedalion

    snirf_path = Path(snirf_path)
    if not snirf_path.exists():
        raise FileNotFoundError(f"SNIRF file not found: {snirf_path}")

    if landmarks is not None:
        from .snirf_utils import add_landmarks_to_snirf
        add_landmarks_to_snirf(snirf_path, landmarks)

    rec = cedalion.io.read_snirf(snirf_path)[0]
    amp = rec["amp"]

    if add_events_from_tsv and (rec.stim is None or len(rec.stim) == 0):
        ev = events_tsv_path(snirf_path)
        if ev is not None:
            try:
                rec.stim = cedalion.io.read_events_from_tsv(ev)
                print(f"[bids] events loaded from {ev.name}")
            except Exception as exc:  # pragma: no cover - depends on cedalion version
                print(f"[bids] could not read {ev.name}: {exc}")

    return rec, amp

def iter_bids_recordings(
    manifest: pd.DataFrame,
    add_events_from_tsv: bool = True,
    landmarks: Optional[dict] = None,
) -> Iterator[tuple[str, "object", "object", dict]]:
    """Yield ``(run_id, rec, amp, meta)`` for every row of a BIDS manifest.

    Memory-friendly: one recording is materialised at a time. ``meta`` is the
    manifest row (minus the path) as a plain dict, ready to stash in a results
    container for later group filtering.
    """
    meta_cols = [c for c in manifest.columns if c != "snirf_path"]
    for row in manifest.itertuples(index=False):
        row_d = row._asdict()
        rec, amp = load_bids_recording(
            row_d["snirf_path"],
            add_events_from_tsv=add_events_from_tsv,
            landmarks=landmarks,
        )
        meta = {k: row_d[k] for k in meta_cols}
        yield row_d["run_id"], rec, amp, meta


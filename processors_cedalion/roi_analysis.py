"""ROI responses (FPA) from Brodmann-labelled image reconstructions.

The image must carry Brodmann labels on its ``parcel`` coordinate
(:func:`processors_cedalion.atlas_brodmann.relabel_image_parcels`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .atlas_brodmann import select_roi_parcels, available_parcels_from_image
from .plotstyle import (CHROMO_COLORS as _CHROMO_COLOR, style_axes, INK,
                        averaged_title, REGION_ABBR)


# ---------------------------------------------------------------------------
# ROI time courses and window means
# ---------------------------------------------------------------------------

def roi_timecourse(
    img_relabeled: xr.DataArray,
    roi,
    hemisphere: str = "both",
    parcel_coord: str = "parcel",
) -> xr.DataArray:
    """Mean over the ROI's vertices -> dims ``(chromo, trial_type, reltime)``.

    ``roi`` is a :data:`BRODMANN_ROIS` key or a list of Brodmann-area numbers.
    The result carries attrs ``roi``, ``n_vertices``, ``parcels``.
    """
    available = available_parcels_from_image(img_relabeled, parcel_coord)
    roi_labels = select_roi_parcels(available, roi, hemisphere)
    if not roi_labels:
        raise ValueError(
            f"ROI {roi!r} (hemisphere={hemisphere}) matched no parcels present on "
            f"the image. Available areas: {available}"
        )

    parcels = np.asarray(img_relabeled.coords[parcel_coord].values)
    mask = np.isin(parcels, roi_labels)
    sub = img_relabeled.isel(vertex=np.where(mask)[0])
    data = sub.pint.dequantify() if hasattr(sub, "pint") and getattr(sub.pint, "units", None) is not None else sub
    tc = data.mean("vertex")
    tc.attrs.update(roi=str(roi), n_vertices=int(mask.sum()), parcels=roi_labels)
    return tc


def roi_window_mean(
    tc: xr.DataArray,
    before: float = 0.0,
    after: float = 17.0,
) -> pd.DataFrame:
    """Mean, peak and time-to-peak over ``reltime in [before, after]``,
    one row per (chromo, trial_type)."""
    sel = tc.sel(reltime=slice(before, after))
    rows = []
    for chromo in sel.chromo.values:
        for tt in sel.trial_type.values:
            vals = np.asarray(sel.sel(chromo=chromo, trial_type=tt).values, dtype=float)
            times = np.asarray(sel.reltime.values, dtype=float)
            k = int(np.abs(vals).argmax()) if vals.size else 0
            rows.append({
                "chromo": str(chromo),
                "trial_type": str(tt),
                "mean": float(np.nanmean(vals)) if vals.size else np.nan,
                "peak": float(vals[k]) if vals.size else np.nan,
                "time_to_peak": float(times[k]) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def build_group_roi_table(
    all_results: dict,
    rois=("FPA",),
    hemisphere: str = "both",
    img_key: str = "img_brain_bl",
    parcel_coord: str = "parcel",
    condition_key: str = "condition",
    subject_key: str = "subject_id",
    before: float = 0.0,
    after: float = 17.0,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Window means for every run -> long table, one row per
    (run, roi, chromo, trial_type).

    Columns: ``run_id, subject, condition, roi, hemisphere, n_vertices, chromo,
    trial_type, mean, peak, time_to_peak``.
    """
    frames = []
    for run_id, res in all_results.items():
        if img_key not in res:
            print(f"[roi] {run_id}: no '{img_key}', skipping.")
            continue
        img = res[img_key]
        if parcel_coord not in img.coords:
            print(f"[roi] {run_id}: no '{parcel_coord}' coord "
                  f"(run relabel_image_parcels), skipping.")
            continue
        for roi in rois:
            try:
                tc = roi_timecourse(img, roi, hemisphere, parcel_coord)
            except ValueError as exc:
                print(f"[roi] {run_id}/{roi}: {exc}")
                continue
            w = roi_window_mean(tc, before, after)
            w.insert(0, "n_vertices", tc.attrs["n_vertices"])
            w.insert(0, "hemisphere", hemisphere)
            w.insert(0, "roi", roi)
            w.insert(0, "condition", res.get(condition_key))
            w.insert(0, "subject", res.get(subject_key, run_id))
            w.insert(0, "run_id", run_id)
            frames.append(w)

    if not frames:
        raise RuntimeError("No ROI responses computed. Check img relabeling & ROIs.")
    df = pd.concat(frames, ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[roi] group ROI table: {len(df)} rows -> {output_path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roi_averaged(
    all_results: dict,
    roi,
    hemisphere: str = "both",
    img_key: str = "img_brain_bl",
    parcel_coord: str = "parcel",
    trial_types=("MC", "REST"),
    condition_key: str = "condition",
    conditions: Optional[list] = None,
    window: Optional[tuple] = (0.0, 17.0),
    ylims: Optional[tuple] = None,
    title: Optional[str] = None,
):
    """ROI HbO/HbR time courses, mean ± SEM across subjects, one panel per
    condition. MC solid, REST dashed; ``window`` is shaded."""
    import matplotlib.pyplot as plt

    per_cond: dict = {}
    for run_id, res in all_results.items():
        if img_key not in res or parcel_coord not in res[img_key].coords:
            continue
        try:
            tc = roi_timecourse(res[img_key], roi, hemisphere, parcel_coord)
        except ValueError:
            continue
        cond = res.get(condition_key, "all")
        per_cond.setdefault(cond, []).append(tc)

    if not per_cond:
        raise RuntimeError("No ROI time courses available (relabel images first).")

    if conditions is None:
        conditions = list(per_cond.keys())
    conditions = [c for c in conditions if c in per_cond]

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False, sharey=True)

    for j, cond in enumerate(conditions):
        ax = axes[0][j]
        stacked = xr.concat(per_cond[cond], dim="subject")
        mean = stacked.mean("subject")
        sem = stacked.std("subject") / np.sqrt(stacked.sizes["subject"])
        t = np.asarray(mean.reltime.values, float)

        for chromo in ("HbO", "HbR"):
            if chromo not in mean.chromo.values:
                continue
            for tt, ls in zip(trial_types, ["-", "--", ":"]):
                if tt not in mean.trial_type.values:
                    continue
                m = np.asarray(mean.sel(chromo=chromo, trial_type=tt).values, float)
                s = np.asarray(sem.sel(chromo=chromo, trial_type=tt).values, float)
                ax.plot(t, m, color=_CHROMO_COLOR[chromo], ls=ls, lw=2.2,
                        label=f"{chromo} {tt}")
                ax.fill_between(t, m - s, m + s, color=_CHROMO_COLOR[chromo],
                                alpha=0.15, linewidth=0)

        if window is not None:
            ax.axvspan(window[0], window[1], color="#9AA0A6", alpha=0.15, zorder=0)
        ax.axhline(0, color=INK, lw=0.7)
        ax.axvline(0, color=INK, lw=0.7, ls=":")
        style_axes(ax, grid_axis="both")
        ax.set_title(f"{cond}  (N={stacked.sizes['subject']})")
        ax.set_xlabel(r"$t_{\mathrm{rel}}$ / s")
        if j == 0:
            ax.set_ylabel(r"$\Delta$ concentration / µM")
        if ylims:
            ax.set_ylim(*ylims)

    axes[0][0].legend(loc="best", fontsize=8.5)
    if title is None:
        title = averaged_title(roi, hemisphere, trial_types)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_roi_contrast_bars(
    group_roi_df: pd.DataFrame,
    rois=("FPA",),
    value_col: str = "mean",
    mc_label: str = "MC",
    rest_label: str = "REST",
    order: Optional[list] = None,
):
    """MC − REST per ROI × chromophore × condition: group mean ± SEM of the
    per-subject contrast."""
    import matplotlib.pyplot as plt

    recs = []
    for (subject, condition, roi, chromo), g in group_roi_df.groupby(
            ["subject", "condition", "roi", "chromo"], dropna=False):
        wide = g.pivot_table(index="subject", columns="trial_type", values=value_col)
        if mc_label in wide.columns and rest_label in wide.columns:
            recs.append({"subject": subject, "condition": condition, "roi": roi,
                         "chromo": chromo,
                         "contrast": float(wide[mc_label].iloc[0] - wide[rest_label].iloc[0])})
    cdf = pd.DataFrame(recs)

    conditions = order or sorted(cdf["condition"].dropna().unique(), key=str)
    fig, axes = plt.subplots(1, len(rois), figsize=(4.6 * len(rois), 4.4), squeeze=False)

    for a, roi in enumerate(rois):
        ax = axes[0][a]
        x = np.arange(len(conditions))
        width = 0.38
        for i, chromo in enumerate(("HbO", "HbR")):
            means, sems = [], []
            for cond in conditions:
                vals = cdf[(cdf.roi == roi) & (cdf.chromo == chromo) &
                           (cdf.condition == cond)]["contrast"].values
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                sems.append(np.nanstd(vals) / np.sqrt(len(vals)) if len(vals) else np.nan)
            ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=3,
                   color=_CHROMO_COLOR[chromo], alpha=0.9, label=chromo,
                   edgecolor="white", linewidth=0.8,
                   error_kw=dict(ecolor=INK, lw=1.0))

        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in conditions])
        ax.set_ylabel("MC − REST contrast / µM")
        ax.set_title(REGION_ABBR.get(str(roi), str(roi)), fontsize=10)
        style_axes(ax, grid_axis="y")
        ax.legend(fontsize=8.5)
    fig.suptitle("MC − REST contrast by ROI", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig, axes

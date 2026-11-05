"""Signal-quality metrics for technical validation: SNR, SCI and CV.

Provides the three channel-wise quality metrics commonly reported in the
Technical Validation section of fNIRS data descriptors, computed per recording
and aggregated across a group, split by an experimental *condition*
(e.g. SITTING vs WALKING):

* **SNR** – signal-to-noise ratio, ``mean(amp)/std(amp)`` per channel &
  wavelength (cedalion ``sigproc.quality.snr``; Huppert 2009).
* **SCI** – scalp-coupling index, cardiac-band cross-wavelength correlation per
  channel (cedalion ``sigproc.quality.sci``; Pollonini 2014/2016).
* **CV** – coefficient of variation, ``100 * std(amp)/mean(amp)`` [%] per channel
  & wavelength. A standard raw-signal quality/motion indicator in ambulatory
  fNIRS. Note CV% and SNR are reciprocal (CV% = 100 / SNR).

The compute helpers depend only on numpy/xarray (CV) or cedalion (SNR/SCI); the
plotting helpers depend on matplotlib. Everything is grouped here because it forms
one self-contained validation workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .plotstyle import CONDITION_PALETTE, style_axes

_METRIC_LABELS = {
    "snr": "SNR (mean/std)",
    "sci": "Scalp Coupling Index",
    "cv": "Coefficient of Variation [%]",
}
# Conventional acceptance thresholds (editable). Good channels: SNR high, SCI
# high, CV low.
_METRIC_THRESH = {"snr": 2.0, "sci": 0.75, "cv": 7.5}
_METRIC_GOOD_SIDE = {"snr": "above", "sci": "above", "cv": "below"}

# ---------------------------------------------------------------------------
# CV (coefficient of variation)
# ---------------------------------------------------------------------------

def compute_cv(amp: xr.DataArray) -> xr.DataArray:
    """Coefficient of variation (%) per channel and wavelength.

    ``CV = 100 * std_time(amp) / mean_time(amp)``. Operates on raw amplitude and
    is unit-agnostic (pint units are stripped internally).

    Parameters
    ----------
    amp : xr.DataArray
        Raw amplitude, dims ``(channel, wavelength, time)`` (order irrelevant).

    Returns
    -------
    xr.DataArray
        CV in percent, dims ``(channel, wavelength)``.
    """
    a = amp.pint.dequantify() if hasattr(amp, "pint") and getattr(amp.pint, "units", None) is not None else amp
    mean = a.mean("time")
    std = a.std("time")
    cv = 100.0 * std / mean.where(mean != 0)
    cv.name = "cv"
    return cv

# ---------------------------------------------------------------------------
# Per-recording quality table
# ---------------------------------------------------------------------------

def channel_quality_table(
    amp: xr.DataArray,
    sci_window_s: float = 5.0,
    sci_thresh: float = 0.75,
    snr_thresh: float = 2.0,
    cv_thresh: float = 7.5,
) -> pd.DataFrame:
    """Compute SNR, SCI and CV for one recording -> tidy per-channel/wavelength table.

    Returns one row per ``(channel, wavelength)`` with columns
    ``channel, wavelength, snr, cv, sci`` (SCI is per channel and repeated across
    wavelengths) plus boolean pass flags ``snr_ok, cv_ok, sci_ok``.
    """
    import cedalion.sigproc.quality as quality
    from cedalion import units

    # SNR & CV are per (channel, wavelength)
    snr, _ = quality.snr(amp, snr_thresh)
    cv = compute_cv(amp)

    # SCI is per channel (averaged over its time windows)
    sci_tw, _ = quality.sci(amp, sci_window_s * units.s, sci_thresh)
    sci_ch = sci_tw.mean("time")

    snr = snr.pint.dequantify() if hasattr(snr, "pint") and getattr(snr.pint, "units", None) is not None else snr

    rows = []
    channels = list(amp.channel.values)
    wavelengths = list(amp.wavelength.values)
    for ch in channels:
        sci_val = float(sci_ch.sel(channel=ch).values)
        for wl in wavelengths:
            snr_val = float(snr.sel(channel=ch, wavelength=wl).values)
            cv_val = float(cv.sel(channel=ch, wavelength=wl).values)
            rows.append(
                {
                    "channel": str(ch),
                    "wavelength": float(wl),
                    "snr": snr_val,
                    "cv": cv_val,
                    "sci": sci_val,
                    "snr_ok": snr_val >= snr_thresh,
                    "cv_ok": 0.0 <= cv_val <= cv_thresh,   # CV < 0: mean amplitude <= 0
                    "sci_ok": sci_val >= sci_thresh,
                }
            )
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Group quality table
# ---------------------------------------------------------------------------

def build_group_quality_table(
    all_results: dict,
    condition_key: str = "condition",
    subject_key: str = "subject_id",
    amp_from: str = "amp",
    sci_window_s: float = 5.0,
    sci_thresh: float = 0.75,
    snr_thresh: float = 2.0,
    cv_thresh: float = 7.5,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Assemble a long-format group quality table (SNR/SCI/CV) for all recordings.

    Parameters
    ----------
    all_results : dict
        ``run_id -> {...}`` with, at minimum, a recording/amplitude accessible via
        ``amp_from`` and the metadata keys ``condition_key`` and ``subject_key``.
        ``amp_from`` may name either a Recording (``res["rec"]`` then ``rec[amp]``)
        or an amplitude DataArray directly (``res["amp"]``).
    condition_key, subject_key : str
        Metadata keys stored alongside each recording (see the BIDS/manifest loaders).
    amp_from : str
        Key holding amplitude data or the Recording (default ``"amp"``; falls back
        to ``res["rec"]["amp"]``).
    output_path : optional
        If given, write the table to CSV.

    Returns
    -------
    pandas.DataFrame
        Columns: ``run_id, subject, condition, channel, wavelength,
        snr, cv, sci, snr_ok, cv_ok, sci_ok``.
    """
    frames = []
    for run_id, res in all_results.items():
        # resolve amplitude
        amp = None
        if amp_from in res and hasattr(res[amp_from], "dims"):
            amp = res[amp_from]
        elif "rec" in res:
            amp = res["rec"]["amp"]
        elif "amp" in res:
            amp = res["amp"]
        if amp is None:
            print(f"[quality] {run_id}: no amplitude data found, skipping.")
            continue

        tbl = channel_quality_table(
            amp,
            sci_window_s=sci_window_s,
            sci_thresh=sci_thresh,
            snr_thresh=snr_thresh,
            cv_thresh=cv_thresh,
        )
        tbl.insert(0, "condition", res.get(condition_key))
        tbl.insert(0, "subject", res.get(subject_key, run_id))
        tbl.insert(0, "run_id", run_id)
        frames.append(tbl)

    if not frames:
        raise RuntimeError("No recordings produced quality metrics.")
    df = pd.concat(frames, ignore_index=True)

    # Record the thresholds actually used so the figures (and any re-summary) can
    # draw the *same* reference lines instead of the hardcoded conventional
    # defaults. Kept on the DataFrame so the caller sets them in one place only.
    df.attrs["thresholds"] = {"snr": snr_thresh, "sci": sci_thresh, "cv": cv_thresh}

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[quality] group quality table: {len(df)} rows -> {output_path}")
    return df

def summarize_group_quality(
    df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Per-condition summary of SNR/SCI/CV.

    Reports mean, SD, the quartiles (Q1 / median / Q3), the IQR and the percentage
    of channels passing the metric's threshold. The quartiles are the robust
    counterpart to mean ± SD, which these metrics need: SNR and CV have heavy
    right tails (a few channels inflate the SD by orders of magnitude), so
    median [Q1, Q3] is what should be reported.
    """
    recs = []
    for cond, sub in df.groupby(condition_col, dropna=False):
        row = {condition_col: cond,
               "n_subjects": sub["subject"].nunique(),
               "n_channel_obs": len(sub)}
        for m in ("snr", "sci", "cv"):
            _metric_stats(sub[m], m, row)
            ok = f"{m}_ok"
            if ok in sub.columns:
                row[f"pct_{m}_pass"] = float(100 * sub[ok].mean())
        recs.append(row)
    return pd.DataFrame(recs)

def _metric_stats(values, name, out: dict) -> dict:
    """mean / SD / quartiles / IQR for one metric, written into ``out``."""
    q1, med, q3 = values.quantile([0.25, 0.50, 0.75])
    out[f"{name}_mean"] = float(values.mean())
    out[f"{name}_sd"] = float(values.std())
    out[f"{name}_q1"] = float(q1)
    out[f"{name}_median"] = float(med)
    out[f"{name}_q3"] = float(q3)
    out[f"{name}_iqr"] = float(q3 - q1)
    return out

def summarize_subject_quality(
    df: pd.DataFrame,
    condition_col: str = "condition",
    subject_col: str = "subject",
    metrics=("snr", "sci", "cv"),
) -> pd.DataFrame:
    """One row per subject x condition — the unit of analysis for group statistics.

    Each subject's channels are collapsed to mean, SD, quartiles (Q1 / median /
    Q3), IQR and the percentage of channels passing the metric threshold. Use the
    medians for statistics: SNR and CV are heavily right-skewed across channels,
    so a subject's mean is dominated by a few bad channels.
    """
    recs = []
    for (subj, cond), sub in df.groupby([subject_col, condition_col], dropna=False):
        row = {subject_col: subj, condition_col: cond,
               "n_channel_obs": len(sub),
               "n_channels": sub["channel"].nunique() if "channel" in sub else np.nan}
        for m in metrics:
            _metric_stats(sub[m], m, row)
            ok = f"{m}_ok"
            if ok in sub.columns:
                row[f"pct_{m}_pass"] = float(100 * sub[ok].mean())
        recs.append(row)

    out = pd.DataFrame(recs)
    key = pd.to_numeric(out[subject_col], errors="coerce")
    out = (out.assign(_k=key).sort_values(["_k", subject_col, condition_col])
              .drop(columns="_k").reset_index(drop=True))
    return out

def pivot_subject_quality(
    subject_df: pd.DataFrame,
    condition_col: str = "condition",
    subject_col: str = "subject",
    conditions=None,
) -> pd.DataFrame:
    """Wide layout of :func:`summarize_subject_quality`: one row per subject.

    Condition is folded into the column names (``snr_median_SITTING`` ...) because
    every subject is measured in both conditions — condition is a **repeated
    measure**, not a grouping variable. Feeding the long table to a stats package
    would make it look like independent groups.
    """
    conds = list(conditions) if conditions is not None else \
        list(pd.unique(subject_df[condition_col].dropna()))
    value_cols = [c for c in subject_df.columns
                  if c not in (subject_col, condition_col)]

    out = pd.DataFrame({subject_col: pd.unique(subject_df[subject_col])})
    for col in value_cols:
        p = subject_df.pivot_table(index=subject_col, columns=condition_col,
                                   values=col, aggfunc="first")
        for c in conds:
            if c in p.columns:
                out[f"{col}_{c}"] = out[subject_col].map(p[c])

    key = pd.to_numeric(out[subject_col], errors="coerce")
    return (out.assign(_k=key).sort_values(["_k", subject_col])
               .drop(columns="_k").reset_index(drop=True))

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _aggregate_by_condition(df, metric, agg):
    """Return {condition: 1D array of values} for a metric under an aggregation."""
    if agg == "subject_channel_mean":
        # average over wavelength (and subjects) per (condition, channel) -> across channels
        g = (df.groupby(["condition", "channel"], dropna=False)[metric]
               .mean().reset_index())
        return {c: sub[metric].values for c, sub in g.groupby("condition", dropna=False)}
    elif agg == "subject_mean":
        # one value per (condition, subject): channel+wavelength mean -> across subjects
        g = (df.groupby(["condition", "subject"], dropna=False)[metric]
               .mean().reset_index())
        return {c: sub[metric].values for c, sub in g.groupby("condition", dropna=False)}
    elif agg == "observation":
        return {c: sub[metric].values for c, sub in df.groupby("condition", dropna=False)}
    raise ValueError(f"unknown agg={agg!r}")

def _robust_ylim(box_data, thr=None, means=None, whis: float = 1.5, pad: float = 0.08):
    """Data-adaptive y-limits: crop extreme outliers, keep the informative part.

    Uses the same Tukey fences as the boxplot whiskers (``whis`` * IQR, applied
    per condition) to find the inlier extent, takes the union across conditions,
    then expands the range to always include the group means and the reference
    threshold so nothing informative is cut off. A few very large/small channel
    values therefore no longer stretch the axis and flatten the boxes (the issue
    seen on the SNR and CV figures).

    Returns ``(lo, hi)`` with a small padding, or ``None`` if there is no finite
    data to bound.
    """
    lows, highs = [], []
    for vals in box_data:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        inliers = v[(v >= q1 - whis * iqr) & (v <= q3 + whis * iqr)]
        if inliers.size == 0:
            inliers = v
        lows.append(float(inliers.min()))
        highs.append(float(inliers.max()))
    if not lows:
        return None

    lo, hi = min(lows), max(highs)
    if means is not None:
        m = np.asarray([x for x in means if np.isfinite(x)], dtype=float)
        if m.size:
            lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
    if thr is not None:
        lo, hi = min(lo, thr), max(hi, thr)

    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi != 0 else 1.0
    return lo - pad * span, hi + pad * span

def plot_quality_by_condition(
    df: pd.DataFrame,
    metric: str,
    agg: str = "subject_channel_mean",
    order: Optional[list] = None,
    threshold: Optional[float] = None,
    ax=None,
    palette=CONDITION_PALETTE,
    title: Optional[str] = None,
    clip_outliers: bool = True,
    whis: float = 1.5,
    ylim: Optional[tuple] = None,
):
    """Plot the distribution of one quality metric per condition.

    A box (IQR + median), a jittered scatter of the underlying values, and the
    group mean (diamond) are drawn for each condition. Use one call per metric to
    obtain separate figures (as required for the validation report).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`build_group_quality_table`.
    metric : {"snr", "sci", "cv"}
    agg : {"subject_channel_mean", "subject_mean", "observation"}
        How to reduce the table before plotting. ``subject_channel_mean``
        (default) averages each channel across subjects/wavelengths and shows the
        spatial distribution — i.e. "averaged across subjects" as requested.
        ``subject_mean`` shows one value per subject. ``observation`` shows every
        channel×subject point.
    threshold : float, optional
        Reference line. If omitted, uses the threshold recorded on ``df`` by
        :func:`build_group_quality_table` (``df.attrs["thresholds"][metric]``),
        falling back to the conventional default for the metric.
    clip_outliers : bool
        If True (default), the y-axis is scaled to the data using Tukey fences
        (see :func:`_robust_ylim`) so a handful of extreme channels — common for
        SNR and CV — do not squash the boxes into an unreadable band. Points that
        fall outside the view are parked at the axis edge with ``▲``/``▼`` markers
        so no data is silently hidden. Set False for a raw autoscaled axis.
    whis : float
        Whisker/​fence multiplier (× IQR) used both for the box whiskers and the
        adaptive y-limits. Default 1.5 (standard Tukey).
    ylim : tuple, optional
        Explicit ``(lo, hi)`` y-limits. Overrides the adaptive scaling.
    """
    import matplotlib.pyplot as plt

    metric = metric.lower()
    if metric not in _METRIC_LABELS:
        raise ValueError(f"metric must be one of {list(_METRIC_LABELS)}")

    data = _aggregate_by_condition(df, metric, agg)
    conditions = order or sorted(data.keys(), key=lambda x: (x is None, str(x)))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1.6 * len(conditions) + 2.5, 4.5))
    else:
        fig = ax.figure

    rng = np.random.default_rng(0)
    positions = np.arange(len(conditions)) + 1
    box_data = [np.asarray(data[c])[~np.isnan(data[c])] for c in conditions]

    bp = ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=False,
                    whis=whis, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
                    boxprops=dict(edgecolor="0.4"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.35)

    means = [float(np.mean(v)) if len(v) else np.nan for v in box_data]
    # Threshold line: explicit arg wins, else the value recorded by
    # build_group_quality_table (df.attrs["thresholds"]), else the conventional default.
    if threshold is not None:
        thr = threshold
    else:
        recorded = getattr(df, "attrs", {}).get("thresholds") or {}
        thr = recorded.get(metric, _METRIC_THRESH.get(metric))

    # Data-adaptive vertical range: crop extreme outliers so the boxes and the
    # bulk of the channels stay readable. The group means and the threshold are
    # always kept in view; out-of-range points are shown parked at the edges.
    if ylim is None and clip_outliers:
        ylim = _robust_ylim(box_data, thr=thr, means=means, whis=whis)

    n_clipped = 0
    for i, vals in enumerate(box_data):
        if not len(vals):
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = palette[i % len(palette)]
        if ylim is not None:
            lo, hi = ylim
            inside = (vals >= lo) & (vals <= hi)
            ax.scatter(positions[i] + jitter[inside], vals[inside], s=14,
                       color=color, alpha=0.7, edgecolor="white",
                       linewidth=0.3, zorder=3)
            above, below = vals > hi, vals < lo
            if above.any():
                ax.scatter(positions[i] + jitter[above],
                           np.full(int(above.sum()), hi), marker="^", s=24,
                           color=color, alpha=0.9, edgecolor="black",
                           linewidth=0.4, zorder=4, clip_on=False)
            if below.any():
                ax.scatter(positions[i] + jitter[below],
                           np.full(int(below.sum()), lo), marker="v", s=24,
                           color=color, alpha=0.9, edgecolor="black",
                           linewidth=0.4, zorder=4, clip_on=False)
            n_clipped += int(above.sum() + below.sum())
        else:
            ax.scatter(positions[i] + jitter, vals, s=14, color=color, alpha=0.7,
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(positions[i], means[i], marker="D", s=70, color="black",
                   zorder=5, label="group mean" if i == 0 else None)

    if thr is not None:
        ax.axhline(thr, color="#B00020", ls=(0, (6, 3)), lw=1.4, alpha=0.9,
                   zorder=1, label=f"threshold ({thr:g})")

    if ylim is not None:
        ax.set_ylim(*ylim)
        if n_clipped:
            ax.text(0.99, 0.015,
                    f"{n_clipped} outlier point(s) beyond axis (▲/▼ at edge)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="0.35")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in conditions])
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_xlabel("condition")
    style_axes(ax, grid_axis="y")
    ax.legend(loc="best", fontsize=8.5)
    if title is None:
        side = _METRIC_GOOD_SIDE.get(metric, "")
        title = f"{_METRIC_LABELS[metric]} by condition  (good = {side} threshold)"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def plot_quality_report(
    df: pd.DataFrame,
    metrics=("snr", "sci", "cv"),
    agg: str = "subject_channel_mean",
    order: Optional[list] = None,
    clip_outliers: bool = True,
    whis: float = 1.5,
    thresholds: Optional[dict] = None,
):
    """Convenience: produce one separate figure per metric.

    Returns a dict ``{metric: (fig, ax)}``. Each metric is a standalone figure so
    they can be saved/embedded independently in the validation notebook.

    The threshold reference line for each metric follows the values recorded by
    :func:`build_group_quality_table` on ``df`` (so changing thresholds there
    moves the lines automatically). Pass ``thresholds={"snr": 10, "cv": 15, ...}``
    to override per metric.

    ``clip_outliers`` / ``whis`` are forwarded to
    :func:`plot_quality_by_condition`; with the default ``clip_outliers=True`` the
    SNR and CV figures auto-scale to the data instead of being dominated by a few
    outlying channels.
    """
    thresholds = thresholds or {}
    out = {}
    for m in metrics:
        out[m] = plot_quality_by_condition(
            df, m, agg=agg, order=order,
            clip_outliers=clip_outliers, whis=whis,
            threshold=thresholds.get(m),
        )
    return out

def plot_quality_panel(
    df: pd.DataFrame,
    metrics=("snr", "sci", "cv"),
    agg: str = "subject_channel_mean",
    order: Optional[list] = None,
    clip_outliers: bool = True,
    whis: float = 1.5,
    thresholds: Optional[dict] = None,
    suptitle: Optional[str] = "Signal quality by condition",
):
    """The three quality indicators (SNR, SCI, CV) side by side on one figure.

    A single-row companion to :func:`plot_quality_report` (which returns one
    standalone figure per metric). Each panel reuses
    :func:`plot_quality_by_condition` — same adaptive y-limits, recorded
    thresholds, outlier markers and shared style — drawn onto a shared 1×N grid.

    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    thresholds = thresholds or {}
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, metrics):
        ptitle = str(m).upper()   # bare metric name: SNR / SCI / CV
        plot_quality_by_condition(
            df, m, agg=agg, order=order, ax=ax,
            clip_outliers=clip_outliers, whis=whis,
            threshold=thresholds.get(m), title=ptitle,
        )
    if suptitle:
        joined = " · ".join(str(m).upper() for m in metrics)
        fig.suptitle(f"{suptitle} - {joined}", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig, axes

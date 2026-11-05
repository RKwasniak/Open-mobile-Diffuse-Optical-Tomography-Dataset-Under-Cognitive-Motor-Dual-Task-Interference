"""Visualization functions for fNIRS data using cedalion plotting API."""
from __future__ import annotations
from pathlib import Path

import pyvista as pv
import xarray as xr
import numpy as np

import cedalion.dataclasses as cdc
import cedalion.vis.blocks as vbx

def _to_uM(da: xr.DataArray) -> xr.DataArray:
    """Plain values in µM. cedalion's image reconstruction already returns µM;
    unquantified input is taken as µM."""
    if getattr(da.pint, "units", None) is not None:
        return da.pint.to("uM").pint.dequantify()
    return da

# ---------------------------------------------------------------------------
# 1. Group brain maps (animated GIF)
# ---------------------------------------------------------------------------

def plot_group_brain_map(
    all_results: dict,
    *,
    trial_type: str,
    view_type: str = "hbo_brain",
    view_positions=("left", "right", "top"),   # 1-3 camera views
    scalar_bar_title: str | None = None,        # colour-bar title
    session: str | None = None,
    group: int | None = None,
    tag: str | None = None,                     # file-name / title label
    img_key: str = "img_brain_bl",
    title: str | None = None,
    out_dir="figs",
    time_range: tuple = (-2, 19, 0.5),
    clim: tuple | None = None,
    fps: int = 6,
    trial_types=("MC", "REST"),
):
    """Group-average brain map for one trial type, as an animated GIF.

    ``tag`` labels the file name and title (default: the active filters, or
    ``"all"``); pass e.g. the condition when ``all_results`` is pre-filtered.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- filter ------------------------------------------------------
    checks = []
    if session    is not None: checks.append(("session",    session))
    if group      is not None: checks.append(("group",      int(group)))

    if checks:
        def flt(sid, res, c=checks):
            return all(res.get(k) == v for k, v in c)
        auto_tag = "_".join(f"{k}-{v}" for k, v in checks)
    else:
        flt, auto_tag = None, "all"
    tag = tag or auto_tag

    # ---- group average -----------------------------------------------
    group_img, n = group_average_image(
        all_results, img_key=img_key, subject_filter=flt, trial_types=trial_types,
    )
    if trial_type not in group_img.trial_type.values:
        raise ValueError(f"trial_type {trial_type!r} not found.")

    X_ts = group_img.sel(trial_type=trial_type)
    if "reltime" in X_ts.dims:
        X_ts = X_ts.rename({"reltime": "time"})
    X_ts = X_ts.transpose("vertex", "chromo", "time")

    chromo_label = "HbO" if "hbo" in view_type else "HbR"

    # auto clim
    if clim is None:
        is_brain = (X_ts.is_brain == True).values
        vals = X_ts.sel(chromo=chromo_label).pint.dequantify().values[is_brain, :]
        scl = float(np.percentile(np.abs(vals), 99))
        if scl < 1e-12:
            scl = 0.3
        clim = (-scl, scl)
        print(f"[group plot] auto clim = ({-scl:.3g}, {scl:.3g})")

    # ---- geometry (shared by all runs; taken from the first) ---------
    ref           = next(iter(all_results.values()))
    head_ijk      = ref["head_ijk"]
    geo3d_snapped = ref["geo3d_snapped"]

    # ---- colour-bar title --------------------------------------------
    if scalar_bar_title is None:
        scalar_bar_title = f"{trial_type} - {chromo_label} [µM]"

    if title is None:
        title = f"Group {chromo_label} — {trial_type} — {tag} (N={n})"
    fname = str(out_dir / f"group_{tag}_N{n}_{trial_type}_{view_type}")

    plot_brain_recon_gif(
        X_ts, head_ijk, geo3d_snapped,
        chromo=chromo_label,
        view_positions=view_positions,
        time_range=time_range,
        clim=clim,
        fps=fps,
        filename=fname,
        scalar_bar_title=scalar_bar_title,
        title_str=title,
    )
    print(f"[group plot] {tag}: N={n}, saved → {fname}.gif")
    return fname

def plot_brain_recon_gif(
    X_ts,
    head,
    geo3d_snapped,
    *,
    chromo: str = "HbO",
    view_positions=("left",),
    time_range: tuple = (-2, 19, 0.5),
    cmap: str = "seismic",
    clim: tuple | None = None,
    wdw_size: tuple | None = None,
    drop_landmarks: bool = True,
    fps: int = 6,
    filename: str = "brain_recon",
    scalar_bar_title: str | None = None,
    title_str: str | None = None,
    show_time_label: bool = True,
    background: str = "white",
):
    """Animated GIF of a reconstructed image from 1, 2 or 3 camera views.

    Renders off-screen over a time window and writes a ``.gif``.

    Parameters
    ----------
    X_ts : xr.DataArray
        Dims (vertex, chromo, time|reltime) with an ``is_brain`` coordinate.
        If ``trial_type`` is present, select it before the call,
        e.g. ``X_ts = img.sel(trial_type="MC")``.
    head : TwoSurfaceHeadModel
    geo3d_snapped : xr.DataArray
    chromo : 'HbO' | 'HbR'
    view_positions : str | sequence
        1-3 views from {'left','right','frontal','back','top','superior'},
        rendered as a single row of subplots.
    time_range : (start, stop, step) [s]. The nearest samples are used;
        duplicates are dropped, order is kept.
    clim : (vmin, vmax) or None for a symmetric ±99th percentile over the
        brain vertices in the time window.
    wdw_size : (w, h) or None -> (520*n_views, 600).
    fps : GIF frames per second.
    filename : output path, with or without ``.gif`` (added if missing).
    scalar_bar_title : text next to the colour bar, e.g. the experimental
        condition. Default ``"<chromo> [<unit>]"``.
    title_str : title on the first subplot.
    show_time_label : stamp ``t = …s`` on every frame.
    background : background colour.

    Returns
    -------
    str : path of the saved GIF.
    """
    # ---- views --------------------------------------------------------
    if isinstance(view_positions, str):
        view_positions = (view_positions,)
    view_positions = list(view_positions)
    n_views = len(view_positions)
    if not (1 <= n_views <= 3):
        raise ValueError(
            f"view_positions must have 1, 2 or 3 elements; got {n_views}"
        )
    if wdw_size is None:
        wdw_size = (520 * n_views, 600)

    # ---- chromophore + frame indices ---------------------------------
    time_dim = "time" if "time" in X_ts.dims else "reltime"
    data = X_ts.sel(chromo=chromo)
    t_vals = data[time_dim].values

    t_start, t_stop, t_step = time_range
    targets = np.arange(t_start, t_stop + 1e-9, t_step)
    raw_idx = [int(np.abs(t_vals - t).argmin()) for t in targets]
    seen, frame_idx = set(), []
    for i in raw_idx:
        if i not in seen:
            seen.add(i); frame_idx.append(i)
    if not frame_idx:
        raise ValueError("time_range gave 0 frames.")

    is_brain = (data.is_brain == True).values

    # pint units, before dequantifying
    try:
        units = data.pint.units
    except Exception:
        units = None
    data_plain = data.pint.dequantify()

    # ---- clim ---------------------------------------------------------
    if clim is None:
        window = data_plain.isel({time_dim: frame_idx}).values[is_brain, :]
        scl = float(np.percentile(np.abs(window), 99))
        clim = (-scl, scl) if scl > 1e-12 else (-0.3, 0.3)

    # ---- colour-bar title --------------------------------------------
    if scalar_bar_title is None:
        if units is not None:
            try:
                unit_str = f"{units:~P}"      # pint short-pretty
            except Exception:
                unit_str = str(units)
            scalar_bar_title = f"{chromo} [{unit_str}]"
        else:
            scalar_bar_title = chromo

    # ---- mesh ---------------------------------------------------------
    brain_vtk = cdc.VTKSurface.from_trimeshsurface(head.brain)
    brain_pv  = pv.wrap(brain_vtk.mesh)
    brain_pv["reco"] = data_plain.isel({time_dim: frame_idx[0]}).values[is_brain]

    if drop_landmarks:
        geo3d_snapped = geo3d_snapped[geo3d_snapped.type != cdc.PointType.LANDMARK]

    # ---- camera ------------------------------------------------------
    cog = head.brain.vertices.mean("label").values
    offsets = {
        "left":     np.array([-350,   50, -50]),
        "right":    np.array([ 350,   50, -50]),
        "frontal":  np.array([   0,  300, 150]),
        "back":     np.array([   0, -300, 150]),
        "top":      np.array([   0,    0, 400]),
        "superior": np.array([   0,    0, 400]),
    }

    # ---- plotter with subplots ---------------------------------------
    plotter = pv.Plotter(shape=(1, n_views), window_size=wdw_size, off_screen=True)
    plotter.set_background(background)

    for i, view in enumerate(view_positions):
        plotter.subplot(0, i)
        plotter.add_mesh(
            brain_pv, scalars="reco", cmap=cmap, clim=clim,
            show_scalar_bar=False,
        )
        vbx.plot_labeled_points(plotter, geo3d_snapped, show_labels=False)

        off = offsets.get(view, offsets["left"])
        plotter.camera.position    = cog + off
        plotter.camera.focal_point = cog
        plotter.camera.up = [0, 1, 0] if view in ("top", "superior") else [0, 0, 1]
        plotter.reset_camera()

        plotter.add_text(view, font_size=10, position="upper_right", color="black")

    # one shared colour bar, bottom of the last subplot
    plotter.subplot(0, n_views - 1)
    plotter.add_scalar_bar(
        title=scalar_bar_title,
        n_labels=5,
        position_x=0.20, position_y=0.04,
        width=0.60, height=0.06,
        title_font_size=14,
        label_font_size=12,
        color="black",
    )

    if title_str:
        plotter.subplot(0, 0)
        plotter.add_text(title_str, font_size=12, position="upper_edge", color="black")

    # ---- write GIF ---------------------------------------------------
    out_path = Path(str(filename))
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.open_gif(str(out_path), fps=fps)

    for t_idx in frame_idx:
        brain_pv["reco"] = data_plain.isel({time_dim: t_idx}).values[is_brain]
        t_actual = float(t_vals[t_idx])

        if show_time_label:
            for i in range(n_views):
                plotter.subplot(0, i)
                plotter.add_text(
                    f"t = {t_actual:5.1f} s",
                    name=f"t_label_{i}",     # replaces the previous label
                    font_size=11,
                    position="lower_left",
                    color="black",
                )

        plotter.write_frame()

    plotter.close()
    print(f"[gif] saved → {out_path}  ({len(frame_idx)} frames @ {fps} fps)")
    return str(out_path)

def group_average_image(
    all_results: dict,
    *,
    img_key: str = "img_brain_bl",
    subject_filter=None,
    trial_types=("MC", "REST"),
    chromo=("HbO", "HbR"),
):
    """Iteratively average reconstructed images across subjects.

    RAM-friendly: keeps one subject in memory at a time. Drops extra
    trial_type/chromo levels not listed, so subjects with stray events
    (e.g. unmapped '4') don't break broadcasts.
    """
    trial_types = list(trial_types)
    chromo = list(chromo)
    accum, template, is_brain, units_, n = None, None, None, None, 0

    for sid, res in all_results.items():
        if subject_filter is not None and not subject_filter(sid, res):
            continue
        img = res[img_key]

        if "trial_type" in img.coords:
            present = [t for t in trial_types if t in img.trial_type.values]
            if present != trial_types:
                missing = set(trial_types) - set(present)
                if missing:
                    print(f"[group_avg] {sid}: missing {missing}, skipping")
                    continue
            extra = [t for t in img.trial_type.values if t not in trial_types]
            if extra:
                print(f"[group_avg] {sid}: dropping trial_type {extra}")
            img = img.sel(trial_type=trial_types)

        if "chromo" in img.coords:
            img = img.sel(chromo=[c for c in chromo if c in img.chromo.values])

        img_plain = img.pint.dequantify().astype("float32")

        if accum is None:
            accum    = img_plain.values.copy()
            template = img_plain
            units_   = img.pint.units
            if "is_brain" in img.coords:
                is_brain = img.is_brain
        else:
            if img_plain.shape != accum.shape:
                raise ValueError(
                    f"Subject {sid}: shape {img_plain.shape} != accum {accum.shape}"
                )
            accum += img_plain.values
        n += 1

    if n == 0:
        raise ValueError("No subjects matched the filter.")

    accum /= n
    img_mean = xr.DataArray(
        accum, dims=template.dims, coords=template.coords,
        attrs=template.attrs, name=template.name,
    )
    if is_brain is not None and "is_brain" not in img_mean.coords:
        img_mean = img_mean.assign_coords(is_brain=is_brain)
    if units_ is not None:
        img_mean = img_mean.pint.quantify(units_)
    return img_mean, n

# ---------------------------------------------------------------------------
# 2. Window-averaged effect map (static 3D)
# ---------------------------------------------------------------------------

def _roi_boundary_meshes(mesh, roi_outlines, head_labeled, parcel_coord):
    """[(label, colour, edge_mesh)] outlining each ROI on the cortical mesh.

    Uses the parcel labels of ``head_labeled`` (brain-vertex order, which matches
    the mesh points) and keeps only the boundary edges, so the effect colours
    inside the ROI stay visible.
    """
    if not roi_outlines:
        return []
    if head_labeled is None:
        raise ValueError("roi_outlines needs head_labeled (a Brodmann-labelled head)")

    from .atlas_brodmann import BRODMANN_ROIS, brodmann_labels_for_head

    labels = np.asarray(brodmann_labels_for_head(head_labeled, parcel_coord)).astype(str)
    if labels.size != mesh.n_points:
        raise ValueError(
            f"parcel labels ({labels.size}) do not match mesh points "
            f"({mesh.n_points}); head model and image are inconsistent.")

    out = []
    for spec in roi_outlines:
        roi, hemi, colour = spec[0], spec[1], spec[2]
        name = spec[3] if len(spec) > 3 else f"{roi} {hemi}"
        bas = BRODMANN_ROIS[roi] if isinstance(roi, str) else list(roi)
        hemis_ = ["left", "right"] if hemi == "both" else [hemi]
        wanted = {f"{h}_BA{n}" for h in hemis_ for n in bas}
        idx = np.flatnonzero(np.isin(labels, list(wanted)))
        if idx.size == 0:
            print(f"[effect3d] WARNING {name}: no vertices matched {sorted(wanted)}")
            continue
        try:
            sub = mesh.extract_points(idx, adjacent_cells=False)
            edges = sub.extract_surface().extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False, non_manifold_edges=False)
        except Exception as exc:                      # pragma: no cover
            print(f"[effect3d] {name}: outline failed ({type(exc).__name__}: {exc})")
            continue
        if edges.n_points == 0:
            print(f"[effect3d] {name}: empty outline, skipped")
            continue
        out.append((name, colour, edges))
    return out

def window_average_image(
    all_results: dict,
    *,
    window: tuple = (0.0, 17.0),
    img_key: str = "img_brain_bl",
    condition: str | None = None,
    condition_key: str = "condition",
    trial_types=("MC", "REST"),
    chromo=("HbO", "HbR"),
):
    """Group mean of the reconstruction averaged over ``reltime in window``
    (same window as the ROI means). Returns ``(img, n_subjects)``, dims
    ``(chromo, vertex, trial_type)``, in µM."""
    flt = None
    if condition is not None:
        def flt(sid, res, c=condition, k=condition_key):
            return res.get(k) == c

    img_mean, n = group_average_image(
        all_results, img_key=img_key, subject_filter=flt,
        trial_types=trial_types, chromo=chromo,
    )
    data = _to_uM(img_mean)
    return data.sel(reltime=slice(*window)).mean("reltime"), n

def plot_window_average_map(
    all_results: dict,
    head,
    geo3d_snapped=None,
    *,
    window: tuple = (0.0, 17.0),
    chromo: str = "HbO",
    trial_type: str = "MC",
    contrast: tuple | None = None,
    condition: str | None = None,
    condition_key: str = "condition",
    img_key: str = "img_brain_bl",
    views=("oblique",),
    camera: tuple = (-45.0, 25.0, 380.0),
    clim: tuple | None = None,
    cmap: str = "seismic",
    percentile: float = 99.0,
    filename=None,
    off_screen: bool = True,
    wdw_size: tuple = (950, 800),
    title: str | None = None,
    show_optodes: bool = False,
    roi_outlines=None,
    head_labeled=None,
    parcel_coord: str = "parcel_brodmann",
    outline_width: float = 4.0,
):
    """Static 3D cortical map of the group-mean change averaged over ``window``.

    Parameters
    ----------
    trial_type : str
        Which trial type to show when ``contrast`` is None.
    contrast : (str, str), optional
        ``("MC", "REST")`` renders the difference MC − REST instead.
    condition : str, optional
        Restrict to subjects whose ``condition_key`` equals this (e.g. "SITTING").
    camera : (azimuth, elevation, distance)
        Azimuth 0 = front (+Y), +90 = right (+X), -90 = left (-X).
    clim : (lo, hi), optional
        Colour limits; by default symmetric around 0 at the given percentile.
        Pass the *same* value across a figure set so the maps are comparable.
    roi_outlines : sequence, optional
        ``(roi, hemisphere, colour)`` or ``(roi, hemisphere, colour, label)`` per
        ROI, drawn as boundary outlines on top of the map. Requires
        ``head_labeled``.
    head_labeled : TwoSurfaceHeadModel, optional
        Brodmann-labelled head model supplying ``parcel_coord`` (needed for
        ``roi_outlines``).

    Returns
    -------
    Path or None
        The saved PNG when rendering off-screen, otherwise None.
    """
    import pyvista as pv

    wanted = tuple(contrast) if contrast else (trial_type,)
    avg, n = window_average_image(
        all_results, window=window, img_key=img_key, condition=condition,
        condition_key=condition_key,
        trial_types=tuple(dict.fromkeys(wanted)) if contrast else ("MC", "REST"),
    )
    avg = avg.sel(chromo=chromo)

    if contrast:
        vals = (avg.sel(trial_type=contrast[0]) - avg.sel(trial_type=contrast[1])).values
        what = f"{contrast[0]} − {contrast[1]}"
    else:
        vals = avg.sel(trial_type=trial_type).values
        what = str(trial_type)
    vals = np.asarray(vals, dtype=float).ravel()

    is_brain = (np.asarray(avg.is_brain.values, dtype=bool)
                if "is_brain" in avg.coords
                else np.ones(vals.size, dtype=bool))
    brain_vals = vals[is_brain]

    if clim is None:
        scl = float(np.nanpercentile(np.abs(brain_vals), percentile))
        if not np.isfinite(scl) or scl < 1e-12:
            scl = 1.0
        clim = (-scl, scl)

    mesh = pv.wrap(cdc.VTKSurface.from_trimeshsurface(head.brain).mesh)
    mesh["effect"] = brain_vals
    cog = np.asarray(head.brain.vertices.mean("label").values,
                     dtype=float).ravel()[:3]

    # ROI boundaries, drawn on top so the effect colours stay visible
    outlines = _roi_boundary_meshes(mesh, roi_outlines, head_labeled, parcel_coord)

    az0, el0, dist = camera
    presets = {"oblique": (az0, el0), "frontal": (0, 10), "left": (-90, 0),
               "right": (90, 0), "top": (0, 80), "back": (180, 10)}

    label = f"Δ{chromo} / µM"
    p = pv.Plotter(off_screen=off_screen, shape=(1, len(views)),
                   window_size=(wdw_size[0] * len(views), wdw_size[1]))
    p.set_background("white")
    for i, view in enumerate(views):
        p.subplot(0, i)
        p.add_mesh(mesh.copy(), scalars="effect", cmap=cmap, clim=clim,
                   smooth_shading=True,
                   scalar_bar_args=dict(title=label, n_labels=3,
                                        title_font_size=12, label_font_size=10,
                                        color="#222222", vertical=False,
                                        position_x=0.25, position_y=0.02,
                                        width=0.5))
        for name, colour, edges in outlines:
            p.add_mesh(edges.copy(), color=colour, line_width=outline_width,
                       render_lines_as_tubes=True, lighting=False)
        if show_optodes and geo3d_snapped is not None:
            geo = geo3d_snapped[geo3d_snapped.type != cdc.PointType.LANDMARK]
            vbx.plot_labeled_points(p, geo, show_labels=False)
        az, el = presets.get(view, (az0, el0))
        a, e = np.radians(az), np.radians(el)
        p.camera.position = cog + np.array([dist * np.cos(e) * np.sin(a),
                                            dist * np.cos(e) * np.cos(a),
                                            dist * np.sin(e)])
        p.camera.focal_point = cog
        p.camera.up = [0, 0, 1]
        p.reset_camera()
        if len(views) > 1:
            p.add_text(view, font_size=10, color="#222222")

    if title is None:
        cond = f" · {condition}" if condition else ""
        title = (f"{chromo} · {what}{cond} · {window[0]:g}–{window[1]:g} s "
                 f"(N={n})")
    p.subplot(0, 0)
    p.add_text(title, position="upper_edge", font_size=11, color="#222222")
    if outlines:
        p.add_legend([(name, colour) for name, colour, _ in outlines],
                     bcolor="white", face="line", size=(0.26, 0.10),
                     loc="upper left")

    print(f"[effect3d] {title} | clim=({clim[0]:.3g}, {clim[1]:.3g}) µM")
    if off_screen:
        out = Path(filename or "effect_map")
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        p.screenshot(str(out)); p.close()
        print(f"[effect3d] saved {out}")
        return out
    p.show()
    return None

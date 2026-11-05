"""Signal processing functions for fNIRS pipeline using cedalion."""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
import cedalion.nirs.cw
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units

# ---------------------------------------------------------------------------
# 1. Rename stimulus events
# ---------------------------------------------------------------------------

def rename_events(rec, mapping: dict[str, str]) -> None:
    """Rename stimulus/event trial_type labels in-place.

    Parameters
    ----------
    rec : cedalion.dataclasses.Recording
        Recording container whose ``stim`` DataFrame will be modified.
    mapping : dict
        Old name -> new name, e.g. {"1": "Control", "2": "FTapping/Left"}.

    Example
    -------
    >>> rename_events(rec, {"1": "Control", "2": "Task"})
    """
    rec.stim.cd.rename_events(mapping)

# ---------------------------------------------------------------------------
# 2. Convert raw amplitude to optical density
# ---------------------------------------------------------------------------

def convert_to_od(rec, method: str = "cedalion", nonpositive: str = "error",
                  interpolate: bool = False) -> xr.DataArray:
    """Convert raw amplitude data to optical density.

    method : {"cedalion", "manual"}  -- both compute OD = -log(I / mean(I)).
    nonpositive : {"error", "abs_floor", "nan"} -- how to handle samples <= 0
        before the log (repair is opt-in, never silent):
          "error"     : keep cedalion's strict behaviour (raise on any <= 0).
          "abs_floor" : abs(negatives) + floor zeros to the smallest positive
                        value (replicates MNE's optical_density; no NaNs).
          "nan"       : set non-positive samples to NaN (cedalion's suggestion);
                        pair with interpolate=True so gaps don't propagate.
    interpolate : if True, interpolate NaNs over time in the resulting OD.

    Stores the result in rec["od"] and returns it.
    """
    amp = rec["amp"]
    n_bad = int((amp <= 0).sum())

    if nonpositive == "error":
        pass
    elif nonpositive == "abs_floor":
        if n_bad:
            amp = np.abs(amp)
            floor = amp.where(amp > 0).min().item()
            if not np.isfinite(floor):
                raise ValueError(
                    "convert_to_od: amplitude data has no positive values at all."
                )
            amp = amp.where(amp > 0, floor)
            print(f"  [convert_to_od] abs_floor: repaired {n_bad} non-positive "
                  f"sample(s) (floor={floor:.4g})")
    elif nonpositive == "nan":
        if n_bad:
            amp = amp.where(amp > 0, np.nan)
            print(f"  [convert_to_od] nan: set {n_bad} non-positive sample(s) to NaN")
    else:
        raise ValueError(
            f"Unknown nonpositive: {nonpositive!r}. Use 'error', 'abs_floor' or 'nan'."
        )

    if method == "cedalion":
        od = cedalion.nirs.cw.int2od(amp)
    elif method == "manual":
        od = -np.log(amp / amp.mean("time"))
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'cedalion' or 'manual'.")

    if interpolate and bool(np.isnan(od).any()):
        n_nan = int(np.isnan(od).sum())
        od = od.interpolate_na("time").ffill("time").bfill("time")
        print(f"  [convert_to_od] interpolated {n_nan} NaN sample(s) in OD")

    rec["od"] = od
    return rec["od"]

# ---------------------------------------------------------------------------
# 3. Signal quality and channel pruning
# ---------------------------------------------------------------------------

def prune_channels(
    rec,
    snr_thresh: float | None = 16.0,
    sci_thresh: float | None = 0.75,
    sci_window=5 * units.s,
    psp_thresh: float | None = 0.1,
    perc_time_clean: float = 0.6,        # min. fraction of clean windows
    sd_threshs: Optional[list] = None,
    amp_threshs: Optional[list] = None,
    operator: str = "all",
) -> tuple[xr.DataArray, list]:
    """Prune channels by combining multiple quality masks.

    Window-based masks (SCI, PSP) are aggregated over time before pruning:
    a channel passes if at least ``perc_time_clean`` fraction of its
    windows exceed the metric threshold. Otherwise a single noisy window
    would discard the entire channel.
    """
    amp = rec["amp"]
    masks = []

    if snr_thresh is not None:
        # SNR mask is already 1D (per channel) – no aggregation needed
        _, snr_mask = quality.snr(amp, snr_thresh)
        masks.append(snr_mask)

    if sci_thresh is not None:
        _, sci_mask = quality.sci(amp, sci_window, sci_thresh)
        # collapse (channel, time) -> (channel,)
        sci_mask_ch = sci_mask.mean("time") >= perc_time_clean
        masks.append(sci_mask_ch)

    if psp_thresh is not None:
        _, psp_mask = quality.psp(amp, sci_window, psp_thresh)
        psp_mask_ch = psp_mask.mean("time") >= perc_time_clean
        masks.append(psp_mask_ch)

    if sd_threshs is not None:
        _, sd_mask = quality.sd_dist(amp, rec.geo3d, sd_threshs)
        masks.append(sd_mask)

    if amp_threshs is not None:
        _, amp_mask = quality.mean_amp(amp, amp_threshs)
        masks.append(amp_mask)

    amp_pruned, drop_list = quality.prune_ch(amp, masks, operator)
    return amp_pruned, drop_list

def drop_channels(ts: xr.DataArray, channels) -> xr.DataArray:
    """``ts`` without the listed channels (e.g. the drop list of :func:`prune_channels`)."""
    drop = {str(ch) for ch in channels}
    keep = np.array([str(ch) not in drop for ch in ts.channel.values])
    return ts.isel(channel=np.flatnonzero(keep))

# ---------------------------------------------------------------------------
# 4. Short-channel regression
# ---------------------------------------------------------------------------

def short_channel_regression(
    rec: cdc.Recording,
    od_key: str = "od",
    max_dist: cedalion.Quantity = 8 * units.mm,
    bad_channels=None,
) -> tuple[xr.DataArray, dict]:
    """Regress the short-channel signal out of every long channel.

    Scholkmann et al. 2014 (Eqs. 26-27), per wavelength on OD::

        alfa = (A_s · A_l) / (A_s · A_s)
        A_l_corrected = A_l - alfa * A_s

    ``A_s`` is the nearest short channel (distance between channel midpoints).
    If that short channel is in ``bad_channels`` (pruned), ``A_s`` is the mean of
    the surviving short channels, so every long channel is corrected.

    Channels with source-detector distance ``<= max_dist`` (max 10 mm) are short;
    they are dropped from the output. Raises ``RuntimeError`` if there are no
    short channels or none survived pruning.

    Returns ``(od_corrected, info)``. ``info``: ``n_short``, ``n_short_ok``,
    ``n_long``, ``short_channels``, ``short_pruned``, ``long_channels``,
    ``long_mean_regressor`` (long channels corrected with the mean),
    ``max_dist_mm``.
    """
    max_dist_mm = float(max_dist.to("mm").magnitude)
    if max_dist_mm > 10.0:
        raise ValueError(f"max_dist must be <= 10 mm (got {max_dist_mm} mm).")
    if od_key not in rec.timeseries:
        raise KeyError(f"rec[{od_key!r}] not found. Run OD conversion first.")

    od = rec[od_key]
    geo3d = rec.geo3d

    src_pos = geo3d.sel(label=od.source.values).pint.to("mm").pint.dequantify().values
    det_pos = geo3d.sel(label=od.detector.values).pint.to("mm").pint.dequantify().values
    sd_dists = np.linalg.norm(src_pos - det_pos, axis=1)         # (n_ch,)
    ch_mid = 0.5 * (src_pos + det_pos)                           # (n_ch, 3)

    short_mask = sd_dists <= max_dist_mm
    long_mask = ~short_mask
    n_short, n_long = int(short_mask.sum()), int(long_mask.sum())
    if n_short == 0:
        raise RuntimeError(f"No short channels found with distance <= {max_dist_mm} mm.")
    if n_long == 0:
        raise RuntimeError("No long channels found.")

    short_channels = od.channel.values[short_mask]
    long_channels = od.channel.values[long_mask]

    bad = {str(ch) for ch in (bad_channels if bad_channels is not None else [])}
    short_ok = np.array([str(ch) not in bad for ch in short_channels])
    if not short_ok.any():
        raise RuntimeError(f"All {n_short} short channels were pruned.")

    # nearest short channel per long channel; pruned -> mean of surviving shorts
    dist_ls = np.linalg.norm(
        ch_mid[long_mask][:, None, :] - ch_mid[short_mask][None, :, :], axis=-1
    )                                                            # (n_long, n_short)
    nearest_short_idx = np.argmin(dist_ls, axis=1)               # (n_long,)
    use_mean = ~short_ok[nearest_short_idx]                      # (n_long,)

    od_long = od.sel(channel=long_channels).transpose("channel", "wavelength", "time")
    od_short = od.sel(channel=short_channels).transpose("channel", "wavelength", "time")

    A_l = od_long.pint.dequantify().values                       # (n_long,  n_wl, n_t)
    A_s_all = od_short.pint.dequantify().values                  # (n_short, n_wl, n_t)
    A_s_mean = A_s_all[short_ok].mean(axis=0)                    # (n_wl, n_t)
    A_s = np.where(use_mean[:, None, None], A_s_mean[None],
                   A_s_all[nearest_short_idx])                   # (n_long,  n_wl, n_t)

    num = np.einsum("cwt,cwt->cw", A_s, A_l)
    den = np.einsum("cwt,cwt->cw", A_s, A_s)
    den_safe = np.where(den == 0.0, 1.0, den)
    alfa = num / den_safe                                        # (n_long, n_wl)

    A_l_corr = A_l - alfa[..., None] * A_s                       # (n_long, n_wl, n_t)

    od_corrected = xr.DataArray(
        A_l_corr,
        dims=od_long.dims,
        coords=od_long.coords,
        attrs=od_long.attrs,
        name=od_long.name,
    )
    if od_long.pint.units is not None:
        od_corrected = od_corrected.pint.quantify(od_long.pint.units)

    info = {
        "n_short": n_short,
        "n_short_ok": int(short_ok.sum()),
        "n_long": n_long,
        "short_channels": list(short_channels),
        "short_pruned": [str(ch) for ch in short_channels[~short_ok]],
        "long_channels": list(long_channels),
        "long_mean_regressor": [str(ch) for ch in long_channels[use_mean]],
        "max_dist_mm": max_dist_mm,
    }
    return od_corrected, info

# ---------------------------------------------------------------------------
# 5. Motion artifact correction — TDDR
# ---------------------------------------------------------------------------

def correct_motion_tddr(rec, od_key: str = "od") -> xr.DataArray:
    """Apply Temporal Derivative Distribution Repair (TDDR).

    Parameters
    ----------
    rec : cedalion.dataclasses.Recording
        Must contain rec[od_key].
    od_key : str
        Key of the OD time series in rec.

    Returns
    -------
    od_tddr : xr.DataArray
        Motion-corrected OD. Also stored in rec["od_tddr"].
    """
    rec["od_tddr"] = motion.tddr(rec[od_key])
    return rec["od_tddr"]

# ---------------------------------------------------------------------------
# 6. Bandpass frequency filter
# ---------------------------------------------------------------------------

def freq_filter(
    rec,
    input_key: str = "od_tddr",
    output_key: str = "od_filtered",
    fmin: float = 0.01,
    fmax: float = 0.5,
    butter_order: int = 4,
) -> xr.DataArray:
    """Apply Butterworth bandpass filter.

    Parameters
    ----------
    rec : cedalion.dataclasses.Recording
        Must contain rec[input_key].
    input_key : str
        Key of the input time series.
    output_key : str
        Key under which the filtered data will be stored.
    fmin, fmax : float
        Lower and upper cutoff frequencies in Hz.
    butter_order : int
        Filter order.

    Returns
    -------
    filtered : xr.DataArray
        Bandpass-filtered time series. Also stored in rec[output_key].
    """
    rec[output_key] = rec[input_key].cd.freq_filter(
        fmin=fmin, fmax=fmax, butter_order=butter_order
    )
    return rec[output_key]

# ---------------------------------------------------------------------------
# 7. Epoch extraction and block averaging
# ---------------------------------------------------------------------------

def compute_epochs(
    rec,
    ts_key: str = "od_filtered",
    trial_types: Optional[list[str]] = None,
    before: float = 5.0,
    after: float = 30.0,
) -> xr.DataArray:
    """Segment continuous data into stimulus-locked epochs.

    Parameters
    ----------
    rec : cedalion.dataclasses.Recording
        Must contain rec[ts_key] and rec.stim.
    ts_key : str
        Key of the time series to epoch.
    trial_types : list of str, optional
        Which trial_type values to keep. If None, use all.
    before, after : float
        Seconds before / after stimulus onset.

    Returns
    -------
    epochs : xr.DataArray
        Epoched data with dimensions (epoch, channel, wavelength, reltime).
    """
    if trial_types is None:
        trial_types = rec.stim["trial_type"].unique().tolist()

    epochs = rec[ts_key].cd.to_epochs(
        rec.stim,
        trial_types,
        before=before * units.s,
        after=after * units.s,
    )
    return epochs

def compute_block_average(
    epochs: xr.DataArray,
) -> xr.DataArray:
    """Baseline-correct epochs and compute block averages per trial_type.

    Parameters
    ----------
    epochs : xr.DataArray
        Output of ``compute_epochs``.

    Returns
    -------
    blockaverage : xr.DataArray
        Mean response per (trial_type, channel, wavelength, reltime).
    """
    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    epochs_blcorrected = epochs - baseline
    blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")
    return blockaverage

# ---------------------------------------------------------------------------
# 8. Head model, registration, forward model, sensitivity
# ---------------------------------------------------------------------------

def load_head_model(head_model_name: str = "colin27"):
    """Load a standard two-surface head model.

    Parameters
    ----------
    head_model_name : {"colin27", "icbm152"}
        Name of the standard head model.

    Returns
    -------
    head_ijk : cedalion.dot.TwoSurfaceHeadModel
        Head model in voxel (ijk) space.
    """
    import cedalion.dot as dot
    head_ijk = dot.get_standard_headmodel(head_model_name)
    return head_ijk

def register_optodes(head_ijk, geo3d_meas):
    """Register optode positions to the scalp surface.

    Uses landmark-based affine alignment followed by snapping to nearest
    scalp vertex.

    Parameters
    ----------
    head_ijk : TwoSurfaceHeadModel
        Head model in ijk space.
    geo3d_meas : xr.DataArray
        Measured 3D optode/landmark coordinates from the recording.

    Returns
    -------
    geo3d_snapped : xr.DataArray
        Optode coordinates snapped to the scalp surface in ijk space.
    """
    geo3d_snapped = head_ijk.align_and_snap_to_scalp(geo3d_meas)
    return geo3d_snapped

def create_forward_model(head_ijk, geo3d_snapped, meas_list):
    """Create a cedalion ForwardModel object.

    Parameters
    ----------
    head_ijk : TwoSurfaceHeadModel
        Head model in ijk space.
    geo3d_snapped : xr.DataArray
        Registered optode coordinates.
    meas_list : pd.DataFrame
        Measurement list from rec._measurement_lists["amp"].

    Returns
    -------
    fwm : cedalion.dot.ForwardModel
        Forward model ready for fluence computation.
    """
    import cedalion.dot as dot
    fwm = dot.ForwardModel(head_ijk, geo3d_snapped, meas_list)
    return fwm

def compute_fluence(
    fwm,
    fluence_path,
    method: str = "mcx",
    precomputed: bool = False,
):
    """Run photon propagation or load precomputed fluence.

    Store fluence files outside the git repository (they can be >1 GB).

    Parameters
    ----------
    fwm : cedalion.dot.ForwardModel
        Forward model (ignored when precomputed=True).
    fluence_path : str or Path
        Path to the fluence HDF5 file.
    method : {"mcx", "nirfaster"}
        Simulation back-end (ignored when precomputed=True).
    precomputed : bool
        If True, skip simulation and expect file at fluence_path.

    Returns
    -------
    fluence_path : Path
    """
    fluence_path = Path(fluence_path)

    if precomputed:
        if not fluence_path.exists():
            raise FileNotFoundError(
                f"precomputed=True but fluence file not found: {fluence_path}"
            )
        print(f"Using precomputed fluence: {fluence_path}")
        return fluence_path

    fluence_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "mcx":
        fwm.compute_fluence_mcx(fluence_path)
    elif method == "nirfaster":
        fwm.compute_fluence_nirfaster(fluence_path)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    print(f"Fluence saved: {fluence_path}")
    return fluence_path

def compute_sensitivity(
    fwm,
    fluence_path,
    sensitivity_path,
    precomputed: bool = False,
):
    """Compute or load the sensitivity (Adot) matrix.

    Parameters
    ----------
    fwm : cedalion.dot.ForwardModel
        Forward model (ignored when precomputed=True).
    fluence_path : str or Path
        Path to the fluence HDF5 file.
    sensitivity_path : str or Path
        Path to save / load the sensitivity HDF5 file.
    precomputed : bool
        If True, skip computation and load existing file.

    Returns
    -------
    Adot : xr.DataArray
    """
    from cedalion.io.forward_model import load_Adot

    sensitivity_path = Path(sensitivity_path)

    if precomputed:
        if not sensitivity_path.exists():
            raise FileNotFoundError(
                f"precomputed=True but sensitivity file not found: {sensitivity_path}"
            )
        print(f"Using precomputed sensitivity: {sensitivity_path}")
        return load_Adot(sensitivity_path)

    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    fwm.compute_sensitivity(Path(fluence_path), sensitivity_path)
    print(f"Sensitivity saved: {sensitivity_path}")
    return load_Adot(sensitivity_path)

def run_image_reconstruction(
    Adot,
    blockaverage,
    recon_mode: str = "mua2conc",
    brain_only: bool = False,
    alpha_meas: float = 0.01,
    alpha_spatial=None,
):
    """DOT image reconstruction of block averages.

    ``Adot`` is reduced to the channels in ``blockaverage`` before the inverse is
    computed, so pruned and short channels do not enter the reconstruction.
    ``recon_mode``: ``"mua2conc"`` (mua per wavelength, then HbO/HbR), ``"mua"``,
    ``"conc"``. Returns the image (chromo × vertex × trial_type × reltime).
    """
    import cedalion.dot as dot

    recon = dot.ImageRecon(
        Adot.sel(channel=blockaverage.channel.values),
        recon_mode=recon_mode,
        brain_only=brain_only,
        alpha_meas=alpha_meas,
        alpha_spatial=alpha_spatial,
        apply_c_meas=False,
        spatial_basis_functions=None,
    )
    img = recon.reconstruct(blockaverage)
    return img

# ---------------------------------------------------------------------------
# 9. Image baseline correction
# ---------------------------------------------------------------------------

def baseline_correct_image(img):
    """Baseline-correct reconstructed image using first time sample.

    Selects only brain vertices and subtracts the first reltime sample.

    Parameters
    ----------
    img : xr.DataArray
        Reconstructed image (chromo, vertex, trial_type, reltime).

    Returns
    -------
    img_brain_bl : xr.DataArray
        Baseline-corrected brain-only image.
    """
    img_brain = img.sel(vertex=img.is_brain)
    baseline = img_brain.isel(reltime=0)
    return img_brain - baseline


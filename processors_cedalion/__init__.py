"""Processors for cedalion-based fNIRS analysis pipeline."""
from .snirf_utils import add_landmarks_to_snirf, DEFAULT_LANDMARKS
from .processing import (
    rename_events,
    convert_to_od,
    prune_channels,
    drop_channels,
    short_channel_regression,
    correct_motion_tddr,
    freq_filter,
    compute_epochs,
    compute_block_average,
    load_head_model,
    register_optodes,
    create_forward_model,
    compute_fluence,
    compute_sensitivity,
    run_image_reconstruction,
    baseline_correct_image,
)
from .visualization import (
    plot_group_brain_map,
    group_average_image,
    window_average_image,
    plot_window_average_map,
)
# --- BIDS group import (Luke et al. 2025, NIRS-BIDS) ------------------------
from .bids_loader import (
    parse_bids_filename,
    find_nirs_files,
    read_participants,
    build_bids_manifest,
    load_bids_recording,
    iter_bids_recordings,
)
# --- Brodmann atlas + ROIs --------------------------------------------------
from .atlas_brodmann import (
    BRODMANN_ROIS,
    load_brodmann_atlas,
    assign_brodmann_to_head,
    relabel_image_parcels,
    select_roi_parcels,
    available_parcels_from_image,
)
# --- quality metrics (SNR / SCI / CV) for technical validation --------------
from .quality_metrics import (
    compute_cv,
    channel_quality_table,
    build_group_quality_table,
    summarize_group_quality,
    plot_quality_by_condition,
    plot_quality_report,
    plot_quality_panel,
)
# --- ROI analysis (FPA) -----------------------------------------------------
from .roi_analysis import (
    roi_timecourse,
    roi_window_mean,
    build_group_roi_table,
    plot_roi_averaged,
    plot_roi_contrast_bars,
)

__all__ = [
    # loading
    "add_landmarks_to_snirf",
    "DEFAULT_LANDMARKS",
    # processing
    "rename_events",
    "convert_to_od",
    "prune_channels",
    "drop_channels",
    "short_channel_regression",
    "correct_motion_tddr",
    "freq_filter",
    "compute_epochs",
    "compute_block_average",
    # DOT / sensitivity
    "load_head_model",
    "register_optodes",
    "create_forward_model",
    "compute_fluence",
    "compute_sensitivity",
    "run_image_reconstruction",
    "baseline_correct_image",
    # visualization
    "plot_group_brain_map",
    "group_average_image",
    "window_average_image",
    "plot_window_average_map",
    # BIDS
    "parse_bids_filename",
    "find_nirs_files",
    "read_participants",
    "build_bids_manifest",
    "load_bids_recording",
    "iter_bids_recordings",
    # Brodmann atlas + ROIs
    "BRODMANN_ROIS",
    "load_brodmann_atlas",
    "assign_brodmann_to_head",
    "relabel_image_parcels",
    "select_roi_parcels",
    "available_parcels_from_image",
    # quality metrics
    "compute_cv",
    "channel_quality_table",
    "build_group_quality_table",
    "summarize_group_quality",
    "plot_quality_by_condition",
    "plot_quality_report",
    "plot_quality_panel",
    # ROI analysis
    "roi_timecourse",
    "roi_window_mean",
    "build_group_roi_table",
    "plot_roi_averaged",
    "plot_roi_contrast_bars",
]

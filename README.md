# Diffuse Optical Tomography under Cognitive–Motor Dual-Task Interference

A reproducible **fNIRS / DOT group-analysis pipeline** built on
[cedalion](https://github.com/ibs-lab/cedalion), packaged as a small Python library
(`processors_cedalion/`) driven by a single technical-validation notebook
(`validation-cedalion.ipynb`).

It takes the NIRS-BIDS recordings of the dataset through preprocessing → **diffuse
optical tomography (DOT) image reconstruction** on a standard head model →
**Brodmann-atlas ROI analysis**, and reports the signal-quality and task-response
evidence that a *Technical Validation* section of a data descriptor needs.

---

## Highlights

- **NIRS-BIDS loader** (`bids_loader`) for the published dataset
  ([NIRS-BIDS](https://doi.org/10.1038/s41597-024-04136-9) layout).
- **Full cedalion pipeline** — OD conversion, SCI-based channel pruning, TDDR motion
  correction, band-pass, short-channel regression, epoching, block averaging,
  registration, forward model, fluence (**MCX** or **NIRFASTer**), sensitivity and
  regularized image reconstruction.
- **Brodmann parcellation** (`atlas_brodmann`) instead of a surface atlas, so ROIs
  are defined directly on Brodmann areas (`left_BA10`, `right_BA10`, …).
- **Signal quality, ROI responses (FPA) and 3D effect maps**, with figures
  sharing one visual style (`plotstyle`).

---

## Repository layout

```
.
├── validation-cedalion.ipynb      # the pipeline + all analyses
├── processors_cedalion/           # the analysis library
│   ├── snirf_utils.py             # inject fiducial landmarks into a SNIRF
│   ├── bids_loader.py             # NIRS-BIDS group import
│   ├── processing.py              # OD, pruning, motion, filtering, short channels, DOT
│   ├── atlas_brodmann.py          # Brodmann atlas assignment + ROI selection
│   ├── quality_metrics.py         # SNR / SCI / CV tables + per-condition figures
│   ├── roi_analysis.py            # ROI time courses, window means, figures
│   ├── visualization.py           # 3D effect maps and group GIFs (PyVista)
│   └── plotstyle.py               # shared publication figure style
├── requirements.txt
├── results/                       # tables and JASP files
└── figs/                          # figures
```

---

## Installation

The only real dependency is **cedalion** (not on PyPI); it pulls in the whole
scientific stack (numpy, pandas, xarray, matplotlib, h5py, pint-xarray, pyvista,
pmcx/MCX …). The library modules add **no** new dependencies.

The results were produced with **Python 3.13** and cedalion **26.5.2.dev3**
(commit `923c4276565b8ffc38b4c355c1719b733cfa1b15`).

**Recommended (conda — most robust for the VTK / OpenCV / MCX binaries):**

```bash
git clone https://github.com/ibs-lab/cedalion.git
cd cedalion
git checkout 923c4276565b8ffc38b4c355c1719b733cfa1b15
conda env create -n cedalion -f environment_dev.yml
conda activate cedalion
pip install -e .
# then, from this project's folder:
pip install -r requirements.txt   # already satisfied; documents direct deps
```

**Pip-only** (works, but `vtk` / `opencv-python` / `pmcx` wheels can be
platform-finicky):

```bash
pip install -r requirements.txt
```

---

## Quickstart

1. Download the dataset from OpenNeuro
   ([ds008544](https://openneuro.org/datasets/ds008544)) and point `BIDS_ROOT` at it.
2. Keep the precomputed forward model (`fluence_colin27.h5`,
   `sensitivity_colin27.h5`, > 1 GB) **outside** the repo and point `DATA_DIR` at
   it. To create the files, set `PRECOMPUTED = False`: the per-run loop then
   simulates them for every run, so set it back to `True` after the first run.
3. Open `validation-cedalion.ipynb`, edit the single **Configuration** cell
   (paths, condition mapping, regularization, event map) and run all cells.

---

## Pipeline (per subject)

```
rename events → optical density → quality / pruning (SCI) → TDDR motion correction
→ band-pass → short-channel regression → epoch → block-average → register optodes
→ forward model → fluence (MCX / NIRFASTer) → sensitivity → image reconstruction
(mua2conc) → baseline-correct → relabel vertices with Brodmann areas
```

The head model and its Brodmann labelling are computed **once** and reused for all
subjects. Baseline correction happens twice: on the epochs (mean of the
pre-stimulus window) and on the reconstructed image (first sample).

**Short-channel regression:** each long channel is regressed on its nearest short
channel; if that short channel was pruned, on the mean of the surviving short
channels. If no short channel survives, the run is left uncorrected.

---

## Analyses

| Part | What | Content |
|------|------|---------|
| **Signal quality** | SNR, SCI, CV per channel, then per participant (median over channels) | Table 1, SITTING vs WALKING tests (Shapiro–Wilk, Wilcoxon), figures per condition |
| **ROI responses** | FPA (BA10), both hemispheres | time courses, window means, MC − REST bars per condition |
| **Effect maps** | window-averaged HbO/HbR on the cortex | MC, REST, MC − REST, per condition, ROI outlines |

---

## Configuration knobs (in the notebook)

| Setting | Meaning |
|---------|---------|
| `BIDS_ROOT`, `BIDS_CONDITION_FROM` | dataset location; BIDS entity that holds the condition (`"task"`: SITTING / WALKING) |
| `SESSION_TO_CONDITION` | optional mapping of those labels onto the two conditions |
| `EVENT_MAP` | stimulus code → trial type, e.g. `{"1": "MC", "2": "REST"}` |
| `HEAD_MODEL` | `"icbm152"` or `"colin27"` |
| `FLUENCE_METHOD` | `"mcx"` or `"nirfaster"` photon-transport backend |
| `RECON_MODE` | image reconstruction mode (`"mua2conc"`) |
| `ALPHA_MEAS`, `ALPHA_SPATIAL` | image-reconstruction regularization |
| `RESP_WINDOW` | averaging window (s) for ROI means and effect maps |
| `ROIS` | ROI names from `BRODMANN_ROIS` (`"FPA"`) |
| quality thresholds | `sci_thresh`, `snr_thresh`, `cv_thresh` — drive both the pass/fail columns **and** the figure reference lines |

---

## Outputs

- **`results/`**
  - written by the notebook: `group_quality_metrics.csv`,
    `group_quality_summary.csv`, the per-participant quality tables
    (`subject_quality_summary.csv`, `subject_quality_wide.csv`), Table 1 and the
    quality tests (`quality/table.csv`, `quality/tests.csv`), and
    `group_roi_window.csv`;
  - JASP checks and the per-hemisphere FPA table: `subject_quality_wide.jasp`
    (quality tests), `group_roi_window_LH_RH.csv` and `roi_window_wide*.csv`
    analysed in `To article 19.10.jasp`.
- **`figs/`** — PNGs (300 dpi): signal quality per participant (`quality/`:
  `panel_median.png` is the figure of the paper; `*_mean.png` use participant
  means), ROI time courses, MC − REST bars, effect maps, and cortical activation
  GIFs per condition.

File names are kept short: GitHub's ZIP repeats the long repository name in the
extracted path, and Windows Explorer cannot extract paths longer than 260
characters.

---

## Notes & caveats

- **ROI definitions are editable.** FPA = {BA10};
  adjust `BRODMANN_ROIS` in `atlas_brodmann.py` to match your montage.
- Check the printed "Brodmann areas present on the cortical surface" list — a
  frontal-only montage will not cover posterior areas.

---

## References

- Cedalion — IBS-Lab: <https://github.com/ibs-lab/cedalion>
- Luke et al., *NIRS-BIDS*, **Sci Data** 12:159 (2025) —
  <https://doi.org/10.1038/s41597-024-04136-9>
- Mai & Majtanik, *Human Brain in Standard MNI Space: A Comprehensive Pocket Atlas*
  (2017) — Brodmann atlas bundled with cedalion.
- Gagnon et al., **NeuroImage** 59(3), 2518–2528 (2012) — short-channel regression.

---

## License

No license file is included yet. Add a `LICENSE` before publishing if you intend
others to reuse the code.

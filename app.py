import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    binary_opening,
    binary_closing,
    binary_dilation,
    remove_small_objects,
    disk,
    label,
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Incucyte-style Migration Analysis", layout="wide")

# Default timepoints (you can edit these in the UI)
DEFAULT_TIMEPOINTS_HOURS = [0, 24, 48, 72]

# Tunable analysis constants (these are the tuned values that gave us
# the closest match to the Incucyte table you shared)
CLAHE_CLIP_LIMIT = 0.05   # stronger local contrast boost for faint cells
GAUSSIAN_SIGMA   = 1      # mild blur to remove camera noise
THRESH_RELAX     = 0.85   # lower -> more permissive, higher -> stricter
MIN_CELL_AREA    = 30     # drop tiny noise
OPEN_DISK        = 2      # morphology open radius
CLOSE_DISK       = 2      # morphology close radius
BORDER_PX        = 20     # thickness of "border ring" around wound ROI


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def load_image_gray(uploaded_file: io.BytesIO) -> np.ndarray:
    """
    Load any brightfield TIFF/JPEG/etc using PIL (handles LZW TIFF),
    return grayscale float [0..1].
    """
    img = Image.open(uploaded_file).convert("RGB")
    arr_rgb = np.array(img)
    gray = rgb2gray(arr_rgb)  # skimage.rgb2gray -> float64 in [0,1]
    return gray


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Make faint/transparent cells more visible:
    1. Adaptive histogram equalization (CLAHE) with slightly higher clip_limit
    2. Light gaussian blur to smooth noise
    """
    eq = exposure.equalize_adapthist(gray, clip_limit=CLAHE_CLIP_LIMIT)
    smooth = gaussian(eq, sigma=GAUSSIAN_SIGMA)
    return smooth


def segment_cells(enhanced_gray: np.ndarray) -> np.ndarray:
    """
    Detect cells as a binary mask.
    Steps:
    - Otsu threshold
    - Relax it (multiply by THRESH_RELAX) so lighter cells survive
    - Morphological open/close
    - Remove small specks
    Returns boolean mask same shape as input.
    """
    otsu_val = threshold_otsu(enhanced_gray)
    relaxed_threshold = otsu_val * THRESH_RELAX

    raw_mask = enhanced_gray > relaxed_threshold

    opened = binary_opening(raw_mask, disk(OPEN_DISK))
    closed = binary_closing(opened, disk(CLOSE_DISK))
    cleaned = remove_small_objects(closed, min_size=MIN_CELL_AREA)

    return cleaned.astype(bool)


def get_wound_roi(cell_mask_t0: np.ndarray) -> np.ndarray:
    """
    Define the wound at T0 as the largest empty gap (no cells).
    We invert the mask to get 'empty', label connected regions,
    and take the biggest region as the wound ROI.
    """
    empty_regions = ~cell_mask_t0
    lbl = label(empty_regions)

    if lbl.max() == 0:
        # fallback: whole frame considered wound (edge case)
        return np.ones_like(cell_mask_t0, dtype=bool)

    # pick the largest connected empty area
    max_area = 0
    wound_region_id = None
    for region_id in range(1, lbl.max() + 1):
        area = np.sum(lbl == region_id)
        if area > max_area:
            max_area = area
            wound_region_id = region_id

    wound_roi = (lbl == wound_region_id)
    return wound_roi


def get_border_roi(wound_roi: np.ndarray) -> np.ndarray:
    """
    Build a 'border ring' around the wound for normalization.
    We dilate the wound mask by BORDER_PX and subtract the wound itself.
    """
    dilated = binary_dilation(wound_roi, disk(BORDER_PX))
    ring = np.logical_and(dilated, ~wound_roi)
    return ring


def compute_incucyte_metrics(cell_masks: List[np.ndarray],
                             wound_roi: np.ndarray,
                             border_roi: np.ndarray) -> pd.DataFrame:
    """
    Compute:
    1. Wound Confluence (%)
       = 100 * (#cell pixels inside wound / wound area)

    2. Relative Wound Density (%)
       Incucyte-style normalization:
       Let:
         W_t = wound cell density at time t
         W_0 = wound cell density at time 0
         B_0 = border cell density at time 0 (the intact monolayer)
       Then:
         RWD_t = 100 * ( (W_t - W_0) / (B_0 - W_0) )

       This forces:
         t=0  -> RWD = 0
         later -> approaches ~50-60% etc.

    We keep wound_roi fixed from T0 and apply to all timepoints.
    """
    wound_area = float(np.sum(wound_roi))
    border_area = float(np.sum(border_roi))

    # baseline (t=0)
    baseline_mask = cell_masks[0]
    w0_cells = np.sum(np.logical_and(baseline_mask, wound_roi))
    b0_cells = np.sum(np.logical_and(baseline_mask, border_roi))

    W_0 = (w0_cells / wound_area) if wound_area > 0 else 0.0
    B_0 = (b0_cells / border_area) if border_area > 0 else 1e-9

    records = []
    for idx, cm in enumerate(cell_masks):
        wound_cells_now = np.sum(np.logical_and(cm, wound_roi))
        border_cells_now = np.sum(np.logical_and(cm, border_roi))

        # Absolute wound confluence % at this time
        wound_confluence_pct = (
            100.0 * (wound_cells_now / wound_area) if wound_area > 0 else 0.0
        )

        # Density in wound right now
        W_t = (wound_cells_now / wound_area) if wound_area > 0 else 0.0

        # Relative Wound Density normalized to t0
        denom = (B_0 - W_0)
        if denom == 0:
            rel_wound_density_pct = 0.0
        else:
            rel_wound_density_pct = 100.0 * ((W_t - W_0) / denom)

        records.append(
            {
                "frame_idx": idx,
                "wound_confluence_pct": wound_confluence_pct,
                "relative_wound_density_pct": rel_wound_density_pct,
            }
        )

    df = pd.DataFrame(records)
    return df


def overlay_debug(rgb_img: np.ndarray,
                  wound_roi: np.ndarray,
                  border_roi: np.ndarray,
                  cell_mask: np.ndarray) -> np.ndarray:
    """
    Visualization to QC segmentation:
      - green: detected cells
      - red: wound ROI
      - yellow: border ROI
    """
    base = rgb_img.astype(np.float32) / 255.0

    cell_overlay = np.stack(
        [np.zeros_like(cell_mask),
         cell_mask.astype(float),
         np.zeros_like(cell_mask)],
        axis=-1,
    )

    wound_overlay = np.stack(
        [wound_roi.astype(float),
         np.zeros_like(wound_roi),
         np.zeros_like(wound_roi)],
        axis=-1,
    )

    border_overlay = np.stack(
        [border_roi.astype(float),
         border_roi.astype(float),
         np.zeros_like(border_roi)],
        axis=-1,
    )

    combo = np.clip(
        base * 0.6 + 0.4 * (cell_overlay + wound_overlay + border_overlay),
        0, 1
    )

    return (combo * 255).astype(np.uint8)


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title("Incucyte-style Wound Healing / Migration Metrics")

st.markdown(
    """
This tool:
1. Segments cells with a tuned pipeline that picks up faint cells.
2. Locks a wound mask from the first timepoint (T0).
3. Calculates:
   - **Wound Confluence (%)**
   - **Relative Wound Density (%)** (Incucyte-style formula)
4. Plots both over time.

Usage:
- Upload multiple brightfield images (same well, different times).
- We'll sort them by filename.
- Enter the timepoints (hours) in order.
"""
)

uploaded_files = st.file_uploader(
    "Upload time-series brightfield images (same well across time)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
)

time_input = st.text_input(
    "Comma-separated hours for each image in sorted order "
    "(default: 0,24,48,72):",
    value=",".join(map(str, DEFAULT_TIMEPOINTS_HOURS)),
)

run_btn = st.button("Run Analysis")

if run_btn and uploaded_files:
    # sort by filename for consistent ordering
    uploaded_sorted = sorted(uploaded_files, key=lambda f: f.name)

    # parse timepoints
    try:
        time_h = [float(x.strip()) for x in time_input.split(",")]
    except Exception:
        time_h = DEFAULT_TIMEPOINTS_HOURS[:]

    # pad / trim time list to match number of images
    while len(time_h) < len(uploaded_sorted):
        # extend assuming constant step
        if len(time_h) >= 2:
            step = time_h[-1] - time_h[-2]
        else:
            step = 12.0
        time_h.append(time_h[-1] + step)
    time_h = time_h[: len(uploaded_sorted)]

    # ----- Process images -----
    rgb_images = []
    enhanced_list = []
    cell_masks = []

    for f in uploaded_sorted:
        pil_img = Image.open(f).convert("RGB")
        rgb_arr = np.array(pil_img)
        rgb_images.append(rgb_arr)

        g = rgb2gray(rgb_arr)
        enh = enhance_contrast(g)
        enhanced_list.append(enh)

        cm = segment_cells(enh)
        cell_masks.append(cm)

    # ----- Build wound + border from T0 -----
    wound_roi = get_wound_roi(cell_masks[0])
    border_roi = get_border_roi(wound_roi)

    total_px = wound_roi.size
    wound_area_pct = 100.0 * np.sum(wound_roi) / total_px

    # ----- Compute metrics -----
    df_metrics = compute_incucyte_metrics(
        cell_masks=cell_masks,
        wound_roi=wound_roi,
        border_roi=border_roi,
    )

    df_metrics["filename"] = [f.name for f in uploaded_sorted]
    df_metrics["time_h"] = time_h

    st.subheader("Quantitative Results (our pipeline)")
    st.write(df_metrics)

    st.markdown(
        f"""
        Wound ROI area ~ {wound_area_pct:.2f}% of the image (from first frame).
        BORDER_PX used for border ring: {BORDER_PX} px.
        """
    )

    # ----- Plots -----
    fig1, ax1 = plt.subplots()
    ax1.plot(df_metrics["time_h"], df_metrics["wound_confluence_pct"], marker="o")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Wound Confluence (%)")
    ax1.set_title("Wound Confluence Over Time")
    ax1.set_ylim(0, 110)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(df_metrics["time_h"], df_metrics["relative_wound_density_pct"], marker="o")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Relative Wound Density (%)")
    ax2.set_title("Relative Wound Density Over Time")
    ax2.set_ylim(0, 120)
    st.pyplot(fig2)

    # ----- QC Overlays -----
    st.subheader("QC Overlays (red=wound, yellow=border, green=cells)")

    for idx, (rgb_arr, cm) in enumerate(zip(rgb_images, cell_masks)):
        ov = overlay_debug(rgb_arr, wound_roi, border_roi, cm)
        st.image(
            ov,
            caption=f"{uploaded_sorted[idx].name} @ {time_h[idx]} h",
            use_column_width=True,
        )

    # Optional: let user download CSV
    csv_bytes = df_metrics.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download our metrics as CSV",
        data=csv_bytes,
        file_name="our_incucyte_style_metrics.csv",
        mime="text/csv",
    )
else:
    st.info("Upload images and click 'Run Analysis'.")

import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    binary_dilation,
    disk,
    label,
)


# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Incucyte-style Migration Analysis", layout="wide")

# You can edit this if you know the capture time (in hours) for each image you upload.
# The assumption here is files[0] = time 0h, files[1] = 12h, etc.
DEFAULT_TIMEPOINTS_HOURS = [0, 12, 24, 36, 48]


# ----------------------------- IMAGE / MASK UTILS -----------------------------
def load_image_as_array(uploaded_file) -> np.ndarray:
    """Load uploaded image -> float grayscale [0..1]."""
    img = Image.open(uploaded_file).convert("RGB")
    gray = rgb2gray(np.array(img))  # 0..1 float
    return gray


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Make faint / lighter cells more visible.
    Steps:
    1. Adaptive histogram equalization (CLAHE via equalize_adapthist)
    2. Gentle gaussian blur to reduce noise
    Result stays 0..1 float.
    """
    # CLAHE: clip_limit small so we don't over-amplify noise
    eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    smooth = gaussian(eq, sigma=1)
    return smooth


def segment_cells(enhanced_gray: np.ndarray,
                  min_cell_area: int = 30,
                  thresh_relax: float = 0.9) -> np.ndarray:
    """
    Convert enhanced grayscale image -> binary mask of 'cells present'.

    - We assume cells are BRIGHTER than background.
    - Otsu gives a threshold; we relax it downward (multiply by thresh_relax < 1)
      so that even faint bright cells above background are counted.
    - Morphological open/close and small-object removal to clean noise.
    """
    otsu_val = threshold_otsu(enhanced_gray)
    relaxed = otsu_val * thresh_relax  # <-- more inclusive for faint cells

    raw_mask = enhanced_gray > relaxed

    # clean up mask
    opened = binary_opening(raw_mask, disk(2))
    closed = binary_closing(opened, disk(2))

    cleaned = remove_small_objects(closed, min_size=min_cell_area)

    return cleaned.astype(bool)


def get_primary_wound_roi(cell_mask_t0: np.ndarray,
                          min_wound_area_px: int = 5000) -> np.ndarray:
    """
    At time 0, wound = the big empty gap (no cells).
    Strategy:
      - Invert the cell mask -> "empty regions"
      - Label connected components
      - Pick the largest component above min_wound_area_px
      - That becomes the canonical wound ROI for ALL timepoints
    """
    empty_regions = ~cell_mask_t0

    lbl = label(empty_regions)
    if lbl.max() == 0:
        # no connected components found, fallback to entire image = wound (shouldn't happen in real scratch assays)
        return np.ones_like(cell_mask_t0, dtype=bool)

    wound_roi = np.zeros_like(cell_mask_t0, dtype=bool)

    # choose largest component that is big enough
    max_area = 0
    max_region_id = None
    for region_id in range(1, lbl.max() + 1):
        region_mask = lbl == region_id
        area = np.sum(region_mask)
        if area > max_area and area >= min_wound_area_px:
            max_area = area
            max_region_id = region_id

    if max_region_id is None:
        # fallback: just take the largest component anyway
        for region_id in range(1, lbl.max() + 1):
            region_mask = lbl == region_id
            area = np.sum(region_mask)
            if area > max_area:
                max_area = area
                max_region_id = region_id

    wound_roi[lbl == max_region_id] = True
    return wound_roi


def get_border_roi(wound_roi: np.ndarray,
                   border_px: int = 30) -> np.ndarray:
    """
    Relative Wound Density needs:
       - wound region
       - bordering monolayer region (reference density)
    We'll get a "ring" around wound by dilating and subtracting.

    border_px controls how thick that ring is.
    """
    dilated = binary_dilation(wound_roi, disk(border_px))
    ring = np.logical_and(dilated, ~wound_roi)
    return ring


def measure_metrics(cell_mask: np.ndarray,
                    wound_roi: np.ndarray,
                    border_roi: np.ndarray) -> Tuple[float, float]:
    """
    Compute:
      - Wound Confluence (%)
        = (% of wound ROI area now occupied by cells)
        = 100 * (#cell_pixels_in_wound / wound_area)

      - Relative Wound Density (%)
        = density_wound / density_border * 100
        where density = (#cell_pixels / area)
    """
    wound_area = np.sum(wound_roi)
    border_area = np.sum(border_roi)

    cell_in_wound = np.sum(np.logical_and(cell_mask, wound_roi))
    cell_in_border = np.sum(np.logical_and(cell_mask, border_roi))

    # avoid divide-by-zero
    wound_confluence_pct = 0.0
    rel_wound_density_pct = 0.0

    if wound_area > 0:
        wound_confluence_pct = 100.0 * (cell_in_wound / wound_area)

    wound_density = (cell_in_wound / wound_area) if wound_area > 0 else 0.0
    border_density = (cell_in_border / border_area) if border_area > 0 else 1e-9

    rel_wound_density_pct = 100.0 * (wound_density / border_density)

    return wound_confluence_pct, rel_wound_density_pct


def overlay_debug(rgb_img: np.ndarray,
                  wound_roi: np.ndarray,
                  border_roi: np.ndarray,
                  cell_mask: np.ndarray) -> np.ndarray:
    """
    Make a quick QC overlay:
      - wound ROI tinted red
      - border ROI tinted yellow
      - detected cells tinted green
    Output is uint8 RGB for preview.
    """
    base = rgb_img.copy().astype(np.float32)

    # green for cells
    cell_overlay = np.stack([np.zeros_like(cell_mask),
                             cell_mask.astype(float),
                             np.zeros_like(cell_mask)], axis=-1)

    # red for wound
    wound_overlay = np.stack([wound_roi.astype(float),
                              np.zeros_like(wound_roi),
                              np.zeros_like(wound_roi)], axis=-1)

    # yellow for border (red+green)
    border_overlay = np.stack([border_roi.astype(float),
                               border_roi.astype(float),
                               np.zeros_like(border_roi)], axis=-1)

    # combine overlays
    combined = base / 255.0
    combined = np.clip(
        combined * 0.6 + 0.4 * (cell_overlay + wound_overlay + border_overlay),
        0, 1
    )
    return (combined * 255).astype(np.uint8)


# ----------------------------- STREAMLIT APP LOGIC -----------------------------
st.title("Incucyte-style Wound Healing / Migration Analysis")

st.markdown(
    """
This app:
1. Segments bright cells (more sensitive to faint cells).
2. Locks the wound region from **time 0h** and reuses it.
3. Measures:
   - **Wound Confluence (%)** = % of wound that's filled with cells.
   - **Relative Wound Density (%)** = cell density in wound vs surrounding monolayer.
4. Outputs a results table like Incucyte.

**Instructions**
- Upload images from a *single well* across time.
- Order doesn't matter; we'll sort by filename.
- First image after sort = Time 0h (baseline wound mask).
"""
)

uploaded_files = st.file_uploader(
    "Upload time-series brightfield images (same well across time)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)

custom_times = st.text_input(
    "Optional: comma-separated timepoints in hours (matches sorted file order). "
    "Leave blank to use default [0,12,24,36,48].",
    value=",".join(map(str, DEFAULT_TIMEPOINTS_HOURS)),
)

run_btn = st.button("Run Analysis")


if run_btn and uploaded_files:
    # Sort files by filename for reproducible order
    uploaded_files_sorted = sorted(uploaded_files, key=lambda f: f.name)

    # Parse timepoints
    try:
        timepoints_hours = [
            float(x.strip()) for x in custom_times.split(",")
        ]
    except Exception:
        timepoints_hours = DEFAULT_TIMEPOINTS_HOURS[:]

    # If user didn't provide enough timepoints, extend or trim
    if len(timepoints_hours) < len(uploaded_files_sorted):
        # extend by repeating last delta
        if len(timepoints_hours) >= 2:
            last_delta = timepoints_hours[-1] - timepoints_hours[-2]
        else:
            last_delta = 12.0
        while len(timepoints_hours) < len(uploaded_files_sorted):
            timepoints_hours.append(timepoints_hours[-1] + last_delta)
    elif len(timepoints_hours) > len(uploaded_files_sorted):
        timepoints_hours = timepoints_hours[:len(uploaded_files_sorted)]

    st.subheader("Step 1. Build wound ROI from T0")

    # --- T0 processing ---
    base_file = uploaded_files_sorted[0]
    base_gray = load_image_as_array(base_file)
    base_enhanced = enhance_contrast(base_gray)
    base_cell_mask = segment_cells(base_enhanced)

    wound_roi = get_primary_wound_roi(base_cell_mask)
    border_roi = get_border_roi(wound_roi, border_px=30)  # tunable ring size

    # QC preview for T0
    base_rgb = np.array(Image.open(base_file).convert("RGB"))
    preview_t0 = overlay_debug(base_rgb, wound_roi, border_roi, base_cell_mask)

    col1, col2 = st.columns(2)
    with col1:
        st.image(base_rgb, caption=f"T0 Raw ({base_file.name})", use_column_width=True)
    with col2:
        st.image(preview_t0, caption="T0 QC Overlay (red=wound, yellow=border, green=cells)", use_column_width=True)

    st.markdown(
        """
        - **wound ROI (red)** is fixed and will be reused for all later timepoints  
        - **border ring (yellow)** is used to estimate background monolayer density  
        """
    )

    # --- Loop through all timepoints ---
    records = []
    qc_overlays = []

    for img_file, t_hr in zip(uploaded_files_sorted, timepoints_hours):
        gray = load_image_as_array(img_file)
        enh = enhance_contrast(gray)
        cell_mask = segment_cells(enhanced_gray=enh)

        wound_confluence_pct, rel_wound_density_pct = measure_metrics(
            cell_mask=cell_mask,
            wound_roi=wound_roi,
            border_roi=border_roi,
        )

        records.append(
            {
                "filename": img_file.name,
                "time_h": t_hr,
                "wound_confluence_pct": wound_confluence_pct,
                "relative_wound_density_pct": rel_wound_density_pct,
            }
        )

        # build QC overlay for this frame
        rgb_img = np.array(Image.open(img_file).convert("RGB"))
        ov = overlay_debug(rgb_img, wound_roi, border_roi, cell_mask)
        qc_overlays.append((t_hr, img_file.name, ov))

    # Results table
    df_results = pd.DataFrame(records).sort_values("time_h").reset_index(drop=True)

    st.subheader("Step 2. Quantitative Results")
    st.write(df_results)

    # Download as CSV
    csv_bytes = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results CSV",
        data=csv_bytes,
        file_name="incucyte_style_metrics.csv",
        mime="text/csv",
    )

    # Plot curves like Incucyte kinetics
    st.subheader("Step 3. Kinetics Plots")

    fig1, ax1 = plt.subplots()
    ax1.plot(df_results["time_h"], df_results["wound_confluence_pct"], marker="o")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Wound Confluence (%)")
    ax1.set_title("Wound Confluence Over Time")
    ax1.set_ylim(0, 110)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(df_results["time_h"], df_results["relative_wound_density_pct"], marker="o")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Relative Wound Density (%)")
    ax2.set_title("Relative Wound Density Over Time")
    ax2.set_ylim(0, 200)
    st.pyplot(fig2)

    st.subheader("Step 4. QC Overlays Across Time")
    st.markdown(
        "Each frame: green=cells the algorithm found, red=original wound ROI from T0, yellow=border ring."
    )
    for t_hr, name, ov in qc_overlays:
        st.image(
            ov,
            caption=f"{name} @ {t_hr}h",
            use_column_width=True,
        )

else:
    st.info("Upload images and click 'Run Analysis' to begin.")

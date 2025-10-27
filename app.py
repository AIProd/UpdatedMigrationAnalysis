import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage import morphology, measure

# ----------------------------- CONFIG ---------------------------------
GAUSSIAN_SIGMA = 1            # blur to denoise before gradients
WOUND_LOW_GRAD_PERCENTILE = 30  # % of lowest gradient values to call "wound-like"
MORPH_KERNEL_RADIUS = 10      # smooth wound edges
MIN_WOUND_SIZE = 500          # drop tiny specks
BAND_THICKNESS_PX = 50        # how far outside wound to sample monolayer
CELL_PERCENTILE = 10          # how generous we are detecting cells (lower = more sensitive)

# ----------------------------- HELPERS --------------------------------


def to_gray(img_pil: Image.Image) -> np.ndarray:
    """PIL -> grayscale float [0..1] -> Gaussian blur."""
    gray = np.array(ImageOps.grayscale(img_pil)).astype(np.float32)
    blur = gaussian(gray, sigma=GAUSSIAN_SIGMA)
    return blur  # float64 after gaussian


def build_wound_mask_from_t0(gray_blur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the wound at t0 using texture: wound = large smooth low-gradient band.
    Returns:
        wound_mask (bool HxW)
        grad0 (float HxW) Sobel gradient image for t0
    """
    grad0 = sobel(gray_blur)

    # Low gradient = smooth gap. Pick the lowest ~30% gradient pixels as wound candidates.
    thr = np.percentile(grad0, WOUND_LOW_GRAD_PERCENTILE)
    wound_cand = grad0 < thr

    # Morphological cleanup
    wound_cand = morphology.remove_small_objects(wound_cand, min_size=MIN_WOUND_SIZE)
    wound_cand = morphology.binary_closing(wound_cand, morphology.disk(MORPH_KERNEL_RADIUS))
    wound_cand = morphology.binary_opening(wound_cand, morphology.disk(MORPH_KERNEL_RADIUS))

    # Keep only the largest connected component (the main wound band)
    labeled, _ = measure.label(wound_cand, return_num=True)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(wound_mask: np.ndarray, thickness_px: int = BAND_THICKNESS_PX) -> np.ndarray:
    """
    Get a ring of 'monolayer reference' around the wound by dilating and subtracting.
    This approximates the intact cell lawn on both sides.
    """
    dilated = morphology.binary_dilation(wound_mask, morphology.disk(thickness_px))
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    baseline_w_frac: float,
    cell_percentile: int = CELL_PERCENTILE,
) -> Dict[str, float]:
    """
    Compute Wound Confluence (%) and Relative Wound Density (%) for ONE timepoint.
    - baseline_w_frac is w(0): wound cell fraction at t0
    """
    grad = sobel(gray_blur)

    # Adaptive threshold for "cells": look at texture strength in the monolayer band.
    thr_cell = np.percentile(grad[band_mask], cell_percentile)

    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    w_frac = wound_cells_mask.sum() / wound_mask.sum()          # wound "cell coverage"
    c_frac = band_cells_mask.sum() / band_mask.sum()            # monolayer "cell coverage"
    w0 = baseline_w_frac

    # IncuCyte-style outputs
    wound_confluence_pct = 100.0 * w_frac
    # avoid /0, clip to 0..100 range for sanity
    rwd_pct = 100.0 * (w_frac - w0) / (c_frac - w0 + 1e-9)
    rwd_pct = float(np.clip(rwd_pct, 0, 100))

    return {
        "wound_confluence_pct": float(wound_confluence_pct),
        "relative_wound_density_pct": float(rwd_pct),
        "w_frac": float(w_frac),
        "c_frac": float(c_frac),
        "thr_cell": float(thr_cell),
    }


def parse_hours_from_name(name: str) -> float:
    """
    Try to grab time info from filename.
    We'll look for patterns like '00d00h00m', '01d00h00m', '24 H', '48H', '72 H', etc.
    Returns hours as float. If nothing found, returns 0.
    """
    # 1) try 'XdYYh' style (01d00h00m)
    m = re.search(r'(\d+)\s*[dD]\s*(\d+)\s*[hH]', name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    # 2) try explicit hours like '24H' or '72 H'
    m = re.search(r'(\d+)\s*[hH]', name)
    if m:
        return float(m.group(1))

    # fallback
    return 0.0


def overlay_debug_rgb(
    img_pil: Image.Image,
    wound_mask: np.ndarray,
    wound_cells_mask: np.ndarray,
    alpha_wound: float = 0.4,
    alpha_cells: float = 0.4,
) -> Image.Image:
    """
    Make an RGB overlay:
      - wound region tinted blue
      - wound cells tinted green
    Just for QC in Streamlit.
    """
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    # blue = wound area
    blue = np.array([0, 0, 255], dtype=np.float32)
    out[wound_mask] = (1 - alpha_wound) * out[wound_mask] + alpha_wound * blue

    # green = cells that have migrated into wound
    green = np.array([0, 255, 0], dtype=np.float32)
    out[wound_cells_mask] = (1 - alpha_cells) * out[wound_cells_mask] + alpha_cells * green

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def run_full_analysis(images: List[Image.Image], names: List[str]) -> Tuple[pd.DataFrame, List[Image.Image]]:
    """
    Core driver:
    1. sort images by time
    2. build wound model from t0
    3. compute metrics for each time
    4. build QC overlays
    """
    # sort by extracted hours
    order = np.argsort([parse_hours_from_name(n) for n in names])
    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [parse_hours_from_name(n) for n in names_sorted]

    # preprocess
    grays = [to_gray(im) for im in images_sorted]

    # wound mask from first (earliest) frame
    wound_mask, grad0 = build_wound_mask_from_t0(grays[0])
    band_mask = make_band_mask(wound_mask, BAND_THICKNESS_PX)

    # baseline wound coverage fraction at t0 (w0)
    grad_first = sobel(grays[0])
    thr_cell_first = np.percentile(grad_first[band_mask], CELL_PERCENTILE)
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)
    w0_frac = wound_cells_first.sum() / wound_mask.sum()

    # loop all timepoints
    rows = []
    debug_overlays = []
    for img_pil, gray_img, hr, nm in zip(images_sorted, grays, hours_sorted, names_sorted):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            baseline_w_frac=w0_frac,
            cell_percentile=CELL_PERCENTILE,
        )

        # Save table row
        rows.append({
            "Image": nm,
            "Hours": hr,
            "Wound Confluence (%)": metrics["wound_confluence_pct"],
            "Relative Wound Density (%)": metrics["relative_wound_density_pct"],
        })

        # make overlay for QC
        grad_now = sobel(gray_img)
        thr_now = np.percentile(grad_now[band_mask], CELL_PERCENTILE)
        wound_cells_now = np.logical_and(wound_mask, grad_now > thr_now)
        overlay_img = overlay_debug_rgb(img_pil, wound_mask, wound_cells_now)
        debug_overlays.append(overlay_img)

    df = pd.DataFrame(rows).sort_values("Hours").reset_index(drop=True)
    return df, debug_overlays


# ----------------------------- STREAMLIT UI ----------------------------

st.set_page_config(page_title="IncuCyte-style Migration Analysis", layout="wide")

st.title("IncuCyte-style Wound Healing Analysis")
st.caption(
    "We lock the wound at t=0, detect invading cells over time, and report "
    "Wound Confluence (%) and Relative Wound Density (%) similar to IncuCyte."
)

uploaded_files = st.file_uploader(
    "Upload images for ONE well / condition (0h, 24h, 48h, 72h...). "
    "Use the same magnification.",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

show_debug = st.checkbox("Show debug overlays (wound in blue, invading cells in green)", value=False)

if uploaded_files:
    # Read images
    imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
    names = [f.name for f in uploaded_files]

    # Run analysis
    df_metrics, overlays = run_full_analysis(imgs, names)

    st.subheader("Results Table")
    st.dataframe(df_metrics.style.format({
        "Wound Confluence (%)": "{:.2f}",
        "Relative Wound Density (%)": "{:.2f}",
        "Hours": "{:.2f}",
    }))

    if show_debug:
        st.subheader("QC Overlays")
        cols = st.columns(2)
        for i, (nm, ov) in enumerate(zip(df_metrics["Image"], overlays)):
            with cols[i % 2]:
                st.text(nm)
                st.image(ov, use_column_width=True)

    st.markdown(
        """
        **Notes / knobs you can tune on top of the script constants:**
        - `CELL_PERCENTILE` ↓ (e.g. 8 instead of 10) will classify even fainter cells as 'present', increasing
          Wound Confluence and RWD numbers.
        - `BAND_THICKNESS_PX`: if your wound is huge or tiny, change how thick the monolayer reference band is.
        - `WOUND_LOW_GRAD_PERCENTILE`: if the wound detection breaks, nudge this (20–40).
        """
    )
else:
    st.info("Upload a full time series for one condition to get started.")

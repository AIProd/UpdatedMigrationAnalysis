import re
from io import BytesIO
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage import morphology, measure


# =========================================
# Configuration
# =========================================

GAUSSIAN_SIGMA = 1                # blur to denoise before gradients
WOUND_LOW_GRAD_PERCENTILE = 30    # % of lowest Sobel gradient values treated as wound-like at t0
MORPH_KERNEL_RADIUS = 10          # radius (px) for binary_opening/closing to smooth wound edges
MIN_WOUND_SIZE = 500              # ignore speckle regions smaller than this when defining wound
BAND_THICKNESS_PX = 50            # px distance outside wound mask to define reference monolayer band
CELL_PERCENTILE = 10              # lower = more sensitive to faint cells, higher = stricter


# =========================================
# Image / mask utilities
# =========================================

def to_gray(image_pil: Image.Image) -> np.ndarray:
    """
    Convert PIL image to grayscale float64 in [0..1] and apply Gaussian blur.
    """
    gray = np.array(ImageOps.grayscale(image_pil)).astype(np.float32)
    blurred = gaussian(gray, sigma=GAUSSIAN_SIGMA)
    return blurred  # float64 after gaussian()


def build_wound_mask_from_t0(gray_blur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the wound region at time 0.
    Assumption:
        - Wound region is relatively smooth / low texture (low Sobel gradient)
        - Surrounding monolayer is more textured (higher gradient)
    Steps:
        1. Sobel gradient
        2. Threshold on low-gradient pixels
        3. Morphological cleanup
        4. Keep only the largest connected component
    Returns:
        wound_mask (bool HxW)
        grad0      (float HxW) Sobel gradient map at t0
    """
    grad0 = sobel(gray_blur)

    thr = np.percentile(grad0, WOUND_LOW_GRAD_PERCENTILE)
    wound_candidate = grad0 < thr

    # remove speckles, smooth edges
    wound_candidate = morphology.remove_small_objects(
        wound_candidate, min_size=MIN_WOUND_SIZE
    )
    wound_candidate = morphology.binary_closing(
        wound_candidate, morphology.disk(MORPH_KERNEL_RADIUS)
    )
    wound_candidate = morphology.binary_opening(
        wound_candidate, morphology.disk(MORPH_KERNEL_RADIUS)
    )

    # keep the largest connected component
    labeled, _ = measure.label(wound_candidate, return_num=True)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(wound_mask: np.ndarray, thickness_px: int = BAND_THICKNESS_PX) -> np.ndarray:
    """
    Build a "reference monolayer band": a ring just outside the wound.
    This approximates intact, confluent cells used for normalization.
    """
    dilated = morphology.binary_dilation(wound_mask, morphology.disk(thickness_px))
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def parse_hours_from_name(name: str) -> float:
    """
    Extract time (in hours) from filename.
    Supports:
        - "00d00h00m", "01d00h00m", etc. -> days + hours
        - "24H", "72 H", etc. -> hours
    Fallback: 0
    """
    # pattern: "<dd>d<hh>h"
    m = re.search(r'(\d+)\s*[dD]\s*(\d+)\s*[hH]', name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    # pattern: "<hh>H"
    m = re.search(r'(\d+)\s*[hH]', name)
    if m:
        return float(m.group(1))

    return 0.0


def overlay_debug_rgb(
    img_pil: Image.Image,
    wound_mask: np.ndarray,
    wound_cells_mask: np.ndarray,
    alpha_wound: float = 0.4,
    alpha_cells: float = 0.4,
) -> Image.Image:
    """
    Build an RGB overlay for QC:
        - Wound region (from t0) tinted blue.
        - Cells detected inside the wound at this timepoint tinted green.
    """
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    blue = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    # wound region
    out[wound_mask] = (1 - alpha_wound) * out[wound_mask] + alpha_wound * blue

    # cells migrated into wound
    out[wound_cells_mask] = (1 - alpha_cells) * out[wound_cells_mask] + alpha_cells * green

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =========================================
# Metric computation
# =========================================

def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    w0_frac: float,
    cell_percentile: int = CELL_PERCENTILE,
) -> Dict[str, float]:
    """
    Compute per-timepoint migration metrics.

    Wound Confluence (%):
        Fraction of the original wound area now containing cells * 100.

    Relative Wound Density (%):
        IncuCyte-style normalization of wound density vs surrounding monolayer:
            RWD(t) = 100 * (w(t) - w(0)) / (c(t) - w(0))
        where
            w(t) = fraction of wound area classified as cells at time t
            c(t) = fraction of band area classified as cells at time t
            w(0) = w0_frac, baseline wound fraction at t=0
    """
    grad = sobel(gray_blur)

    # adaptive cell threshold: look at texture in the surrounding band
    thr_cell = np.percentile(grad[band_mask], cell_percentile)

    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    w_frac = wound_cells_mask.sum() / wound_mask.sum()
    c_frac = band_cells_mask.sum() / band_mask.sum()

    wound_confluence_pct = 100.0 * w_frac

    # avoid division by zero in RWD calc
    rwd_pct = 100.0 * (w_frac - w0_frac) / (c_frac - w0_frac + 1e-9)
    rwd_pct = float(np.clip(rwd_pct, 0, 100))

    return {
        "wound_confluence_pct": float(wound_confluence_pct),
        "relative_wound_density_pct": float(rwd_pct),
        "w_frac": float(w_frac),
        "c_frac": float(c_frac),
        "thr_cell": float(thr_cell),
    }


def run_full_analysis(
    images: List[Image.Image],
    names: List[str],
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    """
    Full per-series analysis:
        1. Sort frames by time.
        2. Build wound mask from earliest timepoint.
        3. Compute metrics for each timepoint.
        4. Generate QC overlays.
    Returns:
        df_metrics: table with Hours, Wound Confluence, Relative Wound Density
        overlays:   list of overlay images aligned with df_metrics rows
    """
    # sort frames by parsed hours
    hours = [parse_hours_from_name(n) for n in names]
    order = np.argsort(hours)

    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [hours[i] for i in order]

    # grayscale + blur for each frame
    gray_series = [to_gray(im) for im in images_sorted]

    # define wound & reference band from t0
    wound_mask, _grad0 = build_wound_mask_from_t0(gray_series[0])
    band_mask = make_band_mask(wound_mask, BAND_THICKNESS_PX)

    # compute baseline wound fraction at t0
    grad_first = sobel(gray_series[0])
    thr_cell_first = np.percentile(grad_first[band_mask], CELL_PERCENTILE)
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)
    w0_frac = wound_cells_first.sum() / wound_mask.sum()

    # iterate over all timepoints
    rows = []
    overlays = []
    for img_pil, gray_img, hr, nm in zip(images_sorted, gray_series, hours_sorted, names_sorted):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            w0_frac=w0_frac,
            cell_percentile=CELL_PERCENTILE,
        )

        # store row
        rows.append({
            "Image": nm,
            "Hours": hr,
            "Wound Confluence (%)": metrics["wound_confluence_pct"],
            "Relative Wound Density (%)": metrics["relative_wound_density_pct"],
        })

        # build QC overlay for display
        grad_now = sobel(gray_img)
        thr_now = np.percentile(grad_now[band_mask], CELL_PERCENTILE)
        wound_cells_now = np.logical_and(wound_mask, grad_now > thr_now)
        ov = overlay_debug_rgb(img_pil, wound_mask, wound_cells_now)
        overlays.append(ov)

    df_metrics = pd.DataFrame(rows).sort_values("Hours").reset_index(drop=True)
    return df_metrics, overlays


# =========================================
# Plotting helpers
# =========================================

def plot_metric(hours: np.ndarray, values: np.ndarray, ylabel: str, title: str):
    """
    Make a simple line+marker plot for a single metric.
    """
    fig, ax = plt.subplots()
    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xlabel("Hours")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Return CSV bytes (UTF-8) suitable for st.download_button.
    """
    return df.to_csv(index=False).encode("utf-8")


# =========================================
# Streamlit UI
# =========================================

st.set_page_config(page_title="Wound Healing Analysis", layout="wide")

st.title("Wound Healing Analysis")
st.write(
    "Quantifies scratch-wound closure over time, reporting Wound Confluence (%) "
    "and Relative Wound Density (%) for a single well/condition."
)

uploaded_files = st.file_uploader(
    "Upload images from one well (e.g. 0h, 24h, 48h, 72h). "
    "All images should be same magnification.",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Load images
    imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
    names = [f.name for f in uploaded_files]

    # Run analysis
    df_metrics, overlays = run_full_analysis(imgs, names)

    # ------------------------
    # Metrics table + download
    # ------------------------
    st.header("Metrics")
    styled = df_metrics.style.format({
        "Hours": "{:.2f}",
        "Wound Confluence (%)": "{:.2f}",
        "Relative Wound Density (%)": "{:.2f}",
    })
    st.dataframe(styled, use_container_width=True)

    csv_data = df_to_csv_bytes(df_metrics)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="wound_metrics.csv",
        mime="text/csv",
    )

    # ------------------------
    # Plots
    # ------------------------
    st.header("Time-Series Plots")

    hours_arr = df_metrics["Hours"].to_numpy(dtype=float)

    confluence_arr = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)
    fig_conf = plot_metric(
        hours_arr,
        confluence_arr,
        ylabel="Wound Confluence (%)",
        title="Wound Confluence vs Time",
    )
    st.pyplot(fig_conf, clear_figure=True)

    rwd_arr = df_metrics["Relative Wound Density (%)"].to_numpy(dtype=float)
    fig_rwd = plot_metric(
        hours_arr,
        rwd_arr,
        ylabel="Relative Wound Density (%)",
        title="Relative Wound Density vs Time",
    )
    st.pyplot(fig_rwd, clear_figure=True)

    # ------------------------
    # Overlays
    # ------------------------
    st.header("Overlay QC")
    st.write(
        "Blue = wound region defined at t0. "
        "Green = detected cells inside that wound region at each timepoint."
    )

    cols = st.columns(2)
    for i, (row, overlay_img) in enumerate(zip(df_metrics.itertuples(index=False), overlays)):
        col = cols[i % 2]
        with col:
            st.caption(f"{row.Image} ({row.Hours:.2f} h)")
            st.image(overlay_img, use_container_width=True)

else:
    st.info("Upload all timepoints from a single well to begin.")

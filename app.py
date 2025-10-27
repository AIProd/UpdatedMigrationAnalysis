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


# =====================================================
# Core image / mask utilities
# =====================================================

def to_gray(image_pil: Image.Image, gaussian_sigma: float) -> np.ndarray:
    """
    Convert PIL image to grayscale float64 in [0..1] and apply Gaussian blur.
    """
    gray = np.array(ImageOps.grayscale(image_pil)).astype(np.float32)
    blurred = gaussian(gray, sigma=gaussian_sigma)
    return blurred  # float64 after gaussian()


def build_wound_mask_from_t0(
    gray_blur: np.ndarray,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the wound region at time 0.

    Steps:
        1. Compute Sobel gradient.
        2. Treat the lowest Nth percentile of gradient magnitude as "smooth gap".
        3. Morphological cleanup.
        4. Keep the single largest connected component as wound.
    Returns:
        wound_mask (bool HxW)
        grad0      (float HxW) Sobel gradient map at t0
    """
    grad0 = sobel(gray_blur)

    thr = np.percentile(grad0, wound_low_grad_percentile)
    wound_candidate = grad0 < thr

    # cleanup / smoothing
    wound_candidate = morphology.remove_small_objects(
        wound_candidate, min_size=min_wound_size
    )
    wound_candidate = morphology.binary_closing(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )
    wound_candidate = morphology.binary_opening(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )

    # largest component only
    labeled, _ = measure.label(wound_candidate, return_num=True)
    sizes = np.bincount(labeled.ravel())
    if sizes.size == 0:
        raise ValueError("No wound-like region detected in first frame.")
    sizes[0] = 0  # ignore background label 0
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(
    wound_mask: np.ndarray,
    band_thickness_px: int,
) -> np.ndarray:
    """
    Build a 'reference band': a ring just outside the wound.
    Used as the confluent monolayer reference for normalization.
    """
    dilated = morphology.binary_dilation(
        wound_mask, morphology.disk(band_thickness_px)
    )
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def parse_hours_from_name(name: str) -> float:
    """
    Extract time (hours) from filename.
    Supports:
        - "01d00h00m" -> days + hours
        - "24H", "72 H" -> hours
    Fallback: 0
    """
    # pattern like "01d00h"
    m = re.search(r'(\d+)\s*[dD]\s*(\d+)\s*[hH]', name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    # pattern like "24H" / "72 H"
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
      - Wound region from t0 tinted blue
      - Cells detected inside the wound at this timepoint tinted green
    """
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    blue = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    out[wound_mask] = (1 - alpha_wound) * out[wound_mask] + alpha_wound * blue
    out[wound_cells_mask] = (
        (1 - alpha_cells) * out[wound_cells_mask] + alpha_cells * green
    )

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =====================================================
# Metric computation
# =====================================================

def _cell_threshold(grad: np.ndarray, band_mask: np.ndarray, cell_percentile: float):
    """
    Adaptive texture threshold:
    Take a given percentile of Sobel gradient values in the reference band.
    Lower percentile -> more sensitive to faint / transparent cells.
    """
    if band_mask.sum() == 0:
        # Fallback: use global grad distribution if band is empty for some reason
        return np.percentile(grad, cell_percentile)
    return np.percentile(grad[band_mask], cell_percentile)


def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    w0_frac: float,
    cell_percentile: float,
) -> Dict[str, float]:
    """
    Compute timepoint migration metrics.

    Wound Confluence (%):
        Fraction of original wound area now classified as cells * 100.

    Relative Wound Density (%):
        RWD(t) = 100 * ( w(t) - w(0) ) / ( c(t) - w(0) )
        where
            w(t) = wound cell fraction at time t
            c(t) = band cell fraction at time t
            w(0) = baseline wound fraction at t=0
    """
    grad = sobel(gray_blur)
    thr_cell = _cell_threshold(grad, band_mask, cell_percentile)

    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    # coverage fractions
    wound_area = wound_mask.sum()
    band_area = band_mask.sum()

    if wound_area == 0:
        raise ValueError("Wound mask area is zero.")
    if band_area == 0:
        # fallback to avoid div-by-zero in normalization
        band_area = 1

    w_frac = wound_cells_mask.sum() / wound_area
    c_frac = band_cells_mask.sum() / band_area

    wound_confluence_pct = 100.0 * w_frac

    # RWD normalization
    denom = (c_frac - w0_frac)
    if abs(denom) < 1e-9:
        rwd_pct = 0.0
    else:
        rwd_pct = 100.0 * (w_frac - w0_frac) / denom

    rwd_pct = float(np.clip(rwd_pct, 0, 100))

    return {
        "wound_confluence_pct": float(wound_confluence_pct),
        "relative_wound_density_pct": float(rwd_pct),
        "w_frac": float(w_frac),
        "c_frac": float(c_frac),
    }


def run_full_analysis(
    images: List[Image.Image],
    names: List[str],
    gaussian_sigma: float,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
    band_thickness_px: int,
    cell_percentile: float,
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    """
    Full analysis pipeline for a single well / condition:
        1. Sort frames by parsed time.
        2. Build wound mask from earliest frame.
        3. Compute metrics (Wound Confluence, Relative Wound Density) at each timepoint.
        4. Generate QC overlays (blue wound, green migrated cells).
    Returns:
        df_metrics: table of results
        overlays:   list of QC overlay images, aligned with df_metrics rows
    """
    # sort frames by inferred hours
    hours_list = [parse_hours_from_name(n) for n in names]
    order = np.argsort(hours_list)

    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [hours_list[i] for i in order]

    # grayscale + blur
    gray_series = [to_gray(im, gaussian_sigma) for im in images_sorted]

    # wound mask from earliest frame (t0)
    wound_mask, _grad0 = build_wound_mask_from_t0(
        gray_series[0],
        wound_low_grad_percentile,
        morph_kernel_radius,
        min_wound_size,
    )

    # reference band
    band_mask = make_band_mask(wound_mask, band_thickness_px)

    # baseline wound fraction at t=0
    grad_first = sobel(gray_series[0])
    thr_cell_first = _cell_threshold(grad_first, band_mask, cell_percentile)
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)
    w0_frac = wound_cells_first.sum() / max(wound_mask.sum(), 1)

    # loop over all timepoints
    rows = []
    overlays = []
    for img_pil, gray_img, hr, nm in zip(images_sorted, gray_series, hours_sorted, names_sorted):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            w0_frac=w0_frac,
            cell_percentile=cell_percentile,
        )

        rows.append({
            "Image": nm,
            "Hours": hr,
            "Wound Confluence (%)": metrics["wound_confluence_pct"],
            "Relative Wound Density (%)": metrics["relative_wound_density_pct"],
        })

        # overlay for QC
        grad_now = sobel(gray_img)
        thr_now = _cell_threshold(grad_now, band_mask, cell_percentile)
        wound_cells_now = np.logical_and(wound_mask, grad_now > thr_now)
        ov = overlay_debug_rgb(img_pil, wound_mask, wound_cells_now)
        overlays.append(ov)

    df_metrics = pd.DataFrame(rows).sort_values("Hours").reset_index(drop=True)
    return df_metrics, overlays


# =====================================================
# Plotting + export helpers
# =====================================================

def plot_metric(
    hours: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
):
    """
    Small line+marker plot for a single metric.
    """
    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xlabel("Hours")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    DataFrame -> CSV bytes for download.
    """
    return df.to_csv(index=False).encode("utf-8")


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="Wound Healing Analysis", layout="wide")

st.title("Wound Healing Analysis")

st.write(
    "Quantifies scratch-wound closure over time for a single well / condition. "
    "Outputs Wound Confluence (%) and Relative Wound Density (%)."
)

with st.form("analysis_form"):
    uploaded_files = st.file_uploader(
        "Upload all timepoints from one well (e.g. 0h, 24h, 48h, 72h). "
        "All images should use the same magnification.",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.markdown("### Advanced settings (optional)")
    st.caption(
        "These parameters control wound detection and cell detection. "
        "Defaults usually work. Adjust only if the wound mask or cell detection looks off."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gaussian_sigma = st.slider(
            "Gaussian blur σ",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Pre-smoothing before edge detection. Higher smooths noise but can soften wound edges.",
        )
        wound_low_grad_percentile = st.slider(
            "Wound smoothness percentile",
            min_value=5,
            max_value=60,
            value=30,
            step=1,
            help="Lower = narrower wound mask; higher = wider wound mask. "
                 "This percentile of lowest Sobel gradient is treated as wound.",
        )

    with col2:
        morph_kernel_radius = st.slider(
            "Wound edge smoothing (px)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Radius used for morphological open/close. "
                 "Larger values produce a smoother, more continuous wound band.",
        )
        band_thickness_px = st.slider(
            "Reference band thickness (px)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Thickness of the ring outside the wound used as the 'healthy monolayer' reference.",
        )

    with col3:
        min_wound_size = st.number_input(
            "Min wound size (px area)",
            min_value=100,
            max_value=200000,
            value=500,
            step=100,
            help="Ignore wound candidates smaller than this. "
                 "Prevents tiny specks from being mis-identified as the wound.",
        )
        cell_percentile = st.slider(
            "Cell texture percentile",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Lower = more sensitive. "
                 "At lower values, faint/transparent cells are still counted as cells.",
        )

    submitted = st.form_submit_button("Analyze")

if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one image series.")
    else:
        # Load images
        imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
        names = [f.name for f in uploaded_files]

        try:
            df_metrics, overlays = run_full_analysis(
                images=imgs,
                names=names,
                gaussian_sigma=gaussian_sigma,
                wound_low_grad_percentile=wound_low_grad_percentile,
                morph_kernel_radius=morph_kernel_radius,
                min_wound_size=min_wound_size,
                band_thickness_px=band_thickness_px,
                cell_percentile=cell_percentile,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        else:
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
            # Plots (smaller figs)
            # ------------------------
            st.header("Time-Series Plots")

            hours_arr = df_metrics["Hours"].to_numpy(dtype=float)

            conf_arr = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)
            fig_conf = plot_metric(
                hours_arr,
                conf_arr,
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
            # Overlays (smaller grid)
            # ------------------------
            st.header("Overlay QC")
            st.caption(
                "Blue: wound region defined at first timepoint.  "
                "Green: detected cells inside that wound region at each timepoint."
            )

            cols = st.columns(3)
            for i, (row, overlay_img) in enumerate(zip(df_metrics.itertuples(index=False), overlays)):
                col = cols[i % 3]
                with col:
                    st.caption(f"{row.Image}  ({row.Hours:.2f} h)")
                    # use_container_width=True keeps the image nicely scaled in the column
                    st.image(overlay_img, use_container_width=True)

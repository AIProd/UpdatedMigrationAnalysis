import io
import os
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
    binary_erosion,
    disk,
)

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Migration Image Analysis", layout="wide")
TIMEPOINTS = [0, 12, 24, 36]
CONCENTRATIONS = ["Control", "8000", "16000", "32000", "64000"]

# ----------------------------- UTILITIES -------------------------------
@st.cache_data(show_spinner=False)
def _downscale(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Limit the largest dimension to max_side px to keep processing fast."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def illumination_correct(gray: np.ndarray, sigma_bg: float) -> np.ndarray:
    """
    Divide by a heavy gaussian blur to flatten illumination, then rescale to [0,1].
    This helps remove bright-center / dark-edge artifacts.
    """
    bg = gaussian(gray, sigma=sigma_bg, preserve_range=True)
    corr = gray / np.clip(bg, 1e-6, None)
    corr = exposure.rescale_intensity(corr, in_range="image", out_range=(0, 1))
    return corr


def local_std(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Local texture std via Gaussian moments.
    Higher std -> 'textured' (typically cells/edges).
    Lower std -> 'smooth' (typically open gap).
    """
    m1 = gaussian(gray, sigma=sigma, preserve_range=True)
    m2 = gaussian(gray * gray, sigma=sigma, preserve_range=True)
    var = np.clip(m2 - m1 * m1, 0, None)
    std = np.sqrt(var)
    std = std / (std.max() + 1e-8)
    return std


def center_roi_mask(h: int, w: int, margin_frac: float) -> np.ndarray:
    """
    Centered rectangular ROI. margin_frac=0 → full frame.
    We crop off a uniform border to avoid glare/text.
    """
    r0, r1 = int(h * margin_frac), int(h * (1 - margin_frac))
    c0, c1 = int(w * margin_frac), int(w * (1 - margin_frac))
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def mask_scale_bar(
    h: int, w: int, width_frac: float, height_frac: float, inset_frac: float = 0.02
) -> np.ndarray:
    """
    Mask a bottom-right rectangle (scale bar / timestamp).
    width_frac & height_frac are fractions of the full image size.
    """
    if width_frac <= 0 or height_frac <= 0:
        return np.zeros((h, w), dtype=bool)
    bw = int(w * width_frac)
    bh = int(h * height_frac)
    r1 = int(h * (1 - inset_frac))
    r0 = max(0, r1 - bh)
    c1 = int(w * (1 - inset_frac))
    c0 = max(0, c1 - bw)
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.38, color=(0, 180, 255)) -> np.ndarray:
    """
    Colorize mask (open gap or filled region etc.) on top of grayscale background.
    """
    out = rgb.copy()
    col = np.zeros_like(out)
    col[..., 0], col[..., 1], col[..., 2] = color
    out = (out * (1 - alpha) + col * alpha * mask[..., None]).astype(np.uint8)
    return out


def segment_open(
    corr_gray01: np.ndarray,
    std_sigma: float,
    sens: float,
    open_r: int,
    close_r: int,
    refine_r: int,
    min_area: int,
    open_class: str,
) -> np.ndarray:
    """
    Texture-based segmentation of 'open wound' vs 'cells'.

    open_class:
      - 'low'  -> interpret LOW texture as OPEN WOUND (typical phase contrast)
      - 'high' -> interpret HIGH texture as OPEN
    sens:
      - negative makes threshold stricter (shrinks the wound mask, counts faint cells as cells)
      - positive makes threshold looser (grows wound mask)

    refine_r:
      - extra erosion (in pixels) to shrink the wound mask,
        so faint/light cells inside the gap are treated as CELLS, not 'open'.
    """
    # 1. local texture map
    s = local_std(corr_gray01, sigma=std_sigma)

    # 2. Otsu threshold, sensitivity shift
    thr = threshold_otsu(s)
    thr = thr * (1.0 + sens)
    if open_class == "low":
        mask = s <= thr
    else:
        mask = s >= thr

    # 3. Morphology cleanup
    if open_r > 0:
        mask = binary_opening(mask, footprint=disk(open_r))
    if close_r > 0:
        mask = binary_closing(mask, footprint=disk(close_r))
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)

    # 4. Extra erosion to shrink the wound area and "promote" faint cells
    if refine_r > 0:
        mask = binary_erosion(mask, footprint=disk(refine_r))

    return mask


def analyze_image(
    pil_image: Image.Image,
    roi_margin: float,
    bg_sigma: float,
    std_sigma: float,
    sens: float,
    open_r: int,
    close_r: int,
    refine_r: int,
    min_area: int,
    open_class: str,
    sbw: float,
    sbh: float,
):
    """
    Process ONE timepoint image.
    Returns dict with:
      - raw_open_pct: % of ROI still labeled 'open'
      - keep_mask: mask of valid analysis region
      - open_valid_mask: wound/open mask restricted to keep_mask
      - overlay_png: PNG bytes of QC overlay (without text label yet)
      - shape: (H, W)
    """
    # --- prep image ---
    rgb = np.array(pil_image.convert("RGB"))
    rgb = _downscale(rgb, max_side=1600)
    gray = rgb2gray(rgb).astype(np.float32)

    # --- illumination correction ---
    corr = illumination_correct(gray, sigma_bg=bg_sigma)  # float [0..1]

    # --- texture segmentation of "open gap" ---
    mask_open_full = segment_open(
        corr_gray01=corr,
        std_sigma=std_sigma,
        sens=sens,
        open_r=open_r,
        close_r=close_r,
        refine_r=refine_r,
        min_area=min_area,
        open_class=open_class,
    )

    # --- ROI (center crop) & scale bar exclusion ---
    h, w = mask_open_full.shape
    roi = center_roi_mask(h, w, roi_margin)
    sb_mask = mask_scale_bar(h, w, width_frac=sbw, height_frac=sbh)
    keep_mask = roi & ~sb_mask  # pixels we trust

    # restrict wound/open mask to "keep"
    open_valid_mask = mask_open_full & keep_mask

    # --- quantify % open ---
    keep_pix = int(keep_mask.sum())
    raw_open_pct = 100.0 * (open_valid_mask.sum() / max(1, keep_pix))

    # --- build QC overlay image ---
    # background = illumination-corrected grayscale
    base = (corr * 255).astype(np.uint8)
    base_rgb = np.repeat(base[..., None], 3, axis=2)

    # colorize the CURRENT "open" mask in cyan/blue
    overlay_img = overlay_mask(base_rgb, open_valid_mask, alpha=0.42, color=(0, 180, 255))

    # draw ROI rectangle in green
    rr0, rr1 = int(h * roi_margin), int(h * (1 - roi_margin))
    cc0, cc1 = int(w * roi_margin), int(w * (1 - roi_margin))
    overlay_img[rr0:rr1, [cc0, cc1 - 1]] = (0, 255, 90)
    overlay_img[[rr0, rr1 - 1], cc0:cc1] = (0, 255, 90)

    # gray out scale-bar area
    overlay_img[sb_mask] = (200, 200, 200)

    # encode PNG (no text yet)
    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")

    return {
        "raw_open_pct": raw_open_pct,
        "keep_mask": keep_mask,
        "open_valid_mask": open_valid_mask,
        "overlay_png": buf.getvalue(),
        "shape": (h, w),
    }


def compute_metrics(results_by_t: dict):
    """
    Compute all time-series metrics:
      - Raw Open %
      - Wound Confluence %
      - Relative Wound Density %
      plus baseline key for normalization.

    Definitions (mirroring IncuCyte-style logic):
      wound ROI = the open area at baseline (earliest timepoint, usually 0h)
      cell region = everything else in the analysis ROI outside that wound ROI

    For each time t:
      w(t) = cell density *inside* the baseline wound ROI at time t
      c(t) = cell density in the surrounding cell region at time t
      Wound Confluence % = 100 * w(t)
      RWD % = 100 * ( w(t) - w(0) ) / ( c(t) - w(0) )
    """
    # Pick baseline = earliest timepoint available (ideally 0h)
    baseline_key = sorted(results_by_t.keys())[0]
    base = results_by_t[baseline_key]

    h0, w0 = base["shape"]

    # baseline wound region = pixels baseline said are "open"
    wound_region_mask0 = base["open_valid_mask"].copy()  # True = wound at baseline
    keep_mask0 = base["keep_mask"].copy()

    # "cell region" baseline = valid analysis area minus the wound baseline
    cell_region_mask0 = keep_mask0 & (~wound_region_mask0)

    wound_region_pix = max(1, int(wound_region_mask0.sum()))
    cell_region_pix = max(1, int(cell_region_mask0.sum()))

    # helper to get densities at any time t
    def densities_at_time(res_t):
        # skip if size changed
        if res_t["shape"] != (h0, w0):
            return np.nan, np.nan

        openmask_t = res_t["open_valid_mask"]  # True means "still open gap"
        cellmask_t = ~openmask_t               # True means "cell present"

        # fraction of wound ROI now filled by cells
        w_t = (cellmask_t & wound_region_mask0).sum() / wound_region_pix

        # fraction of outside-cell region that has cells
        c_t = (cellmask_t & cell_region_mask0).sum() / cell_region_pix

        return w_t, c_t

    # baseline densities
    w0, c0 = densities_at_time(base)

    metrics = {
        "raw_open_pct": {},
        "wound_confluence_pct": {},
        "rwd_pct": {},
    }

    for t, res in results_by_t.items():
        w_t, c_t = densities_at_time(res)

        # Raw Open %
        raw_open = res["raw_open_pct"]

        # Wound Confluence % = % of baseline wound region now occupied by cells
        # (will be low at 0h, higher later)
        if not np.isnan(w_t):
            wound_conf_pct = 100.0 * w_t
        else:
            wound_conf_pct = np.nan

        # Relative Wound Density %:
        # RWD(t) = 100 * ( w(t) - w(0) ) / ( c(t) - w(0) )
        # Guaranteed ~0% at baseline and ~100% when wound matches outside-cell density.
        denom = (c_t - w0) if (not np.isnan(c_t) and not np.isnan(w0)) else np.nan
        if denom is not None and denom != 0 and not np.isnan(denom) and not np.isnan(w_t):
            rwd_pct = 100.0 * (w_t - w0) / denom
        else:
            rwd_pct = np.nan

        metrics["raw_open_pct"][t] = raw_open
        metrics["wound_confluence_pct"][t] = wound_conf_pct
        metrics["rwd_pct"][t] = rwd_pct

    return metrics, baseline_key


def summarize_series(metrics: dict, baseline_key: int):
    """
    Build final DataFrame with:
      Raw Open %
      Relative Open %
      Closure %
      Wound Confluence %
      Relative Wound Density %

    Relative Open % and Closure % are based on Raw Open % normalized
    to the baseline timepoint.
    """
    raw_open = metrics["raw_open_pct"]
    wound_conf = metrics["wound_confluence_pct"]
    rwd = metrics["rwd_pct"]

    baseline_raw = raw_open.get(baseline_key, np.nan)

    rows = []
    for t in sorted(raw_open.keys()):
        raw_val = raw_open[t]
        # relative open vs baseline
        if baseline_raw and not np.isnan(baseline_raw):
            rel_open = (raw_val / baseline_raw) * 100.0
        else:
            rel_open = np.nan

        closure = 100.0 - rel_open if rel_open == rel_open else np.nan

        rows.append(
            {
                "Hours": t,
                "Raw Open %": raw_val,
                "Relative Open %": rel_open,
                "Closure %": closure,
                "Wound Confluence %": wound_conf.get(t, np.nan),
                "Relative Wound Density %": rwd.get(t, np.nan),
            }
        )

    df = pd.DataFrame(rows).set_index("Hours")
    return df, baseline_raw


def _try_load_font(size: int):
    """Try a sane TTF for annotations; fallback to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate_bytes(
    img_bytes: bytes,
    text: str,
    corner: str = "br",
    scale: float = 0.035,
    fg=(255, 221, 0, 255),     # bright yellow
    shadow=(0, 0, 0, 255),     # black outline
):
    """
    Draw a high-contrast label block onto PNG bytes.
    Supports multi-line text.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Font size proportional to width
    fsize = max(16, int(W * scale))
    font = _try_load_font(fsize)

    # Measure text block
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(6, fsize // 4)

    # Corner placement
    if corner == "br":
        x0, y0 = W - tw - 2 * pad - 8, H - th - 2 * pad - 8
    elif corner == "bl":
        x0, y0 = 8, H - th - 2 * pad - 8
    elif corner == "tr":
        x0, y0 = W - tw - 2 * pad - 8, 8
    else:  # "tl"
        x0, y0 = 8, 8
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad

    # Semi-transparent panel
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 150))

    # Text with outline
    tx, ty = x0 + pad, y0 + pad
    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ]:
        draw.text((tx + dx, ty + dy), text, fill=shadow, font=font)
    draw.text((tx, ty), text, fill=fg, font=font)

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


# ------------------------------- UI -----------------------------------
st.title("Migration Image Analysis (Benchmark vs IncuCyte-style metrics)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")

    concentration = st.selectbox("Concentration", CONCENTRATIONS, index=1)

    roi_margin = st.slider(
        "ROI margin (0 = full image)",
        0.00,
        0.25,
        0.00,
        0.01,
        help="Crop edges to avoid artefacts; set 0 for full-frame analysis",
    )

    bg_sigma = st.slider(
        "BG sigma",
        10.0,
        80.0,
        42.0,
        2.0,
        help="Illumination correction blur size",
    )

    std_sigma = st.slider(
        "Texture sigma",
        3.0,
        30.0,
        12.0,
        1.0,
        help="Neighborhood for local texture (cell vs gap)",
    )

    sens = st.slider(
        "Sensitivity",
        -0.35,
        0.35,
        0.00,
        0.01,
        help="More negative = stricter open mask (faint cells count as CELLS)",
    )

    cleanup_label = st.selectbox(
        "Cleanup (opening/closing)",
        ["Light (2/2)", "Med (3/3)", "Strong (4/3)", "Custom (1/1)"],
        index=1,
    )
    map_r = {
        "Light (2/2)": (2, 2),
        "Med (3/3)": (3, 3),
        "Strong (4/3)": (4, 3),
        "Custom (1/1)": (1, 1),
    }
    open_r, close_r = map_r[cleanup_label]

    refine_gap = st.slider(
        "Gap strictness (erosion px)",
        0,
        5,
        1,
        1,
        help="Higher → wound mask shrinks, so faint/light cells in the gap stop being counted as 'open'.",
    )

    open_mode = st.selectbox(
        "Open class",
        ["Low texture", "High texture", "Auto"],
        index=0,
        help="If closure curve looks upside down, try 'High' or 'Auto'",
    )

    sb_mask = st.checkbox("Mask scale bar", value=False)
    sb_width = st.slider(
        "Scale-bar width (frac)", 0.00, 0.30, 0.12, 0.01, disabled=not sb_mask
    )
    sb_height = st.slider(
        "Scale-bar height (frac)", 0.00, 0.20, 0.06, 0.01, disabled=not sb_mask
    )

# Uploads
st.markdown("#### Upload images (same well / field of view)")
u1, u2, u3, u4 = st.columns(4)
uploads = {}
for t, col in zip(TIMEPOINTS, [u1, u2, u3, u4]):
    with col:
        f = st.file_uploader(
            f"{t} h", type=["png", "jpg", "jpeg", "tiff"], key=f"tp{t}", label_visibility="visible"
        )
        if f:
            try:
                img = Image.open(f).convert("RGB")
                uploads[t] = img
                st.image(img, caption=f"{t}h", use_container_width=True)
            except Exception:
                st.error("Could not read image.")

st.divider()
go = st.button("▶️ Analyze", type="primary", use_container_width=True)

# ------------------------------ ANALYSIS ------------------------------
if go and uploads:
    # We'll run segmentation for each uploaded timepoint
    results_by_t = {}

    # pick initial mode: "Auto" starts as "low", may flip later
    chosen_mode = (
        "low"
        if open_mode == "Low texture"
        else ("high" if open_mode == "High texture" else "low")
    )

    for t in sorted(uploads.keys()):
        res_t = analyze_image(
            uploads[t],
            roi_margin=roi_margin,
            bg_sigma=bg_sigma,
            std_sigma=std_sigma,
            sens=sens,
            open_r=open_r,
            close_r=close_r,
            refine_r=refine_gap,
            min_area=600,
            open_class=chosen_mode,
            sbw=(sb_width if sb_mask else 0.0),
            sbh=(sb_height if sb_mask else 0.0),
        )
        results_by_t[t] = res_t

    # AUTO POLARITY CHECK:
    # If later timepoints look "more open than baseline"
    # (which is biologically backwards for closure),
    # flip interpretation to "high texture = open" and recompute.
    if open_mode == "Auto":
        if len(results_by_t) > 1:
            keys_sorted = sorted(results_by_t.keys())
            base_key = keys_sorted[0]
            base_raw = results_by_t[base_key]["raw_open_pct"]

            later_vals = [
                results_by_t[k]["raw_open_pct"]
                for k in keys_sorted
                if k != base_key
            ]
            if later_vals and np.nanmedian(later_vals) > base_raw:
                # Flip to "high"
                results_by_t = {}
                for t in keys_sorted:
                    res_t = analyze_image(
                        uploads[t],
                        roi_margin=roi_margin,
                        bg_sigma=bg_sigma,
                        std_sigma=std_sigma,
                        sens=sens,
                        open_r=open_r,
                        close_r=close_r,
                        refine_r=refine_gap,
                        min_area=600,
                        open_class="high",
                        sbw=(sb_width if sb_mask else 0.0),
                        sbh=(sb_height if sb_mask else 0.0),
                    )
                    results_by_t[t] = res_t

    # --- compute metrics across timepoints (Incucyte-style + ours) ---
    metrics, baseline_key = compute_metrics(results_by_t)
    df, baseline_raw = summarize_series(metrics, baseline_key)

    # === Layout: left (images w/ annotations), right (table + plot) ===
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Detection overlays (annotated)")
        grid_cols = st.columns(2)  # 2×2 grid

        for i, t in enumerate(sorted(results_by_t.keys())):
            # pull metrics for label
            row_exists = t in df.index
            raw_val = df.loc[t, "Raw Open %"] if row_exists else np.nan
            rel_open = df.loc[t, "Relative Open %"] if row_exists else np.nan
            closure = df.loc[t, "Closure %"] if row_exists else np.nan
            wound_conf = (
                df.loc[t, "Wound Confluence %"] if row_exists else np.nan
            )
            rwd_val = (
                df.loc[t, "Relative Wound Density %"] if row_exists else np.nan
            )

            # Build multi-line label
            # Line1: time + raw + closure
            # Line2: confluence + RWD
            if np.isnan(rel_open):
                line1 = f"{t}h — Open {raw_val:.2f}%"
            else:
                line1 = (
                    f"{t}h — Open {raw_val:.2f}% | Close {closure:.1f}%"
                )
            line2 = f"Conf {wound_conf:.1f}% | RWD {rwd_val:.1f}%"
            label_text = line1 + "\n" + line2

            annotated_png = annotate_bytes(
                results_by_t[t]["overlay_png"],
                label_text,
                corner="br",
                scale=0.04,
                fg=(255, 221, 0, 255),
            )

            with grid_cols[i % 2]:
                st.image(annotated_png, use_container_width=True)

    with right:
        st.markdown("#### 📊 Baseline-normalized results + IncuCyte-style metrics")
        st.dataframe(
            df.style.format("{:.2f}"),
            use_container_width=True,
            height=320,
        )

        # Plot Closure % (classic closure curve)
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        xvals = df.index.values
        yvals = df["Closure %"].values.astype(float)

        if np.isfinite(yvals).any():
            ymin = float(np.nanmin(yvals))
            ymax = float(np.nanmax(yvals))
            ylo = min(-10, ymin - 5) if np.isfinite(ymin) else -10
            yhi = max(100, ymax + 5) if np.isfinite(ymax) else 100
        else:
            ylo, yhi = -10, 100

        ax.plot(
            xvals,
            yvals,
            marker="o",
            linewidth=2,
            color="#009E73",
            label=f"{concentration} p/mL",
        )
        ax.set_xlabel("Hours")
        ax.set_ylabel("Closure % (relative to baseline)")
        ax.set_title(
            f"Closure — baseline {baseline_raw:.2f}% open (time {baseline_key}h)",
            fontsize=11,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)

        # CSV download
        csv = df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"results_{concentration}.csv",
            use_container_width=True,
        )

else:
    st.info(
        "Upload at least one image and click **Analyze**. "
        "For baseline-normalized metrics (Closure %, Wound Confluence %, RWD %), "
        "include the earliest timepoint (ideally 0h)."
    )

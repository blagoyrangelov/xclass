"""SED validation: predicted vs. observed HST magnitudes for training sources.

Two validation modes run in sequence:

Mode 1 — Real HSC photometry (preferred):
  Loads the per-source HSC v3 cache from data/query_cache/hsc_training/
  (first choice) or data/query_cache/hsc_phat/ (M31 PHAT cache, fallback).
  Falls back to live MAST queries if neither cache has data.

Mode 2 — Internal SED quality (always runs):
  Compares SED-predicted magnitudes against the PS1 input photometry in the
  two closest filter pairs (F475W vs PS1-g, F814W vs PS1-i) as a consistency
  check. The predicted values should not differ by more than the fitting
  residuals captured in xclass_fit_chi2red.

Outputs
-------
figures/sed_validation_scatter.pdf   — predicted vs. observed scatter, 6 panels
figures/sed_validation_residuals.pdf — residual histograms, 6 panels
figures/sed_validation_quality.pdf   — SED fit quality diagnostics (4 panels)
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATED_CATALOG = PROJECT_ROOT / "data" / "processed" / "translated_catalog.csv"
HSC_TRAINING_CACHE = PROJECT_ROOT / "data" / "query_cache" / "hsc_training"
HSC_PHAT_CACHE = PROJECT_ROOT / "data" / "query_cache" / "hsc_phat"
FIGURES_DIR = PROJECT_ROOT / "figures"

HSC_API_URL = "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/detailed"
SEARCH_RADIUS_ARCSEC = 1.5
HSC_SLEEP_SEC = 0.1
MAX_RETRIES = 2

# PHAT filter columns in translated_catalog  →  short filter name
PHAT_FILTERS: dict[str, str] = {
    "UVIS_F275W_pred": "F275W",
    "UVIS_F336W_pred": "F336W",
    "ACS_F475W_pred": "F475W",
    "ACS_F814W_pred": "F814W",
    "IR_F110W_pred": "F110W",
    "IR_F160W_pred": "F160W",
}

# AB − Vega offset for each PHAT filter (m_AB = m_Vega + delta)
# From Dalcanton et al. 2012 Table A1 / Williams et al. 2014
VEGA_TO_AB: dict[str, float] = {
    "F275W": 1.52,
    "F336W": 1.17,
    "F475W": 0.10,
    "F814W": 0.42,
    "F110W": 0.77,
    "F160W": 1.27,
}

# PS1 zeropoints: all PS1 mags are already in AB (ZP ≈ 3631 Jy)
PS1_AB: dict[str, float] = {
    "ps1_g": 0.0, "ps1_r": 0.0, "ps1_i": 0.0, "ps1_z": 0.0, "ps1_y": 0.0
}

# Closest PS1 ↔ HST filter pairs for internal consistency check
PS1_TO_HST = [
    ("ps1_g", "ACS_F475W_pred", "g vs F475W"),
    ("ps1_i", "ACS_F814W_pred", "i vs F814W"),
]

# ApJ color cycle for classes
CLASS_ORDER = ["AGN", "LMXB", "HMXB", "CV", "LM-STAR", "HM-STAR"]
CLASS_COLORS = {
    "AGN":     "#1f77b4",
    "LMXB":    "#d62728",
    "HMXB":    "#ff7f0e",
    "CV":      "#9467bd",
    "LM-STAR": "#2ca02c",
    "HM-STAR": "#8c564b",
    "SNR":     "#e377c2",
}
CLASS_MARKERS = {
    "AGN": "o", "LMXB": "s", "HMXB": "^",
    "CV": "D", "LM-STAR": "v", "HM-STAR": "P",
}

FILTER_ORDER = ["F275W", "F336W", "F475W", "F814W", "F110W", "F160W"]
FILTER_LABELS = {
    "F275W": r"$F275W$ (WFC3/UVIS)",
    "F336W": r"$F336W$ (WFC3/UVIS)",
    "F475W": r"$F475W$ (ACS/WFC)",
    "F814W": r"$F814W$ (ACS/WFC)",
    "F110W": r"$F110W$ (WFC3/IR)",
    "F160W": r"$F160W$ (WFC3/IR)",
}

log = logging.getLogger("validate_sed")


# ---------------------------------------------------------------------------
# ApJ plot style
# ---------------------------------------------------------------------------
def _apj_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 0.8,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# HSC query helpers
# ---------------------------------------------------------------------------
def _safe_id(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(source_id))


def _cache_path(source_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_safe_id(source_id)}.pkl"


def _load_cache(source_id: str, cache_dir: Path):
    p = _cache_path(source_id, cache_dir)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_cache(source_id: str, cache_dir: Path, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(source_id, cache_dir), "wb") as f:
        pickle.dump(data, f)


def _hsc_live_query(ra: float, dec: float) -> list[dict]:
    """Single attempt at live MAST HSC query; returns [] on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                HSC_API_URL,
                params={
                    "ra": ra, "dec": dec,
                    "radius": SEARCH_RADIUS_ARCSEC / 3600.0,
                    "format": "json", "pagesize": 50000,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                col_names = [c["name"] for c in data.get("info", [])]
                raw_rows = data.get("data", []) or []
                if col_names and raw_rows and isinstance(raw_rows[0], (list, tuple)):
                    return [dict(zip(col_names, r)) for r in raw_rows]
                return raw_rows
            if isinstance(data, list):
                return data
            return []
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                log.debug("HSC live query failed: %s", exc)
    return []


def _query_source_hsc(
    source_id: str,
    ra: float,
    dec: float,
    live_allowed: bool,
) -> list[dict]:
    """Return raw HSC detections for source, with cascading cache lookup."""
    # 1. Prefer hsc_training cache
    rows = _load_cache(source_id, HSC_TRAINING_CACHE)
    if rows is not None:
        return rows
    # 2. Fall back to hsc_phat cache (same filenames)
    rows = _load_cache(source_id, HSC_PHAT_CACHE)
    if rows is not None:
        return rows
    # 3. Live query
    if not live_allowed:
        return []
    time.sleep(HSC_SLEEP_SEC)
    rows = _hsc_live_query(ra, dec)
    _save_cache(source_id, HSC_TRAINING_CACHE, rows)
    return rows


def _extract_hsc_photometry(
    source_id: str,
    ra: float,
    dec: float,
    detections: list[dict],
) -> dict[str, float]:
    """Return {filter: vega_mag} for PHAT filters within search radius."""
    if not detections:
        return {}
    src_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    valid: list[dict] = []
    for det in detections:
        det_ra = det.get("MatchRA") or det.get("RA")
        det_dec = det.get("MatchDec") or det.get("Dec")
        mag = det.get("MagAper2")
        flt = str(det.get("Filter", "")).strip()
        if "/" in flt:
            flt = flt.split("/")[-1]
        if flt not in VEGA_TO_AB:
            continue
        try:
            det_ra, det_dec, mag = float(det_ra), float(det_dec), float(mag)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(det_ra) or not np.isfinite(det_dec) or not np.isfinite(mag):
            continue
        sep = src_coord.separation(
            SkyCoord(ra=det_ra * u.deg, dec=det_dec * u.deg)
        ).arcsec
        if sep <= SEARCH_RADIUS_ARCSEC:
            valid.append({"filter": flt, "mag": mag, "sep": sep,
                          "match_id": det.get("MatchID")})
    if not valid:
        return {}
    det_df = pd.DataFrame(valid)
    best_id = det_df.groupby("match_id")["sep"].mean().idxmin()
    best = det_df[det_df["match_id"] == best_id]
    result: dict[str, float] = {}
    for flt, grp in best.groupby("filter"):
        if flt in VEGA_TO_AB:
            result[flt] = float(np.median(grp["mag"].values))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _apj_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    HSC_TRAINING_CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load translated training catalog
    # ------------------------------------------------------------------
    log.info("Loading translated catalog …")
    df = pd.read_csv(TRANSLATED_CATALOG, low_memory=False)
    pred_cols = list(PHAT_FILTERS.keys())
    has_pred = df[pred_cols].notna().any(axis=1)
    df = df[has_pred & (df["Class"] != "SNR")].reset_index(drop=True)
    df["xray_id_safe"] = df["xray_name"].apply(_safe_id)
    log.info("  %d non-SNR sources with PHAT predictions", len(df))

    # ------------------------------------------------------------------
    # 2. Detect which sources have cached HSC data (training or phat)
    # ------------------------------------------------------------------
    cached_training = {f.stem for f in HSC_TRAINING_CACHE.glob("*.pkl")}
    cached_phat = {f.stem for f in HSC_PHAT_CACHE.glob("*.pkl")}
    already_cached = cached_training | cached_phat

    n_cached = df["xray_id_safe"].isin(already_cached).sum()
    log.info("  %d / %d sources already in HSC cache", n_cached, len(df))

    # Test live connectivity once
    live_ok = _test_live_connectivity()
    if not live_ok:
        log.warning("MAST HSC API not reachable — skipping live queries.")
        log.info("  Using cached data only (%d sources).", n_cached)

    # ------------------------------------------------------------------
    # 3. Query / retrieve HSC for each training source
    # ------------------------------------------------------------------
    hsc_obs: list[dict[str, float]] = []
    n_queried = 0
    for _, row in df.iterrows():
        src_id = row["xray_id_safe"]
        ra = float(row["xray_ra"])
        dec = float(row["xray_dec"])
        detections = _query_source_hsc(src_id, ra, dec, live_ok)
        phot = _extract_hsc_photometry(src_id, ra, dec, detections)
        hsc_obs.append(phot)
        if live_ok:
            n_queried += 1
            if n_queried % 500 == 0:
                n_hit = sum(1 for p in hsc_obs if p)
                log.info("  %d / %d queried  (%d with HSC match)", n_queried, len(df), n_hit)

    n_hsc = sum(1 for p in hsc_obs if p)
    log.info("HSC retrieval complete: %d sources with ≥1 PHAT-filter match", n_hsc)

    # ------------------------------------------------------------------
    # 4. Build HSC comparison table
    # ------------------------------------------------------------------
    hsc_records: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        obs = hsc_obs[i]
        if not obs:
            continue
        for pred_col, flt in PHAT_FILTERS.items():
            if flt not in obs:
                continue
            pred_mag = row.get(pred_col)
            if pred_mag is None or not np.isfinite(float(pred_mag)):
                continue
            ab_mag = obs[flt] + VEGA_TO_AB[flt]
            hsc_records.append({
                "source_id": row["xray_name"],
                "Class": row["Class"],
                "filter": flt,
                "pred_mag": float(pred_mag),
                "obs_mag_ab": ab_mag,
                "obs_mag_vega": obs[flt],
                "residual": float(pred_mag) - ab_mag,
                "source": "HSC",
            })
    cmp_hsc = pd.DataFrame(hsc_records)
    log.info("HSC comparison table: %d pairs from %d unique sources",
             len(cmp_hsc),
             cmp_hsc["source_id"].nunique() if len(cmp_hsc) else 0)

    # ------------------------------------------------------------------
    # 5. Build PS1 internal consistency table (quality-filtered)
    #    Only sources with well-constrained SED fits (chi2_red < CHI2_CUT)
    #    and physical magnitudes are included.
    # ------------------------------------------------------------------
    CHI2_CUT = 10.0  # reduced chi^2 threshold for "good" SED fit
    MAG_MIN, MAG_MAX = 8.0, 32.0  # physical magnitude range (AB)

    has_chi2 = "xclass_fit_chi2red" in df.columns
    if has_chi2:
        good_mask = (
            df["xclass_fit_chi2red"].notna() &
            (df["xclass_fit_chi2red"] < CHI2_CUT)
        )
    else:
        good_mask = pd.Series(True, index=df.index)

    n_good = good_mask.sum()
    log.info(
        "  %d / %d sources pass chi2_red < %.0f quality cut",
        n_good, len(df), CHI2_CUT,
    )

    ps1_records: list[dict] = []
    for pred_col, flt in PHAT_FILTERS.items():
        if flt not in ("F475W", "F814W"):
            continue
        ps1_col = "ps1_g" if flt == "F475W" else "ps1_i"
        sub = df[good_mask][[pred_col, ps1_col, "Class"]].dropna()
        for _, row in sub.iterrows():
            pred_mag = float(row[pred_col])
            obs_mag = float(row[ps1_col])
            if not (np.isfinite(pred_mag) and np.isfinite(obs_mag)):
                continue
            if not (MAG_MIN < pred_mag < MAG_MAX and MAG_MIN < obs_mag < MAG_MAX):
                continue
            ps1_records.append({
                "Class": row["Class"],
                "filter": flt,
                "pred_mag": pred_mag,
                "obs_mag_ab": obs_mag,
                "obs_label": ps1_col,
                "residual": pred_mag - obs_mag,
                "source": f"PS1 ({ps1_col})",
            })
    cmp_ps1 = pd.DataFrame(ps1_records)
    log.info(
        "PS1 consistency table: %d pairs across F475W/F814W (chi2 < %.0f)",
        len(cmp_ps1), CHI2_CUT,
    )

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    _print_summary(cmp_hsc, cmp_ps1, df)

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    _make_scatter_figure(cmp_hsc, cmp_ps1,
                         FIGURES_DIR / "sed_validation_scatter.pdf")
    _make_residual_figure(cmp_hsc, cmp_ps1,
                          FIGURES_DIR / "sed_validation_residuals.pdf")
    _make_quality_figure(df, FIGURES_DIR / "sed_validation_quality.pdf")

    log.info("All figures saved to %s/", FIGURES_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _test_live_connectivity() -> bool:
    try:
        r = requests.get(HSC_API_URL,
                         params={"ra": 10.684, "dec": 41.269,
                                 "radius": 0.0001, "format": "json"},
                         timeout=8)
        return r.status_code == 200
    except Exception:
        return False


def _print_summary(
    cmp_hsc: pd.DataFrame,
    cmp_ps1: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("SED VALIDATION SUMMARY")
    print("=" * 72)

    print(f"\nTraining catalog: {len(df)} non-SNR sources with PHAT predictions")
    print(f"  Classes: {dict(df['Class'].value_counts())}")

    if len(cmp_hsc):
        print(f"\n--- Mode 1: Real HSC photometry ({cmp_hsc['source_id'].nunique()} "
              f"sources, {len(cmp_hsc)} filter-pairs) ---")
        print(f"  Classes: {dict(cmp_hsc.groupby('Class')['source_id'].nunique())}")
        print()
        print("  Per-filter statistics [pred − obs_AB]:")
        for flt, g in cmp_hsc.groupby("filter"):
            r = g["residual"]
            mad = float(np.median(np.abs(r - r.median())))
            print(f"  {flt:6s}  n={len(r):3d}  "
                  f"mean={r.mean():+.3f}  median={r.median():+.3f}  "
                  f"std={r.std():.3f}  MAD={mad:.3f}")
    else:
        print("\n  Mode 1 (HSC real photometry): No data available.")
        print("  Run again with internet access to query MAST HSC v3.")

    if len(cmp_ps1):
        label_chi2 = cmp_ps1["obs_label"].iloc[0] if len(cmp_ps1) else "—"
        print(f"\n--- Mode 2: PS1 internal consistency "
              f"(chi2_red < 10, {len(cmp_ps1)} pairs) ---")
        for flt, g in cmp_ps1.groupby("filter"):
            r = g["residual"]
            mad = float(np.median(np.abs(r - r.median())))
            label = g["obs_label"].iloc[0]
            print(f"  {flt:6s} vs {label:7s}  n={len(r):5d}  "
                  f"mean={r.mean():+.3f}  median={r.median():+.3f}  "
                  f"std={r.std():.3f}  MAD={mad:.3f}")

        if len(cmp_ps1):
            print()
            print("  Per-class mean residuals (Mode 2):")
            pivot = cmp_ps1.pivot_table(
                values="residual", index="Class", columns="filter", aggfunc="mean"
            )
            print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Figure A — Predicted vs. observed scatter (6 panels)
# ---------------------------------------------------------------------------
def _make_scatter_figure(
    cmp_hsc: pd.DataFrame,
    cmp_ps1: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    axes = axes.ravel()

    has_hsc = len(cmp_hsc) > 0
    has_ps1 = len(cmp_ps1) > 0

    for ax, flt in zip(axes, FILTER_ORDER):
        sub_hsc = cmp_hsc[cmp_hsc["filter"] == flt] if has_hsc else pd.DataFrame()
        sub_ps1 = cmp_ps1[cmp_ps1["filter"] == flt] if has_ps1 else pd.DataFrame()

        if sub_hsc.empty and sub_ps1.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            ax.set_title(FILTER_LABELS[flt], pad=4)
            continue

        # Determine axis limits from available data
        all_pred = pd.concat([sub_hsc.get("pred_mag", pd.Series(dtype=float)),
                               sub_ps1.get("pred_mag", pd.Series(dtype=float))]).dropna()
        all_obs = pd.concat([sub_hsc.get("obs_mag_ab", pd.Series(dtype=float)),
                              sub_ps1.get("obs_mag_ab", pd.Series(dtype=float))]).dropna()
        if all_pred.empty or all_obs.empty:
            ax.set_title(FILTER_LABELS[flt], pad=4)
            continue

        lo = min(all_pred.min(), all_obs.min()) - 0.5
        hi = max(all_pred.max(), all_obs.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, zorder=1, label="1:1")

        # Mode 2 (PS1) — plotted first, smaller, lighter
        if not sub_ps1.empty:
            for cls in CLASS_ORDER:
                s = sub_ps1[sub_ps1["Class"] == cls]
                if s.empty:
                    continue
                ax.scatter(
                    s["obs_mag_ab"], s["pred_mag"],
                    s=6, marker=CLASS_MARKERS.get(cls, "o"),
                    color=CLASS_COLORS.get(cls, "gray"),
                    alpha=0.35, linewidths=0.0, zorder=2,
                )

        # Mode 1 (HSC real) — plotted on top, larger, fully opaque
        if not sub_hsc.empty:
            for cls in CLASS_ORDER:
                s = sub_hsc[sub_hsc["Class"] == cls]
                if s.empty:
                    continue
                ax.scatter(
                    s["obs_mag_ab"], s["pred_mag"],
                    s=30, marker=CLASS_MARKERS.get(cls, "o"),
                    color=CLASS_COLORS.get(cls, "gray"),
                    alpha=0.9, linewidths=0.3, edgecolors="k",
                    zorder=3,
                )

        # Annotate offset and scatter (use HSC if available, else PS1)
        sub_stat = sub_hsc if not sub_hsc.empty else sub_ps1
        med_off = sub_stat["residual"].median()
        std_off = sub_stat["residual"].std()
        src_label = "HSC" if not sub_hsc.empty else "PS1"
        ax.text(0.04, 0.96,
                rf"$\Delta={med_off:+.2f}$, $\sigma={std_off:.2f}$" + f"\n({src_label})",
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.8))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(FILTER_LABELS[flt], pad=4)
        ax.set_xlabel(r"$m_{\rm obs,AB}$", labelpad=2)
        ax.set_ylabel(r"$m_{\rm pred,AB}$", labelpad=2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Legend for classes
    handles = [
        plt.Line2D([0], [0], marker=CLASS_MARKERS.get(c, "o"), color="w",
                   markerfacecolor=CLASS_COLORS.get(c, "gray"), markersize=5, label=c)
        for c in CLASS_ORDER
    ]
    # Symbol legend for data source
    legend_extra = [
        plt.scatter([], [], s=6, c="gray", alpha=0.4, label="PS1 input (Mode 2)"),
        plt.scatter([], [], s=30, c="gray", alpha=0.9,
                    edgecolors="k", linewidths=0.3, label="HSC real (Mode 1)"),
    ]
    fig.legend(handles=handles + legend_extra,
               loc="lower center", ncol=min(8, len(handles) + 2),
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))

    note = ("Large circles = real HSC photometry (Mode 1). "
            "Small dots = PS1 input cross-check (Mode 2).")
    fig.text(0.5, -0.07, note, ha="center", fontsize=6.5, color="0.5",
             style="italic")
    fig.suptitle(
        "SED Translation Validation: Predicted vs. Observed HST Magnitudes",
        fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Figure B — Residual histograms (6 panels)
# ---------------------------------------------------------------------------
def _make_residual_figure(
    cmp_hsc: pd.DataFrame,
    cmp_ps1: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    axes = axes.ravel()

    for ax, flt in zip(axes, FILTER_ORDER):
        sub_hsc = cmp_hsc[cmp_hsc["filter"] == flt] if len(cmp_hsc) else pd.DataFrame()
        sub_ps1 = cmp_ps1[cmp_ps1["filter"] == flt] if len(cmp_ps1) else pd.DataFrame()

        ax.set_title(FILTER_LABELS[flt], pad=4)
        ax.set_xlabel(r"$m_{\rm pred} - m_{\rm obs,AB}$  [mag]", labelpad=2)
        ax.set_ylabel("N", labelpad=2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if sub_hsc.empty and sub_ps1.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            continue

        # Determine bin range from all available data
        all_resid = pd.concat([
            sub_hsc.get("residual", pd.Series(dtype=float)),
            sub_ps1.get("residual", pd.Series(dtype=float)),
        ]).dropna()
        if all_resid.empty:
            continue

        p2, p98 = np.percentile(all_resid, [1, 99])
        margin = max(0.5, (p98 - p2) * 0.15)
        bins = np.linspace(p2 - margin, p98 + margin, 45)

        # Mode 2 (PS1) stacked histogram
        if not sub_ps1.empty:
            class_resid_ps1 = [
                sub_ps1[sub_ps1["Class"] == cls]["residual"].dropna().values
                for cls in CLASS_ORDER if cls in sub_ps1["Class"].values
            ]
            class_labels_ps1 = [cls for cls in CLASS_ORDER
                                 if cls in sub_ps1["Class"].values]
            class_cols_ps1 = [CLASS_COLORS[c] for c in class_labels_ps1]
            ax.hist(class_resid_ps1, bins=bins, stacked=True,
                    color=class_cols_ps1, alpha=0.5, linewidth=0,
                    label=[f"{c} (PS1)" for c in class_labels_ps1])

        # Mode 1 (HSC) — step histogram overlay (all classes together for clarity)
        if not sub_hsc.empty:
            ax.hist(sub_hsc["residual"].dropna(), bins=bins,
                    histtype="stepfilled", color="k", alpha=0.6, linewidth=0,
                    label="HSC real")
            ax.hist(sub_hsc["residual"].dropna(), bins=bins,
                    histtype="step", color="k", alpha=1.0, linewidth=1.0)

        # Reference lines: mean, ±1σ, zero
        for sub_stat, ls, lw in [(sub_hsc, "-", 1.4), (sub_ps1, "--", 0.8)]:
            if sub_stat.empty:
                continue
            r = sub_stat["residual"].dropna()
            mu, sig = r.mean(), r.std()
            ax.axvline(mu, color="k", lw=lw, ls=ls)
            ax.axvline(mu - sig, color="k", lw=lw * 0.6, ls=ls)
            ax.axvline(mu + sig, color="k", lw=lw * 0.6, ls=ls)

        ax.axvline(0, color="0.5", lw=0.8, ls=":")

        # Annotation
        ann_parts = []
        if not sub_hsc.empty:
            r = sub_hsc["residual"].dropna()
            ann_parts.append(rf"HSC: $\mu={r.mean():+.2f}$, $\sigma={r.std():.2f}$")
        if not sub_ps1.empty:
            r = sub_ps1["residual"].dropna()
            ann_parts.append(rf"PS1: $\mu={r.mean():+.2f}$, $\sigma={r.std():.2f}$")
        if ann_parts:
            ax.text(0.98, 0.96, "\n".join(ann_parts),
                    transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.8))

    # Legend
    handles_cls = [
        plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c], alpha=0.6, label=c)
        for c in CLASS_ORDER
    ]
    fig.legend(handles=handles_cls,
               loc="lower center", ncol=len(CLASS_ORDER),
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(r"SED Residuals: $m_{\rm pred} - m_{\rm obs,AB}$",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Figure C — SED quality diagnostics
# ---------------------------------------------------------------------------
def _make_quality_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    chi2_col = "xclass_fit_chi2red"
    family_col = "xclass_sed_family"
    nbands_col = "xclass_n_bands_used"

    has_chi2 = chi2_col in df.columns and df[chi2_col].notna().any()

    # Panel 1: log10(χ²_red) distribution by class
    ax = axes[0, 0]
    if has_chi2:
        log_chi2 = np.log10(df[df["Class"] != "SNR"][chi2_col].dropna().clip(1e-2, 1e25))
        bins = np.linspace(-2, 25, 55)
        for cls in CLASS_ORDER:
            sub = df[df["Class"] == cls][chi2_col].dropna()
            lsub = np.log10(sub.clip(1e-2, 1e25))
            if len(lsub) < 2:
                continue
            ax.hist(lsub, bins=bins, histtype="step",
                    color=CLASS_COLORS[cls], lw=1.2, label=cls, density=True)
        ax.set_xlabel(r"$\log_{10}(\chi^2_{\rm red})$", labelpad=2)
        ax.set_ylabel("Density", labelpad=2)
        ax.set_title("SED Fit Quality by Class", pad=4)
        ax.axvline(0.0, color="k", lw=0.8, ls="--", label=r"$\chi^2_{\rm red}=1$")
        ax.axvline(1.0, color="0.5", lw=0.8, ls=":", label="cut = 10")
        ax.legend(fontsize=7, frameon=False)
        # Fraction passing cut
        n_tot = len(df[df["Class"] != "SNR"])
        n_good = (df[df["Class"] != "SNR"][chi2_col].dropna() < 10).sum()
        ax.text(0.97, 0.95, f"{n_good}/{n_tot} pass\n$\\chi^2_{{\\rm red}}<10$",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "no χ² data", transform=ax.transAxes,
                ha="center", va="center")

    # Panel 2: SED family breakdown by class
    ax = axes[0, 1]
    if family_col in df.columns:
        fam_counts = df.groupby(["Class", family_col]).size().unstack(fill_value=0)
        fam_counts = fam_counts.reindex(CLASS_ORDER).fillna(0)
        fam_colors = {
            "pickles": "#1f77b4", "powerlaw": "#d62728",
            "two_component": "#ff7f0e", "none": "#aaa",
        }
        bottom = np.zeros(len(fam_counts))
        for fam in fam_counts.columns:
            vals = fam_counts[fam].values
            ax.bar(fam_counts.index, vals, bottom=bottom,
                   label=fam, color=fam_colors.get(fam, "#999"),
                   edgecolor="white", linewidth=0.3)
            bottom += vals
        ax.set_xlabel("Class", labelpad=2)
        ax.set_ylabel("N sources", labelpad=2)
        ax.set_title("SED Family by Class", pad=4)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no SED family data", transform=ax.transAxes,
                ha="center", va="center")

    # Panel 3: Predicted F475W distribution by class
    ax = axes[1, 0]
    pred_col = "ACS_F475W_pred"
    if pred_col in df.columns:
        bins = np.linspace(10, 35, 50)
        for cls in CLASS_ORDER:
            sub = df[df["Class"] == cls][pred_col].dropna()
            sub = sub[(sub > 10) & (sub < 35)]
            if len(sub) < 2:
                continue
            ax.hist(sub, bins=bins, histtype="step",
                    color=CLASS_COLORS[cls], lw=1.2, label=cls, density=True)
        ax.set_xlabel(r"$m_{\rm pred}$ [AB, F475W]", labelpad=2)
        ax.set_ylabel("Density", labelpad=2)
        ax.set_title("Predicted F475W Distribution", pad=4)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no F475W data", transform=ax.transAxes,
                ha="center", va="center")

    # Panel 4: n_bands_used histogram
    ax = axes[1, 1]
    if nbands_col in df.columns and df[nbands_col].notna().any():
        for cls in CLASS_ORDER:
            sub = df[df["Class"] == cls][nbands_col].dropna()
            if len(sub) < 2:
                continue
            vals, edges = np.histogram(sub, bins=np.arange(0, 12) - 0.5)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.step(centers, vals / vals.sum(), where="mid",
                    color=CLASS_COLORS[cls], lw=1.2, label=cls)
        ax.set_xlabel("N bands used in SED fit", labelpad=2)
        ax.set_ylabel("Fraction", labelpad=2)
        ax.set_title("Bands Used per SED Fit", pad=4)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no n_bands data", transform=ax.transAxes,
                ha="center", va="center")

    for ax in axes.ravel():
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle("SED Translation Quality Diagnostics", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()

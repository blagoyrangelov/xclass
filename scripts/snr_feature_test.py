#!/usr/bin/env python3
"""SNR feature-source diagnostic.

Tests whether the RF classifier exploits a statistical artefact—the fact that
SNRs have *no* SED-translated HST photometry (all NaN) while every other class
has SED-predicted magnitudes—rather than genuine astrophysical properties.

Four experiments are run, all using stratified 5-fold CV with the standard RF:

  A) Baseline — current feature matrix (SNRs: all optical features imputed to
     column median; non-SNRs: SED-translated optical values).

  B) Real-HSC swap — non-SNR sources that have an HSC counterpart within 0.5"
     have their *_pred columns replaced with real HSC magnitudes before the
     feature matrix is built. SNRs remain all-NaN as in baseline.

  C) has_real_hst probe — baseline feature matrix with one extra binary feature:
     1 if ANY of the 6 PHAT-band _pred columns is non-null (non-SNR), 0 otherwise
     (SNR). The Gini importance of this feature flags whether the classifier
     exploits the NaN-pattern as a proxy for SNR identity.

  D) X-ray-only — all HST colour and X-ray/optical ratio features are dropped;
     only logFx_*, HR_*, xray_pos_err_arcsec, normsep_class_xray, and
     xray_significance are used. This measures whether the classifier can still
     identify SNRs from X-ray properties alone.

Output:
  figures/snr_feature_test.pdf — confusion matrices, per-class metrics,
                                  feature-importance comparison, NaN heatmap

Usage::

    cd xclass_project
    python scripts/snr_feature_test.py

"""
from __future__ import annotations

import logging
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xclass import config
from xclass.features import (
    build_feature_matrix,
    compute_gaia_features,
    compute_hardness_ratios,
    compute_hst_colors,
    compute_xray_optical_ratios,
)
from xclass.query import _hsc_fetch_one, _load_cache, _safe_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUT_PDF = FIGURES_DIR / "snr_feature_test.pdf"

HSC_TRAIN_CACHE = config.QUERY_CACHE_DIR / "hsc_training"
HSC_TRAIN_CACHE.mkdir(parents=True, exist_ok=True)

PHAT_FILTERS = config.PHAT_FILTER_SET  # e.g. ['UVIS_F275W', ...]
HSC_FILTER_MAP = {
    "UVIS_F275W": "F275W",
    "UVIS_F336W": "F336W",
    "ACS_F475W":  "F475W",
    "ACS_F814W":  "F814W",
    "IR_F110W":   "F110W",
    "IR_F160W":   "F160W",
}

# Stage-1 class mapping (matching notebook 04b)
STAGE1_MAP = {
    "AGN":      "AGN",
    "LM-STAR":  "STAR",
    "HM-STAR":  "STAR",
    "LMXB":     "OTHER",
    "HMXB":     "OTHER",
    "CV":       "OTHER",
    "SNR":      "SNR",
}

RF_PARAMS = {
    **config.RF_PARAMS,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "translated_catalog.csv"
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))
    log.info("Class distribution:\n%s", df["Class"].value_counts().to_string())
    return df


def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    """Global median imputation (matches prepare_for_ml behaviour)."""
    out = X.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    return y, le


# ─────────────────────────────────────────────────────────────────────────────
# 2. HSC query for non-SNR training sources
# ─────────────────────────────────────────────────────────────────────────────

def _short(filt: str) -> str:
    """'UVIS_F275W' → 'F275W'."""
    return filt.split("_")[-1]


def query_hsc_for_training_sources(
    df: pd.DataFrame,
    search_radius_arcsec: float = 0.5,
    max_per_class: int = 300,
) -> pd.DataFrame:
    """Query HSC for non-SNR training sources using their optical positions.

    Searches a representative sample (up to *max_per_class* per class) to
    keep the runtime manageable.  Results are cached per-source in
    HSC_TRAIN_CACHE so subsequent runs are fast.

    Returns a DataFrame indexed by ``td_canonical_name`` with columns
    hsc_F275W_mag, hsc_F336W_mag, hsc_F475W_mag, hsc_F814W_mag,
    hsc_F110W_mag, hsc_F160W_mag, hsc_n_filters_detected, hsc_match.
    """
    non_snr = df[df["Class"] != "SNR"].copy()

    # Stratified sample
    rng = np.random.default_rng(42)
    sample_rows = []
    for cls, grp in non_snr.groupby("Class"):
        if len(grp) <= max_per_class:
            sample_rows.append(grp)
        else:
            idx = rng.choice(len(grp), size=max_per_class, replace=False)
            sample_rows.append(grp.iloc[idx])
    sample = pd.concat(sample_rows, ignore_index=True)

    log.info(
        "Querying HSC for %d non-SNR training sources (sample, max %d/class)…",
        len(sample), max_per_class,
    )

    phat_short = [_short(f) for f in PHAT_FILTERS]  # F275W, F336W, …
    records = []

    for i, (_, row) in enumerate(sample.iterrows()):
        sid = str(row.get("td_canonical_name", f"src_{i}"))
        ra = float(row["td_ra"]) if pd.notna(row.get("td_ra")) else float(row["ra"])
        dec = float(row["td_dec"]) if pd.notna(row.get("td_dec")) else float(row["dec"])

        detections = _hsc_fetch_one(sid, ra, dec, search_radius_arcsec, HSC_TRAIN_CACHE)

        # Select the closest HSC match (minimum D field = arcsec offset from query position)
        # Then collect all filter magnitudes for that specific MatchID.
        best_match_id = None
        if detections:
            # Find MatchID with smallest D (distance from query centre)
            min_d = float("inf")
            for det in detections:
                d_val = det.get("D")
                if d_val is not None:
                    try:
                        d_f = float(d_val)
                        if d_f < min_d:
                            min_d = d_f
                            best_match_id = det.get("MatchID")
                    except (TypeError, ValueError):
                        pass
            if best_match_id is None:
                # Fallback: use first detection's MatchID
                best_match_id = detections[0].get("MatchID")

        # Collect magnitudes per PHAT filter for the chosen MatchID
        filt_mags: dict[str, list[float]] = {f: [] for f in phat_short}
        for det in detections:
            if det.get("MatchID") != best_match_id:
                continue
            raw_filter = str(det.get("Filter", "")).strip()
            filt_short = raw_filter.split("/")[-1] if "/" in raw_filter else raw_filter
            if filt_short in filt_mags:
                mag = det.get("MagAper2")
                if mag is not None:
                    try:
                        filt_mags[filt_short].append(float(mag))
                    except (TypeError, ValueError):
                        pass

        rec: dict = {"td_canonical_name": sid, "hsc_match": bool(detections)}
        n_detected = 0
        for filt_short in phat_short:
            vals = filt_mags[filt_short]
            if vals:
                rec[f"hsc_{filt_short}_mag"] = float(np.median(vals))
                n_detected += 1
            else:
                rec[f"hsc_{filt_short}_mag"] = np.nan
        rec["hsc_n_filters_detected"] = n_detected
        records.append(rec)

        if (i + 1) % 100 == 0:
            log.info("  … %d / %d sources queried", i + 1, len(sample))

    result = pd.DataFrame(records).set_index("td_canonical_name")
    n_match = result["hsc_match"].sum()
    n_any_filt = (result["hsc_n_filters_detected"] > 0).sum()
    log.info(
        "HSC query complete: %d sources queried, %d with any HSC response, "
        "%d with ≥1 PHAT filter detected",
        len(result), int(n_match), int(n_any_filt),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build feature matrices
# ─────────────────────────────────────────────────────────────────────────────

def _build_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standard feature matrix from translated_catalog."""
    feat_df, feat_names = build_feature_matrix(df, ml_filter_set="PHAT")
    return feat_df, feat_names


def _build_real_hsc(
    df: pd.DataFrame,
    hsc_results: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix where non-SNR sources with HSC matches use real magnitudes.

    For sources in *hsc_results* with at least one PHAT filter detected, the
    corresponding *_pred columns in the catalog are replaced before computing
    the feature matrix.  All other rows (including SNRs and non-SNRs without
    HSC matches) are unchanged.
    """
    df_mod = df.copy()

    # Map td_canonical_name → index in df_mod for fast updates
    name_to_idx = {
        name: idx
        for idx, name in df_mod["td_canonical_name"].items()
    }

    n_replaced = 0
    for src_name, hsc_row in hsc_results.iterrows():
        if not hsc_row.get("hsc_match", False):
            continue
        if hsc_row.get("hsc_n_filters_detected", 0) == 0:
            continue
        if src_name not in name_to_idx:
            continue

        idx = name_to_idx[src_name]
        for phat_col, hsc_short in HSC_FILTER_MAP.items():
            real_mag = hsc_row.get(f"hsc_{hsc_short}_mag", np.nan)
            pred_col = f"{phat_col}_pred"
            if pred_col in df_mod.columns and np.isfinite(real_mag):
                df_mod.at[idx, pred_col] = real_mag
        n_replaced += 1

    log.info(
        "_build_real_hsc: replaced _pred columns for %d sources with real HSC photometry",
        n_replaced,
    )
    feat_df, feat_names = build_feature_matrix(df_mod, ml_filter_set="PHAT")
    return feat_df, feat_names


def _build_with_hst_probe(
    feat_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary feature: has_pred_hst = 1 if any PHAT _pred column is non-null."""
    phat_pred_cols = [f"{f}_pred" for f in PHAT_FILTERS if f"{f}_pred" in df.columns]
    has_pred = (~df[phat_pred_cols].isna().all(axis=1)).astype(float)
    out = feat_df.copy()
    out["has_pred_hst"] = has_pred.values
    return out


def _build_xray_only(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep only X-ray features (no HST colours or X-ray/optical ratios)."""
    xray_only_cols = [
        c for c in feat_df.columns
        if c.startswith(("logFx_", "HR_", "xray_", "normsep_"))
    ]
    return feat_df[xray_only_cols].copy(), xray_only_cols


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-validation engine
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(
    X: pd.DataFrame,
    y_labels: pd.Series,
    label_order: list[str] | None = None,
    n_splits: int = 5,
    tag: str = "",
) -> dict:
    """Stratified k-fold CV.  Returns per-fold and aggregate metrics."""
    y_enc, le = encode_labels(y_labels)
    classes = list(le.classes_)
    if label_order:
        # Reorder to requested display order (only labels present in the data)
        classes = [c for c in label_order if c in classes]

    X_imp = impute_median(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    fold_metrics: list[dict] = []
    all_y_true, all_y_pred = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_imp.values, y_enc)):
        X_tr, X_v = X_imp.values[tr_idx], X_imp.values[val_idx]
        y_tr, y_v = y_enc[tr_idx], y_enc[val_idx]

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_v)

        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": accuracy_score(y_v, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_v, y_pred),
            "f1_macro": f1_score(y_v, y_pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_v, y_pred),
        })
        all_y_true.extend(y_v)
        all_y_pred.extend(y_pred)

    # Aggregate
    metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}
    metrics_std = {f"{k}_std": np.std([f[k] for f in fold_metrics]) for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}

    # Global confusion matrix and per-class report from concatenated folds
    global_cm = confusion_matrix(all_y_true, all_y_pred, normalize="true")
    # Decode labels for report
    y_true_str = le.inverse_transform(np.array(all_y_true))
    y_pred_str = le.inverse_transform(np.array(all_y_pred))
    per_class_report = classification_report(
        y_true_str, y_pred_str, output_dict=True, zero_division=0
    )

    result = {
        "tag": tag,
        "classes": classes,
        "le": le,
        "fold_metrics": fold_metrics,
        "cm": global_cm,
        "y_true_str": y_true_str,
        "y_pred_str": y_pred_str,
        "per_class_report": per_class_report,
        **metrics,
        **metrics_std,
    }
    log.info(
        "[%s] CV acc=%.4f±%.4f  bal_acc=%.4f±%.4f  f1=%.4f±%.4f  mcc=%.4f±%.4f",
        tag,
        metrics["accuracy"], metrics_std["accuracy_std"],
        metrics["balanced_accuracy"], metrics_std["balanced_accuracy_std"],
        metrics["f1_macro"], metrics_std["f1_macro_std"],
        metrics["mcc"], metrics_std["mcc_std"],
    )
    return result


def run_importance_cv(
    X: pd.DataFrame,
    y_labels: pd.Series,
    tag: str = "",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """CV that also accumulates feature importances across folds."""
    y_enc, le = encode_labels(y_labels)
    X_imp = impute_median(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    importance_rows: list[pd.Series] = []
    fold_f1: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_imp.values, y_enc)):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_imp.values[tr_idx], y_enc[tr_idx])
        y_pred = rf.predict(X_imp.values[val_idx])
        fold_f1.append(f1_score(y_enc[val_idx], y_pred, average="macro", zero_division=0))
        importance_rows.append(pd.Series(rf.feature_importances_, index=X_imp.columns))

    imp_df = pd.DataFrame(importance_rows).mean(axis=0).sort_values(ascending=False)
    log.info("[%s] importance CV f1_macro=%.4f±%.4f", tag, np.mean(fold_f1), np.std(fold_f1))
    return imp_df, {"tag": tag, "f1_macro": np.mean(fold_f1)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _snr_metrics(result: dict) -> dict:
    """Extract SNR precision, recall, f1 from per_class_report."""
    rep = result["per_class_report"]
    snr = rep.get("SNR", {})
    return {
        "precision": snr.get("precision", np.nan),
        "recall":    snr.get("recall",    np.nan),
        "f1-score":  snr.get("f1-score",  np.nan),
        "support":   snr.get("support",   np.nan),
    }


def print_summary(results: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("SNR FEATURE SOURCE DIAGNOSTIC — SUMMARY")
    print("=" * 70)
    header = f"{'Experiment':<20} {'Acc':>7} {'BalAcc':>7} {'F1mac':>7} {'SNR-P':>7} {'SNR-R':>7} {'SNR-F1':>7}"
    print(header)
    print("-" * 70)
    for tag, res in results.items():
        snr = _snr_metrics(res)
        print(
            f"{tag:<20} "
            f"{res['accuracy']:7.4f} "
            f"{res['balanced_accuracy']:7.4f} "
            f"{res['f1_macro']:7.4f} "
            f"{snr['precision']:7.4f} "
            f"{snr['recall']:7.4f} "
            f"{snr['f1-score']:7.4f}"
        )
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────

LABEL_ORDER_7 = ["AGN", "LM-STAR", "HM-STAR", "LMXB", "HMXB", "CV", "SNR"]
LABEL_ORDER_S1 = ["AGN", "STAR", "OTHER", "SNR"]


def _plot_cm(ax: plt.Axes, result: dict, title: str) -> None:
    """Plot normalised confusion matrix on *ax*."""
    le = result["le"]
    classes = le.classes_
    cm = result["cm"]

    # Reorder to LABEL_ORDER_7 or LABEL_ORDER_S1 if possible
    preferred = [c for c in LABEL_ORDER_7 if c in classes] or list(classes)
    idx = [list(classes).index(c) for c in preferred if c in classes]
    cm_ord = cm[np.ix_(idx, idx)]

    im = ax.imshow(cm_ord, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks(range(len(preferred)))
    ax.set_yticks(range(len(preferred)))
    ax.set_xticklabels(preferred, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(preferred, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(preferred)):
        for j in range(len(preferred)):
            val = cm_ord[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if val > 0.6 else "black")


def _plot_per_class(ax: plt.Axes, results: dict[str, dict], title: str) -> None:
    """Grouped bar chart: SNR precision / recall / F1 across experiments."""
    tags = list(results.keys())
    metrics_keys = ["precision", "recall", "f1-score"]
    labels = metrics_keys
    x = np.arange(len(tags))
    width = 0.25

    for i, mk in enumerate(metrics_keys):
        vals = [_snr_metrics(results[t]).get(mk, 0.0) for t in tags]
        bars = ax.bar(x + i * width, vals, width, label=mk.capitalize())
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")


def _plot_nan_heatmap(ax: plt.Axes, feat_df: pd.DataFrame, labels: pd.Series) -> None:
    """NaN rate per feature group for SNR vs. non-SNR."""
    snr_mask = labels == "SNR"
    nan_snr   = feat_df[snr_mask].isna().mean()
    nan_other = feat_df[~snr_mask].isna().mean()

    x = np.arange(len(feat_df.columns))
    ax.bar(x - 0.2, nan_snr.values * 100,  width=0.4, label="SNR",     color="tomato",    alpha=0.8)
    ax.bar(x + 0.2, nan_other.values * 100, width=0.4, label="Non-SNR", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_df.columns, rotation=60, ha="right", fontsize=5)
    ax.set_ylabel("NaN rate (%)", fontsize=8)
    ax.set_title("NaN rate per feature: SNR vs. non-SNR", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 110)


def _plot_importance(ax: plt.Axes, imp_df: pd.Series, title: str, top_n: int = 20) -> None:
    top = imp_df.head(top_n)
    colors = [
        "crimson" if "has_pred_hst" in name else "steelblue"
        for name in top.index
    ]
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=7)
    ax.set_xlabel("Mean Gini importance", fontsize=8)
    ax.set_title(title, fontsize=9)


def _plot_overall_metrics(ax: plt.Axes, results: dict[str, dict]) -> None:
    """Bar chart of overall F1-macro across experiments."""
    tags = list(results.keys())
    f1s  = [results[t]["f1_macro"] for t in tags]
    errs = [results[t].get("f1_macro_std", 0.0) for t in tags]
    x = np.arange(len(tags))
    bars = ax.bar(x, f1s, color="steelblue", alpha=0.8, yerr=errs, capsize=4)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-macro (5-fold CV mean)", fontsize=8)
    ax.set_title("Overall classifier performance", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load training data ────────────────────────────────────────────────────
    df = load_training_data()
    labels_7  = df["Class"]
    labels_s1 = df["Class"].map(STAGE1_MAP)

    # ── Artefact summary ──────────────────────────────────────────────────────
    phat_pred_cols = [f"{f}_pred" for f in PHAT_FILTERS if f"{f}_pred" in df.columns]
    snr_mask = labels_7 == "SNR"

    log.info("=== NaN artefact summary ===")
    for col in phat_pred_cols:
        r_snr  = df.loc[snr_mask,  col].isna().mean() * 100
        r_rest = df.loc[~snr_mask, col].isna().mean() * 100
        log.info("  %s: SNR=%.0f%%  non-SNR=%.0f%%", col, r_snr, r_rest)

    # ── Baseline feature matrix ───────────────────────────────────────────────
    log.info("Building baseline feature matrix …")
    feat_baseline, feat_names = _build_baseline(df)

    # ── Experiment A: Baseline CV ─────────────────────────────────────────────
    log.info("\n=== Experiment A: Baseline 7-class CV ===")
    res_A = run_cv(feat_baseline, labels_7, label_order=LABEL_ORDER_7, tag="A-Baseline")

    log.info("\n=== Experiment A-S1: Baseline Stage-1 CV (4 classes) ===")
    res_A_s1 = run_cv(feat_baseline, labels_s1, label_order=LABEL_ORDER_S1, tag="A-Baseline-S1")

    # ── HSC query for non-SNR sources ─────────────────────────────────────────
    log.info("\n=== Querying HSC for non-SNR training sources ===")
    hsc_results = query_hsc_for_training_sources(df)

    n_hsc_any  = int(hsc_results["hsc_match"].sum())
    n_hsc_filt = int((hsc_results["hsc_n_filters_detected"] > 0).sum())
    log.info(
        "HSC results: %d queried, %d matched, %d with ≥1 PHAT filter",
        len(hsc_results), n_hsc_any, n_hsc_filt,
    )

    # Per-class breakdown — use merge to avoid duplicate-index issues
    hsc_flat = hsc_results.reset_index()  # td_canonical_name becomes a column
    df_cls = df[["td_canonical_name", "Class"]].copy()
    hsc_with_cls = df_cls.merge(hsc_flat, on="td_canonical_name", how="left")

    for cls in df["Class"].unique():
        sub = hsc_with_cls[hsc_with_cls["Class"] == cls]
        n_queried = sub["hsc_match"].notna().sum()
        n_match   = int(sub["hsc_match"].fillna(False).sum())
        n_filt    = int((sub["hsc_n_filters_detected"].fillna(0) > 0).sum())
        log.info("  %-8s: %d/%d queried with HSC response, %d with ≥1 PHAT filter",
                 cls, n_match, n_queried, n_filt)

    # ── Experiment B: Real-HSC swap ───────────────────────────────────────────
    log.info("\n=== Experiment B: Real HSC photometry for matched non-SNRs ===")
    feat_hsc, _ = _build_real_hsc(df, hsc_results)
    res_B = run_cv(feat_hsc, labels_7, label_order=LABEL_ORDER_7, tag="B-RealHSC")

    # ── Experiment C: has_pred_hst probe ─────────────────────────────────────
    log.info("\n=== Experiment C: has_pred_hst binary feature probe ===")
    feat_probe = _build_with_hst_probe(feat_baseline, df)
    imp_probe, probe_stats = run_importance_cv(feat_probe, labels_7, tag="C-Probe")

    probe_rank = list(imp_probe.index).index("has_pred_hst") + 1 if "has_pred_hst" in imp_probe.index else -1
    probe_imp  = imp_probe.get("has_pred_hst", np.nan)
    log.info(
        "has_pred_hst: Gini importance = %.4f  rank = %d / %d",
        probe_imp, probe_rank, len(imp_probe),
    )

    # Also run full CV on probe feature set for metrics comparison
    res_C = run_cv(feat_probe, labels_7, label_order=LABEL_ORDER_7, tag="C-Probe")

    # ── Experiment D: X-ray-only features ────────────────────────────────────
    log.info("\n=== Experiment D: X-ray-only features (no optical) ===")
    feat_xray, xray_names = _build_xray_only(feat_baseline)
    log.info("X-ray-only features (%d): %s", len(xray_names), xray_names)
    res_D = run_cv(feat_xray, labels_7, label_order=LABEL_ORDER_7, tag="D-XrayOnly")

    # ── Experiment E: Masquerade test ─────────────────────────────────────────
    # Blank ALL optical features for a random non-SNR subset (same N as SNRs).
    # After median imputation these sources will sit at the same "cluster" as SNRs.
    # Tests: does the classifier confuse median-coloured non-SNR sources for SNR?
    log.info("\n=== Experiment E: Masquerade test (median-masked non-SNRs) ===")
    opt_cols = [c for c in feat_baseline.columns
                if c.startswith(("color_", "logFxFopt_"))]
    n_snr = int((labels_7 == "SNR").sum())
    non_snr_idx = np.where(labels_7.values != "SNR")[0]

    masq_snr_recalls, masq_fake_rates = [], []
    rng_masq = np.random.default_rng(42)
    n_masq_runs = 5

    for seed in range(n_masq_runs):
        rng_masq_run = np.random.default_rng(seed + 100)
        blank_idx = rng_masq_run.choice(non_snr_idx, size=n_snr, replace=False)
        feat_masked = feat_baseline.copy()
        feat_masked.iloc[blank_idx, feat_masked.columns.get_indexer(opt_cols)] = np.nan

        y_enc, le_m = encode_labels(labels_7)
        X_imp_m = impute_median(feat_masked)
        skf_m = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 1)
        snr_code = list(le_m.classes_).index("SNR")

        y_true_m, y_pred_m, fold_fake_rates = [], [], []
        for tr_idx, val_idx in skf_m.split(X_imp_m.values, y_enc):
            rf_m = RandomForestClassifier(**RF_PARAMS)
            rf_m.fit(X_imp_m.values[tr_idx], y_enc[tr_idx])
            preds = rf_m.predict(X_imp_m.values[val_idx])
            y_true_m.extend(y_enc[val_idx])
            y_pred_m.extend(preds)

            # Rate at which masked non-SNR sources in validation fold are called SNR
            val_blank = np.intersect1d(val_idx, blank_idx)
            if len(val_blank) > 0:
                val_blank_local = np.searchsorted(val_idx, val_blank)
                n_called_snr = sum(preds[i] == snr_code for i in val_blank_local)
                fold_fake_rates.append(n_called_snr / len(val_blank))

        y_true_m_str = le_m.inverse_transform(np.array(y_true_m))
        y_pred_m_str = le_m.inverse_transform(np.array(y_pred_m))
        snr_true_m = y_true_m_str == "SNR"
        snr_pred_m = y_pred_m_str == "SNR"
        masq_snr_recalls.append((snr_true_m & snr_pred_m).sum() / snr_true_m.sum())
        masq_fake_rates.append(float(np.mean(fold_fake_rates)) if fold_fake_rates else 0.0)

    masq_snr_recall_mean = float(np.mean(masq_snr_recalls))
    masq_snr_recall_std  = float(np.std(masq_snr_recalls))
    masq_fake_mean = float(np.mean(masq_fake_rates))
    masq_fake_std  = float(np.std(masq_fake_rates))
    background_rate = n_snr / (len(labels_7) - n_snr)

    log.info(
        "Masquerade: SNR recall=%.3f±%.3f  masked non-SNR→SNR rate=%.3f±%.3f  "
        "(background %.3f, enrichment=%.1fx)",
        masq_snr_recall_mean, masq_snr_recall_std,
        masq_fake_mean, masq_fake_std,
        background_rate,
        masq_fake_mean / max(background_rate, 1e-6),
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    all_results = {
        "A-Baseline":  res_A,
        "B-RealHSC":   res_B,
        "C-Probe":     res_C,
        "D-XrayOnly":  res_D,
    }
    print_summary(all_results)

    snr_A = _snr_metrics(res_A)
    snr_B = _snr_metrics(res_B)
    snr_D = _snr_metrics(res_D)

    print("\n=== KEY FINDINGS ===")
    print(f"SNR recall  — Baseline vs RealHSC:   {snr_A['recall']:.3f} → {snr_B['recall']:.3f}  (Δ={snr_B['recall']-snr_A['recall']:+.3f})")
    print(f"SNR F1      — Baseline vs RealHSC:   {snr_A['f1-score']:.3f} → {snr_B['f1-score']:.3f}")
    print(f"SNR recall  — Baseline vs XrayOnly:  {snr_A['recall']:.3f} → {snr_D['recall']:.3f}  (Δ={snr_D['recall']-snr_A['recall']:+.3f})")
    print(f"has_pred_hst: rank={probe_rank}/{len(imp_probe)}  importance={probe_imp:.4f}")
    print()
    print("Masquerade test (non-SNR sources blanked to median, acting as SNR impostors):")
    print(f"  SNR recall (masked run):  {masq_snr_recall_mean:.3f} ± {masq_snr_recall_std:.3f}")
    print(f"  Masked non-SNR → SNR rate:{masq_fake_mean:.3f} ± {masq_fake_std:.3f}")
    print(f"  Background (class-prior): {background_rate:.3f}")
    print(f"  Enrichment factor:        {masq_fake_mean/max(background_rate,1e-6):.1f}x")

    if probe_rank <= 3:
        print("\n⚠  ALERT: has_pred_hst ranks in top 3 → classifier strongly exploits the NaN signature.")
    elif probe_rank <= 7:
        print("\n!  CAUTION: has_pred_hst in top 7 → moderate reliance on photometry completeness signal.")
    else:
        print(f"\n✓  has_pred_hst not in top 7 (rank {probe_rank}/{len(imp_probe)}) → not the primary cue.")

    enrichment = masq_fake_mean / max(background_rate, 1e-6)
    if enrichment > 3:
        print(f"⚠  ARTEFACT CONFIRMED: masked non-SNRs misclassified as SNR at {enrichment:.1f}× "
              f"background rate → classifier exploits the NaN→median optical signature.")
    else:
        print(f"✓  Masquerade enrichment only {enrichment:.1f}× → NaN→median artefact is not a primary driver.")

    if abs(snr_D["recall"] - snr_A["recall"]) < 0.05:
        print("✓  SNR recall robust to optical feature removal → X-ray properties are sufficient.")
    else:
        print(f"!  SNR recall drops {abs(snr_D['recall']-snr_A['recall']):.0%} without optical features "
              f"→ optical features (even if imputed) contribute to SNR identification.")

    # ── Plotting ──────────────────────────────────────────────────────────────
    log.info("\nGenerating figures → %s", OUTPUT_PDF)

    with PdfPages(OUTPUT_PDF) as pdf:

        # Page 1: NaN artefact heatmap + overall metric comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        _plot_nan_heatmap(axes[0], feat_baseline, labels_7)
        _plot_overall_metrics(axes[1], all_results)
        fig.suptitle("SNR Feature-Source Diagnostic", fontsize=12, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Confusion matrices (A, B, D)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        _plot_cm(axes[0], res_A, "A — Baseline")
        _plot_cm(axes[1], res_B, "B — Real HSC swap")
        _plot_cm(axes[2], res_D, "D — X-ray only")
        fig.suptitle("Normalised confusion matrices (5-fold CV)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Per-class SNR metrics comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_per_class(ax, all_results, "SNR per-class metrics across experiments")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Feature importance — baseline (top 20) vs probe (highlighting has_pred_hst)
        imp_baseline, _ = run_importance_cv(feat_baseline, labels_7, tag="importance-baseline")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        _plot_importance(axes[0], imp_baseline, "Gini importance — Baseline (top 20)")
        _plot_importance(axes[1], imp_probe,    "Gini importance — +has_pred_hst (top 20)")
        fig.suptitle("Feature importance comparison (crimson = has_pred_hst probe)", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: HSC match statistics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Per-class HSC match rate (use flat join to avoid duplicate-index issues)
        hsc_stats_rows = []
        for cls in LABEL_ORDER_7:
            sub = hsc_with_cls[hsc_with_cls["Class"] == cls]
            if len(sub) == 0:
                continue
            hsc_stats_rows.append({
                "class": cls,
                "queried": int(sub["hsc_match"].notna().sum()),
                "matched": int(sub["hsc_match"].fillna(False).sum()),
                "with_filter": int((sub["hsc_n_filters_detected"].fillna(0) > 0).sum()),
            })
        hsc_stats = pd.DataFrame(hsc_stats_rows)
        if not hsc_stats.empty:
            x = np.arange(len(hsc_stats))
            axes[0].bar(x - 0.2, hsc_stats["matched"],     width=0.4, label="HSC matched",     color="steelblue", alpha=0.8)
            axes[0].bar(x + 0.2, hsc_stats["with_filter"], width=0.4, label="≥1 PHAT filter",  color="seagreen",  alpha=0.8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(hsc_stats["class"], fontsize=9)
            axes[0].set_ylabel("N sources (sampled)")
            axes[0].set_title("HSC match rate per class (non-SNR sample)")
            axes[0].legend(fontsize=8)

        # Histogram of n_filters_detected
        n_filt_vals = hsc_results["hsc_n_filters_detected"].dropna()
        axes[1].hist(n_filt_vals, bins=range(0, int(n_filt_vals.max()) + 2), color="steelblue", alpha=0.8, edgecolor="white")
        axes[1].set_xlabel("Number of PHAT filters detected")
        axes[1].set_ylabel("N sources")
        axes[1].set_title("HSC filter coverage for queried non-SNR sources")

        fig.suptitle("HSC cross-match statistics for non-SNR training sources", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Masquerade test — bar chart summary
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: SNR recall comparison (baseline vs masquerade run)
        exp_tags   = ["Baseline (A)", "Masked non-SNRs (E)"]
        snr_recalls = [snr_A["recall"], masq_snr_recall_mean]
        snr_recall_err = [0.0, masq_snr_recall_std]
        bars = axes[0].bar(exp_tags, snr_recalls, color=["steelblue", "tomato"], alpha=0.8,
                           yerr=snr_recall_err, capsize=5)
        for bar, val in zip(bars, snr_recalls):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=10)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("SNR recall")
        axes[0].set_title("SNR recall: does masking non-SNR optical features hurt?", fontsize=9)
        axes[0].axhline(snr_A["recall"], color="steelblue", linestyle="--", linewidth=0.8)

        # Right: fake-SNR rate (masked non-SNR → SNR classification rate vs. background)
        cats  = ["Background\n(class prior)", "Masked non-SNR\npredicted as SNR"]
        rates = [background_rate, masq_fake_mean]
        errs  = [0.0, masq_fake_std]
        bars2 = axes[1].bar(cats, rates, color=["grey", "crimson"], alpha=0.8,
                            yerr=errs, capsize=5)
        for bar, val in zip(bars2, rates):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.002, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=10)
        axes[1].set_ylim(0, max(rates) * 1.4)
        axes[1].set_ylabel("Fraction classified as SNR")
        axes[1].set_title(
            f"Masquerade test: {masq_fake_mean/max(background_rate,1e-6):.1f}× enrichment\n"
            f"(n={n_snr} non-SNR sources masked to NaN then imputed to median)",
            fontsize=9,
        )
        axes[1].axhline(background_rate, color="grey", linestyle="--", linewidth=0.8)

        fig.suptitle(
            "Experiment E — Masquerade test: do median-imputed non-SNRs masquerade as SNR?",
            fontsize=10, fontweight="bold",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    log.info("Saved: %s", OUTPUT_PDF)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fix SNR HST photometry in translated_catalog.csv.

SNR training sources have no SED-translated HST magnitudes (all NaN in
the *_pred columns) because they bypass the SED translation stage.  The
original design assumed they would carry real HST photometry in A_F*
columns, but that crossmatch was never executed, leaving all optical
features NaN — an artefact that the classifier exploits.

This script:
1. Queries HSC v3 for all 150 SNR training sources using their
   optical positions (td_ra/td_dec).
2. Populates the six PHAT *_pred columns with the real HSC magnitudes
   for SNRs that have an HSC counterpart within 0.5 arcsec.
3. Saves the corrected translated_catalog.csv.
4. Re-runs 5-fold CV (baseline vs. fixed) and the masquerade test,
   then prints a full summary.

Usage::

    cd xclass_project
    python scripts/fix_snr_hst_photometry.py

"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xclass import config
from xclass.features import build_feature_matrix
from xclass.query import _hsc_fetch_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
OUTPUT_PDF = FIGURES_DIR / "snr_hst_fix.pdf"

HSC_SNR_CACHE = config.QUERY_CACHE_DIR / "hsc_snr_fix"
HSC_SNR_CACHE.mkdir(parents=True, exist_ok=True)

SEARCH_RADIUS_ARCSEC = 0.5

PHAT_FILTERS = config.PHAT_FILTER_SET
# Map PHAT column prefix → short HSC filter name
HSC_FILTER_MAP = {
    "UVIS_F275W": "F275W",
    "UVIS_F336W": "F336W",
    "ACS_F475W":  "F475W",
    "ACS_F814W":  "F814W",
    "IR_F110W":   "F110W",
    "IR_F160W":   "F160W",
}
# Inverse: HSC short name → _pred column name
HSC_TO_PRED = {v: f"{k}_pred" for k, v in HSC_FILTER_MAP.items()}

RF_PARAMS = {
    **config.RF_PARAMS,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
}

STAGE1_MAP = {
    "AGN":     "AGN",
    "LM-STAR": "STAR",
    "HM-STAR": "STAR",
    "LMXB":    "OTHER",
    "HMXB":    "OTHER",
    "CV":      "OTHER",
    "SNR":     "SNR",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def run_cv(X: pd.DataFrame, y_labels: pd.Series, tag: str = "", n_splits: int = 5) -> dict:
    """Stratified k-fold CV with global pre-imputation (matches prepare_for_ml)."""
    le = LabelEncoder()
    y = le.fit_transform(y_labels.astype(str))

    # Global imputation before splitting — matches prepare_for_ml() behaviour
    X_imp = impute_median(X)
    X_arr = X_imp.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    accs, baccs, f1s, mccs = [], [], [], []
    all_y_true, all_y_pred = [], []

    for train_idx, val_idx in skf.split(X_arr, y):
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_arr[train_idx], y[train_idx])
        y_pred = model.predict(X_arr[val_idx])
        y_v = y[val_idx]

        accs.append(accuracy_score(y_v, y_pred))
        baccs.append(balanced_accuracy_score(y_v, y_pred))
        f1s.append(f1_score(y_v, y_pred, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(y_v, y_pred))
        all_y_true.extend(y_v)
        all_y_pred.extend(y_pred)

    # OOF confusion matrix and per-class SNR metrics
    oof_true = np.array(all_y_true)
    oof_pred = np.array(all_y_pred)
    oof_cm = confusion_matrix(oof_true, oof_pred, labels=range(len(le.classes_)))

    y_true_str = le.inverse_transform(oof_true)
    y_pred_str = le.inverse_transform(oof_pred)
    from sklearn.metrics import classification_report
    cr = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    snr_report = cr.get("SNR", {})

    log.info(
        "[%s] CV acc=%.4f±%.4f  bal_acc=%.4f±%.4f  f1=%.4f±%.4f  mcc=%.4f±%.4f",
        tag, np.mean(accs), np.std(accs),
        np.mean(baccs), np.std(baccs),
        np.mean(f1s), np.std(f1s),
        np.mean(mccs), np.std(mccs),
    )
    log.info(
        "[%s] SNR P=%.3f  R=%.3f  F1=%.3f",
        tag,
        snr_report.get("precision", np.nan),
        snr_report.get("recall", np.nan),
        snr_report.get("f1-score", np.nan),
    )

    return {
        "tag": tag,
        "acc": np.mean(accs), "acc_std": np.std(accs),
        "bacc": np.mean(baccs), "bacc_std": np.std(baccs),
        "f1": np.mean(f1s), "f1_std": np.std(f1s),
        "mcc": np.mean(mccs), "mcc_std": np.std(mccs),
        "snr_prec": snr_report.get("precision", np.nan),
        "snr_rec":  snr_report.get("recall",    np.nan),
        "snr_f1":   snr_report.get("f1-score",  np.nan),
        "mean_cm": oof_cm,
        "le": le,
    }


def run_masquerade(feat_df: pd.DataFrame, labels: pd.Series,
                   n_runs: int = 20, n_splits: int = 5) -> dict:
    """Masquerade test: blank optical features for N non-SNR sources.

    Matches snr_feature_test.py Experiment E: global imputation after
    blanking, so blanked sources get the column median (same artefact
    mechanism as real SNRs).
    """
    blank_cols = [c for c in feat_df.columns
                  if c.startswith(("color_", "logFxFopt_"))]

    snr_mask = labels == "SNR"
    non_snr_idx = np.where(~snr_mask.values)[0]
    n_snr = int(snr_mask.sum())
    background_prior = n_snr / len(labels)

    le = LabelEncoder()
    y_all = le.fit_transform(labels.astype(str))
    snr_le = list(le.classes_).index("SNR")

    rng = np.random.default_rng(42)
    snr_recalls, masked_as_snr_rates = [], []

    for run in range(n_runs):
        blank_idx = rng.choice(non_snr_idx, size=n_snr, replace=False)
        feat_masked = feat_df.copy()
        for col in blank_cols:
            if col in feat_masked.columns:
                feat_masked.iloc[blank_idx, feat_masked.columns.get_loc(col)] = np.nan

        # Global imputation (matches baseline) — blanked sources get column median
        X_imp = impute_median(feat_masked)
        X_arr = X_imp.values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=config.RANDOM_STATE + run)
        fold_snr_recalls, fold_masked_rates = [], []

        for train_idx, val_idx in skf.split(X_arr, y_all):
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_arr[train_idx], y_all[train_idx])
            y_pred = model.predict(X_arr[val_idx])
            y_v = y_all[val_idx]

            snr_in_val = (y_v == snr_le).sum()
            if snr_in_val > 0:
                snr_correct = ((y_v == snr_le) & (y_pred == snr_le)).sum()
                fold_snr_recalls.append(snr_correct / snr_in_val)

            val_blanked = np.intersect1d(val_idx, blank_idx)
            if len(val_blanked) > 0:
                val_blanked_local = np.searchsorted(val_idx, val_blanked)
                y_pred_blanked = y_pred[val_blanked_local]
                fold_masked_rates.append((y_pred_blanked == snr_le).mean())

        snr_recalls.append(np.mean(fold_snr_recalls))
        masked_as_snr_rates.append(np.mean(fold_masked_rates))

    enrichment = np.mean(masked_as_snr_rates) / background_prior

    log.info(
        "Masquerade: SNR recall=%.3f±%.3f  masked non-SNR→SNR rate=%.3f±%.3f"
        "  (background %.3f, enrichment=%.1fx)",
        np.mean(snr_recalls), np.std(snr_recalls),
        np.mean(masked_as_snr_rates), np.std(masked_as_snr_rates),
        background_prior, enrichment,
    )
    return {
        "snr_recall_mean": np.mean(snr_recalls),
        "snr_recall_std":  np.std(snr_recalls),
        "masked_snr_rate_mean": np.mean(masked_as_snr_rates),
        "masked_snr_rate_std":  np.std(masked_as_snr_rates),
        "background_prior": background_prior,
        "enrichment": enrichment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HSC query for SNR training sources
# ─────────────────────────────────────────────────────────────────────────────

def _short(filt_key: str) -> str:
    return filt_key.split("_")[-1]


def query_hsc_for_snrs(df: pd.DataFrame) -> pd.DataFrame:
    """Query HSC for all SNR rows using their td_ra/td_dec positions.

    Returns a DataFrame indexed by td_canonical_name with columns
    hsc_F*_mag for each PHAT filter and hsc_n_filters_detected.
    """
    snr_df = df[df["Class"] == "SNR"].copy()
    log.info("Querying HSC for %d SNR training sources…", len(snr_df))

    phat_short = [_short(f) for f in PHAT_FILTERS]
    records = []

    for i, (_, row) in enumerate(snr_df.iterrows()):
        sid = str(row["td_canonical_name"])
        ra  = float(row["td_ra"])
        dec = float(row["td_dec"])

        detections = _hsc_fetch_one(sid, ra, dec, SEARCH_RADIUS_ARCSEC, HSC_SNR_CACHE)

        best_match_id = None
        if detections:
            min_d = float("inf")
            for det in detections:
                try:
                    d_f = float(det.get("D", float("inf")))
                    if d_f < min_d:
                        min_d = d_f
                        best_match_id = det.get("MatchID")
                except (TypeError, ValueError):
                    pass
            if best_match_id is None:
                best_match_id = detections[0].get("MatchID")

        filt_mags: dict[str, list[float]] = {f: [] for f in phat_short}
        for det in detections:
            if det.get("MatchID") != best_match_id:
                continue
            raw_filter = str(det.get("Filter", "")).strip()
            filt_short = raw_filter.split("/")[-1] if "/" in raw_filter else raw_filter
            if filt_short in filt_mags:
                try:
                    mag = float(det["MagAper2"])
                    filt_mags[filt_short].append(mag)
                except (TypeError, ValueError):
                    pass

        rec: dict = {"td_canonical_name": sid, "hsc_match": bool(detections)}
        n_det = 0
        for filt_short in phat_short:
            vals = filt_mags[filt_short]
            if vals:
                rec[f"hsc_{filt_short}_mag"] = float(np.median(vals))
                n_det += 1
            else:
                rec[f"hsc_{filt_short}_mag"] = np.nan
        rec["hsc_n_filters_detected"] = n_det
        records.append(rec)

        if (i + 1) % 25 == 0:
            log.info("  … %d / %d SNRs queried", i + 1, len(snr_df))

    result = pd.DataFrame(records).set_index("td_canonical_name")
    n_match  = result["hsc_match"].sum()
    n_any    = (result["hsc_n_filters_detected"] > 0).sum()
    log.info(
        "HSC SNR query done: %d queried, %d with HSC response, %d with ≥1 PHAT filter",
        len(result), int(n_match), int(n_any),
    )

    for galaxy in ["M33", "M83", "N6946"]:
        mask = result.index.str.contains(galaxy)
        subset = result[mask]
        log.info(
            "  %-6s: %d/%d with HSC response, %d with ≥1 filter",
            galaxy, int(subset["hsc_match"].sum()), len(subset),
            int((subset["hsc_n_filters_detected"] > 0).sum()),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Patch translated catalog
# ─────────────────────────────────────────────────────────────────────────────

def patch_catalog(df: pd.DataFrame, snr_hsc: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Write real HSC magnitudes into *_pred columns for matched SNRs.

    Returns patched DataFrame and count of SNR rows updated.
    """
    patched = df.copy()
    n_updated = 0

    for td_name, hsc_row in snr_hsc.iterrows():
        if hsc_row["hsc_n_filters_detected"] == 0:
            continue

        mask = patched["td_canonical_name"] == td_name
        if mask.sum() == 0:
            continue

        for filt_key, filt_short in HSC_FILTER_MAP.items():
            pred_col = f"{filt_key}_pred"
            hsc_col  = f"hsc_{filt_short}_mag"
            mag = hsc_row.get(hsc_col, np.nan)
            if pd.notna(mag) and pred_col in patched.columns:
                patched.loc[mask, pred_col] = mag

        # Mark as patched in status flag (informational)
        if "xclass_status_flag" in patched.columns:
            patched.loc[mask, "xclass_status_flag"] = "real_hsc_photometry"

        n_updated += mask.sum()

    log.info(
        "Patched %d SNR rows in translated catalog with real HSC magnitudes.",
        n_updated,
    )
    return patched, n_updated


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cm(ax, cm, classes, title):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm / (cm.sum(axis=1, keepdims=True) + 1e-9),
        display_labels=classes,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)


def make_figures(res_baseline, res_fixed, masq_baseline, masq_fixed):
    log.info("\nGenerating figures → %s", OUTPUT_PDF)
    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Confusion matrices (baseline vs fixed)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        classes = res_baseline["le"].classes_
        _plot_cm(axes[0], res_baseline["mean_cm"], classes, "Baseline (SNRs: all NaN→median)")
        _plot_cm(axes[1], res_fixed["mean_cm"],    classes, "Fixed (SNRs: real HSC where available)")
        fig.suptitle("SNR HST Fix: 5-fold CV Confusion Matrices", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Per-class metric comparison
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        metrics = ["acc", "bacc", "f1"]
        metric_labels = ["Accuracy", "Balanced Accuracy", "F1 macro"]
        for ax, m, ml in zip(axes, metrics, metric_labels):
            vals = [res_baseline[m], res_fixed[m]]
            stds = [res_baseline[f"{m}_std"], res_fixed[f"{m}_std"]]
            bars = ax.bar(["Baseline", "Fixed"], vals, yerr=stds,
                          color=["#4C72B0", "#DD8452"], capsize=5)
            ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
            ax.set_title(ml, fontsize=10)
            ax.set_ylabel("Score")
            for bar, v, s in zip(bars, vals, stds):
                ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.003,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("Overall Metrics: Baseline vs Fixed", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: SNR-specific metrics
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        snr_metrics = ["snr_prec", "snr_rec", "snr_f1"]
        snr_labels  = ["SNR Precision", "SNR Recall", "SNR F1"]
        for ax, m, ml in zip(axes, snr_metrics, snr_labels):
            vals = [res_baseline[m], res_fixed[m]]
            ax.bar(["Baseline", "Fixed"], vals,
                   color=["#4C72B0", "#DD8452"])
            ax.set_ylim(0, 1)
            ax.set_title(ml, fontsize=10)
            ax.set_ylabel("Score")
            for i, (v, label) in enumerate(zip(vals, ["Baseline", "Fixed"])):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        fig.suptitle("SNR-Class Metrics: Baseline vs Fixed", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Masquerade test comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        labels_bar = ["Baseline", "Fixed"]
        rates  = [masq_baseline["masked_snr_rate_mean"], masq_fixed["masked_snr_rate_mean"]]
        stds   = [masq_baseline["masked_snr_rate_std"],  masq_fixed["masked_snr_rate_std"]]
        prior  = masq_baseline["background_prior"]
        bars = ax.bar(labels_bar, rates, yerr=stds, color=["#4C72B0", "#DD8452"], capsize=7)
        ax.axhline(prior, color="red", linestyle="--", label=f"Background prior ({prior:.3f})")
        ax.set_ylabel("Fraction of blanked non-SNRs → predicted SNR")
        ax.set_title("Masquerade Test: Enrichment Factor (Baseline vs Fixed)")
        ax.legend()
        for bar, r, s, enr in zip(
            bars, rates, stds,
            [masq_baseline["enrichment"], masq_fixed["enrichment"]],
        ):
            ax.text(bar.get_x() + bar.get_width() / 2, r + s + 0.002,
                    f"{r:.3f}\n({enr:.1f}×)", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    log.info("Saved: %s", OUTPUT_PDF)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    cat_path = config.PROCESSED_DIR / "translated_catalog.csv"
    df = pd.read_csv(cat_path, low_memory=False)
    log.info("Loaded training data: %d rows, %d columns", len(df), len(df.columns))
    log.info("Class distribution:\n%s", df["Class"].value_counts().to_string())

    # 2. NaN summary before fix
    log.info("\n=== NaN summary BEFORE fix ===")
    snr_mask = df["Class"] == "SNR"
    for filt in PHAT_FILTERS:
        pred_col = f"{filt}_pred"
        if pred_col not in df.columns:
            continue
        snr_nan  = df.loc[snr_mask,  pred_col].isna().mean() * 100
        nonsnr_nan = df.loc[~snr_mask, pred_col].isna().mean() * 100
        log.info("  %s: SNR=%.0f%%  non-SNR=%.0f%%", pred_col, snr_nan, nonsnr_nan)

    # 3. Baseline CV
    log.info("\n=== Baseline CV (before fix) ===")
    feat_baseline, _ = build_feature_matrix(df)
    labels = df["Class"]
    res_baseline = run_cv(feat_baseline, labels, tag="Baseline")

    # 4. Baseline masquerade test
    log.info("\n=== Masquerade test: Baseline ===")
    masq_baseline = run_masquerade(feat_baseline, labels, n_runs=5)

    # 5. Query HSC for SNR sources
    log.info("\n=== Querying HSC for SNR training sources ===")
    snr_hsc = query_hsc_for_snrs(df)

    # 6. Patch catalog
    log.info("\n=== Patching translated catalog ===")
    df_fixed, n_patched = patch_catalog(df, snr_hsc)

    # 7. NaN summary after fix
    log.info("\n=== NaN summary AFTER fix ===")
    snr_mask_f = df_fixed["Class"] == "SNR"
    for filt in PHAT_FILTERS:
        pred_col = f"{filt}_pred"
        if pred_col not in df_fixed.columns:
            continue
        snr_nan_after  = df_fixed.loc[snr_mask_f, pred_col].isna().mean() * 100
        log.info("  %s: SNR NaN=%.0f%%  (was 100%%)", pred_col, snr_nan_after)

    # 8. Fixed CV
    log.info("\n=== Fixed CV (after HSC patch) ===")
    feat_fixed, _ = build_feature_matrix(df_fixed)
    res_fixed = run_cv(feat_fixed, df_fixed["Class"], tag="Fixed")

    # 9. Fixed masquerade test
    log.info("\n=== Masquerade test: Fixed ===")
    masq_fixed = run_masquerade(feat_fixed, df_fixed["Class"], n_runs=5)

    # 10. Save patched catalog
    out_path = config.PROCESSED_DIR / "translated_catalog.csv"
    df_fixed.to_csv(out_path, index=False)
    log.info("Saved patched catalog → %s", out_path)

    # 11. Save figures
    make_figures(res_baseline, res_fixed, masq_baseline, masq_fixed)

    # 12. Summary
    snr_hsc_matched = int(snr_hsc["hsc_n_filters_detected"].gt(0).sum())
    print("\n" + "=" * 70)
    print("SNR HST PHOTOMETRY FIX — SUMMARY")
    print("=" * 70)
    print(f"SNR sources queried via HSC:      {len(snr_hsc)}")
    print(f"SNR sources with ≥1 PHAT filter:  {snr_hsc_matched}")
    print(f"SNR rows patched in catalog:      {n_patched}")
    print()
    print(f"{'Experiment':<20} {'Acc':>6} {'BalAcc':>7} {'F1mac':>7}"
          f" {'SNR-P':>7} {'SNR-R':>7} {'SNR-F1':>7}")
    print("-" * 70)
    for r in [res_baseline, res_fixed]:
        print(
            f"{r['tag']:<20} {r['acc']:>6.4f} {r['bacc']:>7.4f} {r['f1']:>7.4f}"
            f" {r['snr_prec']:>7.4f} {r['snr_rec']:>7.4f} {r['snr_f1']:>7.4f}"
        )
    print("=" * 70)
    print()
    print("=== KEY FINDINGS ===")
    print(f"SNR recall  — Baseline vs Fixed:   "
          f"{res_baseline['snr_rec']:.3f} → {res_fixed['snr_rec']:.3f}"
          f"  (Δ={res_fixed['snr_rec'] - res_baseline['snr_rec']:+.3f})")
    print(f"SNR F1      — Baseline vs Fixed:   "
          f"{res_baseline['snr_f1']:.3f} → {res_fixed['snr_f1']:.3f}")
    print()
    print("Masquerade test:")
    for label, m in [("Baseline", masq_baseline), ("Fixed", masq_fixed)]:
        print(f"  {label:<12}: masked non-SNR→SNR rate={m['masked_snr_rate_mean']:.3f}±"
              f"{m['masked_snr_rate_std']:.3f}  enrichment={m['enrichment']:.1f}×")
    print()
    if masq_fixed["enrichment"] < masq_baseline["enrichment"] * 0.5:
        print("✓  Artefact significantly reduced (enrichment dropped >50%).")
    elif masq_fixed["enrichment"] < masq_baseline["enrichment"] * 0.8:
        print("~  Artefact partially reduced.")
    else:
        print("⚠  Artefact persists — SNRs without HSC counterparts still "
              "show all-NaN optical signature.")
    if snr_hsc_matched < len(snr_hsc) // 2:
        print(f"!  Only {snr_hsc_matched}/{len(snr_hsc)} SNRs have HSC coverage — "
              "remaining NaN rows still drive the artefact.")
    print(f"\nFigures saved → {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

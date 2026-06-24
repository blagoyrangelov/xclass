"""SNR photometry-leakage diagnostic.

Referee concern: the classifier may learn the NaN-imputed colour signature
of SNR rows (which have no SED translation) rather than intrinsic source
properties.

Tests
-----
1. Confirm and quantify the NaN leakage in the current feature matrix.
2. Query HSC v3 for Chandra-matched training sources (cached per source).
3. Build a 'real-photometry' feature matrix where HSC-matched sources use
   measured HST magnitudes instead of SED-translated values.
4. 5-fold CV comparison (baseline vs real-photometry):
   per-class precision / recall / F1, emphasis on SNR.
5. Add a binary 'has_real_hst' flag (= 1 for SNRs in baseline) and measure
   Gini importance.  High rank → classifier exploits the NaN pattern.

Output
------
Printed tables + figures/snr_feature_test.pdf
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xclass import config
from xclass.features import build_feature_matrix
from xclass.classifier import cross_validate

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSLATED_PATH = PROJECT_ROOT / "data" / "processed" / "translated_catalog.csv"
HSC_CACHE_DIR   = PROJECT_ROOT / "data" / "query_cache" / "hsc_training"
FIGURES_DIR     = PROJECT_ROOT / "figures"
FIGURES_PDF     = FIGURES_DIR / "snr_feature_test.pdf"

# PHAT filter label → pred column name
HSC_TO_PRED: dict[str, str] = {
    "F275W": "UVIS_F275W_pred",
    "F336W": "UVIS_F336W_pred",
    "F475W": "ACS_F475W_pred",
    "F814W": "ACS_F814W_pred",
    "F110W": "IR_F110W_pred",
    "F160W": "IR_F160W_pred",
}

# Colour feature names built by compute_hst_colors
COLOR_FEATURES = [
    "color_F275W_F336W", "color_F336W_F475W", "color_F475W_F814W",
    "color_F814W_F110W", "color_F110W_F160W",
    "color_F275W_F814W", "color_F275W_F160W",
    "color_F336W_F814W", "color_F475W_F160W",
]
RATIO_FEATURES = ["logFxFopt_B_F475W", "logFxFopt_B_F814W"]
OPT_FEATURES   = COLOR_FEATURES + RATIO_FEATURES  # 11 features all-NaN for SNRs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impute_median(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def _cv_and_metrics(
    df: pd.DataFrame,
    label: str,
    n_splits: int = 5,
) -> tuple[dict, np.ndarray, LabelEncoder]:
    """Build feature matrix → impute → encode → 5-fold CV.

    Returns cv_results dict, mean confusion matrix, label encoder.
    """
    feat_df, _ = build_feature_matrix(df, ml_filter_set="PHAT")
    labels = df["Class"]
    X = _impute_median(feat_df)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(labels.values), index=labels.index)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []
    fold_reports = []
    for _, (tr, vl) in enumerate(skf.split(X, y)):
        model = RandomForestClassifier(**config.RF_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr])
        yp = model.predict(X.iloc[vl])
        all_y_true.extend(y.iloc[vl].tolist())
        all_y_pred.extend(yp.tolist())
        fold_reports.append(
            classification_report(y.iloc[vl], yp, output_dict=True,
                                  target_names=le.classes_, zero_division=0)
        )

    cv_res = cross_validate(X, y, n_splits=n_splits)
    cm = confusion_matrix(all_y_true, all_y_pred,
                          labels=list(range(len(le.classes_))),
                          normalize="true")

    # Aggregate per-class metrics across folds
    per_class: dict[str, dict] = {}
    for cls in le.classes_:
        p_vals = [r.get(cls, {}).get("precision", 0) for r in fold_reports]
        r_vals = [r.get(cls, {}).get("recall", 0) for r in fold_reports]
        f_vals = [r.get(cls, {}).get("f1-score", 0) for r in fold_reports]
        per_class[cls] = {
            "precision_mean": np.mean(p_vals),
            "recall_mean":    np.mean(r_vals),
            "f1_mean":        np.mean(f_vals),
        }

    return cv_res, cm, le, per_class


def _query_hsc_for_training(
    df: pd.DataFrame,
    max_queries_per_class: int = 500,
) -> pd.DataFrame:
    """Run HSC queries for training sources.  Results are cached per source.

    Parameters
    ----------
    df : DataFrame
        translated_catalog rows (all classes).
    max_queries_per_class : int
        Maximum number of NEW queries per non-SNR class (SNRs are always
        queried in full).  Already-cached sources do not count toward this
        limit.
    """
    from xclass.query import query_hsc_for_chandra_sources

    HSC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build query input: deduplicate by Chandra source
    cols_needed = ["xray_name", "xray_ra", "xray_dec", "xray_pos_err_arcsec", "Class"]
    query_df = df[cols_needed].drop_duplicates("xray_name").copy()
    query_df = query_df.rename(columns={
        "xray_name":          "xray_id",
        "xray_ra":            "ra",
        "xray_dec":           "dec",
        "xray_pos_err_arcsec": "pos_err",
    })

    # Identify which sources are already in cache
    cached_ids = {
        Path(f).stem.replace("_", ".").replace("-", "_")
        for f in HSC_CACHE_DIR.iterdir()
        if f.suffix == ".pkl"
    }
    # Note: cache file names are sanitised versions of xray_id, so we
    # compare loosely by checking if the source .pkl file already exists
    # via the query function's internal logic.  We just pass the full set
    # and let caching skip already-done sources.

    # Stratified selection: all SNRs, all small classes, cap large classes
    large_classes = {"AGN", "LM-STAR", "HM-STAR"}
    selected = []
    for cls, grp in query_df.groupby("Class"):
        if cls not in large_classes:
            selected.append(grp)
        else:
            selected.append(grp.sample(
                min(max_queries_per_class, len(grp)),
                random_state=42,
            ))
    query_input = pd.concat(selected, ignore_index=True)
    print(f"  Querying {len(query_input)} unique Chandra sources "
          f"({query_df['Class'].value_counts().to_dict()})")
    print(f"  (HSC cache: {len(list(HSC_CACHE_DIR.glob('*.pkl')))} sources already cached)")

    best_match, _ = query_hsc_for_chandra_sources(
        query_input,
        cache_dir=HSC_CACHE_DIR,
    )
    return best_match


def _inject_real_photometry(
    df: pd.DataFrame,
    hsc_best: pd.DataFrame,
) -> pd.DataFrame:
    """Replace *_pred columns with real HSC mags where available.

    For sources where HSC returns a non-NaN measurement for a filter,
    the corresponding *_pred column is overwritten with the real mag.
    Sources without an HSC match retain their original *_pred values.
    """
    out = df.copy()

    # Build a lookup: xray_name → {filter: (mag, err)}
    hsc_idx = hsc_best.set_index("xray_id") if "xray_id" in hsc_best.columns \
        else hsc_best.set_index(hsc_best.columns[0])

    n_replaced = 0
    for filt_short, pred_col in HSC_TO_PRED.items():
        mag_col = f"hsc_{filt_short}_mag"
        if mag_col not in hsc_idx.columns:
            continue
        for xray_name in out["xray_name"].unique():
            if xray_name not in hsc_idx.index:
                continue
            mag = float(hsc_idx.at[xray_name, mag_col]) if np.isfinite(
                    pd.to_numeric(hsc_idx.at[xray_name, mag_col],
                                  errors="coerce")) else float("nan")
            if not np.isfinite(mag):
                continue
            row_mask = out["xray_name"] == xray_name
            out.loc[row_mask, pred_col] = mag
            n_replaced += row_mask.sum()

    print(f"  _inject_real_photometry: {n_replaced} column-cell replacements "
          f"across {out['xray_name'].nunique()} sources")
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_confusion(ax, cm, classes, title):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if val > 0.5 else "black")
    return im


def _plot_per_class_metrics(
    ax,
    classes,
    base_pc: dict,
    real_pc: dict,
    metric: str,
    title: str,
):
    x = np.arange(len(classes))
    w = 0.35
    base_vals = [base_pc.get(c, {}).get(metric, 0) for c in classes]
    real_vals  = [real_pc.get(c, {}).get(metric, 0) for c in classes]
    ax.bar(x - w / 2, base_vals, w, label="Baseline (SED-translated)", color="steelblue")
    ax.bar(x + w / 2, real_vals,  w, label="Real photometry (HSC)",    color="darkorange")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_mean", ""), fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load data and characterise NaN leakage
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Loading translated catalog …")
    df_all = pd.read_csv(TRANSLATED_PATH, low_memory=False)
    print(f"  {len(df_all)} rows  |  Classes: {df_all['Class'].value_counts().to_dict()}")

    print("\n--- Step 1: NaN leakage characterisation ---")
    feat_df_raw, feat_names = build_feature_matrix(df_all, "PHAT")
    snr_mask   = df_all["Class"] == "SNR"
    non_snr_mk = ~snr_mask

    nan_snr     = feat_df_raw[snr_mask][OPT_FEATURES].isna().all(axis=1).mean()
    nan_nonsnr  = feat_df_raw[non_snr_mk][OPT_FEATURES].isna().all(axis=1).mean()

    print(f"  SNR rows with ALL 11 optical features NaN:     "
          f"{nan_snr:.1%} ({int(nan_snr * snr_mask.sum())}/{snr_mask.sum()})")
    print(f"  Non-SNR rows with ALL 11 optical features NaN: "
          f"{nan_nonsnr:.1%} "
          f"({int(nan_nonsnr * non_snr_mk.sum())}/{non_snr_mk.sum()})")
    print()
    print("  Fraction of SNR rows that are NaN per optical feature:")
    for f in OPT_FEATURES:
        frac = feat_df_raw.loc[snr_mask, f].isna().mean()
        print(f"    {f:<28}  {frac:.3f}")

    # -----------------------------------------------------------------------
    # Step 5 (fast): has_real_hst Gini importance diagnostic
    # -----------------------------------------------------------------------
    print()
    print("--- Step 5: has_real_hst Gini importance (no HSC query needed) ---")
    feat_df_imp = feat_df_raw.copy()
    feat_df_imp["has_real_hst"] = snr_mask.astype(int).values

    X_imp  = _impute_median(feat_df_imp)
    le_imp = LabelEncoder()
    y_imp  = pd.Series(
        le_imp.fit_transform(df_all["Class"].values),
        index=df_all.index,
    )
    rf_imp = RandomForestClassifier(**config.RF_PARAMS)
    rf_imp.fit(X_imp, y_imp)
    imp_series = pd.Series(
        rf_imp.feature_importances_,
        index=X_imp.columns,
    ).sort_values(ascending=False)
    has_hst_rank = imp_series.index.get_loc("has_real_hst") + 1
    has_hst_imp  = imp_series["has_real_hst"]

    print(f"\n  Feature importances (top 15, with has_real_hst highlighted):")
    print(f"  {'Feature':<30}  {'Gini imp':>10}")
    print(f"  {'-'*42}")
    for i, (feat, imp) in enumerate(imp_series.head(15).items()):
        flag = " *** has_real_hst ***" if feat == "has_real_hst" else ""
        print(f"  {feat:<30}  {imp:>10.4f}{flag}")
    print(f"\n  has_real_hst rank: {has_hst_rank}/{len(imp_series)}")
    print(f"  has_real_hst Gini importance: {has_hst_imp:.4f}")
    if has_hst_rank <= 5:
        print("  => HIGH rank: classifier DOES exploit the NaN pattern.")
    elif has_hst_rank <= 10:
        print("  => MODERATE rank: NaN pattern provides some signal.")
    else:
        print("  => LOW rank: classifier does NOT primarily rely on NaN pattern.")

    # -----------------------------------------------------------------------
    # Step 2: HSC queries (attempt live; fall back to simulation if API down)
    # -----------------------------------------------------------------------
    hsc_best = None
    if not args.skip_hsc:
        print("\n--- Step 2: HSC queries for training sources ---")
        try:
            hsc_best = _query_hsc_for_training(
                df_all,
                max_queries_per_class=args.max_per_class,
            )
            n_matched = (hsc_best["hsc_match_status"] != "none").sum()
            if n_matched == 0:
                print("  No HSC matches returned — falling back to simulation tests.")
                hsc_best = None
            else:
                print(f"  HSC results: {len(hsc_best)} queried, "
                      f"{n_matched} matched ({n_matched/len(hsc_best):.1%})")
                hsc_with_class = hsc_best.merge(
                    df_all[["xray_name", "Class"]].drop_duplicates("xray_name").rename(
                        columns={"xray_name": "xray_id"}
                    ),
                    on="xray_id", how="left",
                )
                print("\n  HSC match rate by class:")
                for cls, grp in hsc_with_class.groupby("Class"):
                    matched = (grp["hsc_match_status"] != "none").sum()
                    print(f"    {cls:<10}: {matched:4d}/{len(grp):4d}  "
                          f"({matched/len(grp):.1%})")
        except Exception as exc:
            print(f"  HSC query FAILED ({exc}) — using simulation tests.")
            hsc_best = None
    else:
        print("\n--- Step 2: HSC queries SKIPPED (--skip-hsc) ---")

    # -----------------------------------------------------------------------
    # Step 3: Build modified feature matrices for comparison
    # -----------------------------------------------------------------------
    print("\n--- Step 3: Building comparison datasets ---")

    # 3A: HSC-injection (only if live queries succeeded)
    df_real = None
    if hsc_best is not None:
        hsc_matched = hsc_best[hsc_best["hsc_match_status"] != "none"].copy()
        df_real = _inject_real_photometry(df_all, hsc_matched)
        feat_real_raw, _ = build_feature_matrix(df_real, "PHAT")
        n_real_optical = {}
        for cls in df_real["Class"].unique():
            mk = df_real["Class"] == cls
            n_ok = (~feat_real_raw.loc[mk, COLOR_FEATURES].isna().all(axis=1)).sum()
            n_real_optical[cls] = (n_ok, mk.sum())
        print("  Sources with ≥1 non-NaN colour after HSC injection:")
        for cls, (n_ok, n_tot) in sorted(n_real_optical.items()):
            print(f"    {cls:<10}: {n_ok:4d}/{n_tot:4d}")

    # 3B: Colour-shuffle test — give SNRs colours drawn from non-SNR distribution.
    #   Rationale: if the classifier relies on the NaN→median fingerprint, giving
    #   SNRs realistic (but wrong-class) colours should degrade SNR recall because
    #   the X-ray features alone are insufficient.  If recall holds, the X-ray
    #   properties dominate.
    print("\n  Building colour-shuffle dataset (SNRs get non-SNR colour draws)…")
    rng = np.random.default_rng(42)
    df_shuffled = df_all.copy()
    snr_idx = df_all.index[df_all["Class"] == "SNR"]
    non_snr_idx = df_all.index[df_all["Class"] != "SNR"]
    pred_cols = [f for f in [
        "UVIS_F275W_pred", "UVIS_F336W_pred", "ACS_F475W_pred",
        "ACS_F814W_pred", "IR_F110W_pred", "IR_F160W_pred",
    ] if f in df_all.columns]

    # For each SNR, sample a random non-SNR donor row and copy its pred columns.
    donor_idx = rng.choice(non_snr_idx, size=len(snr_idx), replace=True)
    for snr_i, donor_i in zip(snr_idx, donor_idx):
        for col in pred_cols:
            df_shuffled.at[snr_i, col] = df_all.at[donor_i, col]
    n_shuffled_filled = sum(
        df_shuffled.loc[snr_idx, c].notna().sum() for c in pred_cols
    )
    print(f"    {n_shuffled_filled} pred-column cells filled for {len(snr_idx)} SNRs.")

    # 3C: X-ray-only test — drop all colour features and optical ratios for everyone.
    #   This shows classifier performance on X-ray features alone.
    print("  Building X-ray-only dataset (all HST pred columns set to NaN)…")
    df_xrayonly = df_all.copy()
    for col in pred_cols:
        df_xrayonly[col] = np.nan

    # -----------------------------------------------------------------------
    # Step 4: 5-fold CV comparison
    # -----------------------------------------------------------------------
    print("\n--- Step 4: 5-fold CV ---")

    print("\n  [A] Baseline (SED-translated / NaN-imputed for SNRs)")
    cv_base, cm_base, le_base, pc_base = _cv_and_metrics(df_all, "Baseline")

    if df_real is not None:
        print("\n  [B] Real photometry (HSC-injected where available)")
        cv_real, cm_real, le_real, pc_real = _cv_and_metrics(df_real, "Real-phot")
    else:
        cv_real = cm_real = le_real = pc_real = None

    print("\n  [C] Colour-shuffled (SNRs given non-SNR donor colours)")
    cv_shuf, cm_shuf, le_shuf, pc_shuf = _cv_and_metrics(df_shuffled, "Colour-shuffle")

    print("\n  [D] X-ray-only (all HST pred columns NaN for everyone)")
    cv_xray, cm_xray, le_xray, pc_xray = _cv_and_metrics(df_xrayonly, "X-ray-only")

    # -----------------------------------------------------------------------
    # Print summary tables
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TABLE: 5-fold CV global metrics")
    print("=" * 70)
    hdr = (f"{'Condition':<20} | {'Acc':>7} | {'±':>6} | "
           f"{'BalAcc':>7} | {'±':>6} | {'F1mac':>7} | {'±':>6}")
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep)
    for lbl, cv in [("Baseline", cv_base),
                    ("Real-photometry", cv_real)]:
        if cv is None:
            continue
        print(f"{lbl:<20} | "
              f"{cv['accuracy_mean']:>7.4f} | {cv['accuracy_std']:>6.4f} | "
              f"{cv['balanced_accuracy_mean']:>7.4f} | "
              f"{cv['balanced_accuracy_std']:>6.4f} | "
              f"{cv['f1_macro_mean']:>7.4f} | {cv['f1_macro_std']:>6.4f}")
    print(sep)

    print()
    print("TABLE: Per-class recall (mean across 5 folds)")
    print("=" * 70)
    classes = le_base.classes_
    hdr2 = (f"{'Class':<10} | {'Recall(base)':>12} | "
            + (f"{'Recall(real)':>12} | {'Δ recall':>9}" if pc_real else ""))
    sep2 = "-" * len(hdr2)
    print(sep2); print(hdr2); print(sep2)
    for cls in classes:
        r_b = pc_base.get(cls, {}).get("recall_mean", float("nan"))
        r_r = pc_real.get(cls, {}).get("recall_mean", float("nan")) if pc_real else None
        flag = " <-- SNR" if cls == "SNR" else ""
        if r_r is not None:
            print(f"{cls:<10} | {r_b:>12.4f} | {r_r:>12.4f} | "
                  f"{r_r - r_b:>+9.4f}{flag}")
        else:
            print(f"{cls:<10} | {r_b:>12.4f}{flag}")
    print(sep2)

    print()
    print("TABLE: Per-class precision (mean across 5 folds)")
    print("=" * 70)
    hdr3 = (f"{'Class':<10} | {'Prec(base)':>10} | "
            + (f"{'Prec(real)':>10} | {'Δ prec':>7}" if pc_real else ""))
    sep3 = "-" * len(hdr3)
    print(sep3); print(hdr3); print(sep3)
    for cls in classes:
        p_b = pc_base.get(cls, {}).get("precision_mean", float("nan"))
        p_r = pc_real.get(cls, {}).get("precision_mean", float("nan")) if pc_real else None
        flag = " <-- SNR" if cls == "SNR" else ""
        if p_r is not None:
            print(f"{cls:<10} | {p_b:>10.4f} | {p_r:>10.4f} | "
                  f"{p_r - p_b:>+7.4f}{flag}")
        else:
            print(f"{cls:<10} | {p_b:>10.4f}{flag}")
    print(sep3)

    # -----------------------------------------------------------------------
    # Step 6: Figures
    # -----------------------------------------------------------------------
    print(f"\n--- Step 6: Writing figures to {FIGURES_PDF} ---")
    classes_list = list(classes)

    with PdfPages(FIGURES_PDF) as pdf:

        # --- Page 1: Feature importance with has_real_hst ---
        fig, ax = plt.subplots(figsize=(9, 5))
        top15 = imp_series.head(15)[::-1]
        colours = ["crimson" if f == "has_real_hst" else "steelblue"
                   for f in top15.index]
        ax.barh(range(len(top15)), top15.values, color=colours)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.index, fontsize=8)
        ax.set_xlabel("Gini importance", fontsize=9)
        ax.set_title(
            f"Top-15 feature importances (has_real_hst in red — rank {has_hst_rank})",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # --- Page 2: Confusion matrices ---
        n_panels = 2 if cm_real is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        _plot_confusion(axes[0], cm_base, classes_list, "Baseline (SED-translated)")
        if cm_real is not None:
            _plot_confusion(axes[1], cm_real, classes_list, "Real photometry (HSC-injected)")
        fig.suptitle("Normalised confusion matrices (5-fold CV pooled)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # --- Page 3: Per-class recall comparison ---
        if pc_real is not None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, metric, title in zip(
                axes,
                ["recall_mean", "precision_mean", "f1_mean"],
                ["Recall", "Precision", "F1"],
            ):
                _plot_per_class_metrics(
                    ax, classes_list, pc_base, pc_real, metric, title,
                )
            # Highlight SNR bars
            snr_idx = list(classes_list).index("SNR") if "SNR" in classes_list else None
            if snr_idx is not None:
                for ax in axes:
                    ax.axvline(snr_idx, color="red", lw=1.2, ls="--", alpha=0.5,
                               label="SNR position")
            fig.suptitle("Per-class metrics: Baseline vs Real photometry", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- Page 4: SNR NaN coverage summary ---
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        summary_lines = [
            "SNR photometry-leakage summary",
            "",
            f"SNR sources with ALL 11 optical features NaN:  {nan_snr:.0%}",
            f"Non-SNR sources with ALL 11 optical features NaN: {nan_nonsnr:.0%}",
            "",
            f"has_real_hst Gini rank: {has_hst_rank}/{len(imp_series)}",
            f"has_real_hst Gini importance: {has_hst_imp:.4f}",
            "",
            ("=> HIGH RISK: classifier exploits NaN pattern"
             if has_hst_rank <= 5 else
             "=> MODERATE RISK: NaN pattern provides partial signal"
             if has_hst_rank <= 10 else
             "=> LOW RISK: classifier does not rely on NaN pattern"),
        ]
        ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", family="monospace")
        pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "SNR Photometry-Leakage Diagnostic"
        d["Author"] = "xclass snr_leakage_diagnostic.py"

    print(f"  Figures saved to {FIGURES_PDF}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hsc", action="store_true",
        help="Skip HSC queries entirely; run only the NaN-leakage and "
             "Gini-importance diagnostics.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=500,
        help="Max number of NEW HSC queries for large non-SNR classes "
             "(AGN, LM-STAR, HM-STAR). Default: 500.",
    )
    args = parser.parse_args()
    main(args)

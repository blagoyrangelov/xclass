"""Ablation study: PS1-only SED fitting vs current PS1+2MASS.

Tests whether high chi²_red values in the full SED fits are driven by
cross-survey (PS1 vs 2MASS) photometric inconsistencies.

Steps
-----
1. Re-run SED fitting for all non-SNR training sources using only
   PanSTARRS grizy photometry (2MASS JHKs zeroed out).
2. Compare chi²_red distributions (current vs PS1-only) by class.
3. For sources with PS1-only chi²_red < 10, compare predicted F475W
   and F814W magnitudes to the all-surveys fit.
4. Re-run 5-fold CV using PS1-only translated features.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xclass import config
from xclass.io import load_all_filter_curves, load_pickles_cache, load_agn_composite
from xclass.photometry import translate_catalog
from xclass.features import build_feature_matrix
from xclass.classifier import cross_validate

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impute_median(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def _chi2_stats(series: pd.Series) -> tuple[float, float]:
    finite = series.dropna()
    if len(finite) == 0:
        return float("nan"), float("nan")
    median = float(np.median(finite))
    frac_lt10 = float((finite < 10).mean())
    return median, frac_lt10


def _print_comparison_table(all_df: pd.DataFrame, ps1_df: pd.DataFrame) -> None:
    CLASS_COL = "Class"
    CHI2_COL = "xclass_fit_chi2red"
    NBANDS_COL = "xclass_n_bands_used"

    classes = sorted(all_df[CLASS_COL].unique())
    header = (
        f"{'Class':<10} | {'n(all,≥3b)':>10} | {'n(PS1,≥3b)':>10} | "
        f"{'med chi2(all)':>14} | {'med chi2(PS1)':>14} | "
        f"{'frac<10(all)':>13} | {'frac<10(PS1)':>13}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for cls in classes:
        a = all_df[all_df[CLASS_COL] == cls]
        p = ps1_df[ps1_df[CLASS_COL] == cls]

        # Filter to ≥3 bands used
        a_fit = a[a[NBANDS_COL] >= 3][CHI2_COL]
        p_fit = p[p[NBANDS_COL] >= 3][CHI2_COL]

        med_a, frac_a = _chi2_stats(a_fit)
        med_p, frac_p = _chi2_stats(p_fit)

        print(
            f"{cls:<10} | {len(a_fit):>10} | {len(p_fit):>10} | "
            f"{med_a:>14.1f} | {med_p:>14.1f} | "
            f"{frac_a:>13.3f} | {frac_p:>13.3f}"
        )

    print(sep)
    # Overall (all classes combined)
    a_all = all_df[all_df[NBANDS_COL] >= 3][CHI2_COL]
    p_all = ps1_df[ps1_df[NBANDS_COL] >= 3][CHI2_COL]
    med_a, frac_a = _chi2_stats(a_all)
    med_p, frac_p = _chi2_stats(p_all)
    print(
        f"{'ALL':<10} | {len(a_all):>10} | {len(p_all):>10} | "
        f"{med_a:>14.1f} | {med_p:>14.1f} | "
        f"{frac_a:>13.3f} | {frac_p:>13.3f}"
    )
    print(sep)


def _print_magnitude_comparison(
    all_df: pd.DataFrame,
    ps1_df: pd.DataFrame,
) -> None:
    CHI2_COL = "xclass_fit_chi2red"
    CLASS_COL = "Class"
    FILTERS = [("ACS_F475W", "F475W"), ("ACS_F814W", "F814W")]

    mask_good_ps1 = ps1_df[CHI2_COL] < 10
    mask_good_all = all_df[CHI2_COL] < 10
    mask_both = mask_good_ps1 & mask_good_all

    print(f"\nSources with PS1-only chi2_red < 10: {mask_good_ps1.sum()}")
    print(f"Sources with all-surveys chi2_red < 10: {mask_good_all.sum()}")
    print(f"Sources where both < 10: {mask_both.sum()}")

    for filt_key, filt_label in FILTERS:
        pred_col = f"{filt_key}_pred"
        if pred_col not in all_df.columns or pred_col not in ps1_df.columns:
            print(f"\n{filt_label}: prediction columns not found — skipping")
            continue

        a_mag = all_df.loc[mask_both, pred_col]
        p_mag = ps1_df.loc[mask_both, pred_col]
        diff = p_mag - a_mag
        finite_diff = diff.dropna()

        print(f"\n{filt_label} predictions (PS1-only vs all-surveys), "
              f"n={len(finite_diff)} sources where both chi2_red < 10:")
        print(f"  mean diff (PS1 - all) = {finite_diff.mean():+.4f} mag")
        print(f"  std  diff             = {finite_diff.std():.4f} mag")
        print(f"  |diff| > 0.5 mag      = {(finite_diff.abs() > 0.5).sum()} "
              f"({100*(finite_diff.abs() > 0.5).mean():.1f}%)")

        # Per-class breakdown
        classes = sorted(all_df.loc[mask_both, CLASS_COL].unique())
        if len(classes) > 1:
            print(f"  Per-class mean diff:")
            for cls in classes:
                cls_mask = mask_both & (all_df[CLASS_COL] == cls)
                d = (ps1_df.loc[cls_mask, pred_col] - all_df.loc[cls_mask, pred_col]).dropna()
                if len(d) > 0:
                    print(f"    {cls:<10} n={len(d):4d}  mean={d.mean():+.4f}  "
                          f"std={d.std():.4f}")


def _run_cv(translated: pd.DataFrame, label: str) -> dict:
    CLASS_COL = "Class"
    non_snr = translated[translated[CLASS_COL] != "SNR"].copy()
    feature_df, _ = build_feature_matrix(non_snr, ml_filter_set="PHAT")
    labels = non_snr[CLASS_COL]
    X = _impute_median(feature_df)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(labels.values), index=labels.index)
    print(f"\n  Running 5-fold CV [{label}]  "
          f"(n={len(X)}, {X.shape[1]} features)...")
    return cross_validate(X, y, n_splits=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    TRANSLATED_PATH = PROJECT_ROOT / "data" / "processed" / "translated_catalog.csv"
    TMASS_COLS = ["tmass_j", "tmass_j_err", "tmass_h", "tmass_h_err",
                  "tmass_k", "tmass_k_err"]
    CLASS_COL = "Class"

    # ------------------------------------------------------------------
    # 1. Load existing translated catalog (all-surveys)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Loading translated catalog (all-surveys results)...")
    all_df = pd.read_csv(TRANSLATED_PATH, low_memory=False)
    non_snr_mask = all_df[CLASS_COL] != "SNR"
    all_non_snr = all_df[non_snr_mask].copy()
    print(f"  Total sources: {len(all_df)}  |  non-SNR: {len(all_non_snr)}")

    # ------------------------------------------------------------------
    # 2. Load spectral resources
    # ------------------------------------------------------------------
    print("\nLoading filter curves and spectral templates...")
    all_hst = config.ALL_HST_FILTERS
    hst_filter_curves = load_all_filter_curves(all_hst, config.FILTER_CACHE_DIR)
    all_filter_names = list(hst_filter_curves.keys())
    survey_filter_curves = load_all_filter_curves(config.SURVEY_FILTERS,
                                                  config.FILTER_CACHE_DIR)
    filter_curves = {**survey_filter_curves, **hst_filter_curves}
    pickles_cache = load_pickles_cache(cache_dir=config.SPECTRA_CACHE_DIR)
    agn_composite = load_agn_composite(cache_dir=config.SPECTRA_CACHE_DIR)
    print(f"  Filter curves loaded: {len(filter_curves)}  "
          f"({len(survey_filter_curves)} survey, {len(hst_filter_curves)} HST)")

    # ------------------------------------------------------------------
    # 3. Re-run SED fitting with PS1-only (null out 2MASS columns)
    # ------------------------------------------------------------------
    print("\nBuilding PS1-only input (zeroing 2MASS columns)...")
    ps1_input = all_non_snr.copy()
    for col in TMASS_COLS:
        if col in ps1_input.columns:
            ps1_input[col] = np.nan

    # Drop existing SED result columns so translate_catalog can append fresh ones
    sed_cols = [c for c in ps1_input.columns
                if c.startswith("xclass_") or c.endswith("_pred") or c.endswith("_pred_err")]
    ps1_input = ps1_input.drop(columns=sed_cols, errors="ignore")

    print("Running PS1-only SED translation (this may take a few minutes)...")
    ps1_translated_new = translate_catalog(
        ps1_input,
        filter_curves=filter_curves,
        all_filter_names=all_filter_names,
        pickles_cache=pickles_cache,
        agn_composite=agn_composite,
        n_jobs=-1,
        cache_path=None,  # always recompute
    )
    ps1_non_snr = ps1_translated_new.copy()
    print(f"  PS1-only translation complete: {len(ps1_non_snr)} sources")

    # ------------------------------------------------------------------
    # 4. Chi²_red comparison table
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("TABLE 1: Chi²_red comparison (all-surveys: PS1+2MASS  vs  PS1-only)")
    print("=" * 70)
    _print_comparison_table(all_non_snr, ps1_non_snr)

    # ------------------------------------------------------------------
    # 5. Predicted magnitude comparison (F475W and F814W)
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("TABLE 2: Predicted magnitude consistency (PS1-only chi²_red < 10)")
    print("=" * 70)

    # Align indices: ps1_non_snr has same index as all_non_snr (subsetted)
    # ps1_translated_new was built from ps1_input which has same index
    _print_magnitude_comparison(all_non_snr, ps1_non_snr)

    # ------------------------------------------------------------------
    # 6. 5-fold CV comparison
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("TABLE 3: 5-fold cross-validation (Stage 1 RF, all classes)")
    print("=" * 70)

    # Baseline CV — use existing translated catalog (all surveys)
    cv_all = _run_cv(all_df, "PS1+2MASS (baseline)")

    # PS1-only CV — build a replacement DataFrame: SNR rows from all_df,
    # non-SNR rows from ps1_non_snr, to give build_feature_matrix the
    # full catalog with class labels.
    snr_rows = all_df[all_df[CLASS_COL] == "SNR"].copy()
    ps1_full = pd.concat([ps1_non_snr, snr_rows], ignore_index=True)
    cv_ps1 = _run_cv(ps1_full, "PS1-only")

    print()
    header2 = (
        f"{'Condition':<20} | {'Acc':>7} | {'±':>6} | "
        f"{'BalAcc':>7} | {'±':>6} | {'F1macro':>7} | {'±':>6} | {'MCC':>6}"
    )
    sep2 = "-" * len(header2)
    print(sep2)
    print(header2)
    print(sep2)

    for label, cv in [("PS1+2MASS", cv_all), ("PS1-only", cv_ps1)]:
        print(
            f"{label:<20} | "
            f"{cv['accuracy_mean']:>7.4f} | {cv['accuracy_std']:>6.4f} | "
            f"{cv['balanced_accuracy_mean']:>7.4f} | {cv['balanced_accuracy_std']:>6.4f} | "
            f"{cv['f1_macro_mean']:>7.4f} | {cv['f1_macro_std']:>6.4f} | "
            f"{cv['matthews_corrcoef_mean']:>6.4f}"
        )
    print(sep2)

    print("\nDone.")


if __name__ == "__main__":
    main()

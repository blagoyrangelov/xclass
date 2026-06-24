"""Label-leakage experiment for SED model selection.

Tests whether using the true class label to choose the SED template family
(Pickles / AGN composite / two-component) introduces classifier leakage.

Three conditions
----------------
class-aware   : existing translated_catalog.csv  (class-specific SED model per source)
pickles-blind : all sources translated with Class="LM-STAR"  → primary=Pickles, fallback=BB
bb-blind      : all sources translated with Class="UNKNOWN"  → primary=BB, fallback=BB
                (UNKNOWN is not in config.SED_MODEL_PRIMARY, so both stages fall back to BB)

Each condition: build feature matrix → median impute NaN → 5-fold stratified CV.
Results printed as a side-by-side comparison table.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xclass import config
from xclass.features import build_feature_matrix
from xclass.classifier import cross_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSLATED_CSV    = config.PROCESSED_DIR / "translated_catalog.csv"
XRAY_TRAINING_CSV = config.PROCESSED_DIR / "xray_training_table.csv"
SNR_CSV           = config.PROCESSED_DIR / "snr_ml_catalog.csv"
PICKLES_CACHE     = config.PROCESSED_DIR / "translated_catalog_pickles_blind.csv"
BB_CACHE          = config.PROCESSED_DIR / "translated_catalog_bb_blind.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sed_resources():
    """Load filter curves, Pickles cache, and AGN composite."""
    from xclass.io import load_filter_curves, load_pickles_cache, load_agn_composite

    all_hst_filters = {
        **config.ACS_WFC_FILTERS,
        **config.WFC3_UVIS_FILTERS,
        **config.WFC3_IR_FILTERS,
    }
    hst_filter_curves = load_filter_curves(all_hst_filters, cache_dir=config.FILTER_CACHE_DIR)
    survey_filter_curves = load_filter_curves(config.SURVEY_FILTERS,
                                              cache_dir=config.FILTER_CACHE_DIR)
    filter_curves = {**survey_filter_curves, **hst_filter_curves}
    all_filter_names = list(hst_filter_curves.keys())

    pickles_cache = load_pickles_cache(cache_dir=config.SPECTRA_CACHE_DIR)
    agn_composite = load_agn_composite(cache_dir=config.SPECTRA_CACHE_DIR)

    return filter_curves, all_filter_names, pickles_cache, agn_composite


def _build_snr_rows() -> pd.DataFrame:
    """Return SNR rows in the same schema as xray_training_table."""
    if not SNR_CSV.exists():
        log.warning("SNR catalog not found at %s — skipping SNR rows", SNR_CSV)
        return pd.DataFrame()

    snr = pd.read_csv(SNR_CSV)
    # Minimal columns that translate_catalog needs: Class (or class_label), ra/dec, fluxes.
    # The run_pipeline stage_translate() calls _build_snr_xray_rows() which is a more
    # complex merge; here we just need the class label set so SNR SED family fires.
    # SNRs have xclass_sed_family='none' regardless of class — so the Class value only
    # matters for the class-blind condition where we override it anyway.
    if "Class" not in snr.columns and "class_label" in snr.columns:
        snr = snr.rename(columns={"class_label": "Class"})
    return snr


def _translate_blind(xray_td: pd.DataFrame, blind_class: str,
                     cache_path: Path) -> pd.DataFrame:
    """Run translate_catalog with every source assigned *blind_class*."""
    from xclass.photometry import translate_catalog

    df = xray_td.copy()
    # Override class label — this is the key manipulation for the leakage test
    df["Class"] = blind_class

    filter_curves, all_filter_names, pickles_cache, agn_composite = _load_sed_resources()

    return translate_catalog(
        df,
        filter_curves=filter_curves,
        all_filter_names=all_filter_names,
        pickles_cache=pickles_cache,
        agn_composite=agn_composite,
        cache_path=str(cache_path),
    )


# Stage 1 broad class mapping (matches asymmetric two-stage architecture in notebook 04b)
_STAGE1_CLASS_MAP = {
    "AGN":      "AGN",
    "LM-STAR":  "STAR",
    "HM-STAR":  "STAR",
    "LMXB":     "OTHER",
    "HMXB":     "OTHER",
    "CV":       "OTHER",
    "SNR":      "SNR",
}

_PRODUCTION_FEATURES = [
    "logFx_S", "logFx_M", "logFx_H", "logFx_B",
    "HR_SM", "HR_MH", "HR_SH", "HR_BM",
    "color_F275W_F336W", "color_F336W_F475W", "color_F475W_F814W",
    "color_F814W_F110W", "color_F110W_F160W",
    "color_F275W_F814W", "color_F275W_F160W", "color_F336W_F814W", "color_F475W_F160W",
    "logFxFopt_B_F475W", "logFxFopt_B_F814W",
]


def _cv_from_catalog(df: pd.DataFrame, label: str) -> dict:
    """Build feature matrix, restrict to production 19 features, impute, run 5-fold CV."""
    log.info("Building feature matrix for condition: %s", label)
    feature_df, _ = build_feature_matrix(df, ml_filter_set="PHAT")

    if "Class" in df.columns:
        y_raw = df.loc[feature_df.index, "Class"]
    elif "class_label" in df.columns:
        y_raw = df.loc[feature_df.index, "class_label"]
    else:
        raise ValueError("No class label column found in catalog")

    # Map 7 fine classes → 4 broad Stage 1 classes to match production CV conditions
    y = y_raw.map(_STAGE1_CLASS_MAP)
    unmapped = y.isna()
    if unmapped.any():
        log.warning("Unmapped classes: %s", y_raw[unmapped].unique())

    # Restrict to the exact 19 features the production model was trained on.
    # This excludes match quality, significance, and GAIA features (all 100% NaN
    # for training data) so the baseline CV reproduces the paper's reported metrics.
    available = [c for c in _PRODUCTION_FEATURES if c in feature_df.columns]
    missing = set(_PRODUCTION_FEATURES) - set(available)
    if missing:
        log.warning("Missing production features: %s", missing)
    X = feature_df[available].copy()

    log.info("%s: %d sources, %d features, Stage1 class dist:\n%s",
             label, len(X), len(available), y.value_counts().to_string())

    # Median imputation — same strategy as prepare_for_ml()
    for col in X.columns:
        med = X[col].median()
        if X[col].isna().any() and pd.notna(med):
            X[col] = X[col].fillna(med)
        elif X[col].isna().any():
            X[col] = X[col].fillna(0.0)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y.dropna())
    X = X.loc[y.dropna().index]

    log.info("Running 5-fold stratified CV for: %s", label)
    metrics = cross_validate(X, pd.Series(y_enc, index=X.index), n_splits=5)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}

    # ── Condition 1: class-aware baseline ────────────────────────────────────
    log.info("=== Condition 1: class-aware (existing translated_catalog.csv) ===")
    df_aware = pd.read_csv(TRANSLATED_CSV)
    results["class-aware"] = _cv_from_catalog(df_aware, "class-aware")

    # ── Load xray_training_table for the two blind conditions ─────────────────
    log.info("Loading xray_training_table.csv")
    xray_td = pd.read_csv(XRAY_TRAINING_CSV)
    log.info("xray_training_table: %d rows", len(xray_td))

    # ── Condition 2: Pickles-blind (Class="LM-STAR") ──────────────────────────
    log.info("=== Condition 2: Pickles-blind (Class=LM-STAR for all sources) ===")
    if PICKLES_CACHE.exists():
        log.info("Using cached Pickles-blind translation: %s", PICKLES_CACHE)
        df_pickles = pd.read_csv(PICKLES_CACHE)
        # Restore true labels from xray_training_table for CV
        df_pickles["Class"] = xray_td["Class"].values
    else:
        df_pickles = _translate_blind(xray_td, "LM-STAR", PICKLES_CACHE)
        # Restore true class labels for CV (we need correct y for CV scoring)
        df_pickles["Class"] = xray_td["Class"].values

    results["pickles-blind"] = _cv_from_catalog(df_pickles, "pickles-blind")

    # ── Condition 3: Blackbody-blind (Class="UNKNOWN") ────────────────────────
    log.info("=== Condition 3: BB-blind (Class=UNKNOWN for all sources) ===")
    if BB_CACHE.exists():
        log.info("Using cached BB-blind translation: %s", BB_CACHE)
        df_bb = pd.read_csv(BB_CACHE)
        df_bb["Class"] = xray_td["Class"].values
    else:
        df_bb = _translate_blind(xray_td, "UNKNOWN", BB_CACHE)
        df_bb["Class"] = xray_td["Class"].values

    results["bb-blind"] = _cv_from_catalog(df_bb, "bb-blind")

    # ── Print comparison table ────────────────────────────────────────────────
    metrics_order = [
        ("accuracy_mean",           "accuracy_std",           "Accuracy"),
        ("balanced_accuracy_mean",  "balanced_accuracy_std",  "Bal. Accuracy"),
        ("f1_macro_mean",           "f1_macro_std",           "Macro F1"),
        ("matthews_corrcoef_mean",  "matthews_corrcoef_std",  "MCC"),
    ]
    conditions = ["class-aware", "pickles-blind", "bb-blind"]

    col_w = 22
    header = f"{'Metric':<18}" + "".join(f"{c:>{col_w}}" for c in conditions)
    sep = "-" * len(header)

    print("\n")
    print("=" * len(header))
    print("  LABEL LEAKAGE EXPERIMENT — 5-fold Stratified CV (n=5)")
    print("=" * len(header))
    print(header)
    print(sep)

    for mean_key, std_key, label in metrics_order:
        row = f"{label:<18}"
        for cond in conditions:
            m = results[cond][mean_key]
            s = results[cond][std_key]
            row += f"{f'{m:.3f} ± {s:.3f}':>{col_w}}"
        print(row)

    print(sep)
    print()
    print("Conditions:")
    print("  class-aware   : SED model selected by true class label (current pipeline)")
    print("  pickles-blind : all sources fitted with Pickles atlas (Class=LM-STAR forced)")
    print("  bb-blind      : all sources fitted with blackbody (Class=UNKNOWN → BB fallback)")
    print()
    print("Notes:")
    print("  - CV uses 4 broad Stage 1 labels (AGN / OTHER / SNR / STAR) matching production setup.")
    print("  - class-aware: 12264 sources (includes 150 SNR); blind variants: 12114 (no SNR rows).")
    print("  - 19 production features used (X-ray log-fluxes, hardness ratios, 9 HST colors,")
    print("    2 X-ray/optical ratios); match quality, significance, and GAIA excluded.")
    print()


if __name__ == "__main__":
    main()

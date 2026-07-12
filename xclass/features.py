"""xclass.features — Feature engineering for ML classification.

Computes hardness ratios, HST colours, and X-ray/optical flux ratios,
then assembles them into the ML feature matrix (22 features, five groups).

Functions
---------
compute_hardness_ratios
compute_hst_colors
compute_xray_optical_ratios
build_feature_matrix
prepare_for_ml
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from xclass import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# X-ray hardness ratios
# ---------------------------------------------------------------------------


def _safe_hr(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute (a - b) / (a + b), returning NaN when denominator <= 0."""
    denom = a + b
    hr = (a - b) / denom
    hr[~np.isfinite(denom) | (denom == 0)] = float("nan")
    return hr


def _get_flux_col(df: pd.DataFrame, short: str) -> pd.Series:
    """Return a flux band series, trying multiple column-naming conventions.

    Checks in order:
      1. ``Fx_{short}``          — training-data convention
      2. ``flux_aper90_avg_{lower}`` — CSC 2.1.1 convention
    Returns an all-NaN series if neither is present.
    """
    candidates = [f"Fx_{short}", f"flux_aper90_avg_{short.lower()}"]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(float("nan"), index=df.index, dtype=float)


def compute_hardness_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute X-ray hardness ratios from soft/medium/hard/broad fluxes.

    Formulae::

        HR_SM = (Fx_S - Fx_M) / (Fx_S + Fx_M)
        HR_MH = (Fx_M - Fx_H) / (Fx_M + Fx_H)
        HR_SH = (Fx_S - Fx_H) / (Fx_S + Fx_H)
        HR_BM = (Fx_B - Fx_M) / (Fx_B + Fx_M)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain X-ray band flux columns in one of the supported naming
        conventions: ``Fx_S/M/H/B`` (training data) or
        ``flux_aper90_avg_s/m/h/b`` (CSC 2.1.1 application data).
        Missing bands produce all-NaN hardness ratios.

    Returns
    -------
    pd.DataFrame
        New columns appended: HR_SM, HR_MH, HR_SH, HR_BM.
        NaN when denominator is zero or both inputs are NaN.
    """
    out = df.copy()
    s = _get_flux_col(df, "S")
    m = _get_flux_col(df, "M")
    h = _get_flux_col(df, "H")
    b = _get_flux_col(df, "B")

    out["HR_SM"] = _safe_hr(s, m)
    out["HR_MH"] = _safe_hr(m, h)
    out["HR_SH"] = _safe_hr(s, h)
    out["HR_BM"] = _safe_hr(b, m)
    return out


# ---------------------------------------------------------------------------
# HST colours
# ---------------------------------------------------------------------------


def _short_name(filt_label: str) -> str:
    """Strip instrument prefix from filter label: 'UVIS_F275W' -> 'F275W'."""
    return filt_label.split("_")[-1]


def compute_hst_colors(
    df: pd.DataFrame,
    filter_prefix_list: list[str],
) -> pd.DataFrame:
    """Compute HST colours from predicted magnitudes.

    Consecutive colours (e.g. F275W-F336W, F336W-F475W, …) plus
    wide-baseline colours (F275W-F814W, F275W-F160W, F336W-F814W).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``{filter}_pred`` columns for each label in
        *filter_prefix_list*.
    filter_prefix_list : list of str
        Ordered filter labels (e.g. ``['UVIS_F275W', 'UVIS_F336W', ...]``).

    Returns
    -------
    pd.DataFrame
        New colour columns appended, named e.g. ``color_F275W_F336W``.
        NaN when either input magnitude is NaN.
    """
    out = df.copy()

    # Filter labels that have pred columns in the DataFrame
    available = [
        f for f in filter_prefix_list
        if f"{f}_pred" in df.columns
    ]

    if len(available) < 2:
        return out

    # Build magnitude series (NaN if column missing)
    mags: dict[str, pd.Series] = {}
    for filt in available:
        col = f"{filt}_pred"
        mags[filt] = pd.to_numeric(df[col], errors="coerce")

    # Consecutive colours
    for i in range(len(available) - 1):
        f1, f2 = available[i], available[i + 1]
        col_name = f"color_{_short_name(f1)}_{_short_name(f2)}"
        out[col_name] = mags[f1] - mags[f2]

    # Wide-baseline colours (only if relevant filters are present)
    _WIDE_PAIRS = [
        # (blue, red)
        ("UVIS_F275W", "ACS_F814W"),
        ("UVIS_F275W", "IR_F160W"),
        ("UVIS_F336W", "ACS_F814W"),
        ("ACS_F475W", "IR_F160W"),
    ]
    for f_blue, f_red in _WIDE_PAIRS:
        if f_blue in mags and f_red in mags:
            col_name = f"color_{_short_name(f_blue)}_{_short_name(f_red)}"
            if col_name not in out.columns:
                out[col_name] = mags[f_blue] - mags[f_red]

    return out


# ---------------------------------------------------------------------------
# X-ray / optical flux ratios
# ---------------------------------------------------------------------------


def compute_xray_optical_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute X-ray/optical flux ratios and log-flux features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Fx_S, Fx_M, Fx_H, Fx_B and ACS_F475W_pred,
        ACS_F814W_pred (or equivalent predicted magnitude columns).

    Returns
    -------
    pd.DataFrame
        New columns: logFx_B, logFx_S, logFx_H,
        logFxFopt_B_F475W, logFxFopt_B_F814W.
        NaN for non-positive inputs.
    """
    out = df.copy()

    def _safe_log(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        result = pd.Series(np.where(s > 0, np.log10(s), float("nan")), index=s.index)
        return result

    out["logFx_B"] = _safe_log(_get_flux_col(df, "B"))
    out["logFx_S"] = _safe_log(_get_flux_col(df, "S"))
    out["logFx_M"] = _safe_log(_get_flux_col(df, "M"))
    out["logFx_H"] = _safe_log(_get_flux_col(df, "H"))

    # X-ray to optical flux ratio: log10(Fx / f_nu_opt)
    # f_nu_opt from AB mag: f_nu = 10^(-(mag + 48.60) / 2.5)
    _AB_ZP = 48.60
    fx_b = _get_flux_col(df, "B")

    for opt_col, ratio_col in [
        ("ACS_F475W_pred", "logFxFopt_B_F475W"),
        ("ACS_F814W_pred", "logFxFopt_B_F814W"),
    ]:
        if opt_col in df.columns:
            mag = pd.to_numeric(df[opt_col], errors="coerce")
            fnu_opt = 10.0 ** (-(mag + _AB_ZP) / 2.5)
            valid = (fx_b > 0) & (fnu_opt > 0)
            ratio = pd.Series(float("nan"), index=df.index, dtype=float)
            ratio[valid] = np.log10(fx_b[valid] / fnu_opt[valid])
            out[ratio_col] = ratio
        else:
            out[ratio_col] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    ml_filter_set: str = "PHAT",
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the full ML feature matrix.

    Feature groups (in order):
    1. X-ray (log): logFx_S, logFx_M, logFx_H, logFx_B
    2. Hardness ratios: HR_SM, HR_MH, HR_SH, HR_BM
    3. HST colours (from *ml_filter_set*)
    4. X-ray/optical ratios: logFxFopt_B_F475W, logFxFopt_B_F814W
    5. Match quality: xray_pos_err_arcsec, normsep_class_xray, significance

    This yields the production feature matrix of 22 features in five groups.
    Raw magnitudes, raw fluxes, RA/Dec, and source IDs are excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset with all intermediate columns.
    ml_filter_set : str
        Key in ``config.ML_FILTER_SETS``, e.g. 'PHAT' or 'ACS_ONLY'.

    Returns
    -------
    feature_df : pd.DataFrame
        Feature columns only (same row index as *df*).
    feature_names : list of str
        Ordered list of feature column names.
    """
    # Step 1: Compute all derived features
    tmp = compute_hardness_ratios(df)
    tmp = compute_xray_optical_ratios(tmp)

    # HST colors for chosen filter set
    filter_list = config.ML_FILTER_SETS.get(ml_filter_set, config.PHAT_FILTER_SET)
    tmp = compute_hst_colors(tmp, filter_list)

    # Step 2: Identify all color columns just created
    color_cols = [c for c in tmp.columns if c.startswith("color_")]

    # Step 3: Build ordered feature list
    feature_names: list[str] = []

    _xray_log = ["logFx_S", "logFx_M", "logFx_H", "logFx_B"]
    _hr = ["HR_SM", "HR_MH", "HR_SH", "HR_BM"]
    _ratios = ["logFxFopt_B_F475W", "logFxFopt_B_F814W"]
    _match = ["xray_pos_err_arcsec", "normsep_class_xray"]
    # Standardise significance column name to "xray_significance" so that
    # feature names are consistent between training and application data.
    # CSC application data has "significance"; training data has "xray_significance".
    if "significance" in tmp.columns and "xray_significance" not in tmp.columns:
        tmp = tmp.rename(columns={"significance": "xray_significance"})
    _sig_col = "xray_significance" if "xray_significance" in tmp.columns else None

    for col in _xray_log + _hr + color_cols + _ratios + _match:
        if col in tmp.columns:
            feature_names.append(col)

    if _sig_col:
        feature_names.append(_sig_col)

    # Only include columns that exist
    feature_names = [c for c in feature_names if c in tmp.columns]
    feature_df = tmp[feature_names].copy()

    log.info(
        "build_feature_matrix: %d features, %d rows", len(feature_names), len(feature_df)
    )
    return feature_df, feature_names


# ---------------------------------------------------------------------------
# Train/val/test split (no SMOTE; balanced_subsample handles imbalance)
# ---------------------------------------------------------------------------


def prepare_for_ml(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    test_frac: float = 0.10,
    val_frac: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Split data into stratified train/val/test sets.

    Class imbalance is handled downstream by the Random Forest's
    ``class_weight="balanced_subsample"`` (production configuration); no
    SMOTE oversampling is applied.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix (from ``build_feature_matrix``).
    labels : pd.Series
        Class labels aligned with *feature_df*.
    test_frac : float
        Fraction for test split.
    val_frac : float
        Fraction for validation split (of the non-test portion).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Keys: X_train, X_val, X_test, y_train, y_val, y_test,
        label_encoder, feature_names, class_weights.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    # Impute NaN with column median
    X = feature_df.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Normalise labels to a plain numpy array of strings for sklearn compatibility
    if isinstance(labels, pd.Series):
        _index = labels.index
        y = pd.array(labels, dtype=object)  # avoid Arrow backend issues
    else:
        _index = pd.RangeIndex(len(labels))
        y = np.asarray(labels, dtype=object)

    # Encode labels
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=_index)

    # Stratified train / (val + test) split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_enc,
        test_size=test_frac + val_frac,
        stratify=y_enc,
        random_state=random_state,
    )

    # Validation vs test from the tmp portion
    val_share = val_frac / (test_frac + val_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - val_share,
        stratify=y_tmp,
        random_state=random_state,
    )

    # No SMOTE: class imbalance is handled by the RF's balanced_subsample
    # class weighting (production configuration).

    # Class weights for balanced training
    classes = np.unique(y_train.values)
    weights = compute_class_weight("balanced", classes=classes, y=y_train.values)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    log.info(
        "prepare_for_ml: train=%d  val=%d  test=%d  features=%d",
        len(X_train), len(X_val), len(X_test), len(feature_df.columns),
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": le,
        "feature_names": list(feature_df.columns),
        "class_weights": class_weights,
    }

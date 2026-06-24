"""Tests for xclass.features — feature engineering.

All tests use synthetic data and do not require network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xray_df(**overrides) -> pd.DataFrame:
    """Create a minimal DataFrame with X-ray flux columns."""
    base = {
        "Fx_S": [1e-14],
        "Fx_M": [2e-14],
        "Fx_H": [3e-14],
        "Fx_B": [6e-14],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# compute_hardness_ratios tests
# ---------------------------------------------------------------------------


def test_hardness_ratios_known_input():
    """HR_SM = (S - M) / (S + M) for known S, M values."""
    from xclass.features import compute_hardness_ratios

    df = _make_xray_df(Fx_S=[3.0], Fx_M=[1.0], Fx_H=[0.5], Fx_B=[4.5])
    result = compute_hardness_ratios(df)

    # HR_SM = (3 - 1) / (3 + 1) = 0.5
    assert abs(result["HR_SM"].iloc[0] - 0.5) < 1e-10
    # HR_MH = (1 - 0.5) / (1 + 0.5) = 1/3
    assert abs(result["HR_MH"].iloc[0] - (1.0 / 3.0)) < 1e-10


def test_hardness_ratios_nan_when_both_inputs_nan():
    """HR is NaN when both numerator inputs are NaN."""
    from xclass.features import compute_hardness_ratios

    df = _make_xray_df(
        Fx_S=[float("nan")],
        Fx_M=[float("nan")],
        Fx_H=[1e-14],
        Fx_B=[1e-14],
    )
    result = compute_hardness_ratios(df)
    assert np.isnan(result["HR_SM"].iloc[0])


def test_hardness_ratios_nan_when_denominator_zero():
    """HR is NaN when S + M = 0 (denominator zero)."""
    from xclass.features import compute_hardness_ratios

    df = _make_xray_df(Fx_S=[0.0], Fx_M=[0.0], Fx_H=[1e-14], Fx_B=[1e-14])
    result = compute_hardness_ratios(df)
    assert np.isnan(result["HR_SM"].iloc[0])


# ---------------------------------------------------------------------------
# compute_hst_colors tests
# ---------------------------------------------------------------------------


def test_hst_colors_consecutive():
    """Consecutive colour F275W-F336W equals mag_F275W - mag_F336W."""
    from xclass.features import compute_hst_colors

    df = pd.DataFrame(
        {
            "UVIS_F275W_pred": [23.0],
            "UVIS_F336W_pred": [22.5],
            "ACS_F475W_pred": [22.0],
            "ACS_F814W_pred": [21.0],
            "IR_F110W_pred": [20.5],
            "IR_F160W_pred": [20.0],
        }
    )
    filter_list = [
        "UVIS_F275W", "UVIS_F336W", "ACS_F475W", "ACS_F814W", "IR_F110W", "IR_F160W"
    ]
    result = compute_hst_colors(df, filter_list)

    assert "color_F275W_F336W" in result.columns
    assert abs(result["color_F275W_F336W"].iloc[0] - 0.5) < 1e-10


def test_hst_colors_nan_propagation():
    """Colour is NaN when either input magnitude is NaN."""
    from xclass.features import compute_hst_colors

    df = pd.DataFrame(
        {
            "UVIS_F275W_pred": [float("nan")],
            "UVIS_F336W_pred": [22.5],
            "ACS_F475W_pred": [22.0],
            "ACS_F814W_pred": [21.0],
            "IR_F110W_pred": [20.5],
            "IR_F160W_pred": [20.0],
        }
    )
    filter_list = [
        "UVIS_F275W", "UVIS_F336W", "ACS_F475W", "ACS_F814W", "IR_F110W", "IR_F160W"
    ]
    result = compute_hst_colors(df, filter_list)
    assert np.isnan(result["color_F275W_F336W"].iloc[0])


# ---------------------------------------------------------------------------
# compute_xray_optical_ratios tests
# ---------------------------------------------------------------------------


def test_xray_optical_ratio_formula():
    """logFx_B = log10(Fx_B) for a known positive value."""
    from xclass.features import compute_xray_optical_ratios

    Fx_B = 1e-13
    df = _make_xray_df(Fx_B=[Fx_B])
    # Also add a dummy optical magnitude column
    df["ACS_F475W_pred"] = [22.0]
    df["ACS_F814W_pred"] = [21.0]

    result = compute_xray_optical_ratios(df)
    expected_logFx_B = np.log10(Fx_B)
    assert abs(result["logFx_B"].iloc[0] - expected_logFx_B) < 1e-10


def test_xray_optical_ratio_nan_for_nonpositive():
    """logFx_B is NaN for zero or negative flux."""
    from xclass.features import compute_xray_optical_ratios

    df = _make_xray_df(Fx_B=[0.0])
    df["ACS_F475W_pred"] = [22.0]
    df["ACS_F814W_pred"] = [21.0]
    result = compute_xray_optical_ratios(df)
    assert np.isnan(result["logFx_B"].iloc[0])

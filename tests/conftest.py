"""Shared pytest fixtures for xclass tests.

All tests must pass without network access; this module provides
synthetic data and HTTP mocks to ensure isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic training dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_td_df() -> pd.DataFrame:
    """5-row training dataset with one source per major class."""
    return pd.DataFrame(
        {
            "name": ["Sco X-1", "Cyg X-1", "SS Cyg", "3C 273", "HIP 12345"],
            "Class": ["LMXB", "HMXB", "CV", "AGN", "LM-STAR"],
            "ra": [244.980, 299.590, 325.678, 187.278, 40.123],
            "dec": [-15.640, 35.201, 43.586, 2.052, 20.456],
            "source_catalog": ["Liu2007", "Liu2006", "Downes", "Veron", "Skiff"],
            "source_ref": ["Liu+2007", "Liu+2006", "Downes+2001", "Veron+2010", "Skiff2014"],
            "label_confidence": [1.0, 1.0, 1.0, 1.0, 1.0],
            "SpType": [None, "B0Ib", None, None, "K2V"],
        }
    )


@pytest.fixture
def synthetic_td_df_stars() -> pd.DataFrame:
    """DataFrame with a mix of spectral types for SpType classification tests."""
    return pd.DataFrame(
        {
            "name": [f"Star_{i}" for i in range(10)],
            "SpType": ["O5V", "B2Ib", "A0V", "F5V", "G2V", "K5V", "M0V", "B0", "O9III", "K7"],
            "expected_class": [
                "HM-STAR", "HM-STAR", "LM-STAR", "LM-STAR", "LM-STAR",
                "LM-STAR", "LM-STAR", "HM-STAR", "HM-STAR", "LM-STAR",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Synthetic Chandra catalog
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chandra_df() -> pd.DataFrame:
    """3-row Chandra Source Catalog-like DataFrame."""
    return pd.DataFrame(
        {
            "name": ["2CXO J161546.5-232327", "2CXO J194430.8+354757", "2CXO J215252.7+433549"],
            "ra": [243.944, 296.128, 328.219],
            "dec": [-23.391, 35.799, 43.597],
            "err_ellipse_r0": [0.5, 1.2, 0.8],
            "err_ellipse_r1": [0.4, 0.9, 0.7],
            "significance": [10.0, 5.5, 3.5],
            "flux_aper90_avg_s": [1e-14, 5e-15, 2e-15],
            "flux_aper90_avg_lolim_s": [8e-15, 3e-15, 1e-15],
            "flux_aper90_avg_hilim_s": [1.2e-14, 7e-15, 3e-15],
            "flux_aper90_avg_m": [2e-14, 1e-14, 4e-15],
            "flux_aper90_avg_lolim_m": [1.5e-14, 8e-15, 3e-15],
            "flux_aper90_avg_hilim_m": [2.5e-14, 1.2e-14, 5e-15],
            "flux_aper90_avg_h": [3e-14, 1.5e-14, 6e-15],
            "flux_aper90_avg_lolim_h": [2e-14, 1e-14, 4e-15],
            "flux_aper90_avg_hilim_h": [4e-14, 2e-14, 8e-15],
            "flux_aper90_avg_b": [6e-14, 3e-14, 1.2e-14],
            "flux_aper90_avg_lolim_b": [5e-14, 2.5e-14, 1e-14],
            "flux_aper90_avg_hilim_b": [7e-14, 3.5e-14, 1.4e-14],
        }
    )


# ---------------------------------------------------------------------------
# Filter curve fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_filter_curve() -> tuple[np.ndarray, np.ndarray]:
    """Flat (top-hat) bandpass from 4000–5000 Å for unit testing."""
    wave = np.linspace(4000.0, 5000.0, 100)
    thru = np.ones(100)
    return wave, thru


@pytest.fixture
def mock_filter_curves(flat_filter_curve) -> dict:
    """Dict of mock filter curves for the PHAT filter set."""
    wave, thru = flat_filter_curve
    labels = [
        "UVIS_F275W", "UVIS_F336W", "ACS_F475W",
        "ACS_F814W", "IR_F110W", "IR_F160W",
    ]
    return {label: (wave.copy(), thru.copy()) for label in labels}


# ---------------------------------------------------------------------------
# HTTP / network mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_requests_get():
    """Patch requests.get to prevent real HTTP calls in tests."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {}
    mock_resp.content = b""
    with patch("requests.get", return_value=mock_resp) as mock_get:
        yield mock_get


@pytest.fixture
def mock_vizier():
    """Patch astroquery.vizier.Vizier to prevent network calls."""
    with patch("astroquery.vizier.Vizier") as mock_viz:
        yield mock_viz


@pytest.fixture
def mock_gaia_tap():
    """Patch astroquery.gaia.Gaia to prevent TAP queries."""
    with patch("astroquery.gaia.Gaia") as mock_gaia:
        yield mock_gaia

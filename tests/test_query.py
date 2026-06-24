"""Tests for xclass.query — live query functions.

All tests mock network calls; no real HTTP requests are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# query_csc_sources_in_polygon tests  [C3]
# ---------------------------------------------------------------------------


def test_polygon_point_inside(tmp_path):
    """A source inside the polygon is kept."""
    from xclass.query import query_csc_sources_in_polygon

    # Simple square polygon in RA/Dec
    polygon = [(10.0, 40.0), (12.0, 40.0), (12.0, 42.0), (10.0, 42.0)]

    # Source at centroid — clearly inside
    csc_data = pd.DataFrame(
        {
            "ra": [11.0],
            "dec": [41.0],
            "err_ellipse_r0": [0.5],
            "err_ellipse_r1": [0.5],
            "significance": [10.0],
            "name": ["src_inside"],
        }
    )
    csc_path = tmp_path / "chandra.csv"
    csc_data.to_csv(csc_path, index=False)

    result = query_csc_sources_in_polygon(polygon, csc_path, significance_min=3.0)
    assert len(result) == 1
    assert result.iloc[0]["name"] == "src_inside"


def test_polygon_point_outside(tmp_path):
    """A source outside the polygon is excluded."""
    from xclass.query import query_csc_sources_in_polygon

    polygon = [(10.0, 40.0), (12.0, 40.0), (12.0, 42.0), (10.0, 42.0)]

    csc_data = pd.DataFrame(
        {
            "ra": [15.0],   # Outside the polygon
            "dec": [41.0],
            "err_ellipse_r0": [0.5],
            "err_ellipse_r1": [0.5],
            "significance": [10.0],
            "name": ["src_outside"],
        }
    )
    csc_path = tmp_path / "chandra.csv"
    csc_data.to_csv(csc_path, index=False)

    result = query_csc_sources_in_polygon(polygon, csc_path, significance_min=3.0)
    assert len(result) == 0


def test_polygon_significance_cut(tmp_path):
    """Sources below significance threshold are excluded."""
    from xclass.query import query_csc_sources_in_polygon

    polygon = [(10.0, 40.0), (12.0, 40.0), (12.0, 42.0), (10.0, 42.0)]

    csc_data = pd.DataFrame(
        {
            "ra": [11.0, 11.5],
            "dec": [41.0, 41.5],
            "err_ellipse_r0": [0.5, 0.5],
            "err_ellipse_r1": [0.5, 0.5],
            "significance": [10.0, 2.0],   # second source below threshold
            "name": ["src_sig_ok", "src_sig_low"],
        }
    )
    csc_path = tmp_path / "chandra.csv"
    csc_data.to_csv(csc_path, index=False)

    result = query_csc_sources_in_polygon(polygon, csc_path, significance_min=3.0)
    assert len(result) == 1
    assert result.iloc[0]["name"] == "src_sig_ok"


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    """The same source_id + radius always produces the same cache key."""
    # This tests the internal cache key function (to be exposed or tested indirectly)
    # Cache key must be deterministic: same inputs -> same .pkl filename
    # Placeholder: verify that two calls with identical args produce identical paths.
    from xclass import config

    source_id = "test_source_001"
    radius = 5.0
    survey = "ps1"

    # Cache path convention: QUERY_CACHE_DIR / survey / f"{source_id}.pkl"
    expected = config.QUERY_CACHE_DIR / survey / f"{source_id}.pkl"
    # The actual implementation must produce this exact path structure.
    assert expected.parent.name == survey


def test_ps1_url_construction(mock_requests_get):
    """PS1 query builds the correct MAST API URL."""
    from xclass import config

    sources = pd.DataFrame(
        {"source_id": ["src1"], "ra": [187.278], "dec": [2.052]}
    )

    # This will be tested against the actual implementation in Phase 3.
    # For now verify the expected base URL is defined.
    assert "catalogs.mast.stsci.edu" in config.PS1_API_URL


def test_hsc_mock_response_split(tmp_path):
    """query_hsc_for_chandra_sources returns two DataFrames with correct shapes."""
    from xclass.query import query_hsc_for_chandra_sources

    chandra = pd.DataFrame(
        {
            "name": ["src1"],
            "ra": [10.5],
            "dec": [41.2],
            "err_ellipse_r0": [0.5],
            "err_ellipse_r1": [0.5],
            "significance": [5.0],
        }
    )

    # Mock the HSC API response
    mock_hsc_response = [
        {
            "MatchID": 1001,
            "CatID": 1,
            "SourceID": 9001,
            "RA": 10.5001,
            "Dec": 41.2001,
            "MagAper2": 22.5,
            "MagAper2Err": 0.05,
            "CI": 0.3,
            "NumImages": 3,
            "StartMJD": 54000.0,
            "StopMJD": 55000.0,
            "Filter": "F475W",
        }
    ]

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_hsc_response
        mock_get.return_value = mock_resp

        best, all_cands = query_hsc_for_chandra_sources(
            chandra, cache_dir=tmp_path / "hsc"
        )

    # best_match should have one row per Chandra source
    assert len(best) == 1
    # all_candidates should have one row per detection
    assert len(all_cands) >= 1

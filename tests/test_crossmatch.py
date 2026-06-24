"""Tests for xclass.crossmatch — positional matching utilities.

All tests use synthetic data and do not require network access.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# effective_pos_err_arcsec tests
# ---------------------------------------------------------------------------


def test_effective_pos_err_both_valid():
    """sqrt((r0^2 + r1^2) / 2) for two valid inputs."""
    from xclass.crossmatch import effective_pos_err_arcsec

    r0, r1 = 3.0, 4.0
    expected = math.sqrt((9.0 + 16.0) / 2.0)  # sqrt(12.5) = 3.536...
    result = effective_pos_err_arcsec(r0, r1)
    assert abs(result - expected) < 1e-10


def test_effective_pos_err_one_nan():
    """When one input is NaN, use the non-NaN value alone."""
    from xclass.crossmatch import effective_pos_err_arcsec

    # If r1 is NaN, result should be based on r0 only
    result = effective_pos_err_arcsec(3.0, float("nan"))
    # Expected: sqrt((3^2 + 3^2) / 2) = 3 (both set to r0), or just r0 — implementation-defined
    assert np.isfinite(result)
    assert result > 0


def test_effective_pos_err_both_nan():
    """Both NaN inputs -> NaN output."""
    from xclass.crossmatch import effective_pos_err_arcsec

    result = effective_pos_err_arcsec(float("nan"), float("nan"))
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# search_radius_arcsec tests
# ---------------------------------------------------------------------------


def test_search_radius_nsigma_scaling():
    """Search radius = nsigma * pos_err when above floor."""
    from xclass.crossmatch import search_radius_arcsec

    pos_err = 2.0
    nsigma = 3.0
    floor = 0.5
    expected = nsigma * pos_err  # 6.0, well above floor
    assert abs(search_radius_arcsec(pos_err, nsigma, floor) - expected) < 1e-10


def test_search_radius_floor_applied():
    """Search radius is clipped to floor when nsigma * pos_err < floor."""
    from xclass.crossmatch import search_radius_arcsec

    pos_err = 0.1
    nsigma = 3.0
    floor = 0.5
    # nsigma * pos_err = 0.3 < floor=0.5 -> should return 0.5
    assert abs(search_radius_arcsec(pos_err, nsigma, floor) - floor) < 1e-10


def test_search_radius_exact_floor():
    """When nsigma * pos_err exactly equals the floor, return floor."""
    from xclass.crossmatch import search_radius_arcsec

    pos_err = 0.5 / 3.0  # nsigma * pos_err = 0.5 = floor exactly
    assert abs(search_radius_arcsec(pos_err, nsigma=3.0, floor=0.5) - 0.5) < 1e-10

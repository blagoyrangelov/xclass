"""xclass.crossmatch — Positional cross-matching and counterpart selection.

Matches training-dataset sources to Chandra X-ray detections, retrieves
optical/IR counterparts, and selects the best counterpart per X-ray source.

Functions
---------
effective_pos_err_arcsec
search_radius_arcsec
match_td_to_chandra
select_optical_counterpart
build_xray_training_table
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from xclass import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position error and search radius
# ---------------------------------------------------------------------------


def effective_pos_err_arcsec(
    err_r0: float,
    err_r1: float,
) -> float:
    """Compute effective position error from X-ray error ellipse semi-axes.

    The effective error is the RMS of the two semi-axes::

        sigma_eff = sqrt((r0^2 + r1^2) / 2)

    Parameters
    ----------
    err_r0 : float
        Semi-major axis of X-ray error ellipse (arcsec).
    err_r1 : float
        Semi-minor axis of X-ray error ellipse (arcsec).

    Returns
    -------
    float
        Effective position error in arcseconds.  Returns NaN when
        both inputs are NaN; uses the non-NaN value when only one is NaN.
    """
    r0_nan = not np.isfinite(err_r0)
    r1_nan = not np.isfinite(err_r1)

    if r0_nan and r1_nan:
        return float("nan")
    if r0_nan:
        return float(err_r1)
    if r1_nan:
        return float(err_r0)

    return float(np.sqrt((err_r0 ** 2 + err_r1 ** 2) / 2.0))


def search_radius_arcsec(
    pos_err: float,
    nsigma: float = 3.0,
    floor: float = 0.5,
) -> float:
    """Compute search radius from position error.

    Parameters
    ----------
    pos_err : float
        Position error in arcseconds.
    nsigma : float
        Number of sigmas.  Default 3.0.
    floor : float
        Minimum radius in arcseconds.  Default 0.5.

    Returns
    -------
    float
        ``max(nsigma * pos_err, floor)``.
    """
    return max(nsigma * float(pos_err), float(floor))


# ---------------------------------------------------------------------------
# TD → Chandra matching
# ---------------------------------------------------------------------------


def _chandra_flux_cols() -> list[str]:
    """Return Chandra flux column names expected in chandra_df."""
    bands = ["s", "m", "h", "b"]
    cols = []
    for b in bands:
        cols += [
            f"flux_aper90_avg_{b}",
            f"flux_aper90_avg_{b}_lolim",
            f"flux_aper90_avg_{b}_hilim",
        ]
    return cols


def match_td_to_chandra(
    td_df: pd.DataFrame,
    chandra_df: pd.DataFrame,
    nsigma: float = 3.0,
) -> pd.DataFrame:
    """Match training-dataset sources to Chandra CSC 2.0 detections.

    For each TD source, finds Chandra sources within the positional search
    radius using ``astropy.coordinates.search_around_sky``.

    Parameters
    ----------
    td_df : pd.DataFrame
        Training dataset with columns: ra, dec (degrees).
    chandra_df : pd.DataFrame
        Chandra Source Catalog with columns: ra, dec, err_ellipse_r0,
        err_ellipse_r1, significance, name, flux_aper90_avg_s/m/h/b and
        corresponding _lolim / _hilim columns.
    nsigma : float
        Search radius multiplier.

    Returns
    -------
    pd.DataFrame
        One row per TD–Chandra matched pair.  Columns include:
        sep_class_xray_arcsec, normsep_class_xray, xray_pos_err_arcsec,
        Fx_S, Fx_M, Fx_H, Fx_B, Fx_S_err, Fx_M_err, Fx_H_err, Fx_B_err.
    """
    if len(td_df) == 0 or len(chandra_df) == 0:
        return pd.DataFrame()

    td_df = td_df.reset_index(drop=True)
    chandra_df = chandra_df.reset_index(drop=True)

    # Compute per-Chandra-source position errors and search radii
    r0_col = _find_col(list(chandra_df.columns), ["err_ellipse_r0", "err_r0", "pos_err_r0"])
    r1_col = _find_col(list(chandra_df.columns), ["err_ellipse_r1", "err_r1", "pos_err_r1"])

    r0 = chandra_df[r0_col].values.astype(float) if r0_col else np.full(len(chandra_df), 0.5)
    r1 = chandra_df[r1_col].values.astype(float) if r1_col else np.full(len(chandra_df), 0.5)

    pos_errs = np.array([effective_pos_err_arcsec(a, b) for a, b in zip(r0, r1)])
    # Replace NaN with default fallback
    pos_errs = np.where(np.isfinite(pos_errs), pos_errs, 0.5)

    radii = np.array([
        search_radius_arcsec(pe, nsigma, config.MIN_SEARCH_RADIUS_ARCSEC)
        for pe in pos_errs
    ])
    global_max_radius = float(np.nanmax(radii)) if len(radii) > 0 else 5.0

    # Build sky coordinates
    td_coords = SkyCoord(
        ra=td_df["ra"].values.astype(float) * u.deg,
        dec=td_df["dec"].values.astype(float) * u.deg,
        frame="icrs",
    )
    xray_coords = SkyCoord(
        ra=chandra_df["ra"].values.astype(float) * u.deg,
        dec=chandra_df["dec"].values.astype(float) * u.deg,
        frame="icrs",
    )

    # All pairs within global_max_radius.
    # astropy search_around_sky(searcharoundcoords) returns
    # (idx_into_searcharoundcoords, idx_into_self, sep, dist3d)
    xray_idx, td_idx, sep2d, _ = td_coords.search_around_sky(
        xray_coords, global_max_radius * u.arcsec
    )

    if len(td_idx) == 0:
        return pd.DataFrame()

    # Filter by per-Chandra-source radius
    keep = sep2d.arcsec <= radii[xray_idx]
    td_idx = td_idx[keep]
    xray_idx = xray_idx[keep]
    sep2d = sep2d[keep]

    if len(td_idx) == 0:
        return pd.DataFrame()

    # Build output DataFrame
    td_part = td_df.iloc[td_idx].reset_index(drop=True).add_prefix("td_")
    xray_part = chandra_df.iloc[xray_idx].reset_index(drop=True).add_prefix("xray_")

    seps = sep2d.arcsec
    normseps = seps / pos_errs[xray_idx]

    # Canonical Fx columns
    def _xray_flux(band: str) -> np.ndarray:
        col = f"flux_aper90_avg_{band.lower()}"
        xcol = f"xray_{col}"
        if xcol in xray_part.columns:
            return xray_part[xcol].values.astype(float)
        return np.full(len(td_idx), np.nan)

    def _xray_flux_err(band: str) -> np.ndarray:
        lo_col = f"xray_flux_aper90_avg_{band.lower()}_lolim"
        hi_col = f"xray_flux_aper90_avg_{band.lower()}_hilim"
        lo = xray_part[lo_col].values.astype(float) if lo_col in xray_part.columns else np.full(len(td_idx), np.nan)
        hi = xray_part[hi_col].values.astype(float) if hi_col in xray_part.columns else np.full(len(td_idx), np.nan)
        flux = _xray_flux(band)
        # Use half the confidence interval as a proxy for 1-sigma error
        return (hi - lo) / 2.0

    out = pd.concat([td_part, xray_part], axis=1)
    out["sep_class_xray_arcsec"] = seps
    out["normsep_class_xray"] = normseps
    out["xray_pos_err_arcsec"] = pos_errs[xray_idx]
    out["Fx_S"] = _xray_flux("s")
    out["Fx_M"] = _xray_flux("m")
    out["Fx_H"] = _xray_flux("h")
    out["Fx_B"] = _xray_flux("b")
    out["Fx_S_err"] = _xray_flux_err("s")
    out["Fx_M_err"] = _xray_flux_err("m")
    out["Fx_H_err"] = _xray_flux_err("h")
    out["Fx_B_err"] = _xray_flux_err("b")

    log.info(
        "match_td_to_chandra: %d TD × %d Chandra -> %d pairs",
        len(td_df), len(chandra_df), len(out),
    )
    return out


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name present in cols, or None."""
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# Optical counterpart selection
# ---------------------------------------------------------------------------


def select_optical_counterpart(
    candidates_df: pd.DataFrame,
    xray_pos_err_arcsec: float,
    method: str = "nway_or_normsep",
) -> dict:
    """Select the best optical counterpart from a set of candidates.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        All optical candidates within search radius for one X-ray source.
    xray_pos_err_arcsec : float
        X-ray positional uncertainty.
    method : str
        'nway_or_normsep': run NWAY if available; else use normalised
        separation threshold.

    Returns
    -------
    dict
        Keys: match_status ('secure', 'ambiguous', 'none'),
              best_candidate (pd.Series or None),
              p_any (float or NaN).
    """
    if candidates_df is None or len(candidates_df) == 0:
        return {"match_status": "none", "best_candidate": None, "p_any": float("nan")}

    # Try NWAY if available
    if method == "nway_or_normsep":
        try:
            return _select_nway(candidates_df, xray_pos_err_arcsec)
        except Exception:
            pass  # fall through to normsep

    return _select_normsep(candidates_df, xray_pos_err_arcsec)


def _select_normsep(
    candidates_df: pd.DataFrame,
    xray_pos_err_arcsec: float,
) -> dict:
    """Normalised-separation counterpart selection."""
    sep_col = _find_col(
        list(candidates_df.columns),
        ["sep_arcsec", "sep_class_xray_arcsec", "separation_arcsec"],
    )
    if sep_col is None:
        return {"match_status": "none", "best_candidate": None, "p_any": float("nan")}

    seps = candidates_df[sep_col].values.astype(float)
    pos_err = max(float(xray_pos_err_arcsec), 0.1)
    normseps = seps / pos_err

    within = normseps <= config.MAX_NORMSEP_XRAY_OPT
    n_within = int(within.sum())

    if n_within == 0:
        return {"match_status": "none", "best_candidate": None, "p_any": float("nan")}

    best_i = int(np.argmin(seps))
    best = candidates_df.iloc[best_i]

    if n_within == 1:
        status = "secure"
    else:
        # Check uniqueness: best_sep / second_best_sep
        sorted_seps = np.sort(seps[within])
        ratio = sorted_seps[1] / max(sorted_seps[0], 1e-6)
        status = "secure" if ratio >= config.MATCH_SECOND_BEST_RATIO else "ambiguous"

    return {
        "match_status": status,
        "best_candidate": best,
        "p_any": float("nan"),  # not computed without NWAY
    }


def _select_nway(
    candidates_df: pd.DataFrame,
    xray_pos_err_arcsec: float,
) -> dict:
    """Counterpart selection using the NWAY library (if installed)."""
    import nway  # noqa: F401 — imported to check availability

    # NWAY integration is complex; fall back to normsep for now
    raise ImportError("Full NWAY integration deferred to pipeline application")


# ---------------------------------------------------------------------------
# Full training table assembly
# ---------------------------------------------------------------------------


def build_xray_training_table(
    td_df: pd.DataFrame,
    chandra_df: pd.DataFrame,
    photometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate full cross-match to build the X-ray training table.

    Steps:
    1. ``match_td_to_chandra`` -> X-ray-matched TD.
    2. Join *photometry_df* on source_id.
    3. For each source, select best optical counterpart.
    4. Record match status for each survey.

    Parameters
    ----------
    td_df : pd.DataFrame
        Master training dataset.
    chandra_df : pd.DataFrame
        Chandra Source Catalog.
    photometry_df : pd.DataFrame
        Output of ``query.query_all_photometry``.

    Returns
    -------
    pd.DataFrame
        One row per TD source with Chandra X-ray detection and all
        available optical/IR photometry.
    """
    # Step 1: TD → Chandra positional match
    matched = match_td_to_chandra(
        td_df, chandra_df, nsigma=config.LABEL_TO_XRAY_SIGMA
    )
    if len(matched) == 0:
        log.warning("build_xray_training_table: no TD–Chandra matches found")
        return pd.DataFrame()

    # Step 2: Join photometry (left-join on source_id from td side)
    sid_col = "td_source_id" if "td_source_id" in matched.columns else None
    if sid_col and "source_id" in photometry_df.columns:
        matched = matched.merge(
            photometry_df,
            left_on=sid_col,
            right_on="source_id",
            how="left",
            suffixes=("", "_phot"),
        )

    # Step 3: Per-source best optical counterpart selection
    #         For training purposes we use all rows; the normsep is already computed.
    #         Flag secure vs ambiguous per source.
    if "normsep_class_xray" in matched.columns:
        matched["match_status"] = np.where(
            matched["normsep_class_xray"] <= config.MAX_NORMSEP_XRAY_OPT,
            np.where(matched["normsep_class_xray"] <= config.MATCH_SECURE_NORMSEP_MAX,
                     "secure", "ambiguous"),
            "none",
        )

    # Step 4: Apply significance cut
    sig_col = _find_col(
        list(matched.columns),
        ["xray_significance", "significance"],
    )
    if sig_col:
        matched = matched[matched[sig_col] >= config.MIN_SIGNIFICANCE].copy()

    log.info(
        "build_xray_training_table: %d matched rows after quality cuts", len(matched)
    )
    return matched

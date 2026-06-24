"""xclass.query — Live photometric queries with caching.  [C1, C3]

All external survey queries live here.  Every function checks a local disk
cache before making a network request.  Network calls use tenacity retry
with exponential back-off.

Functions
---------
query_panstarrs
query_2mass
query_all_photometry
query_csc_sources_in_polygon      [C3]
query_hsc_for_chandra_sources     [C3]
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from xclass import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared caching helpers
# ---------------------------------------------------------------------------

def _safe_id(source_id: str) -> str:
    """Sanitize source_id for use as a filename."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(source_id))


def _cache_file(source_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_safe_id(source_id)}.pkl"


def _load_cache(source_id: str, cache_dir: Path):
    """Return cached payload or None on cache miss / corrupt file."""
    p = _cache_file(source_id, cache_dir)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_cache(source_id: str, cache_dir: Path, data) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(_cache_file(source_id, cache_dir), "wb") as f:
        pickle.dump(data, f)


def _find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name present in *columns*, else None."""
    for c in candidates:
        if c in columns:
            return c
    return None


# ---------------------------------------------------------------------------
# PanSTARRS DR2 (MAST API)
# ---------------------------------------------------------------------------

def _ps1_fetch_one(
    source_id: str,
    ra: float,
    dec: float,
    radius_arcsec: float,
    cache_dir: Path,
) -> dict:
    """Single-source PS1 query with per-source disk caching."""
    cached = _load_cache(source_id, cache_dir)
    if cached is not None:
        return cached

    if dec < config.PS1_DEC_LIMIT_DEG:
        result = {"source_id": source_id, "ps1_available": False, "n_ps1_candidates": 0}
        _save_cache(source_id, cache_dir, result)
        return result

    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(
        stop=stop_after_attempt(config.QUERY_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.QUERY_RETRY_WAIT_SEC,
            min=config.QUERY_RETRY_WAIT_SEC,
            max=60,
        ),
        reraise=True,
    )
    def _do_get():
        # URL already ends in .json (format in path, per MAST example notebook).
        # Columns must be wrapped in [...] brackets per the MAST API spec.
        resp = requests.get(
            config.PS1_API_URL,
            params={
                "ra": ra,
                "dec": dec,
                "radius": radius_arcsec / 3600.0,  # MAST expects degrees
                "columns": "[{}]".format(",".join(config.PS1_COLUMNS)),
                "nDetections.gt": 1,
            },
            timeout=config.QUERY_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        data = _do_get()
    except Exception as exc:
        log.warning("PS1 query failed for source %s: %s", source_id, exc)
        result = {"source_id": source_id, "ps1_available": False, "n_ps1_candidates": 0}
        _save_cache(source_id, cache_dir, result)
        return result

    # MAST JSON response: {"info": [{"name": col, ...}, ...], "data": [[v1, v2, ...], ...]}
    # Convert list-of-lists to list-of-dicts using column names from "info".
    raw_rows = data.get("data", []) if isinstance(data, dict) else (data or [])
    col_names = [c["name"] for c in data.get("info", [])] if isinstance(data, dict) else []
    if col_names and raw_rows and isinstance(raw_rows[0], (list, tuple)):
        rows = [dict(zip(col_names, r)) for r in raw_rows]
    else:
        rows = raw_rows  # already dicts (older API format)
    n_cands = len(rows)

    if not rows:
        result = {"source_id": source_id, "ps1_available": False, "n_ps1_candidates": 0}
        _save_cache(source_id, cache_dir, result)
        return result

    # Select nearest candidate
    src_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    best_sep = np.inf
    best = None
    for row in rows:
        row_ra = _to_float(row.get("raMean"))
        row_dec = _to_float(row.get("decMean"))
        if not (np.isfinite(row_ra) and np.isfinite(row_dec)):
            continue
        sep = src_coord.separation(
            SkyCoord(ra=row_ra * u.deg, dec=row_dec * u.deg)
        ).arcsec
        if sep < best_sep:
            best_sep, best = sep, row

    if best is None:
        result = {"source_id": source_id, "ps1_available": False, "n_ps1_candidates": n_cands}
    else:
        result = {
            "source_id": source_id,
            "ps1_obj_id": best.get("objID"),
            "ps1_ra": _to_float(best.get("raMean")),
            "ps1_dec": _to_float(best.get("decMean")),
            "ps1_g": _to_float(best.get("gMeanPSFMag")),
            "ps1_g_err": _to_float(best.get("gMeanPSFMagErr")),
            "ps1_r": _to_float(best.get("rMeanPSFMag")),
            "ps1_r_err": _to_float(best.get("rMeanPSFMagErr")),
            "ps1_i": _to_float(best.get("iMeanPSFMag")),
            "ps1_i_err": _to_float(best.get("iMeanPSFMagErr")),
            "ps1_z": _to_float(best.get("zMeanPSFMag")),
            "ps1_z_err": _to_float(best.get("zMeanPSFMagErr")),
            "ps1_y": _to_float(best.get("yMeanPSFMag")),
            "ps1_y_err": _to_float(best.get("yMeanPSFMagErr")),
            "sep_source_ps1_arcsec": best_sep,
            "n_ps1_candidates": n_cands,
            "ps1_available": True,
        }

    _save_cache(source_id, cache_dir, result)
    return result


def _to_float(val) -> float:
    """Safely convert a value to float, returning NaN on failure."""
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def query_panstarrs(
    sources_df: pd.DataFrame,
    radius_arcsec: float = 5.0,
    cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Query PanSTARRS DR2 (MAST API) for each source.

    Parameters
    ----------
    sources_df : pd.DataFrame
        Input with columns: source_id, ra, dec.
    radius_arcsec : float
        Search radius in arcseconds.
    cache_dir : str or Path, optional
        Cache directory.  Defaults to ``config.QUERY_CACHE_DIR / 'ps1'``.

    Returns
    -------
    pd.DataFrame
        Columns: source_id, ps1_obj_id, ps1_ra, ps1_dec, ps1_g, ps1_g_err,
        ps1_r, ps1_r_err, ps1_i, ps1_i_err, ps1_z, ps1_z_err, ps1_y,
        ps1_y_err, sep_source_ps1_arcsec, n_ps1_candidates, ps1_available.

    Notes
    -----
    Sources with Dec < ``config.PS1_DEC_LIMIT_DEG`` are flagged
    ``ps1_available=False`` and not queried.
    """
    cdir = Path(cache_dir) if cache_dir else config.QUERY_CACHE_DIR / "ps1"
    cdir.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=4, prefer="threads")(
        delayed(_ps1_fetch_one)(
            str(row["source_id"]),
            float(row["ra"]),
            float(row["dec"]),
            radius_arcsec,
            cdir,
        )
        for _, row in sources_df.iterrows()
    )

    out = pd.DataFrame(results)
    n_with = int(out["ps1_available"].sum()) if "ps1_available" in out.columns else 0
    log.info("query_panstarrs: %d sources, %d with PS1 match", len(sources_df), n_with)
    return out


# ---------------------------------------------------------------------------
# 2MASS All-Sky PSC (VizieR)
# ---------------------------------------------------------------------------

def _tmass_fetch_one(
    source_id: str,
    ra: float,
    dec: float,
    radius_arcsec: float,
    cache_dir: Path,
) -> list[dict]:
    """Single-source 2MASS VizieR query with caching."""
    cached = _load_cache(source_id, cache_dir)
    if cached is not None:
        return cached

    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    from tenacity import retry, stop_after_attempt, wait_exponential

    v = Vizier(columns=config.TMASS_COLUMNS, row_limit=50)

    @retry(
        stop=stop_after_attempt(config.QUERY_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.QUERY_RETRY_WAIT_SEC,
            min=config.QUERY_RETRY_WAIT_SEC,
            max=60,
        ),
        reraise=True,
    )
    def _query():
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        return v.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=config.TMASS_VIZIER_ID,
        )

    try:
        result_list = _query()
    except Exception as exc:
        log.warning("2MASS query failed for source %s: %s", source_id, exc)
        _save_cache(source_id, cache_dir, [])
        return []

    if not result_list or len(result_list) == 0:
        _save_cache(source_id, cache_dir, [])
        return []

    tbl = result_list[0]
    if len(tbl) == 0:
        _save_cache(source_id, cache_dir, [])
        return []

    src_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    rows_out = []
    import numpy.ma as ma

    for row in tbl:
        def _get(col):
            v = row[col]
            if isinstance(v, ma.core.MaskedConstant):
                return np.nan
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        row_ra = _get("RAJ2000")
        row_dec = _get("DEJ2000")
        sep = (
            src_coord.separation(
                SkyCoord(ra=row_ra * u.deg, dec=row_dec * u.deg)
            ).arcsec
            if np.isfinite(row_ra) and np.isfinite(row_dec)
            else np.nan
        )

        tmass_id = str(row["_2MASS"]) if "_2MASS" in tbl.colnames else ""
        qflg = str(row["Qflg"]) if "Qflg" in tbl.colnames else ""

        rows_out.append({
            "source_id": source_id,
            "tmass_id": tmass_id,
            "tmass_ra": row_ra,
            "tmass_dec": row_dec,
            "tmass_j": _get("Jmag"),
            "tmass_j_err": _get("e_Jmag"),
            "tmass_h": _get("Hmag"),
            "tmass_h_err": _get("e_Hmag"),
            "tmass_k": _get("Kmag"),
            "tmass_k_err": _get("e_Kmag"),
            "tmass_qflg": qflg,
            "sep_source_tmass_arcsec": sep,
            "tmass_available": True,
        })

    _save_cache(source_id, cache_dir, rows_out)
    return rows_out


def query_2mass(
    sources_df: pd.DataFrame,
    radius_arcsec: float = 5.0,
    cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Query 2MASS All-Sky PSC via astroquery VizieR for each source.

    Parameters
    ----------
    sources_df : pd.DataFrame
        Input with columns: source_id, ra, dec.
    radius_arcsec : float
        Search radius in arcseconds.
    cache_dir : str or Path, optional
        Cache directory.  Defaults to ``config.QUERY_CACHE_DIR / '2mass'``.

    Returns
    -------
    pd.DataFrame
        Columns: source_id, tmass_id, tmass_ra, tmass_dec, tmass_j,
        tmass_j_err, tmass_h, tmass_h_err, tmass_k, tmass_k_err,
        tmass_qflg, sep_source_tmass_arcsec, n_tmass_candidates,
        tmass_available.
        One row per candidate.
    """
    cdir = Path(cache_dir) if cache_dir else config.QUERY_CACHE_DIR / "2mass"
    cdir.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed

    results_nested = Parallel(n_jobs=4, prefer="threads")(
        delayed(_tmass_fetch_one)(
            str(row["source_id"]),
            float(row["ra"]),
            float(row["dec"]),
            radius_arcsec,
            cdir,
        )
        for _, row in sources_df.iterrows()
    )

    all_rows: list[dict] = []
    for i, candidates in enumerate(results_nested):
        sid = str(sources_df.iloc[i]["source_id"])
        if candidates:
            n = len(candidates)
            for c in candidates:
                c["n_tmass_candidates"] = n
            all_rows.extend(candidates)
        else:
            all_rows.append({
                "source_id": sid,
                "tmass_available": False,
                "n_tmass_candidates": 0,
            })

    out = pd.DataFrame(all_rows)
    n_with = int(out["tmass_available"].sum()) if "tmass_available" in out.columns else 0
    log.info("query_2mass: %d sources, %d with 2MASS match", len(sources_df), n_with)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def query_all_photometry(
    sources_df: pd.DataFrame,
    radius_arcsec: float = 5.0,
    surveys: list[str] | None = None,
    cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Orchestrate queries to PS1 and 2MASS for all sources.

    PS1 and 2MASS queries run in parallel via joblib. These are the two
    ground-based surveys used as SED-translation inputs in production.

    Parameters
    ----------
    sources_df : pd.DataFrame
        Input with columns: source_id, ra, dec.
    radius_arcsec : float
        Search radius passed to each survey query.
    surveys : list of str, optional
        Which surveys to query.  Defaults to ``['ps1', '2mass']``.
    cache_dir : str or Path, optional
        Root cache directory.  Survey-specific subdirs created automatically.

    Returns
    -------
    pd.DataFrame
        *sources_df* with all photometric columns merged in.
        Prints summary: n sources, n with PS1, n with 2MASS,
        n with both, n with none.
    """
    active_surveys = surveys or ["ps1", "2mass"]
    root_cache = Path(cache_dir) if cache_dir else config.QUERY_CACHE_DIR
    root_cache.mkdir(parents=True, exist_ok=True)

    # Run PS1 and 2MASS in parallel threads
    from joblib import Parallel, delayed

    def _run_ps1():
        if "ps1" not in active_surveys:
            return None
        return query_panstarrs(sources_df, radius_arcsec, root_cache / "ps1")

    def _run_2mass():
        if "2mass" not in active_surveys:
            return None
        return query_2mass(sources_df, radius_arcsec, root_cache / "2mass")

    log.info("query_all_photometry: %d sources, surveys=%s", len(sources_df), active_surveys)

    ps1_df, tmass_df = Parallel(n_jobs=2, prefer="threads")(
        [delayed(_run_ps1)(), delayed(_run_2mass)()]
    )

    # Merge best PS1 match (one row per source) onto sources_df
    out = sources_df.copy()

    if ps1_df is not None and len(ps1_df) > 0:
        # PS1 already returns one row per source
        ps1_best = ps1_df.drop_duplicates(subset=["source_id"])
        out = out.merge(ps1_best, on="source_id", how="left", suffixes=("", "_ps1"))

    if tmass_df is not None and len(tmass_df) > 0:
        # Pick nearest 2MASS candidate per source
        tmass_with_sep = tmass_df.dropna(subset=["sep_source_tmass_arcsec"])
        if len(tmass_with_sep) > 0:
            tmass_best = (
                tmass_with_sep
                .sort_values("sep_source_tmass_arcsec")
                .drop_duplicates(subset=["source_id"])
            )
        else:
            tmass_best = tmass_df.drop_duplicates(subset=["source_id"])
        out = out.merge(tmass_best, on="source_id", how="left", suffixes=("", "_tmass"))

    # Summary
    n_ps1 = int(out["ps1_available"].sum()) if "ps1_available" in out.columns else 0
    n_tmass = int(out["tmass_available"].sum()) if "tmass_available" in out.columns else 0

    def _has(col):
        return out.get(col, pd.Series(False)).astype(bool)

    n_both = 0
    if "ps1_available" in out.columns and "tmass_available" in out.columns:
        n_both = int((_has("ps1_available") & _has("tmass_available")).sum())
    n_none = int(
        (~_has("ps1_available") & ~_has("tmass_available")).sum()
    )

    log.info(
        "query_all_photometry summary: total=%d  PS1=%d  2MASS=%d  both=%d  none=%d",
        len(out), n_ps1, n_tmass, n_both, n_none,
    )
    print(
        f"Photometry query summary ({len(out)} sources):\n"
        f"  PS1:   {n_ps1}\n"
        f"  2MASS: {n_tmass}\n"
        f"  Both:  {n_both}\n"
        f"  None:  {n_none}"
    )
    return out


# ---------------------------------------------------------------------------
# [C3] PHAT-specific queries — implemented in Phase 6
# ---------------------------------------------------------------------------


def query_csc_sources_in_polygon(
    polygon_vertices: list[tuple[float, float]],
    csc_csv_path: str | Path,
    significance_min: float = 3.0,
) -> pd.DataFrame:
    """Filter a Chandra Source Catalog CSV to sources inside a sky polygon.  [C3]

    Uses ``shapely.geometry.Polygon`` for the footprint and
    ``shapely.geometry.Point`` per source.

    Parameters
    ----------
    polygon_vertices : list of (ra_deg, dec_deg) tuples
        Polygon vertices in degrees, ICRS.
    csc_csv_path : str or Path
        Path to user-provided CSC CSV.
    significance_min : float
        Minimum detection significance.

    Returns
    -------
    pd.DataFrame
        Filtered CSC rows with ``in_phat_footprint=True`` appended.
        Logs: total sources, inside polygon, passing significance cut.
    """
    from shapely.geometry import Point
    from shapely.geometry import Polygon as ShapelyPolygon

    df = pd.read_csv(Path(csc_csv_path))
    n_total = len(df)
    log.info("query_csc_sources_in_polygon: %d total sources in CSV", n_total)

    poly = ShapelyPolygon(polygon_vertices)

    ra_col = _find_col(list(df.columns), ["ra", "RA", "RAdeg", "RAJ2000"])
    dec_col = _find_col(list(df.columns), ["dec", "Dec", "DE", "DEdeg", "DEJ2000"])

    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Cannot identify RA/Dec columns in {csc_csv_path}. "
            f"Available: {list(df.columns)}"
        )

    inside = np.array(
        [
            poly.contains(Point(float(row[ra_col]), float(row[dec_col])))
            for _, row in df.iterrows()
        ]
    )
    df_inside = df[inside].copy()
    n_inside = len(df_inside)
    log.info("query_csc_sources_in_polygon: %d inside polygon", n_inside)

    sig_col = _find_col(
        list(df_inside.columns), ["significance", "sig", "Significance"]
    )
    if sig_col is not None and len(df_inside) > 0:
        df_inside = df_inside[
            df_inside[sig_col].astype(float) >= significance_min
        ].copy()
    n_sig = len(df_inside)
    log.info(
        "query_csc_sources_in_polygon: %d pass significance >= %.1f",
        n_sig,
        significance_min,
    )

    df_inside["in_phat_footprint"] = True
    return df_inside.reset_index(drop=True)


def _hsc_fetch_one(
    xray_id: str,
    ra: float,
    dec: float,
    search_radius_arcsec: float,
    cache_dir: Path,
) -> list[dict]:
    """Single-source HSC v3 REST query with per-source disk caching."""
    cached = _load_cache(xray_id, cache_dir)
    if cached is not None:
        return cached

    import time

    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(
        stop=stop_after_attempt(config.QUERY_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.QUERY_RETRY_WAIT_SEC,
            min=config.QUERY_RETRY_WAIT_SEC,
            max=60,
        ),
        reraise=True,
    )
    def _do_get():
        resp = requests.get(
            config.HSC_API_URL,
            params={
                "ra": ra,
                "dec": dec,
                # New MAST API expects radius in degrees
                "radius": search_radius_arcsec / 3600.0,
                "format": "json",
                # Request large page to avoid silent truncation in dense fields
                "pagesize": 50000,
            },
            timeout=config.QUERY_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        time.sleep(config.HSC_REQUEST_SLEEP_SEC)
        data = _do_get()
    except Exception as exc:
        log.warning("HSC query failed for source %s: %s", xray_id, exc)
        _save_cache(xray_id, cache_dir, [])
        return []

    # New MAST API format: {"info": [{name, ...}, ...], "data": [[v1, v2, ...], ...]}
    # Convert to list-of-dicts keyed by column name.
    if isinstance(data, dict):
        col_names = [c["name"] for c in data.get("info", [])]
        raw_rows = data.get("data", []) or []
        if col_names and raw_rows and isinstance(raw_rows[0], (list, tuple)):
            rows = [dict(zip(col_names, r)) for r in raw_rows]
        else:
            rows = raw_rows  # already dicts
    elif isinstance(data, list):
        rows = data  # old API format (flat list of dicts) — keep for compat
    else:
        rows = []

    _save_cache(xray_id, cache_dir, rows)
    return rows


def query_hsc_for_chandra_sources(
    chandra_df: pd.DataFrame,
    search_radius_factor: float = 1.0,
    cache_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query HSC v3 for HST counterparts of Chandra sources.  [C3]

    For each Chandra source the search radius is::

        max(MIN_SEARCH_RADIUS_ARCSEC, 3.0 * xray_pos_err_arcsec) * search_radius_factor

    Parameters
    ----------
    chandra_df : pd.DataFrame
        Chandra sources (output of ``query_csc_sources_in_polygon``).
    search_radius_factor : float
        Multiplier on the per-source search radius.
    cache_dir : str or Path, optional
        Cache directory.  Defaults to ``config.QUERY_CACHE_DIR / 'hsc'``.

    Returns
    -------
    hsc_best_match_df : pd.DataFrame
        One row per Chandra source — nearest HSC match (or NaN if none).
        Columns: xray_id, xray_ra, xray_dec, xray_pos_err_arcsec,
        hsc_match_id, hsc_ra, hsc_dec, hsc_sep_arcsec, hsc_sep_normsep,
        hsc_n_candidates, hsc_F275W_mag, hsc_F275W_err, hsc_F336W_mag,
        hsc_F336W_err, hsc_F475W_mag, hsc_F475W_err, hsc_F814W_mag,
        hsc_F814W_err, hsc_F110W_mag, hsc_F110W_err, hsc_F160W_mag,
        hsc_F160W_err, hsc_n_filters_detected, hsc_match_status.

        ``hsc_match_status`` values: 'unique', 'nearest', 'none'.

    hsc_all_candidates_df : pd.DataFrame
        One row per HSC detection per Chandra source at filter level.
        Columns: xray_id, xray_ra, xray_dec, xray_pos_err_arcsec,
        hsc_match_id, hsc_ra, hsc_dec, hsc_sep_arcsec, hsc_normsep,
        hsc_filter, hsc_mag, hsc_mag_err, hsc_n_images, hsc_start_mjd,
        hsc_stop_mjd, hsc_catalog_id.
    """
    cdir = Path(cache_dir) if cache_dir else config.QUERY_CACHE_DIR / "hsc"
    cdir.mkdir(parents=True, exist_ok=True)

    cols = list(chandra_df.columns)
    ra_col = _find_col(cols, ["ra", "RA", "RAdeg", "RAJ2000"]) or "ra"
    dec_col = _find_col(cols, ["dec", "Dec", "DE", "DEdeg", "DEJ2000"]) or "dec"
    id_col = _find_col(cols, ["name", "source_id", "xray_id", "id"]) or "name"
    err0_col = _find_col(cols, ["err_ellipse_r0", "pos_err_r0", "pos_err"])
    err1_col = _find_col(cols, ["err_ellipse_r1", "pos_err_r1"])

    # PHAT HST filters we pivot into per-source photometry columns
    phat_filters = ["F275W", "F336W", "F475W", "F814W", "F110W", "F160W"]

    all_candidates: list[dict] = []
    best_matches: list[dict] = []

    for _, row in chandra_df.iterrows():
        xray_id = str(row[id_col])
        xray_ra = float(row[ra_col])
        xray_dec = float(row[dec_col])

        # Effective position error (arcsec)
        r0 = float(row[err0_col]) if err0_col and pd.notna(row[err0_col]) else np.nan
        r1 = float(row[err1_col]) if err1_col and pd.notna(row[err1_col]) else np.nan
        if np.isfinite(r0) and np.isfinite(r1):
            pos_err = float(np.sqrt((r0 ** 2 + r1 ** 2) / 2.0))
        elif np.isfinite(r0):
            pos_err = r0
        elif np.isfinite(r1):
            pos_err = r1
        else:
            pos_err = config.MIN_SEARCH_RADIUS_ARCSEC

        search_r = (
            max(config.MIN_SEARCH_RADIUS_ARCSEC, config.LABEL_TO_XRAY_SIGMA * pos_err)
            * search_radius_factor
        )

        detections = _hsc_fetch_one(xray_id, xray_ra, xray_dec, search_r, cdir)

        src_coord = SkyCoord(ra=xray_ra * u.deg, dec=xray_dec * u.deg)
        det_rows: list[dict] = []

        for det in detections:
            # New MAST API uses MatchRA/MatchDec; old API used RA/Dec.
            det_ra = _to_float(det.get("MatchRA") or det.get("RA"))
            det_dec = _to_float(det.get("MatchDec") or det.get("Dec"))
            sep_arcsec = (
                src_coord.separation(
                    SkyCoord(ra=det_ra * u.deg, dec=det_dec * u.deg)
                ).arcsec
                if np.isfinite(det_ra) and np.isfinite(det_dec)
                else np.nan
            )
            normsep = (
                sep_arcsec / pos_err
                if pos_err > 0 and np.isfinite(sep_arcsec)
                else np.nan
            )

            # Normalise filter name to short form (e.g. "WFC3/UVIS/F475W" -> "F475W")
            filter_name = str(det.get("Filter", "")).strip()
            if "/" in filter_name:
                filter_name = filter_name.split("/")[-1]

            det_rows.append(
                {
                    "xray_id": xray_id,
                    "xray_ra": xray_ra,
                    "xray_dec": xray_dec,
                    "xray_pos_err_arcsec": pos_err,
                    "hsc_match_id": det.get("MatchID"),
                    "hsc_ra": det_ra,
                    "hsc_dec": det_dec,
                    "hsc_sep_arcsec": sep_arcsec,
                    "hsc_normsep": normsep,
                    "hsc_filter": filter_name,
                    "hsc_mag": _to_float(det.get("MagAper2")),
                    # MagAper2Err not provided by new MAST detailed API
                    "hsc_mag_err": _to_float(det.get("MagAper2Err")),
                    # NumImages not in new API (each row is one detection image)
                    "hsc_n_images": _to_float(det.get("NumImages", 1)),
                    "hsc_start_mjd": _to_float(det.get("StartMJD")),
                    "hsc_stop_mjd": _to_float(det.get("StopMJD")),
                    "hsc_catalog_id": det.get("CatID"),
                }
            )

        all_candidates.extend(det_rows)

        # --- Best-match row (one per Chandra source) ---
        best_row: dict = {
            "xray_id": xray_id,
            "xray_ra": xray_ra,
            "xray_dec": xray_dec,
            "xray_pos_err_arcsec": pos_err,
            "hsc_n_candidates": len(det_rows),
        }

        if not det_rows:
            best_row.update(
                {
                    "hsc_match_id": np.nan,
                    "hsc_ra": np.nan,
                    "hsc_dec": np.nan,
                    "hsc_sep_arcsec": np.nan,
                    "hsc_sep_normsep": np.nan,
                    "hsc_match_status": "none",
                    "hsc_n_filters_detected": 0,
                }
            )
            for f in phat_filters:
                best_row[f"hsc_{f}_mag"] = np.nan
                best_row[f"hsc_{f}_err"] = np.nan
        else:
            det_df = pd.DataFrame(det_rows)

            # Pick best HSC MatchID by mean separation across its detections
            match_seps = (
                det_df.groupby("hsc_match_id")["hsc_sep_arcsec"]
                .mean()
                .reset_index()
                .sort_values("hsc_sep_arcsec")
            )
            n_matches = len(match_seps)
            best_match_id = match_seps.iloc[0]["hsc_match_id"]
            best_sep = float(match_seps.iloc[0]["hsc_sep_arcsec"])

            if n_matches == 1:
                match_status = "unique"
            else:
                second_sep = float(match_seps.iloc[1]["hsc_sep_arcsec"])
                ratio = second_sep / best_sep if best_sep > 0 else np.inf
                match_status = (
                    "nearest" if ratio < config.MATCH_SECOND_BEST_RATIO else "unique"
                )

            best_dets = det_df[det_df["hsc_match_id"] == best_match_id]
            best_ra = float(best_dets["hsc_ra"].iloc[0])
            best_dec = float(best_dets["hsc_dec"].iloc[0])
            best_normsep = best_sep / pos_err if pos_err > 0 else np.nan

            # Pivot per-filter photometry (take first detection per filter)
            filter_phot: dict[str, tuple[float, float]] = {}
            for _, d in best_dets.iterrows():
                fn = str(d["hsc_filter"])
                if fn in phat_filters and fn not in filter_phot:
                    filter_phot[fn] = (float(d["hsc_mag"]), float(d["hsc_mag_err"]))

            best_row.update(
                {
                    "hsc_match_id": best_match_id,
                    "hsc_ra": best_ra,
                    "hsc_dec": best_dec,
                    "hsc_sep_arcsec": best_sep,
                    "hsc_sep_normsep": best_normsep,
                    "hsc_match_status": match_status,
                    "hsc_n_filters_detected": len(filter_phot),
                }
            )
            for f in phat_filters:
                if f in filter_phot:
                    best_row[f"hsc_{f}_mag"] = filter_phot[f][0]
                    best_row[f"hsc_{f}_err"] = filter_phot[f][1]
                else:
                    best_row[f"hsc_{f}_mag"] = np.nan
                    best_row[f"hsc_{f}_err"] = np.nan

        best_matches.append(best_row)

    best_df = pd.DataFrame(best_matches)
    all_df = (
        pd.DataFrame(all_candidates)
        if all_candidates
        else pd.DataFrame(
            columns=[
                "xray_id",
                "xray_ra",
                "xray_dec",
                "xray_pos_err_arcsec",
                "hsc_match_id",
                "hsc_ra",
                "hsc_dec",
                "hsc_sep_arcsec",
                "hsc_normsep",
                "hsc_filter",
                "hsc_mag",
                "hsc_mag_err",
                "hsc_n_images",
                "hsc_start_mjd",
                "hsc_stop_mjd",
                "hsc_catalog_id",
            ]
        )
    )

    n_matched = (
        int((best_df["hsc_match_status"] != "none").sum()) if len(best_df) > 0 else 0
    )
    log.info(
        "query_hsc_for_chandra_sources: %d sources, %d with HSC match",
        len(chandra_df),
        n_matched,
    )
    return best_df, all_df

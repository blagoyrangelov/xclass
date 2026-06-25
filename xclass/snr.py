"""xclass.snr — Extragalactic supernova remnant (SNR) catalog assembly.

Queries literature SNR catalogs from VizieR, deduplicates within each
host galaxy, cross-matches to Chandra X-ray detections, and builds a
training-dataset-ready SNR catalog.

Functions
---------
fetch_snr_literature
dedup_snr_catalog
crossmatch_snr_with_hst
build_snr_ml_catalog
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from xclass import config
from xclass.catalog import UnionFind, _find_col, _parse_radec
from xclass.io import load_catalog

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-galaxy catalog fetching helpers
# ---------------------------------------------------------------------------

# (vizier_table_id, catalog_tag, selection_method)
# Catalog IDs verified against reference notebook 1_2_SNR_Catalog.ipynb.
_SNR_CATALOG_ENTRIES: dict[str, list[tuple[str, str, str]]] = {
    "M33": [
        ("J/ApJS/187/495/table3", "Long2010_M33_opt",  "optical"),   # Long+ 2010
    ],
    "M83": [
        ("J/ApJS/203/8",          "Blair2012_M83_opt", "optical"),   # Blair+ 2012
    ],
    "NGC 6946": [
        ("J/ApJ/916/58/table3",   "Long2021_N6946_opt", "optical"),  # Long+ 2021
    ],
}

# Reverse map: aliases -> canonical galaxy name
_HOST_NORM = {k.upper().replace(" ", ""): v
              for k, v in config.HOST_STANDARDIZATION.items()}


def _canonicalise_host(name: str) -> str:
    key = name.upper().replace(" ", "")
    return _HOST_NORM.get(key, name)


def _fetch_snr_table(
    table_id: str,
    catalog_tag: str,
    host_galaxy: str,
    selection_method: str,
) -> Optional[pd.DataFrame]:
    """Fetch one SNR VizieR table and return a minimal standardised DataFrame."""
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=["**"])
    try:
        result = v.get_catalogs(table_id)
    except Exception as exc:
        log.warning("VizieR fetch failed for %s: %s", table_id, exc)
        return None

    if not result or len(result) == 0:
        log.warning("VizieR returned no results for %s", table_id)
        return None

    tbl = result[0]

    try:
        ra, dec = _parse_radec(tbl)
    except ValueError as exc:
        log.warning("Cannot parse RA/Dec for %s: %s", table_id, exc)
        return None

    # Source name: try common name columns
    name_col = _find_col(list(tbl.colnames), ["Name", "Seq", "SNR", "ID", "[L2010]"])
    if name_col:
        names = np.asarray(tbl[name_col], dtype=str)
    else:
        names = np.array([f"{catalog_tag}-{i}" for i in range(len(tbl))])

    # Prefix names with catalog_tag to keep them unique
    names = np.array([f"{catalog_tag}_{nm}" for nm in names])

    # Diameter (arcsec) if available
    diam_col = _find_col(list(tbl.colnames), ["Diam", "Dmaj", "Size", "Rad"])
    diam = np.full(len(tbl), np.nan)
    if diam_col:
        try:
            import numpy.ma as ma
            raw = np.asarray(tbl[diam_col])
            if isinstance(raw, ma.MaskedArray):
                raw = raw.filled(np.nan)
            diam = raw.astype(float)
        except Exception:
            pass

    log.info("Fetched %d SNRs from %s (%s)", len(tbl), table_id, host_galaxy)
    return pd.DataFrame({
        "source_name": names,
        "ra": ra,
        "dec": dec,
        "host_galaxy": host_galaxy,
        "catalog_tag": catalog_tag,
        "selection_method": selection_method,
        "diam_arcsec": diam,
        "class_label": "SNR",
        "label_confidence": 1.0,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_snr_literature(
    target_galaxies: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Query VizieR for SNR literature catalogs across target galaxies.

    Parameters
    ----------
    target_galaxies : list of str, optional
        Galaxies to include.  Defaults to ``config.TARGET_GALAXIES``
        minus M31 (which has no SNR VizieR catalog in the default set).

    Returns
    -------
    pd.DataFrame
        Columns: source_name, ra, dec, host_galaxy, catalog_tag,
        selection_method, class_label, label_confidence.
    """
    galaxies = target_galaxies or list(_SNR_CATALOG_ENTRIES.keys())
    frames: list[pd.DataFrame] = []

    for galaxy in galaxies:
        canonical = _canonicalise_host(galaxy)
        entries = _SNR_CATALOG_ENTRIES.get(canonical, [])
        if not entries:
            log.warning("No SNR catalog entries configured for galaxy '%s'", galaxy)
            continue

        for table_id, tag, method in entries:
            df_tbl = _fetch_snr_table(table_id, tag, canonical, method)
            if df_tbl is not None and len(df_tbl) > 0:
                frames.append(df_tbl)

    if not frames:
        warnings.warn(
            "fetch_snr_literature: no SNR catalogs could be fetched.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=[
            "source_name", "ra", "dec", "host_galaxy", "catalog_tag",
            "selection_method", "class_label", "label_confidence",
        ])

    result = pd.concat(frames, ignore_index=True)
    log.info(
        "fetch_snr_literature: %d total SNR rows across %d galaxies",
        len(result), result["host_galaxy"].nunique(),
    )
    return result


def dedup_snr_catalog(
    df: pd.DataFrame,
    radius_arcsec: float = 2.0,
) -> pd.DataFrame:
    """Deduplicate SNR catalog within each host galaxy.

    Uses Union-Find with sky proximity grouping. Where multiple entries
    exist for the same SNR, the canonical row prefers X-ray positions
    over optical over radio (determined by ``selection_method``).

    Parameters
    ----------
    df : pd.DataFrame
        Raw SNR catalog from :func:`fetch_snr_literature`.
    radius_arcsec : float
        Matching radius for merging duplicates.

    Returns
    -------
    pd.DataFrame
        Deduplicated catalog (one row per unique SNR).
        Columns: source_name, ra, dec, host_galaxy, selection_method,
        diam_arcsec, class_label, label_confidence, n_input_rows,
        input_catalog_tags.
    """
    df = df.reset_index(drop=True)

    # Priority for canonical position selection (lower = preferred)
    _METHOD_PRIORITY = {"xray": 0, "xray+optical": 0, "optical": 1, "radio": 2}

    rows: list[dict] = []

    for galaxy, gdf in df.groupby("host_galaxy"):
        gdf = gdf.reset_index(drop=True)
        n = len(gdf)

        ra = gdf["ra"].values.astype(float)
        dec = gdf["dec"].values.astype(float)
        valid = np.isfinite(ra) & np.isfinite(dec)

        uf = UnionFind(n)

        if valid.sum() >= 2:
            coords = SkyCoord(
                ra=ra[valid] * u.deg,
                dec=dec[valid] * u.deg,
                frame="icrs",
            )
            valid_idx = np.where(valid)[0]
            try:
                i1, i2, sep2d, _ = coords.search_around_sky(
                    coords, radius_arcsec * u.arcsec
                )
            except Exception as exc:
                log.warning("search_around_sky failed for %s: %s", galaxy, exc)
                i1, i2, sep2d = [], [], []

            for il, jl in zip(i1, i2):
                if il < jl:
                    uf.union(int(valid_idx[il]), int(valid_idx[jl]))

        for _, members in uf.groups().items():
            sub = gdf.iloc[members]

            # Select canonical row: prefer xray, then optical, then radio
            priorities = sub["selection_method"].map(
                lambda m: _METHOD_PRIORITY.get(m, 99)
            )
            best_idx = priorities.idxmin()
            best = sub.loc[best_idx]

            tags = "; ".join(sub["catalog_tag"].unique())
            rows.append({
                "source_name": best["source_name"],
                "ra": best["ra"],
                "dec": best["dec"],
                "host_galaxy": galaxy,
                "selection_method": best["selection_method"],
                "diam_arcsec": (
                    sub["diam_arcsec"].dropna().mean()
                    if "diam_arcsec" in sub.columns else np.nan
                ),
                "class_label": "SNR",
                "label_confidence": 1.0,
                "n_input_rows": len(members),
                "input_catalog_tags": tags,
            })

    result = pd.DataFrame(rows)
    log.info(
        "dedup_snr_catalog: %d raw rows -> %d unique SNRs", len(df), len(result)
    )
    return result


def crossmatch_snr_with_hst(
    snr_df: pd.DataFrame,
    hst_catalog_path: str | Path,
) -> pd.DataFrame:
    """Match SNRs to a user-provided HST source catalog (1.0 arcsec radius).

    Parameters
    ----------
    snr_df : pd.DataFrame
        Deduplicated SNR catalog.
    hst_catalog_path : str or Path
        Path to HST source catalog (CSV or FITS) with columns ra, dec.

    Returns
    -------
    pd.DataFrame
        *snr_df* with columns appended: hst_match_id, hst_ra, hst_dec,
        hst_sep_arcsec, hst_match_status.
    """
    hst = load_catalog(hst_catalog_path)
    snr_coords = SkyCoord(
        ra=snr_df["ra"].values * u.deg,
        dec=snr_df["dec"].values * u.deg,
        frame="icrs",
    )
    hst_ra_col = _find_col(list(hst.columns), ["ra", "RA", "RAdeg", "RAJ2000"])
    hst_dec_col = _find_col(list(hst.columns), ["dec", "Dec", "DE", "DEdeg", "DEJ2000"])
    if hst_ra_col is None or hst_dec_col is None:
        raise ValueError("HST catalog is missing RA/Dec columns.")

    hst_coords = SkyCoord(
        ra=hst[hst_ra_col].values * u.deg,
        dec=hst[hst_dec_col].values * u.deg,
        frame="icrs",
    )

    idx, sep2d, _ = snr_coords.match_to_catalog_sky(hst_coords)
    match_radius = 1.0 * u.arcsec

    out = snr_df.copy()
    matched = sep2d <= match_radius

    name_col = _find_col(list(hst.columns), ["name", "Name", "ID", "id"])

    out["hst_match_id"] = np.where(
        matched,
        hst[name_col].values[idx] if name_col else idx.astype(str),
        "",
    )
    out["hst_ra"] = np.where(matched, hst[hst_ra_col].values[idx], np.nan)
    out["hst_dec"] = np.where(matched, hst[hst_dec_col].values[idx], np.nan)
    out["hst_sep_arcsec"] = np.where(matched, sep2d.arcsec, np.nan)
    out["hst_match_status"] = np.where(matched, "matched", "none")

    n_match = int(matched.sum())
    log.info("crossmatch_snr_with_hst: %d/%d SNRs matched to HST", n_match, len(snr_df))
    return out


def build_snr_ml_catalog(snr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert deduplicated SNR catalog to the main TD schema.

    SNRs use real HST photometry (prefixed 'A_') rather than SED predictions.
    X-ray fluxes and significance are set to NaN if not available from
    cross-matching; they are populated by notebook 02 after Chandra matching.

    Parameters
    ----------
    snr_df : pd.DataFrame
        Deduplicated SNR catalog (from :func:`dedup_snr_catalog`).

    Returns
    -------
    pd.DataFrame
        Columns matching main TD schema: class_label, ra, dec, host_galaxy,
        Fx_S/M/H/B (NaN), significance (NaN), plus any HST magnitude columns
        present in the input (A_F* prefix).
    """
    # Handle the empty-input case: if no SNR catalogs were fetched (e.g. all
    # VizieR queries failed due to a network outage), snr_df has no columns and
    # indexing ["ra"] would raise KeyError. Return a correctly-shaped empty
    # catalog so the pipeline can proceed without SNR sources rather than crash.
    if snr_df.empty:
        log.warning(
            "build_snr_ml_catalog: input SNR catalog is empty; "
            "returning empty SNR catalog (0 sources). This usually means the "
            "SNR literature fetch failed — check network/VizieR availability."
        )
        base_cols = [
            "class_label", "ra", "dec", "host_galaxy", "source_name",
            "label_confidence", "Fx_S", "Fx_M", "Fx_H", "Fx_B", "significance",
        ]
        hst_mag_cols = [c for c in snr_df.columns if c.startswith("A_F")]
        return pd.DataFrame(columns=base_cols + hst_mag_cols)

    hst_mag_cols = [c for c in snr_df.columns if c.startswith("A_F")]

    out = pd.DataFrame({
        "class_label": "SNR",
        "ra": snr_df["ra"].values,
        "dec": snr_df["dec"].values,
        "host_galaxy": snr_df["host_galaxy"].values,
        "source_name": snr_df["source_name"].values,
        "label_confidence": snr_df["label_confidence"].values,
        "Fx_S": np.nan,
        "Fx_M": np.nan,
        "Fx_H": np.nan,
        "Fx_B": np.nan,
        "significance": np.nan,
    })

    for col in hst_mag_cols:
        out[col] = snr_df[col].values

    log.info("build_snr_ml_catalog: %d SNR rows in ML schema", len(out))
    return out

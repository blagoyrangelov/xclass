"""xclass.catalog — Training dataset assembly and deduplication.

Fetches literature catalogs from VizieR, standardises them to a common schema,
and assembles a deduplicated master training dataset.

Functions
---------
normalize_name
classify_sptype
fetch_vizier_catalog
build_td_catalog
build_master_resolved_table
describe_td

Classes
-------
UnionFind
"""

from __future__ import annotations

import inspect
import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from xclass import config
from xclass.io import load_catalog, save_catalog

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_name(name) -> str:
    """Strip, uppercase, and remove non-alphanumeric characters from a name.

    Parameters
    ----------
    name : str or float
        Raw source name.

    Returns
    -------
    str
        Normalised name (alphanumeric uppercase only).
        Returns ``''`` for NaN, None, or whitespace-only input.
    """
    if name is None:
        return ""
    if isinstance(name, float) and np.isnan(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""
    return _NON_ALNUM.sub("", s.upper())


def classify_sptype(sptype) -> Optional[str]:
    """Classify a spectral type string as 'HM-STAR' or 'LM-STAR'.

    O and B spectral classes -> 'HM-STAR'.  All others (A, F, G, K, M, and
    unknown) -> 'LM-STAR', per spec rule (unknown types default to LM-STAR).

    Parameters
    ----------
    sptype : str or float
        Raw spectral type (e.g. 'K5V', 'B2Ib', 'O5').

    Returns
    -------
    str or None
        'HM-STAR', 'LM-STAR', or None for NaN/empty input.
    """
    if sptype is None:
        return None
    if isinstance(sptype, float) and np.isnan(sptype):
        return None
    s = str(sptype).strip()
    if not s:
        return None
    first = s[0].upper()
    if first in config.HM_SPTYPE_PREFIXES:
        return "HM-STAR"
    return "LM-STAR"


# ---------------------------------------------------------------------------
# VizieR fetching
# ---------------------------------------------------------------------------


def fetch_vizier_catalog(
    catalog_id: str,
    columns: Optional[list[str]] = None,
    row_limit: int = -1,
) -> "astropy.table.Table":  # noqa: F821
    """Fetch a catalog table from VizieR.

    Parameters
    ----------
    catalog_id : str
        VizieR catalog identifier (e.g. 'J/A+A/469/807').
    columns : list of str, optional
        Column names to retrieve.  None means all columns.
    row_limit : int
        Maximum rows.  -1 means no limit.

    Returns
    -------
    astropy.table.Table
        First table returned by VizieR for this catalog.

    Raises
    ------
    RuntimeError
        If VizieR returns no tables for the catalog ID.
    """
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=row_limit, columns=columns or ["**"])
    try:
        result = v.get_catalogs(catalog_id)
    except Exception as exc:
        raise RuntimeError(
            f"VizieR query failed for catalog '{catalog_id}': {exc}"
        ) from exc

    if not result:
        raise RuntimeError(
            f"VizieR returned no tables for catalog '{catalog_id}'."
        )

    tbl = result[0]
    log.info("Fetched %d rows from VizieR catalog %s", len(tbl), catalog_id)
    return tbl


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

_RA_CANDIDATES = ["RAdeg", "_RA", "RA_ICRS", "RAJ2000", "RA"]
_DEC_CANDIDATES = ["DEdeg", "_DE", "DE_ICRS", "DEJ2000", "DE"]
_NAME_CANDIDATES = [
    "Name", "ID", "LMXB", "HMXB", "Simbad", "Simbad_",
    "WR", "HD", "HIP", "2MASS", "APOGEE", "SimbadName",
]


def _find_col(colnames: list[str], candidates: list[str]) -> Optional[str]:
    col_set = set(colnames)
    for c in candidates:
        if c in col_set:
            return c
    return None


def _parse_radec(tbl) -> tuple[np.ndarray, np.ndarray]:
    """Extract RA, Dec (decimal degrees) from an astropy Table."""
    colnames = list(tbl.colnames)
    ra_col = _find_col(colnames, _RA_CANDIDATES)
    dec_col = _find_col(colnames, _DEC_CANDIDATES)

    if ra_col and dec_col:
        ra_raw = np.asarray(tbl[ra_col])
        dec_raw = np.asarray(tbl[dec_col])
        # Sexagesimal string check
        if ra_raw.dtype.kind in ("U", "S", "O"):
            sc = SkyCoord(ra_raw, dec_raw, unit=(u.hourangle, u.deg), frame="icrs")
            return sc.ra.deg, sc.dec.deg
        # Masked array handling
        try:
            import numpy.ma as ma
            if isinstance(ra_raw, ma.MaskedArray):
                ra_raw = ra_raw.filled(np.nan)
            if isinstance(dec_raw, ma.MaskedArray):
                dec_raw = dec_raw.filled(np.nan)
        except Exception:
            pass
        return ra_raw.astype(float), dec_raw.astype(float)

    raise ValueError(
        f"Could not find RA/Dec columns.  Available: {colnames}"
    )


def _parse_name_col(tbl, extra: Optional[list[str]] = None) -> np.ndarray:
    colnames = list(tbl.colnames)
    col = _find_col(colnames, (extra or []) + _NAME_CANDIDATES)
    if col is None:
        log.warning("No name column found; using running index")
        return np.array([f"src_{i}" for i in range(len(tbl))])
    return np.asarray(tbl[col], dtype=str)


# ---------------------------------------------------------------------------
# Per-catalog standardisation handlers
# ---------------------------------------------------------------------------


def _std_xrb(tbl, class_label: str, catalog_id: str, ref: str) -> pd.DataFrame:
    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["Name", "System", "ID"])
    return pd.DataFrame({
        "name": names, "Class": class_label,
        "ra": ra, "dec": dec,
        "source_catalog": catalog_id, "source_ref": ref,
        "label_confidence": 1.0, "SpType": np.nan,
    })


def _std_cv(tbl, catalog_id: str, ref: str) -> pd.DataFrame:
    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["Name", "ID"])
    return pd.DataFrame({
        "name": names, "Class": "CV",
        "ra": ra, "dec": dec,
        "source_catalog": catalog_id, "source_ref": ref,
        "label_confidence": 1.0, "SpType": np.nan,
    })


def _std_agn(tbl, catalog_id: str, ref: str) -> pd.DataFrame:
    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["Name", "Simbad"])
    return pd.DataFrame({
        "name": names, "Class": "AGN",
        "ra": ra, "dec": dec,
        "source_catalog": catalog_id, "source_ref": ref,
        "label_confidence": 1.0, "SpType": np.nan,
    })


_SKIFF_MAX_MAG = 23.0          # magnitude cut matching reference notebook
_SKIFF_AMBIGUOUS = ("?", ":", "/", "+")  # tokens marking ambiguous sptypes
_DROP_ORION = True


def _std_skiff(tbl, catalog_id: str, ref: str) -> pd.DataFrame:
    import numpy.ma as ma

    df = tbl.to_pandas().copy()

    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["Name", "HD", "HIP"])

    sptype_col = _find_col(list(tbl.colnames), ["SpType", "MK", "Sp"])
    if sptype_col:
        sptypes = np.asarray(tbl[sptype_col], dtype=str)
    else:
        log.warning("Skiff catalog: SpType column not found")
        return pd.DataFrame(columns=["name", "Class", "ra", "dec",
                                     "source_catalog", "source_ref",
                                     "label_confidence", "SpType"])

    # Magnitude filter (match reference: V <= 23)
    mag_col = _find_col(list(tbl.colnames), ["Vmag", "V", "mag"])
    if mag_col:
        mag_raw = np.asarray(tbl[mag_col])
        if isinstance(mag_raw, np.ma.MaskedArray):
            mag_raw = mag_raw.filled(np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mags = pd.to_numeric(pd.Series(mag_raw), errors="coerce").values
    else:
        mags = np.full(len(sptypes), np.nan)

    rows = []
    for i, (name, sp, r, d, mag) in enumerate(
        zip(names, sptypes, ra, dec, mags)
    ):
        sp_str = str(sp).strip().upper()
        if not sp_str or sp_str in ("", "NAN", "--"):
            continue
        if any(tok in sp_str for tok in _SKIFF_AMBIGUOUS):
            continue
        if _DROP_ORION and "ORION" in sp_str:
            continue
        cls = classify_sptype(sp_str)
        if cls is None:
            continue
        if not np.isnan(mag) and mag > _SKIFF_MAX_MAG:
            continue
        if not (np.isfinite(r) and np.isfinite(d)):
            continue
        rows.append({
            "name": str(name), "Class": cls,
            "ra": float(r), "dec": float(d),
            "source_catalog": catalog_id, "source_ref": ref,
            "label_confidence": 1.0, "SpType": sp_str,
        })

    result = pd.DataFrame(rows)
    log.info("Skiff after filtering: %d / %d rows kept", len(result), len(sptypes))
    return result


def _std_apogee(tbl, catalog_id: str, ref: str) -> pd.DataFrame:
    """APOGEE-2 DR16 LM-STAR handler.

    Requires Teff and logg to be present (matching reference notebook
    ``build_apogee_lmstars``).  Also applies VSCATTER stability filter.
    """
    df = tbl.to_pandas().copy()

    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["APOGEE_ID", "2MASS", "APOGEE", "ID"])

    # Quality columns
    teff_col = _find_col(list(tbl.colnames), ["Teff", "TEFF", "teff"])
    logg_col = _find_col(list(tbl.colnames), ["logg", "LOGG"])
    vscatter_col = _find_col(list(tbl.colnames), ["VSCATTER", "Vscatter"])
    verr_col = _find_col(list(tbl.colnames), ["VERR_MED", "Verr_med"])

    def _to_float(col):
        if col is None:
            return np.full(len(df), np.nan)
        raw = np.asarray(tbl[col])
        if isinstance(raw, np.ma.MaskedArray):
            raw = raw.filled(np.nan)
        return pd.to_numeric(pd.Series(raw), errors="coerce").values

    teff = _to_float(teff_col)
    logg = _to_float(logg_col)
    vscatter = _to_float(vscatter_col)
    verr = _to_float(verr_col)

    # Keep rows with valid Teff and logg (matches reference filter)
    valid_teff = np.isfinite(teff)
    valid_logg = np.isfinite(logg)

    # VSCATTER stability: if present, require VSCATTER <= 1.0
    if not np.all(np.isnan(vscatter)):
        stable = np.isnan(vscatter) | (vscatter <= 1.0)
    else:
        stable = np.ones(len(df), dtype=bool)

    mask = valid_teff & valid_logg & stable & np.isfinite(ra) & np.isfinite(dec)

    result = pd.DataFrame({
        "name": names[mask], "Class": "LM-STAR",
        "ra": ra[mask], "dec": dec[mask],
        "source_catalog": catalog_id, "source_ref": ref,
        "label_confidence": 1.0, "SpType": np.nan,
    })
    log.info("APOGEE after filtering: %d / %d rows kept", len(result), len(df))
    return result


def _std_wr(tbl, catalog_id: str, ref: str) -> pd.DataFrame:
    ra, dec = _parse_radec(tbl)
    names = _parse_name_col(tbl, ["WR", "Name", "Star"])
    return pd.DataFrame({
        "name": names, "Class": "HM-STAR",
        "ra": ra, "dec": dec,
        "source_catalog": catalog_id, "source_ref": ref,
        "label_confidence": 1.0, "SpType": "WN/WC",
    })


# (vizier_table_path, handler_fn, (catalog_id, class_or_marker, ref_label))
_CATALOG_DISPATCH: dict[str, tuple] = {
    "J/A+A/469/807":  ("J/A+A/469/807",   _std_xrb,    ("J/A+A/469/807",  "LMXB",    "Liu+2007")),
    "J/A+A/455/1165": ("J/A+A/455/1165",  _std_xrb,    ("J/A+A/455/1165", "HMXB",    "Liu+2006")),
    "B/cb/lmxbdata":  ("B/cb/lmxbdata",   _std_xrb,    ("B/cb/lmxbdata",  "LMXB",    "Ritter&Kolb2003")),
    "B/cb/cbdata":    ("B/cb/cbdata",     _std_cv,     ("B/cb/cbdata",    "CV",      "Ritter&Kolb2003")),
    "V/123A/cv":      ("V/123A/cv",       _std_cv,     ("V/123A/cv",      "CV",      "Downes+2001")),
    "VII/258":        ("VII/258",         _std_agn,    ("VII/258",        "AGN",     "VeronCetty+2010")),
    "B/mk":           ("B/mk",           _std_skiff,  ("B/mk",           "STAR",    "Skiff2014")),
    "III/284":        ("III/284",         _std_apogee, ("III/284",        "LM-STAR", "APOGEE2+DR16")),
    "III/215":        ("III/215",         _std_wr,     ("III/215",        "HM-STAR", "vanderHucht2001")),
    "J/A+A/458/453":  ("J/A+A/458/453",   _std_wr,     ("J/A+A/458/453",  "HM-STAR", "vanderHucht2006")),
}


def _call_handler(fn, args_tuple, tbl) -> pd.DataFrame:
    """Dispatch to handler, injecting class_label only if the handler takes 4 args."""
    n_params = len(inspect.signature(fn).parameters)
    catalog_id, class_label, ref = args_tuple
    if n_params == 4:
        return fn(tbl, class_label, catalog_id, ref)
    return fn(tbl, catalog_id, ref)


def _fetch_and_standardise(catalog_id: str) -> Optional[pd.DataFrame]:
    if catalog_id not in _CATALOG_DISPATCH:
        log.warning("No handler for catalog %s; skipping.", catalog_id)
        return None
    table_id, handler_fn, handler_args = _CATALOG_DISPATCH[catalog_id]
    try:
        tbl = fetch_vizier_catalog(table_id)
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", catalog_id, exc)
        return None
    try:
        df = _call_handler(handler_fn, handler_args, tbl)
    except Exception as exc:
        log.warning("Failed to standardise %s: %s", catalog_id, exc)
        return None
    if df is None or len(df) == 0:
        log.warning("Catalog %s: 0 rows after standardisation.", catalog_id)
        return None
    log.info("Standardised %d rows from %s", len(df), catalog_id)
    return df


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set (Union-Find) with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            result.setdefault(self.find(i), []).append(i)
        return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def build_master_resolved_table(
    df: pd.DataFrame,
    match_radius_arcsec: float = 5.0,
) -> pd.DataFrame:
    """Deduplicate and resolve a catalog into unique physical objects.

    Algorithm:
    1. For each Class subset, merge rows whose normalised names are identical
       (and non-empty).
    2. Use astropy search_around_sky within *match_radius_arcsec*; for each
       pair in the same Class merge if: same norm_name OR either name empty
       OR angular separation <= 1 arcsec.
    3. Per group: canonical_name = shortest name, position = median RA/Dec,
       concatenate catalogs/refs.

    Parameters
    ----------
    df : pd.DataFrame
        Columns required: name, Class, ra, dec, source_catalog, source_ref,
        label_confidence.  Optional: SpType.
    match_radius_arcsec : float
        Maximum sky proximity for proximity-based merging.

    Returns
    -------
    pd.DataFrame
        One row per resolved object with columns: master_group, Class,
        canonical_name, name_norm, ra, dec, n_input_rows, input_names,
        source_catalogs, source_refs, label_confidences, SpType, input_row_ids.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    norm_names = np.array([normalize_name(x) for x in df["name"]])
    uf = UnionFind(n)

    # Group by Class for efficiency
    class_groups: dict[str, list[int]] = {}
    for i, cls in enumerate(df["Class"]):
        class_groups.setdefault(str(cls), []).append(i)

    one_arcsec = 1.0 * u.arcsec

    for cls, idx_list in class_groups.items():
        if len(idx_list) < 2:
            continue

        sub_idx = np.array(idx_list)
        sub_norm = norm_names[sub_idx]

        # --- Name-based merge (no distance constraint) ---
        name_map: dict[str, list[int]] = {}
        for gi, nm in zip(sub_idx, sub_norm):
            if nm:
                name_map.setdefault(nm, []).append(int(gi))
        for members in name_map.values():
            for k in range(1, len(members)):
                uf.union(members[0], members[k])

        # --- Position-based merge ---
        sub_ra = df["ra"].values[sub_idx].astype(float)
        sub_dec = df["dec"].values[sub_idx].astype(float)
        valid = np.isfinite(sub_ra) & np.isfinite(sub_dec)
        if valid.sum() < 2:
            continue

        coords = SkyCoord(
            ra=sub_ra[valid] * u.deg,
            dec=sub_dec[valid] * u.deg,
            frame="icrs",
        )
        valid_global = sub_idx[valid]

        try:
            idx1, idx2, sep2d, _ = coords.search_around_sky(
                coords, match_radius_arcsec * u.arcsec
            )
        except Exception as exc:
            log.warning("search_around_sky failed for class %s: %s", cls, exc)
            continue

        for i_loc, j_loc, sep in zip(idx1, idx2, sep2d):
            if i_loc >= j_loc:
                continue
            ig = int(valid_global[i_loc])
            jg = int(valid_global[j_loc])
            ni, nj = norm_names[ig], norm_names[jg]
            same_norm = ni != "" and nj != "" and ni == nj
            either_empty = ni == "" or nj == ""
            close = sep <= one_arcsec
            if same_norm or either_empty or close:
                uf.union(ig, jg)

    # Aggregate groups
    def _join_unique(col_vals) -> str:
        seen: dict[str, None] = {}
        for v in col_vals:
            s = str(v)
            if pd.notna(v) and s not in ("nan", ""):
                seen[s] = None
        return "; ".join(seen)

    rows = []
    for gid, (_, members) in enumerate(uf.groups().items()):
        sub = df.iloc[members]
        non_empty_names = [str(nm) for nm in sub["name"] if normalize_name(nm)]
        canonical = min(non_empty_names, key=len) if non_empty_names else ""

        ra_v = sub["ra"].dropna().values.astype(float)
        dec_v = sub["dec"].dropna().values.astype(float)

        sptypes = (
            sub["SpType"].dropna() if "SpType" in sub.columns else pd.Series(dtype=str)
        )
        sptype_val = sptypes.iloc[0] if len(sptypes) else np.nan

        rows.append({
            "master_group": gid,
            "Class": str(sub["Class"].iloc[0]),
            "canonical_name": canonical,
            "name_norm": normalize_name(canonical),
            "ra": float(np.median(ra_v)) if len(ra_v) else np.nan,
            "dec": float(np.median(dec_v)) if len(dec_v) else np.nan,
            "n_input_rows": len(members),
            "input_names": "; ".join(str(nm) for nm in sub["name"]),
            "source_catalogs": _join_unique(sub["source_catalog"]),
            "source_refs": _join_unique(sub["source_ref"]),
            "label_confidences": _join_unique(sub["label_confidence"]),
            "SpType": sptype_val,
            "input_row_ids": "; ".join(str(m) for m in members),
        })

    result = pd.DataFrame(rows)
    log.info(
        "build_master_resolved_table: %d rows -> %d resolved objects", n, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Top-level catalog builder
# ---------------------------------------------------------------------------


def build_td_catalog(
    output_path: Optional[str | Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build the master training-dataset catalog from VizieR sources.

    Parameters
    ----------
    output_path : str or Path, optional
        Path to save (and cache) the result.
    use_cache : bool
        If True and *output_path* exists, load from file instead of querying.

    Returns
    -------
    pd.DataFrame
        Master resolved training dataset.
    """
    if use_cache and output_path is not None:
        p = Path(output_path)
        if p.exists():
            log.info("Loading TD catalog from cache: %s", p)
            return load_catalog(p)

    catalog_ids = [entry[0] for entry in config.VIZIER_CATALOGS]
    frames: list[pd.DataFrame] = []
    for cid in catalog_ids:
        df_cat = _fetch_and_standardise(cid)
        if df_cat is not None and len(df_cat) > 0:
            frames.append(df_cat)

    if not frames:
        raise RuntimeError("No training catalogs could be fetched from VizieR.")

    all_rows = pd.concat(frames, ignore_index=True)
    log.info("Combined %d rows from %d catalogs", len(all_rows), len(frames))

    before = len(all_rows)
    all_rows = all_rows.drop_duplicates(
        subset=["name", "Class", "source_catalog"], keep="first"
    )
    log.info("Light dedup: %d -> %d rows", before, len(all_rows))

    resolved = build_master_resolved_table(
        all_rows, match_radius_arcsec=config.MASTER_MATCH_RADIUS_ARCSEC
    )

    if output_path is not None:
        save_catalog(resolved, Path(output_path))

    return resolved


def describe_td(df: pd.DataFrame) -> None:
    """Print class counts, total rows, and missing fraction for key columns.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset to describe.
    """
    print(f"\n{'='*55}")
    print("  Training Dataset Summary")
    print(f"{'='*55}")
    print(f"  Total rows: {len(df):,}")
    print()
    print("  Class counts:")
    col = "Class" if "Class" in df.columns else "class_label"
    counts = df[col].value_counts()
    for cls in config.ALL_CLASSES:
        print(f"    {cls:<12} {counts.get(cls, 0):>6,}")
    extra = {k: v for k, v in counts.items() if k not in config.ALL_CLASSES}
    for cls, n in extra.items():
        print(f"    {cls:<12} {n:>6,}  (other)")
    print()
    key_cols = ["ra", "dec", "SpType", "source_catalogs", "source_refs"]
    present = [c for c in key_cols if c in df.columns]
    if present:
        print("  Missing fraction:")
        for c in present:
            print(f"    {c:<22} {df[c].isna().mean():.1%}")
    print(f"{'='*55}\n")

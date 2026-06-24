"""xclass.io — Data loading, saving, caching, and external resource retrieval.

All file I/O and network resource loading for spectra and filter curves lives
here.  Other modules must not import astroquery or requests directly for
resource loading; use the functions below instead.

Functions
---------
load_catalog
save_catalog
load_filter_curve
get_filter_pivot
load_all_filter_curves
load_pickles_spectrum
load_agn_composite
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------


def load_catalog(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a catalog from CSV or FITS, auto-detected by file suffix.

    Parameters
    ----------
    path : str or Path
        File path.  Suffix must be ``.csv``, ``.fits``, or ``.fit``.
    **kwargs
        Passed through to ``pd.read_csv`` or ``astropy.table.Table.read``.

    Returns
    -------
    pd.DataFrame
        Loaded catalog.  Masked columns from FITS are converted to plain
        float arrays with NaN filling.

    Raises
    ------
    ValueError
        If the file suffix is not recognised.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
        log.info("Loaded %d rows from %s", len(df), path)
        return df

    if suffix in {".fits", ".fit"}:
        from astropy.table import Table

        tbl = Table.read(path, **kwargs)
        # Convert masked columns to plain numpy arrays with NaN
        for col in tbl.colnames:
            if hasattr(tbl[col], "filled"):
                try:
                    tbl[col] = tbl[col].filled(np.nan)
                except (TypeError, ValueError):
                    # Non-numeric masked column — fill with empty string
                    try:
                        tbl[col] = tbl[col].filled("")
                    except Exception:
                        pass
        df = tbl.to_pandas()
        log.info("Loaded %d rows from %s (FITS)", len(df), path)
        return df

    raise ValueError(
        f"Unrecognised file suffix '{suffix}' for {path}. "
        "Expected .csv, .fits, or .fit"
    )


def save_catalog(df: pd.DataFrame, path: str | Path, overwrite: bool = True) -> None:
    """Save a DataFrame to CSV or FITS.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : str or Path
        Destination path.  Suffix determines format (``.csv`` or ``.fits``).
    overwrite : bool
        If *True* (default), overwrite an existing file.  If *False* and the
        file exists, raise ``FileExistsError``.

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is False.
    ValueError
        If the file suffix is not recognised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Pass overwrite=True.")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        log.info("Saved %d rows to %s", len(df), path)
        return

    if suffix in {".fits", ".fit"}:
        from astropy.table import Table

        tbl = Table.from_pandas(df)
        tbl.write(str(path), overwrite=overwrite)
        log.info("Saved %d rows to %s (FITS)", len(df), path)
        return

    raise ValueError(
        f"Unrecognised file suffix '{suffix}' for {path}. "
        "Expected .csv or .fits"
    )


# ---------------------------------------------------------------------------
# Filter curves
# ---------------------------------------------------------------------------


def load_filter_curve(
    filter_id: str,
    cache_dir: str | Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load an HST filter transmission curve, using local cache if available.

    The curve is downloaded from the SVO Filter Profile Service via
    ``astroquery.svo_fps`` and cached as a pickle file.

    Parameters
    ----------
    filter_id : str
        SVO filter identifier, e.g. ``"HST/ACS_WFC.F475W"``.
    cache_dir : str or Path
        Directory for caching filter curves.

    Returns
    -------
    wave_AA : np.ndarray or None
        Wavelength array in Angstroms.
    throughput : np.ndarray or None
        Dimensionless throughput in [0, 1].
        Returns ``(None, None)`` if download or parsing fails.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filter_id for use as filename (replace / and . with _)
    safe_name = filter_id.replace("/", "_").replace(".", "_")
    cache_path = cache_dir / f"{safe_name}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            wave_AA, thru = pickle.load(fh)
        log.debug("Filter curve cache hit: %s", filter_id)
        return wave_AA, thru

    # Download from SVO
    try:
        from astroquery.svo_fps import SvoFps  # type: ignore[import]

        tbl = SvoFps.get_transmission_data(filter_id)
        wave_AA = np.asarray(tbl["Wavelength"], dtype=float)
        thru = np.asarray(tbl["Transmission"], dtype=float)

        # Clamp throughput to [0, 1]
        thru = np.clip(thru, 0.0, 1.0)

        # Sort by wavelength (SVO usually returns sorted, but be safe)
        order = np.argsort(wave_AA)
        wave_AA, thru = wave_AA[order], thru[order]

        with open(cache_path, "wb") as fh:
            pickle.dump((wave_AA, thru), fh)

        log.info("Downloaded and cached filter curve: %s", filter_id)
        return wave_AA, thru

    except Exception as exc:
        warnings.warn(
            f"Failed to load filter curve for '{filter_id}': {exc}",
            stacklevel=2,
        )
        log.warning("Filter curve download failed for %s: %s", filter_id, exc)
        return None, None


def get_filter_pivot(wave_AA: np.ndarray, thru: np.ndarray) -> float:
    """Compute the pivot wavelength of a filter.

    The pivot wavelength is defined as::

        lambda_p = sqrt( integral(T dlambda) / integral(T / lambda^2 dlambda) )

    Parameters
    ----------
    wave_AA : np.ndarray
        Wavelength array in Angstroms.
    thru : np.ndarray
        Dimensionless throughput.

    Returns
    -------
    float
        Pivot wavelength in Angstroms.
    """
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy ≥2.0 renamed trapz→trapezoid
    num = _trapz(thru, wave_AA)
    denom = _trapz(thru / wave_AA**2, wave_AA)
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(num / denom))


def load_all_filter_curves(
    filter_dict: dict[str, str],
    cache_dir: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all filters in *filter_dict*, using cache when available.

    Parameters
    ----------
    filter_dict : dict[str, str]
        Mapping of short filter label to SVO filter ID,
        e.g. ``{"ACS_F475W": "HST/ACS_WFC.F475W"}``.
    cache_dir : str or Path
        Directory for caching filter curves.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of short label to ``(wave_AA, throughput)``.
        Filters that could not be loaded are **omitted** from the result
        (a warning is logged for each failure).
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    failed: list[str] = []

    for label, svo_id in filter_dict.items():
        wave, thru = load_filter_curve(svo_id, cache_dir)
        if wave is None:
            failed.append(label)
        else:
            curves[label] = (wave, thru)

    if failed:
        log.warning(
            "Failed to load %d filter(s): %s", len(failed), ", ".join(failed)
        )
    log.info(
        "Loaded %d/%d filter curves successfully.",
        len(curves),
        len(filter_dict),
    )
    return curves


# ---------------------------------------------------------------------------
# Stellar spectral templates (Pickles atlas)
# ---------------------------------------------------------------------------

# Map spectral type keys to Pickles FITS filenames (without extension)
# The atlas is available from STScI via synphot or direct download.
PICKLES_FILENAME_MAP: dict[str, str] = {
    "O5": "pickles_uk_1",
    "O9": "pickles_uk_2",
    "B0": "pickles_uk_3",
    "B2": "pickles_uk_5",
    "B5": "pickles_uk_6",
    "B8": "pickles_uk_7",
    "B9": "pickles_uk_8",
    "A0": "pickles_uk_9",
    "A2": "pickles_uk_10",
    "A5": "pickles_uk_11",
    "F0": "pickles_uk_12",
    "F5": "pickles_uk_14",
    "F8": "pickles_uk_15",
    "G0": "pickles_uk_16",
    "G2": "pickles_uk_18",
    "G5": "pickles_uk_20",
    "G8": "pickles_uk_23",
    "K0": "pickles_uk_26",
    "K2": "pickles_uk_28",
    "K5": "pickles_uk_31",
    "K7": "pickles_uk_33",
    "M0": "pickles_uk_34",
    "M2": "pickles_uk_36",
    "M3": "pickles_uk_37",
    "M4": "pickles_uk_38",
    "M5": "pickles_uk_40",
    "M6": "pickles_uk_41",
}

_PICKLES_STScI_BASE = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/"
)
_PICKLES_STScI_BASE_LEGACY = (
    "https://archive.stsci.edu/pub/cdbs/grid/pickles/dat_uvk/"
)


def load_pickles_spectrum(
    sptype_key: str,
    cache_dir: str | Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a Pickles stellar spectrum as (wavelength_AA, f_nu).

    Tries synphot (PYSYN_CDBS) first; falls back to direct STScI download.
    Results are cached locally.

    Parameters
    ----------
    sptype_key : str
        Spectral type key from ``PICKLES_FILENAME_MAP``, e.g. ``"K5"``.
    cache_dir : str or Path
        Directory for caching spectra.

    Returns
    -------
    wave_AA : np.ndarray or None
    fnu : np.ndarray or None
        Flux density in f_nu (arbitrary normalisation).
        Returns ``(None, None)`` on failure.

    Raises
    ------
    ValueError
        If *sptype_key* is not in ``PICKLES_FILENAME_MAP``.
    """
    if sptype_key not in PICKLES_FILENAME_MAP:
        raise ValueError(
            f"Unknown Pickles spectral type key: '{sptype_key}'. "
            f"Valid keys: {sorted(PICKLES_FILENAME_MAP)}"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = PICKLES_FILENAME_MAP[sptype_key]
    cache_path = cache_dir / f"{fname}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            wave_AA, fnu = pickle.load(fh)
        log.debug("Pickles cache hit: %s (%s)", sptype_key, fname)
        return wave_AA, fnu

    # --- Try synphot first ---
    if os.environ.get("PYSYN_CDBS"):
        try:
            import synphot  # type: ignore[import]
            from synphot.spectrum import SourceSpectrum  # type: ignore[import]

            sp = SourceSpectrum.from_file(
                f"$PYSYN_CDBS/grid/pickles/dat_uvk/{fname}.fits"
            )
            wave_AA = sp.waveset.to("AA").value  # type: ignore[union-attr]
            fnu = sp(sp.waveset, flux_unit="Jy").value  # type: ignore[arg-type]
            _cache_pickles(cache_path, wave_AA, fnu)
            log.info("Loaded Pickles %s via synphot", sptype_key)
            return wave_AA, fnu
        except Exception as exc:
            log.debug("synphot load failed for %s: %s; trying direct download", sptype_key, exc)

    # --- Fall back to direct STScI download (try new path, then legacy) ---
    try:
        import requests
        from astropy.io import fits as afits

        resp = None
        for base in (_PICKLES_STScI_BASE, _PICKLES_STScI_BASE_LEGACY):
            url = f"{base}{fname}.fits"
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    resp = r
                    break
                log.debug("Pickles URL %s returned %s", url, r.status_code)
            except Exception as _exc:
                log.debug("Pickles URL %s failed: %s", url, _exc)
        if resp is None:
            raise RuntimeError(f"All Pickles download URLs failed for {fname}")
        resp.raise_for_status()

        fits_path = cache_dir / f"{fname}.fits"
        fits_path.write_bytes(resp.content)

        with afits.open(fits_path) as hdul:
            wave_AA = np.asarray(hdul[1].data["WAVELENGTH"], dtype=float)
            fnu = np.asarray(hdul[1].data["FLUX"], dtype=float)

        _cache_pickles(cache_path, wave_AA, fnu)
        log.info("Downloaded Pickles %s from STScI", sptype_key)
        return wave_AA, fnu

    except Exception as exc:
        warnings.warn(
            f"Failed to load Pickles spectrum for '{sptype_key}': {exc}",
            stacklevel=2,
        )
        log.warning("Pickles download failed for %s: %s", sptype_key, exc)
        return None, None


def _cache_pickles(path: Path, wave_AA: np.ndarray, fnu: np.ndarray) -> None:
    """Persist Pickles spectrum to local pickle cache."""
    with open(path, "wb") as fh:
        pickle.dump((wave_AA, fnu), fh)


def load_pickles_cache(
    cache_dir: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all Pickles spectral templates into a dict keyed by spectral type.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching spectra.  Defaults to ``config.SPECTRA_CACHE_DIR``.

    Returns
    -------
    dict
        Mapping of spectral-type key (e.g. ``"K5"``) to
        ``(wave_AA, fnu)`` arrays.  Entries that fail to load are omitted.
    """
    from xclass import config as _cfg

    cdir = Path(cache_dir) if cache_dir else _cfg.SPECTRA_CACHE_DIR
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in PICKLES_FILENAME_MAP:
        wave, fnu = load_pickles_spectrum(key, cdir)
        if wave is not None:
            cache[key] = (wave, fnu)
    log.info("load_pickles_cache: loaded %d/%d Pickles templates",
             len(cache), len(PICKLES_FILENAME_MAP))
    return cache


# Convenience alias used by run_pipeline.py
load_filter_curves = load_all_filter_curves


# ---------------------------------------------------------------------------
# AGN composite spectrum (Vanden Berk et al. 2001)
# ---------------------------------------------------------------------------

_AGN_COMPOSITE_URLS: list[str] = [
    # STScI CDBS — new path (hlsps/reference-atlases)
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/agn/vandenberk.fits",
    # Legacy STScI path (may redirect or 404 depending on server)
    "https://archive.stsci.edu/pub/cdbs/grid/agn/vandenberk.fits",
]
_AGN_COMPOSITE_CACHE_NAME = "vandenberk2001_agn_composite.pkl"


def load_agn_composite(
    cache_dir: str | Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load the Vanden Berk 2001 SDSS AGN composite spectrum.

    Tries synphot first, then local cache, then direct download.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for caching the spectrum.

    Returns
    -------
    wave_AA : np.ndarray or None
    fnu : np.ndarray or None
        Flux density in f_nu (arbitrary normalisation).
        Returns ``(None, None)`` after all attempts fail.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _AGN_COMPOSITE_CACHE_NAME

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            wave_AA, fnu = pickle.load(fh)
        log.debug("AGN composite cache hit")
        return wave_AA, fnu

    # --- Try synphot ---
    if os.environ.get("PYSYN_CDBS"):
        try:
            from synphot.spectrum import SourceSpectrum  # type: ignore[import]

            sp = SourceSpectrum.from_file("$PYSYN_CDBS/grid/agn/vandenberk.fits")
            wave_AA = sp.waveset.to("AA").value  # type: ignore[union-attr]
            fnu = sp(sp.waveset, flux_unit="Jy").value  # type: ignore[arg-type]
            with open(cache_path, "wb") as fh:
                pickle.dump((wave_AA, fnu), fh)
            log.info("Loaded AGN composite via synphot")
            return wave_AA, fnu
        except Exception as exc:
            log.debug("synphot AGN composite failed: %s", exc)

    # --- Try direct download ---
    for url in _AGN_COMPOSITE_URLS:
        try:
            import requests
            from astropy.io import fits as afits

            resp = requests.get(url, timeout=60)
            resp.raise_for_status()

            fits_path = cache_dir / "vandenberk.fits"
            fits_path.write_bytes(resp.content)

            with afits.open(fits_path) as hdul:
                data = hdul[1].data
                wave_AA = np.asarray(data["WAVELENGTH"], dtype=float)
                fnu = np.asarray(data["FLUX"], dtype=float)

            with open(cache_path, "wb") as fh:
                pickle.dump((wave_AA, fnu), fh)
            log.info("Downloaded AGN composite from %s", url)
            return wave_AA, fnu

        except Exception as exc:
            log.debug("AGN composite download failed from %s: %s", url, exc)

    warnings.warn(
        "Failed to load Vanden Berk 2001 AGN composite spectrum after all attempts. "
        "SED fitting for AGN will fall back to power-law grid.",
        stacklevel=2,
    )
    return None, None

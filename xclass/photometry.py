"""xclass.photometry — Filter convolution, AB magnitudes, HST translation.  [C2]

Converts SED models to synthetic photometry and translates observed
ground-based photometry into predicted HST magnitudes for the full
universal filter set.

Functions
---------
convolve_sed_through_filter
fnu_to_abmag
abmag_to_fnu
magerr_to_fnuerr
translate_source_to_hst
translate_catalog
fnu_to_nuFnu
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xclass import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SNR HSC bypass — SNRs carry real HST photometry (nothing to SED-translate);
# their six PHAT *_pred magnitudes are filled directly from HSC v3.
# ---------------------------------------------------------------------------
# PHAT _pred column prefix -> HSC v3 short filter name.
_SNR_HSC_FILTER_MAP: dict[str, str] = {
    "UVIS_F275W": "F275W",
    "UVIS_F336W": "F336W",
    "ACS_F475W":  "F475W",
    "ACS_F814W":  "F814W",
    "IR_F110W":   "F110W",
    "IR_F160W":   "F160W",
}
_SNR_HSC_SEARCH_RADIUS_ARCSEC: float = 0.5

# AB system zero-point constant (CGS): m = -2.5*log10(f_nu) - 48.60
# where f_nu is in erg/s/cm^2/Hz
_AB_ZP = 48.60

# ln(10)/2.5 factor for mag-error to fnu-error conversion
_LN10_OVER_2P5 = np.log(10.0) / 2.5

# Columns in the cross-matched TD DataFrame and their photometric survey
# identifiers.  The 'zp_key' must be a key in config.ZEROPOINTS_JY or
# the special value 'ab' (AB system: ZP = 3631 Jy).
_SURVEY_BANDS: list[dict] = [
    {"col": "ps1_g", "err_col": "ps1_g_err", "zp_key": "ps1_g"},
    {"col": "ps1_r", "err_col": "ps1_r_err", "zp_key": "ps1_r"},
    {"col": "ps1_i", "err_col": "ps1_i_err", "zp_key": "ps1_i"},
    {"col": "ps1_z", "err_col": "ps1_z_err", "zp_key": "ps1_z"},
    {"col": "ps1_y", "err_col": "ps1_y_err", "zp_key": "ps1_y"},
    {"col": "tmass_j", "err_col": "tmass_j_err", "zp_key": "tmass_j"},
    {"col": "tmass_h", "err_col": "tmass_h_err", "zp_key": "tmass_h"},
    {"col": "tmass_k", "err_col": "tmass_k_err", "zp_key": "tmass_k"},
]

# Conversion factors from Vega/AB systems to CGS f_nu (erg/s/cm^2/Hz)
# For AB: f_nu(Jy) = 3631 * 10^(-mag/2.5)
#         f_nu(CGS) = f_nu(Jy) * 1e-23
# For Vega-based (2MASS): f_nu(Jy) = ZP_Jy * 10^(-mag/2.5)
_AB_ZP_JY = 3631.0


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


def fnu_to_abmag(fnu: float) -> float:
    """Convert f_nu (erg/s/cm^2/Hz) to AB magnitude.

    Parameters
    ----------
    fnu : float
        Flux density.

    Returns
    -------
    float
        AB magnitude: ``m = -2.5 * log10(fnu) - 48.60``.
        Returns NaN for non-positive *fnu*.
    """
    if not np.isfinite(fnu) or fnu <= 0:
        return float("nan")
    return -2.5 * np.log10(fnu) - _AB_ZP


def abmag_to_fnu(mag: float) -> float:
    """Convert AB magnitude to f_nu (erg/s/cm^2/Hz).

    Parameters
    ----------
    mag : float
        AB magnitude.

    Returns
    -------
    float
        Flux density: ``10^(-(mag + 48.60) / 2.5)``.
    """
    if not np.isfinite(mag):
        return float("nan")
    return 10.0 ** (-(mag + _AB_ZP) / 2.5)


def magerr_to_fnuerr(mag: float, mag_err: float) -> float:
    """Convert magnitude error to f_nu error via linearisation.

    Parameters
    ----------
    mag : float
        AB magnitude.
    mag_err : float
        1-sigma magnitude uncertainty.

    Returns
    -------
    float
        Flux uncertainty: ``f_nu * (ln(10) / 2.5) * mag_err``.
    """
    fnu = abmag_to_fnu(mag)
    if not np.isfinite(fnu) or fnu <= 0:
        return float("nan")
    return fnu * _LN10_OVER_2P5 * float(mag_err)


def convolve_sed_through_filter(
    fnu: np.ndarray,
    wave_AA: np.ndarray,
    filter_wave: np.ndarray,
    filter_thru: np.ndarray,
) -> float:
    """Compute mean flux through a bandpass filter.

    Uses the mean-flux convention::

        <f_nu> = integral(f_nu * T * dlambda) / integral(T * dlambda)

    Integration via ``numpy.trapz``.

    Parameters
    ----------
    fnu : np.ndarray
        Spectral flux density on *wave_AA* grid.
    wave_AA : np.ndarray
        Wavelength grid in Angstroms.
    filter_wave : np.ndarray
        Filter wavelength grid in Angstroms.
    filter_thru : np.ndarray
        Filter throughput in [0, 1].

    Returns
    -------
    float
        Mean flux density through the filter (same units as *fnu*).
        Returns NaN if the filter does not overlap the SED grid.
    """
    mask = (wave_AA >= filter_wave.min()) & (wave_AA <= filter_wave.max())
    if mask.sum() < 2:
        return float("nan")

    w = wave_AA[mask]
    f = fnu[mask]
    t = np.interp(w, filter_wave, filter_thru, left=0.0, right=0.0)

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy ≥2.0 renamed trapz→trapezoid
    denom = float(_trapz(t, w))
    if denom <= 0:
        return float("nan")

    numer = float(_trapz(f * t, w))
    return numer / denom


def fnu_to_nuFnu(fnu: float, pivot_angstrom: float) -> float:
    """Convert f_nu to nu * f_nu for X-ray/optical SED ratio features.

    Parameters
    ----------
    fnu : float
        Spectral flux density in f_nu.
    pivot_angstrom : float
        Filter pivot wavelength in Angstroms.

    Returns
    -------
    float
        ``nu * f_nu`` where ``nu = c / lambda``.
    """
    from xclass.sed import C_AA
    nu = C_AA / float(pivot_angstrom)
    return float(fnu) * nu


# ---------------------------------------------------------------------------
# Per-source SED translation
# ---------------------------------------------------------------------------


def _mag_to_fnu_cgs(mag: float, zp_key: str) -> float:
    """Convert observed magnitude to f_nu in CGS using survey zeropoint."""
    if not np.isfinite(mag):
        return float("nan")
    # Guard against overflow: mag << 0 (very bright) or mag > ~900 (too faint)
    # 10^(-mag/2.5) overflows float64 for mag < ~-1800; clamp to a safe range.
    if mag < -100 or mag > 100:
        return float("nan")
    zp_jy = config.ZEROPOINTS_JY.get(zp_key, _AB_ZP_JY)
    # f_nu in Jy, then convert to CGS (erg/s/cm^2/Hz)
    fnu_jy = zp_jy * 10.0 ** (-mag / 2.5)
    return fnu_jy * 1e-23  # Jy -> erg/s/cm^2/Hz


def _fnu_cgs_to_abmag(fnu_cgs: float) -> float:
    """Convert CGS f_nu to AB magnitude."""
    return fnu_to_abmag(fnu_cgs)


def translate_source_to_hst(
    row: pd.Series,
    filter_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    all_filter_names: list[str],
    pickles_cache: dict,
    agn_composite: Optional[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Translate one training-dataset row to predicted HST magnitudes.

    Runs ``sed.fit_sed`` then evaluates the best model on a dense wavelength
    grid and convolves through every filter in *all_filter_names*.

    UV filters (F275W, F336W) have ``config.UV_SYSTEMATIC_ERR_MAG`` added
    in quadrature to the predicted uncertainty.

    Parameters
    ----------
    row : pd.Series
        One row from the cross-matched training dataset.
    filter_curves : dict
        Full filter curve dictionary (input survey + HST output filters).
    all_filter_names : list of str
        Ordered list of output filter labels (HST filters to predict).
    pickles_cache : dict
        Pre-loaded Pickles spectra.
    agn_composite : tuple or None
        Pre-loaded AGN composite spectrum.

    Returns
    -------
    dict
        Keys for each filter: ``{filter}_pred``, ``{filter}_pred_err``.
        Plus SED metadata: xclass_sed_family, xclass_sed_param,
        xclass_fit_chi2red, xclass_n_bands_used, xclass_av_used,
        xclass_status_flag, xclass_bands_used.
    """
    from xclass.sed import fit_sed

    # Build observed magnitudes from survey columns present in row
    obs_mags: dict[str, float] = {}
    obs_errs: dict[str, float] = {}

    for band_info in _SURVEY_BANDS:
        col = band_info["col"]
        err_col = band_info["err_col"]

        if col not in row.index:
            continue
        mag = float(row[col]) if row[col] is not None else float("nan")
        if not np.isfinite(mag):
            continue

        # Default uncertainty when not provided
        err = 0.1
        if err_col and err_col in row.index:
            e = float(row[err_col]) if row[err_col] is not None else float("nan")
            if np.isfinite(e) and e > 0:
                err = e

        # Convert Vega/AB mag to CGS fnu, then back to AB mag for consistent fitting
        zp_key = band_info["zp_key"]
        fnu_cgs = _mag_to_fnu_cgs(mag, zp_key)
        if not np.isfinite(fnu_cgs) or fnu_cgs <= 0:
            continue

        # Store as "effective AB mag" for fitting (all in CGS now)
        eff_abmag = fnu_to_abmag(fnu_cgs)
        if np.isfinite(eff_abmag) and col in filter_curves:
            obs_mags[col] = eff_abmag
            obs_errs[col] = err

    # Source class and spectral type
    source_class = str(row.get("Class", row.get("class_label", "LM-STAR")))
    sptype = str(row.get("SpType", "")) if "SpType" in row.index else None

    # Extinction is not applied in the production SED fit (A_V = 0).
    av = 0.0

    # Fit SED
    sed_result = fit_sed(
        obs_mags=obs_mags,
        obs_errs=obs_errs,
        filter_curves=filter_curves,
        source_class=source_class,
        sptype=sptype,
        pickles_cache=pickles_cache,
        agn_composite=agn_composite,
        av=av,
    )

    # Evaluate SED through each output HST filter
    fnu_model = sed_result["fnu_model"]
    wave_dense = np.arange(
        config.SED_WAVE_MIN_AA,
        config.SED_WAVE_MAX_AA + config.SED_WAVE_STEP_AA,
        config.SED_WAVE_STEP_AA,
        dtype=float,
    )

    result: dict = {
        "xclass_sed_family": sed_result["sed_family"],
        "xclass_sed_param": sed_result["sed_param"],
        "xclass_fit_chi2red": sed_result["chi2_reduced"],
        "xclass_n_bands_used": sed_result["n_bands_used"],
        "xclass_av_used": sed_result["av_used"],
        "xclass_status_flag": sed_result["status_flag"],
        "xclass_bands_used": ";".join(obs_mags.keys()),
    }

    chi2r = sed_result.get("chi2_reduced", float("nan"))
    base_err = 0.1 * (np.sqrt(chi2r) if np.isfinite(chi2r) and chi2r > 0 else 1.0)
    base_err = min(base_err, 1.0)  # cap at 1 mag

    for filt_name in all_filter_names:
        pred_col = f"{filt_name}_pred"
        err_col = f"{filt_name}_pred_err"

        if filt_name not in filter_curves:
            result[pred_col] = float("nan")
            result[err_col] = float("nan")
            continue

        fw, ft = filter_curves[filt_name]
        syn_fnu = convolve_sed_through_filter(fnu_model, wave_dense, fw, ft)
        pred_mag = fnu_to_abmag(syn_fnu) if np.isfinite(syn_fnu) and syn_fnu > 0 else float("nan")

        # UV systematic error (F275W, F336W)
        is_uv = "F275" in filt_name or "F336" in filt_name
        pred_err = float(
            np.sqrt(base_err ** 2 + config.UV_SYSTEMATIC_ERR_MAG ** 2)
            if is_uv else base_err
        )

        result[pred_col] = pred_mag
        result[err_col] = pred_err

    return result


# ---------------------------------------------------------------------------
# Catalog-level translation  [C2]
# ---------------------------------------------------------------------------


def _snr_row_mask(out: pd.DataFrame) -> pd.Series:
    """Boolean mask of SNR rows, robust to whether the label lives in ``Class``,
    ``td_Class`` or ``class_label`` (a fresh crossmatch leaves SNR rows with only
    ``td_Class`` set)."""
    mask = pd.Series(False, index=out.index)
    for col in ("Class", "td_Class", "class_label"):
        if col in out.columns:
            mask = mask | (out[col].astype(str) == "SNR")
    return mask


def apply_snr_hsc_bypass(
    out: pd.DataFrame,
    cache_dir: str | Path,
    search_radius_arcsec: float = _SNR_HSC_SEARCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Fill the six PHAT ``*_pred`` columns for SNR rows with real HSC v3 magnitudes.

    SNRs bypass SED translation — they already carry real HST photometry — so we
    query HSC v3 at each SNR's optical position (``td_ra``/``td_dec``) and slot the
    magnitudes straight into the ``_pred`` columns (nearest match by ``D``; median
    ``MagAper2`` per filter within 0.5"), matching the historical
    ``scripts/fix_snr_hst_photometry.py`` behaviour.

    Fails loudly rather than silently dropping the SNR class:

    * **partial** coverage -> ``WARNING`` naming how many SNRs were lost and why;
    * **total** SNR loss    -> ``RuntimeError`` (a 3-class model would not reproduce
      the published 4-class catalog).

    Network failures (``_hsc_fetch_one`` returns ``None``) are counted separately
    from genuine no-counterpart responses so an HSC outage is unambiguous.  *out*
    is modified in place and returned.
    """
    from xclass.query import _hsc_fetch_one

    snr_mask = _snr_row_mask(out)
    n_snr = int(snr_mask.sum())
    if n_snr == 0:
        log.info("SNR HSC bypass: no SNR rows in catalog — nothing to do.")
        return out
    if "td_canonical_name" not in out.columns or \
            "td_ra" not in out.columns or "td_dec" not in out.columns:
        raise RuntimeError(
            "SNR HSC bypass: catalog has SNR rows but is missing "
            "td_canonical_name / td_ra / td_dec — cannot query HSC. "
            "Was the SNR crossmatch (_build_snr_xray_rows) run before translate?"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    phat = {k: v for k, v in _SNR_HSC_FILTER_MAP.items() if f"{k}_pred" in out.columns}

    # Coerce target dtypes so scalar assignment is safe under pandas' strict
    # setitem (>=2.x): the _pred columns must accept floats, and the status flag
    # column (int-typed from SED fitting) must accept the 'real_hsc_photometry'
    # string marker.
    for pred_key in phat:
        col = f"{pred_key}_pred"
        if not pd.api.types.is_float_dtype(out[col].dtype):
            out[col] = out[col].astype(float)
    if "xclass_status_flag" in out.columns:
        out["xclass_status_flag"] = out["xclass_status_flag"].astype(object)

    log.info("SNR HSC bypass: querying HSC v3 for %d SNR sources (cache=%s)…",
             n_snr, cache_dir)

    n_patched = n_net_fail = n_no_counterpart = 0
    for idx in out.index[snr_mask]:
        sid = str(out.at[idx, "td_canonical_name"])
        try:
            ra = float(out.at[idx, "td_ra"]); dec = float(out.at[idx, "td_dec"])
        except (TypeError, ValueError):
            n_no_counterpart += 1
            continue

        dets = _hsc_fetch_one(sid, ra, dec, search_radius_arcsec, cache_dir)
        if dets is None:                    # network / API failure (not cached)
            n_net_fail += 1
            continue
        if not dets:                        # genuine empty response (cached)
            n_no_counterpart += 1
            continue

        # nearest match by 'D'
        best_id, min_d = None, float("inf")
        for det in dets:
            try:
                d_f = float(det.get("D", float("inf")))
            except (TypeError, ValueError):
                continue
            if d_f < min_d:
                min_d, best_id = d_f, det.get("MatchID")
        if best_id is None:
            best_id = dets[0].get("MatchID")

        mags: dict[str, list[float]] = {v: [] for v in phat.values()}
        for det in dets:
            if det.get("MatchID") != best_id:
                continue
            raw = str(det.get("Filter", "")).strip()
            fshort = raw.split("/")[-1] if "/" in raw else raw
            if fshort in mags:
                try:
                    mags[fshort].append(float(det["MagAper2"]))
                except (TypeError, ValueError, KeyError):
                    pass

        n_filt = 0
        for pred_key, fshort in phat.items():
            vals = mags[fshort]
            if vals:
                out.at[idx, f"{pred_key}_pred"] = float(np.median(vals))
                n_filt += 1
        if n_filt > 0:
            n_patched += 1
            if "xclass_status_flag" in out.columns:
                out.at[idx, "xclass_status_flag"] = "real_hsc_photometry"
        else:
            n_no_counterpart += 1

    n_lost = n_snr - n_patched
    log.info("SNR HSC bypass: %d/%d SNRs patched with >=1 PHAT filter "
             "(%d no-counterpart, %d network failures).",
             n_patched, n_snr, n_no_counterpart, n_net_fail)

    if n_patched == 0:
        # Total SNR loss -> the optical-baseline cut would drop the whole SNR class,
        # silently yielding a 3-class model.  Refuse instead.
        if n_net_fail > 0:
            raise RuntimeError(
                f"SNR HSC bypass: ALL {n_snr} SNR HSC queries failed "
                f"({n_net_fail} network/API errors, {n_no_counterpart} empty). HSC "
                f"(MAST) appears UNREACHABLE, so the SNR class would be dropped and "
                f"the run would NOT reproduce the published 4-class catalog. Re-run "
                f"when HSC is reachable, or point --snr-hsc-cache at a populated "
                f"cache (e.g. data/query_cache/hsc_snr_fix)."
            )
        raise RuntimeError(
            f"SNR HSC bypass: 0/{n_snr} SNRs have any HSC PHAT-filter counterpart. "
            f"The SNR class would be dropped, giving a 3-class model that does NOT "
            f"reproduce the published catalog. Check SNR positions / HSC coverage."
        )
    if n_lost > 0:
        log.warning(
            "SNR HSC bypass: %d/%d SNRs have NO usable HSC PHAT photometry "
            "(%d network failures, %d genuine no-counterpart) and will be dropped by "
            "the optical-baseline cut. The published run recovered 108/150; results "
            "reproduce the paper only insofar as SNR coverage matches. If the losses "
            "are network failures, re-run when HSC is reachable (failures are NOT "
            "cached) or supply --snr-hsc-cache.",
            n_lost, n_snr, n_net_fail, n_no_counterpart,
        )
    return out


def _translate_cache_fingerprint(
    df: pd.DataFrame,
    all_filter_names: list[str],
    agn_composite: Optional[tuple[np.ndarray, np.ndarray]],
) -> str:
    """Content fingerprint of the *inputs* to a translation run.

    A cache is only reused if its sidecar manifest carries a matching fingerprint,
    so a stale cache built from different inputs (e.g. a pre-SNR snapshot) can never
    be silently reused on mere path existence.
    """
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(",".join(map(str, sorted(map(str, df.columns)))).encode())
    try:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except Exception:
        for c in ("td_canonical_name", "td_ra", "td_dec", "Class", "td_Class"):
            if c in df.columns:
                h.update(df[c].astype(str).str.cat(sep="|").encode())
    h.update(",".join(map(str, config.PHAT_FILTER_SET)).encode())
    h.update(",".join(map(str, all_filter_names)).encode())
    h.update(f"agn_composite={agn_composite is not None}".encode())
    # Bump this tag whenever the translation logic changes semantics.
    h.update(b"translate_logic=snr_hsc_bypass_v1")
    return h.hexdigest()


def _cache_manifest_path(cache_path: str | Path) -> Path:
    return Path(str(cache_path) + ".manifest.json")


def _read_cache_manifest(cache_path: str | Path) -> Optional[dict]:
    p = _cache_manifest_path(cache_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache_manifest(cache_path: str | Path, fingerprint: str, n_rows: int) -> None:
    try:
        _cache_manifest_path(cache_path).write_text(
            json.dumps({"fingerprint": fingerprint, "n_rows": int(n_rows)})
        )
    except Exception as exc:
        log.warning("translate_catalog: could not write cache manifest: %s", exc)


def translate_catalog(
    df: pd.DataFrame,
    filter_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    all_filter_names: list[str],
    pickles_cache: dict,
    agn_composite: Optional[tuple[np.ndarray, np.ndarray]],
    n_jobs: int = -1,
    cache_path: Optional[str] = None,
    force: bool = False,
    snr_hsc: bool = True,
    snr_hsc_cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Translate all rows in *df* to the universal HST filter set.  [C2]

    Applies ``translate_source_to_hst`` to every row using
    ``joblib.Parallel`` with a ``tqdm`` progress bar.

    Cache reuse is gated on a content *fingerprint* of the inputs (not mere path
    existence): a cache is only reused if its sidecar ``*.manifest.json`` carries a
    fingerprint matching the current inputs.  Pass ``force=True``
    (``--force-retranslate``) to bypass the cache unconditionally.

    Parameters
    ----------
    df : pd.DataFrame
        Cross-matched training dataset.
    filter_curves : dict
        Full filter curve dictionary.
    all_filter_names : list of str
        Ordered list of output filter labels.
    pickles_cache : dict
        Pre-loaded Pickles spectra.
    agn_composite : tuple or None
        Pre-loaded AGN composite spectrum.
    n_jobs : int
        Passed to ``joblib.Parallel``.  -1 uses all available cores.
    cache_path : str, optional
        Path for result caching (CSV or FITS).  A ``<cache_path>.manifest.json``
        fingerprint sidecar is written alongside it.
    force : bool
        If True, ignore any existing cache and retranslate.
    snr_hsc : bool
        If True (default), fill SNR ``_pred`` columns from HSC v3 (see
        ``apply_snr_hsc_bypass``).  SNRs bypass SED fitting.
    snr_hsc_cache_dir : str or Path, optional
        Per-source HSC cache directory for the SNR bypass.  Defaults to
        ``config.QUERY_CACHE_DIR / 'hsc_snr_fix'``.

    Returns
    -------
    pd.DataFrame
        *df* with all predicted HST columns appended.

    Notes
    -----
    Logs: N sources translated, N per SED family, mean chi2_reduced per
    class, missing fraction per output filter.
    """
    from xclass.io import load_catalog, save_catalog

    fingerprint = _translate_cache_fingerprint(df, all_filter_names, agn_composite)

    # Reuse the cache ONLY when the input fingerprint matches — never on mere path
    # existence (which silently reused stale/pre-SNR caches).
    if cache_path and Path(cache_path).exists():
        if force:
            log.warning("translate_catalog: --force-retranslate set — ignoring "
                        "existing cache %s and retranslating.", cache_path)
        else:
            manifest = _read_cache_manifest(cache_path)
            p = Path(cache_path)
            import time as _time
            mtime = _time.strftime("%Y-%m-%d %H:%M:%S",
                                   _time.localtime(p.stat().st_mtime))
            if manifest and manifest.get("fingerprint") == fingerprint:
                log.warning(
                    "translate_catalog: REUSING cache %s (%s rows, mtime %s) — input "
                    "fingerprint matches. Pass --force-retranslate to override.",
                    cache_path, manifest.get("n_rows", "?"), mtime,
                )
                return load_catalog(cache_path)
            reason = ("no/unreadable manifest" if manifest is None
                      else "input fingerprint changed")
            log.warning(
                "translate_catalog: IGNORING stale cache %s (%s; mtime %s) — the "
                "inputs do not match this cache. Retranslating.",
                cache_path, reason, mtime,
            )

    from joblib import Parallel, delayed
    try:
        from tqdm import tqdm
        _wrap = lambda it: tqdm(it, desc="SED translation", total=len(df))
    except ImportError:
        _wrap = iter

    log.info("translate_catalog: translating %d sources", len(df))

    rows_list = [row for _, row in df.iterrows()]
    results = Parallel(n_jobs=n_jobs)(
        delayed(translate_source_to_hst)(
            row, filter_curves, all_filter_names, pickles_cache, agn_composite
        )
        for row in _wrap(rows_list)
    )

    pred_df = pd.DataFrame(results, index=df.index)
    out = pd.concat([df, pred_df], axis=1)

    # SNR bypass: SNRs carry real HST photometry (nothing to SED-translate); slot
    # HSC v3 magnitudes directly into the _pred columns.  Fails loudly on total loss.
    if snr_hsc:
        cdir = snr_hsc_cache_dir or (config.QUERY_CACHE_DIR / "hsc_snr_fix")
        out = apply_snr_hsc_bypass(out, cdir)

    # Summary logging
    if "xclass_sed_family" in pred_df.columns:
        family_counts = pred_df["xclass_sed_family"].value_counts()
        log.info("SED families: %s", family_counts.to_dict())

    class_col = next((c for c in ["Class", "class_label"] if c in out.columns), None)
    if class_col and "xclass_fit_chi2red" in out.columns:
        mean_chi2 = out.groupby(class_col)["xclass_fit_chi2red"].mean()
        log.info("Mean chi2_reduced per class:\n%s", mean_chi2.to_string())

    for filt_name in all_filter_names:
        col = f"{filt_name}_pred"
        if col in out.columns:
            missing_frac = out[col].isna().mean()
            if missing_frac > 0.5:
                log.warning("Filter %s: %.0f%% missing predictions", filt_name, 100 * missing_frac)

    if cache_path:
        save_catalog(out, cache_path)
        _write_cache_manifest(cache_path, fingerprint, len(out))
        log.info("translate_catalog: saved to %s (+ manifest)", cache_path)

    return out

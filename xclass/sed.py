"""xclass.sed — Spectral energy distribution (SED) models and fitting.

Provides physical SED models (blackbody, power-law, Pickles stellar spectra,
AGN composite) and a grid-search chi-squared fitter that selects the best
model for each source class.

Functions
---------
blackbody_fnu
powerlaw_fnu
match_sptype_to_pickles
fit_sed
normalize_sed_to_band
compute_chi2

Constants
---------
PICKLES_SPTYPE_MAP
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Physical constants (CGS / convenience units)
# Populated from astropy.constants at import time.
try:
    from astropy import constants as _const

    C_CGS: float = _const.c.cgs.value          # cm/s
    H_ERG: float = _const.h.cgs.value           # erg·s
    KB_ERG: float = _const.k_B.cgs.value        # erg/K
    C_AA: float = _const.c.to("AA/s").value      # Angstrom/s
except Exception:
    # Fallback values in case astropy is not installed during development
    C_CGS = 2.99792458e10
    H_ERG = 6.62607015e-27
    KB_ERG = 1.380649e-16
    C_AA = 2.99792458e18

# Mapping from spectral type key to Pickles atlas filename stub.
# The complete set of required keys is listed here; io.py holds the full map.
PICKLES_SPTYPE_MAP: dict[str, str] = {
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

# Ordered spectral type sequence within each class, for nearest-match lookup.
_SPTYPE_SEQUENCE: dict[str, list[str]] = {
    "O": ["O5", "O9"],
    "B": ["B0", "B2", "B5", "B8", "B9"],
    "A": ["A0", "A2", "A5"],
    "F": ["F0", "F5", "F8"],
    "G": ["G0", "G2", "G5", "G8"],
    "K": ["K0", "K2", "K5", "K7"],
    "M": ["M0", "M2", "M3", "M4", "M5", "M6"],
}


# ---------------------------------------------------------------------------
# SED model functions
# ---------------------------------------------------------------------------


def blackbody_fnu(wavelength_AA: np.ndarray, T: float) -> np.ndarray:
    """Compute Planck blackbody f_nu at a given temperature.

    Parameters
    ----------
    wavelength_AA : np.ndarray
        Wavelengths in Angstroms.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Spectral flux density in f_nu (arbitrary units).
        The exponential is clipped at 700 to prevent overflow.
    """
    wave_cm = wavelength_AA * 1e-8  # Angstrom -> cm
    exponent = H_ERG * C_CGS / (wave_cm * KB_ERG * T)
    exponent = np.clip(exponent, 0.0, 700.0)
    # f_nu = 2hc/lambda^3 * 1/(exp(hc/lambda kT) - 1)
    fnu = (2.0 * H_ERG * C_CGS / wave_cm ** 3) / (np.exp(exponent) - 1.0)
    return fnu


def powerlaw_fnu(
    wavelength_AA: np.ndarray,
    alpha: float,
    lam_ref: float = 5500.0,
) -> np.ndarray:
    """Compute a power-law SED f_nu ~ nu^alpha.

    Parameters
    ----------
    wavelength_AA : np.ndarray
        Wavelengths in Angstroms.
    alpha : float
        Spectral index (f_nu ~ nu^alpha).
    lam_ref : float
        Reference wavelength for normalisation (Angstroms).  Default 5500 Å.

    Returns
    -------
    np.ndarray
        Power-law SED normalised to 1.0 at *lam_ref*.
    """
    # nu ~ 1/lambda, so f_nu ~ (1/lambda)^alpha = lambda^(-alpha)
    # normalised: f_nu / f_nu(lam_ref) = (lam_ref / lambda)^alpha
    return (lam_ref / wavelength_AA) ** alpha


def match_sptype_to_pickles(sptype_str: str) -> Optional[str]:
    """Find the nearest Pickles spectral type key for a raw SpType string.

    Algorithm: strip luminosity class -> take first 2 characters ->
    find nearest type in the same spectral class.

    Parameters
    ----------
    sptype_str : str
        Raw spectral type (e.g. 'K5V', 'B2Ib', 'G0').

    Returns
    -------
    str or None
        Nearest key in ``PICKLES_SPTYPE_MAP`` (e.g. 'K5'), or None if the
        spectral class cannot be determined.
    """
    if not sptype_str or not isinstance(sptype_str, str):
        return None

    s = sptype_str.strip().upper()
    if not s:
        return None

    # Extract spectral class letter
    sp_class = s[0]
    if sp_class not in _SPTYPE_SEQUENCE:
        return None

    # Extract numeric subtype (may be absent → default to mid-class)
    subtype_str = ""
    for ch in s[1:]:
        if ch.isdigit() or ch == ".":
            subtype_str += ch
        else:
            break

    try:
        subtype = float(subtype_str) if subtype_str else 5.0
    except ValueError:
        subtype = 5.0

    # Find nearest key in the sequence
    candidates = _SPTYPE_SEQUENCE[sp_class]
    best = None
    best_dist = np.inf
    for cand in candidates:
        cand_sub = float(cand[1:]) if len(cand) > 1 else 0.0
        dist = abs(subtype - cand_sub)
        if dist < best_dist:
            best_dist = dist
            best = cand

    return best


# ---------------------------------------------------------------------------
# Internal fitting helpers
# ---------------------------------------------------------------------------

def _convolve_model(fnu_model: np.ndarray, wave_dense: np.ndarray,
                    filter_wave: np.ndarray, filter_thru: np.ndarray) -> float:
    """Convolve model fnu through a single filter. Returns NaN on failure."""
    mask = (wave_dense >= filter_wave.min()) & (wave_dense <= filter_wave.max())
    if mask.sum() < 2:
        return float("nan")
    w = wave_dense[mask]
    f = fnu_model[mask]
    t = np.interp(w, filter_wave, filter_thru, left=0.0, right=0.0)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy ≥2.0 renamed trapz→trapezoid
    denom = _trapz(t, w)
    if denom <= 0:
        return float("nan")
    return float(_trapz(f * t, w) / denom)


def _normalize_to_band(fnu_model: np.ndarray, wave_dense: np.ndarray,
                        filter_curves: dict, norm_band: str, obs_fnu: float) -> Optional[np.ndarray]:
    """Return fnu_model scaled so its convolution through norm_band equals obs_fnu."""
    fw, ft = filter_curves[norm_band]
    syn = _convolve_model(fnu_model, wave_dense, fw, ft)
    if not np.isfinite(syn) or syn <= 0:
        return None
    return fnu_model * (obs_fnu / syn)


def _compute_chi2(fnu_model: np.ndarray, wave_dense: np.ndarray,
                  filter_curves: dict, obs_fnus: dict, obs_fnu_errs: dict,
                  norm_band: str) -> tuple[float, int]:
    """Compute chi2 and number of fitted bands (excluding normalization band)."""
    chi2 = 0.0
    n = 0
    for band, obs_fnu in obs_fnus.items():
        if band == norm_band or band not in filter_curves:
            continue
        fw, ft = filter_curves[band]
        syn = _convolve_model(fnu_model, wave_dense, fw, ft)
        err = obs_fnu_errs.get(band, obs_fnu * 0.1)
        if np.isfinite(syn) and syn > 0 and np.isfinite(err) and err > 0:
            chi2 += ((obs_fnu - syn) / err) ** 2
            n += 1
    dof = max(n, 1)
    return chi2 / dof, n


def _fit_blackbody_grid(
    wave_dense: np.ndarray,
    obs_fnus: dict, obs_fnu_errs: dict,
    filter_curves: dict, norm_band: str,
) -> dict:
    from xclass import config

    best_chi2 = np.inf
    best_T = float("nan")
    best_fnu = None

    for T in config.T_GRID:
        fnu_raw = blackbody_fnu(wave_dense, T)
        fnu = _normalize_to_band(fnu_raw, wave_dense, filter_curves, norm_band,
                                  obs_fnus[norm_band])
        if fnu is None:
            continue
        chi2, _ = _compute_chi2(fnu, wave_dense, filter_curves, obs_fnus, obs_fnu_errs, norm_band)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_T = float(T)
            best_fnu = fnu

    if best_fnu is None:
        return None
    return {
        "sed_family": "blackbody",
        "sed_param": best_T,
        "chi2_reduced": best_chi2,
        "fnu_model": best_fnu,
    }


def _fit_powerlaw_grid(
    wave_dense: np.ndarray,
    obs_fnus: dict, obs_fnu_errs: dict,
    filter_curves: dict, norm_band: str,
) -> Optional[dict]:
    from xclass import config

    best_chi2 = np.inf
    best_alpha = float("nan")
    best_fnu = None

    for alpha in config.ALPHA_GRID:
        fnu_raw = powerlaw_fnu(wave_dense, alpha)
        fnu = _normalize_to_band(fnu_raw, wave_dense, filter_curves, norm_band,
                                  obs_fnus[norm_band])
        if fnu is None:
            continue
        chi2, _ = _compute_chi2(fnu, wave_dense, filter_curves, obs_fnus, obs_fnu_errs, norm_band)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_alpha = float(alpha)
            best_fnu = fnu

    if best_fnu is None:
        return None
    return {
        "sed_family": "powerlaw",
        "sed_param": best_alpha,
        "chi2_reduced": best_chi2,
        "fnu_model": best_fnu,
    }


def _fit_pickles(
    wave_dense: np.ndarray,
    obs_fnus: dict, obs_fnu_errs: dict,
    filter_curves: dict, norm_band: str,
    sptype: Optional[str],
    pickles_cache: Optional[dict],
) -> Optional[dict]:
    if not pickles_cache:
        return None

    # Determine which Pickles spectra to try
    keys_to_try: list[str] = []
    if sptype:
        k = match_sptype_to_pickles(sptype)
        if k:
            keys_to_try.append(k)
    # Also try all cached keys
    keys_to_try.extend(k for k in pickles_cache if k not in keys_to_try)

    best_chi2 = np.inf
    best_key = None
    best_fnu = None

    for key in keys_to_try:
        spec = pickles_cache.get(key)
        if spec is None:
            continue
        pk_wave, pk_fnu = spec
        if pk_wave is None or pk_fnu is None:
            continue
        # Interpolate Pickles spectrum onto dense grid
        fnu_raw = np.interp(wave_dense, pk_wave, pk_fnu, left=0.0, right=0.0)
        fnu = _normalize_to_band(fnu_raw, wave_dense, filter_curves, norm_band,
                                  obs_fnus[norm_band])
        if fnu is None:
            continue
        chi2, _ = _compute_chi2(fnu, wave_dense, filter_curves, obs_fnus, obs_fnu_errs, norm_band)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_key = key
            best_fnu = fnu

    if best_fnu is None:
        return None
    return {
        "sed_family": "pickles",
        "sed_param": best_key,
        "chi2_reduced": best_chi2,
        "fnu_model": best_fnu,
    }


def _fit_agn_composite(
    wave_dense: np.ndarray,
    obs_fnus: dict, obs_fnu_errs: dict,
    filter_curves: dict, norm_band: str,
    agn_composite: Optional[tuple],
) -> Optional[dict]:
    if agn_composite is None:
        return None
    agn_wave, agn_fnu = agn_composite
    if agn_wave is None or agn_fnu is None:
        return None

    fnu_raw = np.interp(wave_dense, agn_wave, agn_fnu, left=0.0, right=0.0)
    fnu = _normalize_to_band(fnu_raw, wave_dense, filter_curves, norm_band,
                              obs_fnus[norm_band])
    if fnu is None:
        return None
    chi2, _ = _compute_chi2(fnu, wave_dense, filter_curves, obs_fnus, obs_fnu_errs, norm_band)
    return {
        "sed_family": "agn_composite",
        "sed_param": float("nan"),
        "chi2_reduced": chi2,
        "fnu_model": fnu,
    }


def _fit_two_component(
    wave_dense: np.ndarray,
    obs_fnus: dict, obs_fnu_errs: dict,
    filter_curves: dict, norm_band: str,
    star_sptype: str,
    pickles_cache: Optional[dict],
    disk_fraction_grid: list[float],
) -> Optional[dict]:
    """Two-component SED: Pickles star + flat-spectrum disk.

    If Pickles spectrum not available, falls back to blackbody for the star.
    """
    from xclass import config

    # Get star component
    star_fnu = None
    star_key = None
    if pickles_cache:
        k = match_sptype_to_pickles(star_sptype)
        if k and k in pickles_cache:
            spec = pickles_cache[k]
            if spec is not None:
                pk_wave, pk_fnu_raw = spec
                if pk_wave is not None and pk_fnu_raw is not None:
                    star_fnu = np.interp(wave_dense, pk_wave, pk_fnu_raw, left=0.0, right=0.0)
                    star_key = k

    if star_fnu is None:
        # Fallback: representative blackbody for star type
        T_star = 4500.0 if "K" in star_sptype.upper() else 20000.0
        star_fnu = blackbody_fnu(wave_dense, T_star)

    # Normalize star component to ~1 at reference (5500 AA)
    ref_idx = np.argmin(np.abs(wave_dense - 5500.0))
    star_norm = star_fnu[ref_idx] if star_fnu[ref_idx] > 0 else 1.0
    star_fnu = star_fnu / star_norm

    # Flat disk: f_nu = constant (power-law alpha=0)
    disk_fnu = np.ones_like(wave_dense)

    best_chi2 = np.inf
    best_disk_frac = float("nan")
    best_fnu = None

    for disk_frac in disk_fraction_grid:
        fnu_raw = (1.0 - disk_frac) * star_fnu + disk_frac * disk_fnu
        fnu = _normalize_to_band(fnu_raw, wave_dense, filter_curves, norm_band,
                                  obs_fnus[norm_band])
        if fnu is None:
            continue
        chi2, _ = _compute_chi2(fnu, wave_dense, filter_curves, obs_fnus, obs_fnu_errs, norm_band)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_disk_frac = float(disk_frac)
            best_fnu = fnu

    if best_fnu is None:
        return None
    return {
        "sed_family": "two_component",
        "sed_param": best_disk_frac,
        "chi2_reduced": best_chi2,
        "fnu_model": best_fnu,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_sed_to_band(
    fnu: np.ndarray,
    wave_AA: np.ndarray,
    band_wave: np.ndarray,
    band_thru: np.ndarray,
    obs_fnu: float,
) -> np.ndarray:
    """Scale an SED so its synthetic flux through *band* matches *obs_fnu*.

    Parameters
    ----------
    fnu : np.ndarray
        Model SED flux density.
    wave_AA : np.ndarray
        Wavelength grid corresponding to *fnu*.
    band_wave : np.ndarray
        Filter wavelength grid.
    band_thru : np.ndarray
        Filter throughput.
    obs_fnu : float
        Observed flux density to match (same units as *fnu*).

    Returns
    -------
    np.ndarray
        Re-normalised SED.
    """
    syn = _convolve_model(fnu, wave_AA, band_wave, band_thru)
    if not np.isfinite(syn) or syn <= 0:
        return fnu.copy()
    return fnu * (obs_fnu / syn)


def compute_chi2(
    fnu_model: np.ndarray,
    wave_dense: np.ndarray,
    filter_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_fnus: dict[str, float],
    obs_fnu_errs: dict[str, float],
) -> float:
    """Compute reduced chi-squared between model and observed fluxes.

    Parameters
    ----------
    fnu_model : np.ndarray
        Model SED on *wave_dense* grid.
    wave_dense : np.ndarray
        Dense wavelength grid (Angstroms).
    filter_curves : dict
        Band label -> (wave_AA, throughput).
    obs_fnus : dict
        Band label -> observed f_nu.
    obs_fnu_errs : dict
        Band label -> f_nu uncertainty.

    Returns
    -------
    float
        Reduced chi-squared (chi2 / degrees of freedom).
        Returns NaN if fewer than 2 bands are available.
    """
    chi2 = 0.0
    n = 0
    for band, obs_fnu in obs_fnus.items():
        if band not in filter_curves:
            continue
        fw, ft = filter_curves[band]
        syn = _convolve_model(fnu_model, wave_dense, fw, ft)
        err = obs_fnu_errs.get(band, obs_fnu * 0.1)
        if np.isfinite(syn) and syn > 0 and np.isfinite(err) and err > 0:
            chi2 += ((obs_fnu - syn) / err) ** 2
            n += 1

    if n < 2:
        return float("nan")
    return chi2 / (n - 1)


def fit_sed(
    obs_mags: dict[str, float],
    obs_errs: dict[str, float],
    filter_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    source_class: str,
    sptype: Optional[str] = None,
    pickles_cache: Optional[dict] = None,
    agn_composite: Optional[tuple[np.ndarray, np.ndarray]] = None,
    disk_fraction_grid: Optional[list[float]] = None,
    av: float = 0.0,
) -> dict:
    """Fit the best SED model to observed photometry.

    Parameters
    ----------
    obs_mags : dict
        Band label -> observed AB magnitude.
    obs_errs : dict
        Band label -> photometric uncertainty (magnitudes).
    filter_curves : dict
        Band label -> (wave_AA, throughput).
    source_class : str
        One of the seven xclass source classes.
    sptype : str, optional
        Spectral type string for star classes.
    pickles_cache : dict, optional
        Pre-loaded Pickles spectra {sptype_key: (wave_AA, fnu)}.
    agn_composite : tuple, optional
        Pre-loaded Vanden Berk spectrum (wave_AA, fnu).
    disk_fraction_grid : list of float, optional
        Disk fraction values for two-component models.
        Defaults to ``config.DISK_FRACTION_GRID``.
    av : float
        V-band extinction in magnitudes (not applied in production; A_V = 0).

    Returns
    -------
    dict
        Keys: sed_family, sed_param, chi2_reduced, n_bands_used,
        av_used, status_flag, fnu_model (np.ndarray on dense wavelength grid).
    """
    from xclass import config
    # Local import to avoid circular dependency at module load time
    from xclass.photometry import abmag_to_fnu, magerr_to_fnuerr

    if disk_fraction_grid is None:
        disk_fraction_grid = config.DISK_FRACTION_GRID

    wave_dense = np.arange(
        config.SED_WAVE_MIN_AA,
        config.SED_WAVE_MAX_AA + config.SED_WAVE_STEP_AA,
        config.SED_WAVE_STEP_AA,
        dtype=float,
    )

    # Convert mags to fnu, filter out non-finite
    obs_fnus: dict[str, float] = {}
    obs_fnu_errs: dict[str, float] = {}
    for band in obs_mags:
        if band not in filter_curves:
            continue
        mag = obs_mags[band]
        err = obs_errs.get(band, 0.1)
        if not (np.isfinite(mag) and np.isfinite(err)):
            continue
        fnu = abmag_to_fnu(mag)
        fnu_err = magerr_to_fnuerr(mag, err)
        if fnu > 0 and np.isfinite(fnu) and fnu_err > 0:
            obs_fnus[band] = fnu
            obs_fnu_errs[band] = fnu_err

    n_bands = len(obs_fnus)
    _empty = np.zeros_like(wave_dense)

    if n_bands < config.MIN_BANDS_FOR_FIT:
        return {
            "sed_family": "none",
            "sed_param": float("nan"),
            "chi2_reduced": float("nan"),
            "n_bands_used": n_bands,
            "av_used": av,
            "status_flag": 1,
            "fnu_model": _empty,
        }

    # Normalization band: brightest (highest fnu)
    norm_band = max(obs_fnus, key=obs_fnus.get)

    primary = config.SED_MODEL_PRIMARY.get(source_class, "blackbody")
    fallback = config.SED_MODEL_FALLBACK.get(source_class, "blackbody")

    def _run(model_type: str) -> Optional[dict]:
        if model_type == "none":
            return None
        if model_type == "blackbody":
            return _fit_blackbody_grid(wave_dense, obs_fnus, obs_fnu_errs, filter_curves, norm_band)
        if model_type == "powerlaw":
            return _fit_powerlaw_grid(wave_dense, obs_fnus, obs_fnu_errs, filter_curves, norm_band)
        if model_type == "pickles":
            return _fit_pickles(wave_dense, obs_fnus, obs_fnu_errs, filter_curves, norm_band,
                                sptype, pickles_cache)
        if model_type == "agn_composite":
            return _fit_agn_composite(wave_dense, obs_fnus, obs_fnu_errs, filter_curves, norm_band,
                                      agn_composite)
        if model_type.startswith("two_component"):
            star_sp = (config.LMXB_CV_STAR_SPTYPE if "k5" in model_type.lower()
                       else config.HMXB_STAR_SPTYPE)
            return _fit_two_component(wave_dense, obs_fnus, obs_fnu_errs, filter_curves, norm_band,
                                      star_sp, pickles_cache, disk_fraction_grid)
        log.debug("fit_sed: unknown model type '%s'", model_type)
        return None

    result = None
    status = 0

    if primary != "none":
        try:
            result = _run(primary)
        except Exception as exc:
            log.debug("fit_sed primary model '%s' raised: %s", primary, exc)

    if result is None and fallback != "none" and fallback != primary:
        try:
            result = _run(fallback)
            if result is not None:
                status = 2
        except Exception as exc:
            log.debug("fit_sed fallback model '%s' raised: %s", fallback, exc)

    if result is None:
        return {
            "sed_family": "none",
            "sed_param": float("nan"),
            "chi2_reduced": float("nan"),
            "n_bands_used": n_bands,
            "av_used": av,
            "status_flag": 3,
            "fnu_model": _empty,
        }

    result["n_bands_used"] = n_bands
    result["av_used"] = av
    result["status_flag"] = status
    return result

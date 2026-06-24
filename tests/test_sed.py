"""Tests for xclass.sed — SED models and fitting.

All tests are self-contained and do not require network access.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# blackbody_fnu tests
# ---------------------------------------------------------------------------


def test_blackbody_wien_peak():
    """Blackbody SED peaks near Wien's displacement law wavelength.

    Wien: lambda_peak = 2.898e7 / T  (Angstroms, T in Kelvin).
    Tolerance: 5%.
    """
    from xclass.sed import blackbody_fnu

    T = 6000.0  # Solar-like temperature
    wave = np.linspace(1000.0, 30000.0, 5000)
    fnu = blackbody_fnu(wave, T)

    # Convert to f_lambda ~ fnu / lambda^2 to find peak
    flam = fnu / wave**2
    peak_wave = wave[np.argmax(flam)]
    wien_peak = 2.898e7 / T

    assert abs(peak_wave - wien_peak) / wien_peak < 0.05


def test_blackbody_no_overflow():
    """blackbody_fnu should not raise for very hot or very cold temperatures."""
    from xclass.sed import blackbody_fnu

    wave = np.array([1000.0, 5000.0, 10000.0])
    # Very hot star — exponent clipping tested
    fnu_hot = blackbody_fnu(wave, 50000.0)
    assert np.all(np.isfinite(fnu_hot))

    # Very cool object
    fnu_cool = blackbody_fnu(wave, 2000.0)
    assert np.all(np.isfinite(fnu_cool))


# ---------------------------------------------------------------------------
# powerlaw_fnu tests
# ---------------------------------------------------------------------------


def test_powerlaw_slope():
    """Power-law SED has correct log-log slope (spectral index = alpha)."""
    from xclass.sed import powerlaw_fnu

    wave = np.array([3000.0, 6000.0])  # 2x wavelength ratio
    alpha = -1.0  # f_nu ~ nu^alpha, nu ~ 1/lambda => f_nu ~ lambda^(-alpha)

    fnu = powerlaw_fnu(wave, alpha, lam_ref=5500.0)
    # log(f2/f1) / log(nu1/nu2) should equal alpha
    nu = 2.998e18 / wave  # Hz
    log_ratio_fnu = np.log10(fnu[0] / fnu[1])
    log_ratio_nu = np.log10(nu[0] / nu[1])
    slope = log_ratio_fnu / log_ratio_nu
    assert abs(slope - alpha) < 0.01


def test_powerlaw_normalised_at_ref():
    """Power-law SED equals 1.0 at the reference wavelength."""
    from xclass.sed import powerlaw_fnu

    lam_ref = 5500.0
    wave = np.array([lam_ref])
    fnu = powerlaw_fnu(wave, alpha=0.5, lam_ref=lam_ref)
    assert abs(fnu[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# filter convolution tests (via photometry.py)
# ---------------------------------------------------------------------------


def test_convolve_flat_sed(flat_filter_curve):
    """Flat SED through a bandpass returns a constant equal to the SED value."""
    from xclass.photometry import convolve_sed_through_filter

    filter_wave, filter_thru = flat_filter_curve  # 4000-5000 Å, flat
    wave = np.linspace(3000.0, 6000.0, 1000)
    fnu = np.ones_like(wave) * 2.5  # flat SED at 2.5 (arbitrary units)

    result = convolve_sed_through_filter(fnu, wave, filter_wave, filter_thru)
    assert abs(result - 2.5) < 1e-6


# ---------------------------------------------------------------------------
# AB magnitude round-trip tests
# ---------------------------------------------------------------------------


def test_abmag_roundtrip():
    """fnu_to_abmag(abmag_to_fnu(m)) == m to 1e-10 tolerance."""
    from xclass.photometry import abmag_to_fnu, fnu_to_abmag

    for mag in [15.0, 20.0, 25.0, 30.0]:
        fnu = abmag_to_fnu(mag)
        recovered = fnu_to_abmag(fnu)
        assert abs(recovered - mag) < 1e-10, f"Round-trip failed for mag={mag}"


def test_abmag_non_positive_returns_nan():
    """fnu_to_abmag(0) and fnu_to_abmag(-1) return NaN."""
    from xclass.photometry import fnu_to_abmag

    assert np.isnan(fnu_to_abmag(0.0))
    assert np.isnan(fnu_to_abmag(-1.0))


# ---------------------------------------------------------------------------
# fit_sed recovery test
# ---------------------------------------------------------------------------


def test_fit_sed_recovers_blackbody_temperature(flat_filter_curve):
    """fit_sed recovers a known blackbody temperature within 5%.

    Synthetic photometry is generated from a T=7000 K blackbody and then
    fit; the recovered temperature should match to within 5%.
    """
    from xclass.photometry import convolve_sed_through_filter, fnu_to_abmag
    from xclass.sed import blackbody_fnu, fit_sed

    T_true = 7000.0
    wave_dense = np.linspace(1000.0, 25000.0, 5000)
    fnu_true = blackbody_fnu(wave_dense, T_true)

    # Build two synthetic bands with known centroids
    band_a_wave = np.linspace(4000.0, 5000.0, 100)
    band_b_wave = np.linspace(7000.0, 8500.0, 100)
    band_thru = np.ones(100)

    fnu_a = convolve_sed_through_filter(fnu_true, wave_dense, band_a_wave, band_thru)
    fnu_b = convolve_sed_through_filter(fnu_true, wave_dense, band_b_wave, band_thru)

    obs_mags = {
        "band_a": fnu_to_abmag(fnu_a) if not np.isnan(fnu_a) else 20.0,
        "band_b": fnu_to_abmag(fnu_b) if not np.isnan(fnu_b) else 21.0,
        "band_c": fnu_to_abmag(fnu_b * 0.8),  # third band to meet MIN_BANDS_FOR_FIT
    }
    obs_errs = {k: 0.05 for k in obs_mags}
    filter_curves = {
        "band_a": (band_a_wave, band_thru),
        "band_b": (band_b_wave, band_thru),
        "band_c": (band_b_wave * 1.1, band_thru),
    }

    result = fit_sed(obs_mags, obs_errs, filter_curves, source_class="LM-STAR")
    T_fit = result["sed_param"]
    assert abs(T_fit - T_true) / T_true < 0.05, (
        f"Recovered T={T_fit:.0f} K, expected {T_true:.0f} K"
    )

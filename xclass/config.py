"""xclass configuration — all constants, paths, and class definitions.

No logic lives here; this module is pure data. Import from this module
throughout the package; never hard-code numbers in other modules.
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Anchor all paths to the project root so the package works regardless of the
# current working directory (e.g. when run from notebooks/ or scripts/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
QUERY_CACHE_DIR = DATA_DIR / "query_cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
FILTER_CACHE_DIR = DATA_DIR / "filter_cache"
SPECTRA_CACHE_DIR = DATA_DIR / "spectra_cache"

TARGETS_DIR = _PROJECT_ROOT / "targets"
# Per-galaxy application outputs live under targets/{galaxy_name}/processed/
# and targets/{galaxy_name}/plots/.  Training pipeline data goes in data/ only.

# ---------------------------------------------------------------------------
# Source classes
# ---------------------------------------------------------------------------

ALL_CLASSES: list[str] = ["AGN", "LMXB", "HMXB", "CV", "LM-STAR", "HM-STAR", "SNR"]

# Classes whose SEDs are modelled with blackbody (or two-component with BB/Pickles)
BB_CLASSES: list[str] = ["LMXB", "HMXB", "CV", "LM-STAR", "HM-STAR"]

# Classes whose SEDs are modelled with a power law (or AGN composite)
PL_CLASSES: list[str] = ["AGN"]

SNR_CLASS: list[str] = ["SNR"]

# ---------------------------------------------------------------------------
# VizieR training-dataset catalog IDs
# ---------------------------------------------------------------------------

# Each entry: (vizier_catalog_id, assigned_class, short_label)
VIZIER_CATALOGS: list[tuple[str, str, str]] = [
    # LMXBs
    ("J/A+A/469/807", "LMXB", "Liu2007_LMXB"),
    ("B/cb/lmxbdata", "LMXB", "RitterKolb_LMXB"),
    # HMXBs
    ("J/A+A/455/1165", "HMXB", "Liu2006_HMXB"),
    # CVs
    ("B/cb/cbdata", "CV", "RitterKolb_CV"),
    ("V/123A/cv", "CV", "Downes_CV"),
    # AGN
    ("VII/258", "AGN", "VeronCetty2010_AGN"),
    # Stars (mixed — SpType split applied in catalog.py)
    ("B/mk", "STAR", "Skiff2014_stars"),
    ("III/284", "LM-STAR", "APOGEE2_DR16"),
    ("III/215", "HM-STAR", "vanderHucht2001_WR"),
    ("J/A+A/458/453", "HM-STAR", "vanderHucht2006_WR_annex"),
]

# ---------------------------------------------------------------------------
# Spectral-type classification rule
# ---------------------------------------------------------------------------

# SpType strings whose first non-whitespace character is in HM_SPTYPE_PREFIXES
# are classified as HM-STAR; everything else (including unknown) is LM-STAR.
HM_SPTYPE_PREFIXES: frozenset[str] = frozenset({"O", "B"})

# ---------------------------------------------------------------------------
# SNR target galaxies
# ---------------------------------------------------------------------------

TARGET_GALAXIES: list[str] = ["M31", "M33", "M51", "M83", "NGC 6946"]

# Canonical host-galaxy name mapping — normalise aliases to canonical forms
HOST_STANDARDIZATION: dict[str, str] = {
    # M31
    "M31": "M31",
    "NGC 224": "M31",
    "NGC224": "M31",
    "MESSIER 031": "M31",
    # M33
    "M33": "M33",
    "NGC 598": "M33",
    "NGC598": "M33",
    "MESSIER 033": "M33",
    # M51
    "M51": "M51",
    "NGC 5194": "M51",
    "NGC5194": "M51",
    "MESSIER 051": "M51",
    # M83
    "M83": "M83",
    "NGC 5236": "M83",
    "NGC5236": "M83",
    "MESSIER 083": "M83",
    # NGC 6946
    "NGC 6946": "NGC 6946",
    "NGC6946": "NGC 6946",
    "UGC 11597": "NGC 6946",
}

# VizieR SNR catalog IDs per galaxy
SNR_VIZIER_CATALOGS: dict[str, list[tuple[str, str]]] = {
    "M33": [
        ("J/ApJS/187/495", "Long2010_M33"),   # optical + X-ray
        ("J/AJ/148/127", "LeeLee2014_M33"),
    ],
    "M31": [
        ("J/A+A/544/A144", "Sasaki2012_M31"),
        ("J/ApJS/239/13", "Sasaki2018_M31"),
    ],
    "M83": [
        ("J/AJ/129/790", "BlairLong2004_M83"),
    ],
    "M51": [
        ("J/ApJS/109/333", "MatonickFesen1997_M51"),
    ],
    "NGC 6946": [
        ("J/ApJS/109/333", "MatonickFesen1997_NGC6946"),
        ("J/AJ/127/2850", "Lacey2001_NGC6946"),
    ],
}

# ---------------------------------------------------------------------------
# [C1] Live query parameters
# ---------------------------------------------------------------------------

QUERY_SEARCH_RADIUS_ARCSEC: float = 5.0
QUERY_MAX_RETRIES: int = 1        # 1 = single attempt, no retry (fast-fail for broken APIs like MAST PS1)
QUERY_RETRY_WAIT_SEC: float = 5.0
QUERY_TIMEOUT_SEC: float = 30.0  # reduced from 60s to fail faster on network errors

# PanSTARRS (MAST API)
PS1_API_URL: str = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
PS1_COLUMNS: list[str] = [
    "objID",
    "raMean",
    "decMean",
    "gMeanPSFMag",
    "gMeanPSFMagErr",
    "rMeanPSFMag",
    "rMeanPSFMagErr",
    "iMeanPSFMag",
    "iMeanPSFMagErr",
    "zMeanPSFMag",
    "zMeanPSFMagErr",
    "yMeanPSFMag",
    "yMeanPSFMagErr",
    "nDetections",
]
# PS1 does not cover Dec < -30 deg
PS1_DEC_LIMIT_DEG: float = -30.0

# 2MASS (astroquery.vizier, catalog II/246)
TMASS_VIZIER_ID: str = "II/246/out"
TMASS_COLUMNS: list[str] = [
    "RAJ2000",
    "DEJ2000",
    "2MASS",
    "Jmag",
    "e_Jmag",
    "Hmag",
    "e_Hmag",
    "Kmag",
    "e_Kmag",
    "Qflg",
]

# [C3] Hubble Source Catalog v3 (MAST catalogs API — moved from hsc.stsci.edu)
# Detailed endpoint: one row per detection per HST image per filter.
# Response format: {"info": [{name, ...}, ...], "data": [[v1, v2, ...], ...]}
# Radius passed in degrees.
HSC_API_URL: str = "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/detailed"
HSC_COLUMNS: list[str] = [
    "MatchID",
    "CatID",
    "SourceID",
    "MatchRA",
    "MatchDec",
    "MagAper2",
    "CI",
    "StartMJD",
    "StopMJD",
    "Filter",
]
HSC_REQUEST_SLEEP_SEC: float = 0.1  # polite delay between HSC requests

# ---------------------------------------------------------------------------
# [C2] Universal HST output filter sets
# ---------------------------------------------------------------------------
# Keys: short label used in column names (e.g. "ACS_F475W" -> ACS_F475W_pred)
# Values: SVO filter service IDs used to download transmission curves

ACS_WFC_FILTERS: dict[str, str] = {
    "ACS_F435W": "HST/ACS_WFC.F435W",
    "ACS_F475W": "HST/ACS_WFC.F475W",
    "ACS_F502N": "HST/ACS_WFC.F502N",
    "ACS_F555W": "HST/ACS_WFC.F555W",
    "ACS_F606W": "HST/ACS_WFC.F606W",
    "ACS_F625W": "HST/ACS_WFC.F625W",
    "ACS_F658N": "HST/ACS_WFC.F658N",
    "ACS_F775W": "HST/ACS_WFC.F775W",
    "ACS_F814W": "HST/ACS_WFC.F814W",
    "ACS_F850LP": "HST/ACS_WFC.F850LP",
}

WFC3_UVIS_FILTERS: dict[str, str] = {
    "UVIS_F218W": "HST/WFC3_UVIS1.F218W",
    "UVIS_F225W": "HST/WFC3_UVIS1.F225W",
    "UVIS_F275W": "HST/WFC3_UVIS1.F275W",
    "UVIS_F336W": "HST/WFC3_UVIS1.F336W",
    "UVIS_F390M": "HST/WFC3_UVIS1.F390M",
    "UVIS_F390W": "HST/WFC3_UVIS1.F390W",
    "UVIS_F410M": "HST/WFC3_UVIS1.F410M",
    "UVIS_F438W": "HST/WFC3_UVIS1.F438W",
    "UVIS_F467M": "HST/WFC3_UVIS1.F467M",
    "UVIS_F475W": "HST/WFC3_UVIS1.F475W",
    "UVIS_F475X": "HST/WFC3_UVIS1.F475X",
    "UVIS_F547M": "HST/WFC3_UVIS1.F547M",
    "UVIS_F555W": "HST/WFC3_UVIS1.F555W",
    "UVIS_F606W": "HST/WFC3_UVIS1.F606W",
    "UVIS_F621M": "HST/WFC3_UVIS1.F621M",
    "UVIS_F625W": "HST/WFC3_UVIS1.F625W",
    "UVIS_F689M": "HST/WFC3_UVIS1.F689M",
    "UVIS_F763M": "HST/WFC3_UVIS1.F763M",
    "UVIS_F775W": "HST/WFC3_UVIS1.F775W",
    "UVIS_F814W": "HST/WFC3_UVIS1.F814W",
    "UVIS_F845M": "HST/WFC3_UVIS1.F845M",
}

WFC3_IR_FILTERS: dict[str, str] = {
    "IR_F098M": "HST/WFC3_IR.F098M",
    "IR_F105W": "HST/WFC3_IR.F105W",
    "IR_F110W": "HST/WFC3_IR.F110W",
    "IR_F125W": "HST/WFC3_IR.F125W",
    "IR_F127M": "HST/WFC3_IR.F127M",
    "IR_F139M": "HST/WFC3_IR.F139M",
    "IR_F140W": "HST/WFC3_IR.F140W",
    "IR_F153M": "HST/WFC3_IR.F153M",
    "IR_F160W": "HST/WFC3_IR.F160W",
}

# Full universal set (~40 filters)
ALL_HST_FILTERS: dict[str, str] = {
    **ACS_WFC_FILTERS,
    **WFC3_UVIS_FILTERS,
    **WFC3_IR_FILTERS,
}

# Output column naming convention
# ACS/WFC  -> ACS_{filter}_pred,  ACS_{filter}_pred_err
# WFC3/UVIS-> UVIS_{filter}_pred, UVIS_{filter}_pred_err
# WFC3/IR  -> IR_{filter}_pred,   IR_{filter}_pred_err

# [C3] PHAT survey filter set (used in notebook 05 and as default ML set)
PHAT_FILTER_SET: list[str] = [
    "UVIS_F275W",
    "UVIS_F336W",
    "ACS_F475W",
    "ACS_F814W",
    "IR_F110W",
    "IR_F160W",
]

# ML filter set options
ML_FILTER_SETS: dict[str, list[str]] = {
    "PHAT": PHAT_FILTER_SET,
    "ACS_ONLY": [
        "ACS_F435W",
        "ACS_F475W",
        "ACS_F555W",
        "ACS_F606W",
        "ACS_F814W",
    ],
    # 'CUSTOM' reads from ML_CUSTOM_FILTERS below
}
ML_CUSTOM_FILTERS: list[str] = []  # user-editable

# ---------------------------------------------------------------------------
# Input survey filter curves — SVO Filter Profile Service IDs
# These are the ground-based survey bands used as SED fitting inputs.
# Keys match the column names in the cross-matched training table.
# ---------------------------------------------------------------------------

SURVEY_FILTERS: dict[str, str] = {
    # PanSTARRS DR2 (AB)
    "ps1_g": "PAN-STARRS/PS1.g",
    "ps1_r": "PAN-STARRS/PS1.r",
    "ps1_i": "PAN-STARRS/PS1.i",
    "ps1_z": "PAN-STARRS/PS1.z",
    "ps1_y": "PAN-STARRS/PS1.y",
    # 2MASS (Vega)
    "tmass_j": "2MASS/2MASS.J",
    "tmass_h": "2MASS/2MASS.H",
    "tmass_k": "2MASS/2MASS.Ks",
}

# ---------------------------------------------------------------------------
# Input photometry zeropoints (Jy)  — AB (PanSTARRS) or Vega (2MASS)
# ---------------------------------------------------------------------------

ZEROPOINTS_JY: dict[str, float] = {
    # PanSTARRS (AB)
    "ps1_g": 3631.0,
    "ps1_r": 3631.0,
    "ps1_i": 3631.0,
    "ps1_z": 3631.0,
    "ps1_y": 3631.0,
    # 2MASS (Vega)
    "tmass_j": 1594.0,
    "tmass_h": 1024.0,
    "tmass_k": 666.7,
}

# ---------------------------------------------------------------------------
# SED fitting parameters
# ---------------------------------------------------------------------------

# Blackbody temperature grid (Kelvin)
T_GRID: np.ndarray = np.concatenate(
    [np.arange(2000, 10000, 250), np.arange(10000, 50001, 1000)]
)

# Power-law spectral index grid (f_nu ~ nu^alpha)
ALPHA_GRID: np.ndarray = np.arange(-2.5, 1.51, 0.05)

# Disk fraction grid for two-component (star + disk) models
DISK_FRACTION_GRID: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

MIN_BANDS_FOR_FIT: int = 3  # minimum photometric bands required for SED fit

# Extra systematic uncertainty added in quadrature to UV predictions
UV_SYSTEMATIC_ERR_MAG: float = 0.4  # magnitudes; applied to F275W and F336W

# Dense SED wavelength grid for convolution (Angstrom)
SED_WAVE_MIN_AA: float = 100.0
SED_WAVE_MAX_AA: float = 25000.0
SED_WAVE_STEP_AA: float = 5.0

# ---------------------------------------------------------------------------
# SED model assignments by class
# ---------------------------------------------------------------------------
# Used in sed.py and photometry.py to choose which model to fit first.

SED_MODEL_PRIMARY: dict[str, str] = {
    "LM-STAR": "pickles",
    "HM-STAR": "pickles",
    "AGN": "agn_composite",
    "LMXB": "two_component_k5_disk",
    "CV": "two_component_k5_disk",
    "HMXB": "two_component_b2_disk",
    "SNR": "none",  # SNR uses real HST photometry; no SED translation
}

SED_MODEL_FALLBACK: dict[str, str] = {
    "LM-STAR": "blackbody",
    "HM-STAR": "blackbody",
    "AGN": "powerlaw",
    "LMXB": "blackbody",
    "CV": "blackbody",
    "HMXB": "blackbody",
    "SNR": "none",
}

# Pickles anchor spectral types for two-component models
LMXB_CV_STAR_SPTYPE: str = "K5"   # cool companion in LMXB / CV
HMXB_STAR_SPTYPE: str = "B2"      # massive donor in HMXB

# ---------------------------------------------------------------------------
# [C3] PHAT footprint polygon
# ---------------------------------------------------------------------------
# RA/Dec vertices in degrees, ICRS.  Defines the survey boundary used to
# filter Chandra sources for notebook 05.

PHAT_POLYGON_DEG: list[tuple[float, float]] = [
    (10.9996, 41.1082),
    (11.9463, 42.1880),
    (11.6884, 42.4500),
    (10.6200, 41.3000),
]

# ---------------------------------------------------------------------------
# Crossmatching parameters
# ---------------------------------------------------------------------------

LABEL_TO_XRAY_SIGMA: float = 3.0       # sigma multiplier for X-ray search radius
MIN_SEARCH_RADIUS_ARCSEC: float = 0.5  # floor on search radius
MAX_NORMSEP_XRAY_OPT: float = 3.0      # max normalised sep for optical counterpart
MATCH_SECURE_NORMSEP_MAX: float = 2.0   # normsep threshold for a "secure" counterpart match
MATCH_SECOND_BEST_RATIO: float = 1.5    # best/second-best sep ratio for match uniqueness

# ---------------------------------------------------------------------------
# Quality cuts
# ---------------------------------------------------------------------------

MIN_SIGNIFICANCE: float = 3.0  # minimum Chandra source significance to include

# ---------------------------------------------------------------------------
# ML parameters
# ---------------------------------------------------------------------------

TEST_FRACTION: float = 0.10
VAL_FRACTION: float = 0.20
RANDOM_STATE: int = 42

RF_PARAMS: dict = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
}

# Pivot X-ray energy for nuFnu calculations (broad band ~0.5-7 keV midpoint)
PIVOT_XRAY_HZ: float = 1.45e18  # Hz  (~6 keV)

# ---------------------------------------------------------------------------
# Deduplication radii
# ---------------------------------------------------------------------------

MASTER_MATCH_RADIUS_ARCSEC: float = 5.0   # TD source dedup radius
SNR_DEDUP_RADIUS_ARCSEC: float = 2.0      # SNR catalog dedup radius

# ---------------------------------------------------------------------------
# Diagnostics color scheme
# ---------------------------------------------------------------------------
# Use these colours consistently in all plots.

CLASS_COLORS: dict[str, str] = {
    "AGN": "#E41A1C",
    "LMXB": "#377EB8",
    "HMXB": "#4DAF4A",
    "CV": "#984EA3",
    "LM-STAR": "#FF7F00",
    "HM-STAR": "#A65628",
    "SNR": "#F781BF",
}

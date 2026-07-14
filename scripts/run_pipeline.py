"""xclass pipeline CLI — Phase 7.

Usage
-----
Run the full training pipeline::

    python scripts/run_pipeline.py --stage all

Run individual stages::

    python scripts/run_pipeline.py --stage build_td
    python scripts/run_pipeline.py --stage crossmatch
    python scripts/run_pipeline.py --stage translate
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage apply --target M31_PHAT

Override config values with a YAML file::

    python scripts/run_pipeline.py --config my_config.yaml --stage train

Dry-run (print what would happen, no side effects)::

    python scripts/run_pipeline.py --stage all --dry-run

Select HST filter set::

    python scripts/run_pipeline.py --stage train --filter-set ACS_ONLY
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as ``python scripts/run_pipeline.py``
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPTS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

log = logging.getLogger("xclass.run_pipeline")

# ---------------------------------------------------------------------------
# Valid stages
# ---------------------------------------------------------------------------

STAGES = [
    "build_td",   # Phase 1–2: build training dataset from VizieR + SNR catalogs
    "crossmatch", # Phase 3: query photometry + crossmatch to Chandra sources
    "translate",  # Phase 4: SED-translate survey photometry to HST filter set
    "train",      # Phase 5: build optical baseline + train production Stage 1 & 2
    "evaluate",   # Phase 5: 5-fold two-stage CV + production figures (paper numbers)
    "apply",      # Phase 6: apply to target galaxy (requires --target)
    "all",        # run all training stages in order (build_td→crossmatch→translate→train)
]


# ---------------------------------------------------------------------------
# Config YAML override helper
# ---------------------------------------------------------------------------

def _apply_yaml_overrides(yaml_path: Path) -> None:
    """Load a YAML file and patch ``xclass.config`` with its key-value pairs."""
    import yaml  # optional dependency; only needed with --config

    from xclass import config as cfg

    with open(yaml_path) as f:
        overrides = yaml.safe_load(f) or {}

    for key, value in overrides.items():
        if hasattr(cfg, key):
            log.info("Config override: %s = %r", key, value)
            setattr(cfg, key, value)
        else:
            log.warning("Config key not found (ignored): %s", key)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage_build_td(dry_run: bool = False) -> None:
    """Build the training dataset from VizieR catalogs and SNR literature."""
    log.info("=== Stage: build_td ===")
    if dry_run:
        log.info("[dry-run] Would call catalog.build_master_resolved_table() "
                 "and snr.build_snr_ml_catalog()")
        return

    from xclass.catalog import build_master_resolved_table, build_td_catalog
    from xclass.snr import fetch_snr_literature, dedup_snr_catalog, build_snr_ml_catalog
    from xclass import config

    td_df = build_td_catalog()
    log.info("Training dataset: %d sources", len(td_df))

    snr_raw = fetch_snr_literature()
    log.info("SNR literature: %d rows fetched", len(snr_raw))
    snr_dedup = dedup_snr_catalog(snr_raw)
    log.info("SNR after dedup: %d sources", len(snr_dedup))
    snr_df = build_snr_ml_catalog(snr_dedup)
    log.info("SNR ML catalog: %d sources", len(snr_df))

    out_path = config.PROCESSED_DIR / "training_set.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    td_df.to_csv(out_path, index=False)
    log.info("Saved training set to %s", out_path)

    snr_path = config.PROCESSED_DIR / "snr_ml_catalog.csv"
    snr_df.to_csv(snr_path, index=False)
    log.info("Saved SNR catalog to %s", snr_path)


def stage_crossmatch(dry_run: bool = False) -> None:
    """Query photometry + crossmatch training set to Chandra sources."""
    log.info("=== Stage: crossmatch ===")
    if dry_run:
        log.info("[dry-run] Would call query.query_all_photometry() and "
                 "crossmatch.build_xray_training_table()")
        return

    from xclass import config
    from xclass.query import query_all_photometry
    from xclass.crossmatch import match_td_to_chandra, build_xray_training_table

    td_df = _load_csv(config.PROCESSED_DIR / "training_set.csv", "training set")
    chandra_df = _load_csv(config.RAW_DIR / "chandra_catalog.csv", "Chandra catalog")

    # Add source_id from canonical_name (required by query functions)
    if "source_id" not in td_df.columns:
        id_col = next((c for c in ("canonical_name", "master_group", "name_norm")
                       if c in td_df.columns), None)
        td_df["source_id"] = td_df[id_col].astype(str) if id_col else td_df.index.astype(str)

    # Step 1: positional match TD → Chandra first (cheap sky query on full set)
    log.info("Matching TD (%d) to Chandra (%d)...", len(td_df), len(chandra_df))
    matched_pairs = match_td_to_chandra(td_df, chandra_df,
                                        nsigma=config.LABEL_TO_XRAY_SIGMA)
    log.info("TD–Chandra pairs: %d", len(matched_pairs))

    if len(matched_pairs) == 0:
        log.error("No TD–Chandra matches — check RA/Dec columns and catalog overlap.")
        return

    # Step 2: query photometry only for matched TD sources (not all 1M+)
    matched_ids = matched_pairs["td_source_id"].unique()
    td_matched = td_df[td_df["source_id"].isin(matched_ids)].copy()
    log.info("Querying photometry for %d matched TD sources...", len(td_matched))
    # GAIA individual TAP jobs are too slow for 13k sources (~5h); skip for now.
    # PS1 (11k matches) and 2MASS (9k matches) are already cached from prior run.
    # Re-enable gaia by changing surveys to None once bulk GAIA querying is implemented.
    phot_df = query_all_photometry(
        td_matched,
        cache_dir=config.QUERY_CACHE_DIR,
        surveys=["ps1", "2mass"],
    )
    log.info("Photometry table: %d rows", len(phot_df))

    # Step 3: build final X-ray training table (internally re-runs match + joins phot)
    xray_td = build_xray_training_table(td_df, chandra_df, phot_df)
    log.info("X-ray training table: %d rows", len(xray_td))

    out_path = config.PROCESSED_DIR / "xray_training_table.csv"
    xray_td.to_csv(out_path, index=False)
    log.info("Saved to %s", out_path)


def _build_snr_xray_rows(snr_path: Path, chandra_path: Path,
                         match_radius_arcsec: float = 3.0) -> "pd.DataFrame":
    """Crossmatch SNR ML catalog against Chandra and return xray_training_table rows.

    Only SNRs with a Chandra detection within *match_radius_arcsec* are included.
    All non-X-ray columns are left as NaN so that the SED translator assigns them
    the ``none`` SED family (no optical photometry → no translated HST magnitudes).
    """
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not snr_path.exists():
        log.warning("SNR catalog not found: %s — skipping SNR merge", snr_path)
        return pd.DataFrame()

    snr = pd.read_csv(snr_path)
    chandra = pd.read_csv(chandra_path, low_memory=False)

    # ── positional crossmatch ─────────────────────────────────────────────────
    snr_coords = SkyCoord(ra=snr["ra"].values * u.deg,
                          dec=snr["dec"].values * u.deg)
    ch_coords = SkyCoord(ra=chandra["ra"].values * u.deg,
                         dec=chandra["dec"].values * u.deg)

    idx, sep, _ = snr_coords.match_to_catalog_sky(ch_coords)
    matched_mask = sep.arcsec < match_radius_arcsec

    n_matched = matched_mask.sum()
    log.info("SNR–Chandra crossmatch at %.1f arcsec: %d / %d SNRs matched",
             match_radius_arcsec, n_matched, len(snr))
    if n_matched == 0:
        return pd.DataFrame()

    snr_m = snr[matched_mask].copy().reset_index(drop=True)
    ch_m = chandra.iloc[idx[matched_mask]].copy().reset_index(drop=True)
    sep_m = sep[matched_mask].arcsec

    # ── build rows in xray_training_table schema ──────────────────────────────
    pos_err_arcsec = ch_m["err_ellipse_r0"].values  # semi-major axis as proxy
    # Guard against zero error (use 1 arcsec as floor)
    pos_err_arcsec = np.where(pos_err_arcsec > 0, pos_err_arcsec, 1.0)

    rows = pd.DataFrame({
        # Training label — set BOTH the prefixed td_Class and the canonical Class
        # column.  A fresh concat leaves Class=NaN for SNR rows otherwise, which
        # both hides them from the SNR HSC bypass and breaks STAGE1_MAP at train.
        "Class":               "SNR",
        "td_Class":            "SNR",
        "td_canonical_name":   snr_m["source_name"].values,
        "td_name_norm":        snr_m["source_name"].str.lower().values,
        "td_ra":               snr_m["ra"].values,
        "td_dec":              snr_m["dec"].values,
        "td_n_input_rows":     1,
        # Chandra X-ray columns
        "xray_name":               ch_m["name"].values,
        "xray_ra":                 ch_m["ra"].values,
        "xray_dec":                ch_m["dec"].values,
        "xray_err_ellipse_r0":     ch_m["err_ellipse_r0"].values,
        "xray_err_ellipse_r1":     ch_m["err_ellipse_r1"].values,
        "xray_err_ellipse_ang":    ch_m["err_ellipse_ang"].values,
        "xray_significance":       ch_m["significance"].values,
        "xray_likelihood_class":   ch_m["likelihood_class"].values,
        "xray_conf_flag":          ch_m["conf_flag"].values,
        "xray_sat_src_flag":       ch_m["sat_src_flag"].values,
        "xray_streak_src_flag":    ch_m["streak_src_flag"].values,
        "xray_flux_aper90_avg_s":  ch_m["flux_aper90_avg_s"].values,
        "xray_flux_aper90_avg_m":  ch_m["flux_aper90_avg_m"].values,
        "xray_flux_aper90_avg_h":  ch_m["flux_aper90_avg_h"].values,
        "xray_flux_aper90_avg_b":  ch_m["flux_aper90_avg_b"].values,
        # Positional match quality
        "sep_class_xray_arcsec":   sep_m,
        "xray_pos_err_arcsec":     pos_err_arcsec,
        "normsep_class_xray":      sep_m / pos_err_arcsec,
        # Derived flux columns (mirrors of xray_flux_aper90_avg_*)
        "Fx_S":  ch_m["flux_aper90_avg_s"].values,
        "Fx_M":  ch_m["flux_aper90_avg_m"].values,
        "Fx_H":  ch_m["flux_aper90_avg_h"].values,
        "Fx_B":  ch_m["flux_aper90_avg_b"].values,
    })
    log.info("Built %d SNR rows for xray_training_table", len(rows))
    return rows


def stage_translate(filter_set: str = "PHAT", dry_run: bool = False,
                    force_retranslate: bool = False,
                    snr_hsc_cache: "Path | None" = None) -> None:
    """SED-translate survey photometry to HST filter set (SNRs filled from HSC v3)."""
    log.info("=== Stage: translate (filter_set=%s) ===", filter_set)
    if dry_run:
        log.info("[dry-run] Would call photometry.translate_catalog() "
                 "with filter_set=%s (force_retranslate=%s, snr_hsc_cache=%s)",
                 filter_set, force_retranslate, snr_hsc_cache)
        return

    import pandas as pd
    from xclass import config
    from xclass.photometry import translate_catalog
    from xclass.io import load_filter_curves, load_pickles_cache, load_agn_composite

    xray_td = _load_csv(config.PROCESSED_DIR / "xray_training_table.csv",
                        "X-ray training table")

    # ── Merge SNR Chandra counterparts ────────────────────────────────────────
    snr_rows = _build_snr_xray_rows(
        snr_path=config.PROCESSED_DIR / "snr_ml_catalog.csv",
        chandra_path=config.RAW_DIR / "chandra_catalog.csv",
    )
    if len(snr_rows) > 0:
        xray_td = pd.concat([xray_td, snr_rows], ignore_index=True)
        log.info("xray_training_table after SNR merge: %d rows", len(xray_td))

    # Load HST output filter curves (SED translation targets)
    all_hst_filters = {**config.ACS_WFC_FILTERS, **config.WFC3_UVIS_FILTERS,
                       **config.WFC3_IR_FILTERS}
    hst_filter_curves = load_filter_curves(all_hst_filters, cache_dir=config.FILTER_CACHE_DIR)
    all_filter_names = list(hst_filter_curves.keys())

    # Load input survey filter curves (needed for SED fitting)
    survey_filter_curves = load_filter_curves(config.SURVEY_FILTERS,
                                              cache_dir=config.FILTER_CACHE_DIR)

    # Merge: survey filters first (fitting inputs), then HST (prediction outputs)
    filter_curves = {**survey_filter_curves, **hst_filter_curves}

    pickles_cache = load_pickles_cache(cache_dir=config.SPECTRA_CACHE_DIR)
    agn_composite = load_agn_composite(cache_dir=config.SPECTRA_CACHE_DIR)

    translated = translate_catalog(
        xray_td,
        filter_curves=filter_curves,
        all_filter_names=all_filter_names,
        pickles_cache=pickles_cache,
        agn_composite=agn_composite,
        cache_path=config.PROCESSED_DIR / "sed_translation_cache.csv",
        force=force_retranslate,
        snr_hsc_cache_dir=snr_hsc_cache,
    )
    log.info("Translated catalog: %d rows", len(translated))

    out_path = config.PROCESSED_DIR / "translated_catalog.csv"
    translated.to_csv(out_path, index=False)
    log.info("Saved to %s", out_path)

    # ── Run manifest (instrumentation only — records what actually happened) ───
    from datetime import datetime
    from xclass import manifest as _manifest

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # AGN composite present iff a real (wave, flux) tuple was loaded.  If it is
    # None / (None, None), the AGN fit silently fell back to a power law.
    agn_ok = agn_composite is not None and agn_composite[0] is not None
    # Live SNR HSC counters (patched / no-counterpart / network failures) attached
    # by apply_snr_hsc_bypass.  Absent when translate reused its cache, in which
    # case counts are derived from the catalog and this is flagged as a cache hit.
    snr_counts = translated.attrs.get("xclass_snr_hsc")
    cache_hit = snr_counts is None
    # Optical-baseline preview: same ">=1 non-NaN PHAT _pred" cut used to build the
    # production catalog, computed read-only (writes nothing).
    pred_cols = [f"{f}_pred" for f in config.PHAT_FILTER_SET
                 if f"{f}_pred" in translated.columns]
    n_optical = (int((translated[pred_cols].notna().sum(axis=1) >= 1).sum())
                 if pred_cols else None)

    m = _manifest.build_translate_manifest(
        translated, timestamp=ts, n_optical=n_optical,
        agn_composite_available=agn_ok, snr_counts=snr_counts,
        cache_hit=cache_hit, force=force_retranslate,
    )
    mpath = m.write(config.PROCESSED_DIR)
    print(m.format_summary())
    log.info("Run manifest written to %s", mpath)


def stage_train(filter_set: str = "PHAT", dry_run: bool = False) -> None:
    """Train + save the production two-stage classifier (optical baseline N=11,374,
    no SMOTE, balanced_subsample, 3-class Stage 2 = LMXB/HMXB/CV, saves *_optical_v2).

    Thin dispatch into ``xclass.pipeline`` (single implementation; no logic here).
    """
    log.info("=== Stage: train (production config; filter_set=%s) ===", filter_set)
    if dry_run:
        log.info("[dry-run] xclass.pipeline.build_optical_baseline() (if needed) "
                 "+ train_production()")
        return

    from xclass import pipeline
    if not pipeline.OPTICAL_CATALOG.exists():
        log.info("Optical baseline catalog absent — building from %s",
                 pipeline.FULL_CATALOG.name)
        pipeline.build_optical_baseline()
    pipeline.train_production()
    log.info("Production models saved (suffix '_%s')", pipeline.MODEL_SUFFIX)


def stage_evaluate(filter_set: str = "PHAT", dry_run: bool = False) -> None:
    """Run the production 5-fold two-stage CV and regenerate the production figures.

    Thin dispatch into ``xclass.pipeline.evaluate_production``. Prints the headline
    paper numbers (balanced accuracy, macro F1, per-class LMXB/HMXB F1).
    """
    log.info("=== Stage: evaluate (production 5-fold CV) ===")
    if dry_run:
        log.info("[dry-run] xclass.pipeline.evaluate_production()")
        return

    from xclass import pipeline
    if not pipeline.OPTICAL_CATALOG.exists():
        pipeline.build_optical_baseline()
    res = pipeline.evaluate_production()
    fp = res["fp"]["all"]; pc = res["per_class"]
    log.info("Full pipeline: acc=%.4f  bacc=%.4f  f1_macro=%.4f",
             fp["acc"], fp["bacc"], fp["f1"])
    for cls in ("LMXB", "HMXB"):
        row = pc.get(cls, {})
        log.info("  %s: P=%.3f R=%.3f F1=%.3f", cls,
                 row.get("precision", 0), row.get("recall", 0), row.get("f1-score", 0))


def stage_apply(target: str, filter_set: str = "PHAT", dry_run: bool = False) -> None:
    """Apply the production 3-class classifier to a target galaxy footprint.

    Thin dispatch into ``xclass.pipeline.apply_production`` (single implementation).
    """
    log.info("=== Stage: apply (target=%s, filter_set=%s) ===", target, filter_set)
    from xclass import config

    csc_csv = config.TARGETS_DIR / target / "chandra_catalog.csv"
    if not csc_csv.exists():
        log.error("CSC CSV not found: %s", csc_csv)
        sys.exit(1)
    if dry_run:
        log.info("[dry-run] xclass.pipeline.apply_production(%r) using %s", target, csc_csv)
        return

    from xclass import pipeline
    out = pipeline.apply_production(target, filter_set=filter_set)
    log.info("Done — %d sources classified; outputs in %s",
             len(out), config.TARGETS_DIR / target / "processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path, label: str):
    import pandas as pd

    if not path.exists():
        log.error("%s not found: %s", label, path)
        sys.exit(1)
    df = pd.read_csv(path)
    log.info("Loaded %s: %d rows from %s", label, len(df), path)
    return df


def _value_counts(arr):
    import numpy as np

    vals, counts = np.unique(arr, return_counts=True)
    return vals.tolist(), counts.tolist()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="xclass X-ray source classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        required=True,
        help="Pipeline stage to run (use 'all' for full training run).",
    )
    parser.add_argument(
        "--target",
        default="M31_PHAT",
        help="Target galaxy name (used by the 'apply' stage). Default: M31_PHAT.",
    )
    parser.add_argument(
        "--filter-set",
        default="PHAT",
        dest="filter_set",
        help="HST filter set for feature engineering. Default: PHAT.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        dest="config_yaml",
        help="Optional YAML file with config overrides (e.g. RF_PARAMS, RANDOM_STATE).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would happen but do not execute any pipeline code.",
    )
    parser.add_argument(
        "--force-retranslate",
        action="store_true",
        default=False,
        help="Translate stage: ignore any existing SED-translation cache and "
             "retranslate from scratch (defeats stale-cache reuse).",
    )
    parser.add_argument(
        "--snr-hsc-cache",
        type=Path,
        default=None,
        help="Translate stage: directory of per-source HSC v3 caches for the SNR "
             "photometry bypass (offline reproducibility). Default: "
             "data/query_cache/hsc_snr_fix.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    return parser


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Apply YAML config overrides before importing pipeline modules
    if args.config_yaml is not None:
        _apply_yaml_overrides(args.config_yaml)

    dry = args.dry_run
    if dry:
        log.info("*** DRY-RUN MODE — no files will be written ***")

    stage = args.stage
    fs = args.filter_set

    if stage == "all":
        stage_build_td(dry_run=dry)
        stage_crossmatch(dry_run=dry)
        stage_translate(filter_set=fs, dry_run=dry,
                        force_retranslate=args.force_retranslate,
                        snr_hsc_cache=args.snr_hsc_cache)
        stage_train(filter_set=fs, dry_run=dry)
    elif stage == "build_td":
        stage_build_td(dry_run=dry)
    elif stage == "crossmatch":
        stage_crossmatch(dry_run=dry)
    elif stage == "translate":
        stage_translate(filter_set=fs, dry_run=dry,
                        force_retranslate=args.force_retranslate,
                        snr_hsc_cache=args.snr_hsc_cache)
    elif stage == "train":
        stage_train(filter_set=fs, dry_run=dry)
    elif stage == "evaluate":
        stage_evaluate(filter_set=fs, dry_run=dry)
    elif stage == "apply":
        stage_apply(target=args.target, filter_set=fs, dry_run=dry)
    else:
        parser.error(f"Unknown stage: {stage}")

    log.info("Pipeline stage '%s' complete.", stage)


if __name__ == "__main__":
    main()

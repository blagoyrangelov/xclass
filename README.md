# XClass

**Multi-wavelength classification of extragalactic X-ray point sources.**

XClass is a machine-learning pipeline that classifies Chandra X-ray Observatory
point sources in nearby galaxies into seven physical source classes — AGN, LMXB,
HMXB, CV, low-mass stars, high-mass stars, and SNRs — using SED-translated
photometry from PanSTARRS and 2MASS (plus HSC for the application field), fed
into a two-stage Random Forest classifier. The pipeline ingests Chandra Source
Catalog (CSC 2.1.1) detections, fits a per-source spectral energy distribution to
translate ground-based photometry into HST PHAT bandpasses, builds a uniform
22-feature matrix, and produces calibrated class probabilities for each source.

### Production configuration (as published)

- **Training baseline:** optical baseline, **N = 11,374** sources
  (`translated_catalog_optical.csv`).
- **SED-translation inputs:** **PanSTARRS grizy + 2MASS JHKs only** (no Gaia).
- **Feature matrix:** **22 features in five groups** — X-ray log-fluxes (4),
  hardness ratios (4), HST colours (9), X-ray/optical ratios (2), match
  quality (3). No Gaia astrometric features.
- **Classifier:** two-stage Random Forest. Stage 1 → AGN / STAR / SNR / OTHER;
  Stage 2 resolves OTHER into the **3 classes LMXB / HMXB / CV**.
- **Class imbalance:** handled by the Random Forest's
  `class_weight="balanced_subsample"` — **no SMOTE**.
- **Cross-validated performance:** balanced accuracy ≈ 0.90 (5-fold).

## Paper reference

Rangelov, B. et al. 2026, *The Astrophysical Journal*, **submitted**.
*"Automated multi-wavelength classification of extragalactic X-ray point
sources in nearby spiral galaxies."*

## Installation

```bash
git clone https://github.com/blagoyrangelov/xclass.git
cd xclass
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.11+ is required. The pipeline depends on `astropy`, `astroquery`,
`scikit-learn`, `synphot`, and the standard scientific stack.

## Reproducing the paper

The published results (optical baseline, no SMOTE, 3-class Stage 2) are produced
by the single CLI entry point, `scripts/run_pipeline.py`. The production logic
lives in the package module `xclass/pipeline.py`:

```bash
python scripts/run_pipeline.py --stage evaluate   # 5-fold CV + paper figures
python scripts/run_pipeline.py --stage train      # train + save production models
```

These reproduce the cross-validation metrics, the confusion matrices (LMXB→CV =
0.26), the per-class numbers (Table 5), the calibration diagram, and the
feature-importance figure in `figures/`.

## Pipeline stages (CLI)

The end-to-end pipeline is also exposed as CLI subcommands of
`scripts/run_pipeline.py`:

```bash
# 1. Build the training catalog from VizieR (LMXB, HMXB, CV, AGN, stars, SNR)
python scripts/run_pipeline.py --stage build_td

# 2. Crossmatch training sources to Chandra CSC 2.1.1 + query PanSTARRS/2MASS
python scripts/run_pipeline.py --stage crossmatch

# 3. Translate per-source SEDs to PHAT-equivalent magnitudes.
#    SNRs carry real HST photometry (nothing to SED-translate), so this stage
#    also fetches their six PHAT magnitudes from HSC v3 (MAST) and writes them
#    straight into the *_pred columns — this is what keeps the SNR class in the
#    training set. Requires network access to MAST during this stage.
python scripts/run_pipeline.py --stage translate

# 4. Train the production two-stage classifier (optical baseline, no SMOTE,
#    3-class Stage 2). Verified to reproduce the paper numbers
#    (balanced accuracy 0.90, macro F1 0.91, LMXB F1 0.76, HMXB F1 0.95).
python scripts/run_pipeline.py --stage train

# 5. Apply the trained classifier to a target field (e.g. M31 PHAT)
python scripts/run_pipeline.py --stage apply --target M31_PHAT
```

The four training stages (`build_td → crossmatch → translate → train`) reproduce
the published catalog (**11,374** sources, all four Stage-1 classes) and metrics
end to end — no out-of-band scripts are required.

**Translate-stage reproducibility notes:**

- **SNR HSC photometry.** The `translate` stage queries HSC v3 for every SNR. If
  HSC/MAST is unreachable the stage **fails loudly** rather than silently dropping
  the SNR class (which would yield a 3-class model that does not match the paper).
  For offline / outage-resilient reproduction, point it at a pre-fetched HSC cache:
  `--stage translate --snr-hsc-cache path/to/hsc_snr_fix` (the published SNR HSC
  responses are provided in the Zenodo deposit under `data/query_cache/hsc_snr_fix/`).
- **Translation cache.** `translate` caches its result in
  `data/processed/sed_translation_cache.csv` and reuses it **only** when a
  fingerprint of the inputs matches (a stale cache from different inputs is ignored,
  not silently reused). This cache is *not* shipped; force a clean retranslation
  with `--stage translate --force-retranslate`.

## Data and trained models

Input data and trained models are **not** bundled in this repository. The
training catalog (`translated_catalog_optical.csv`) and the production models
(`*_rf_optical_v2`) are available from the Zenodo deposit
[10.5281/zenodo.20838124](https://doi.org/10.5281/zenodo.20838124). The public input catalogs — Chandra CSC 2.1.1, PanSTARRS DR2,
2MASS PSC, HSC v3, and the VizieR training labels — are cited in the paper and
obtained from their respective archives.

## Directory structure

```
xclass/                  Python package — modules implementing each stage
├── config.py            All numerical parameters (no logic)
├── catalog.py           VizieR fetching and training-catalog assembly
├── snr.py               SNR catalog handling
├── query.py             Multi-survey photometry queries with caching
├── crossmatch.py        Positional matching against Chandra CSC
├── sed.py               SED model library (blackbody, power-law, Pickles, AGN)
├── photometry.py        Filter convolution and SED-to-photometry translation
├── features.py          Feature engineering and ML preprocessing
├── classifier.py        Two-stage Random Forest training and prediction
├── diagnostics.py       Plots and reports
└── io.py                Filter and spectral-template I/O

scripts/                 Pipeline CLI + production eval scripts + diagnostics
tests/                   pytest test suite
figures/                 Publication PDF figures
data/                    Created at runtime (gitignored)
```

The `data/` tree is created on first run and is not tracked in git. It contains:

- `data/raw/` — input catalogs (e.g. the Chandra CSC 2.1.1 export)
- `data/query_cache/` — per-source pickle cache for survey queries
- `data/processed/` — pipeline outputs (training catalog, translated catalog, …)
- `data/models/` — saved `.joblib` classifier files
- `data/filter_cache/`, `data/spectra_cache/` — SVO filter and spectral atlas
- `data/targets/{galaxy}/` — per-galaxy application outputs

## Configuration

All numerical parameters — feature filter set, blackbody temperature grid,
Chandra significance threshold, RF hyperparameters (`balanced_subsample`
weighting), UV systematic error budget, query retry counts — live in
`xclass/config.py`. There is no separate config file; edit the constants
there to retune.

Paths are anchored to the package root via `_PROJECT_ROOT`, so the install is
machine-independent once `pip install -e .` has been run.

## Citation

If you use XClass in your work, please cite:

```bibtex
@article{Rangelov2026,
    author       = {Rangelov, B. and {others}},
    title        = {Automated multi-wavelength classification of extragalactic
                    X-ray point sources in nearby spiral galaxies},
    journal      = {The Astrophysical Journal},
    year         = {2026},
    note         = {submitted}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contact

Open an issue on the project's GitHub repository for questions, bug reports,
or contributions.

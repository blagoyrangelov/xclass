"""Run manifest — records what the pipeline *actually did*.

The XClass pipeline has historically produced several *silent* substitutions,
each discovered only by accident:

    * the AGN composite template silently falling back to a power law;
    * HSC network failures silently recorded as "no counterpart";
    * 2MASS/PS1 network failures silently recorded as non-detections;
    * stale translate caches silently reused.

This module makes such degradation impossible to miss.  Each pipeline stage
builds a :class:`RunManifest`, writes it to
``data/processed/run_manifest_<stage>_<timestamp>.json``, and prints a
human-readable summary whose first section is an ``ALERTS`` block listing every
fallback that fired, every external resource that failed, and every class that
is absent.  If the ALERTS block is empty the run was clean.

The manifest is *pure instrumentation*: it reads the pipeline's outputs and
never changes a computed value.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from xclass import config

# Libraries whose version drift has (or could) move a published metric.
_TRACKED_LIBS = (
    "numpy", "scipy", "pandas", "scikit-learn",
    "astropy", "astroquery", "synphot", "joblib", "tenacity",
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    """Best-effort installed version of *name* (``'not installed'`` if absent)."""
    try:
        from importlib.metadata import version as _v
        return _v(name)
    except Exception:
        # Fall back to importing the module and reading __version__.
        mod_name = name.replace("-", "_")
        if mod_name == "scikit_learn":
            mod_name = "sklearn"
        try:
            import importlib
            return getattr(importlib.import_module(mod_name), "__version__", "unknown")
        except Exception:
            return "not installed"


def _git_commit() -> Optional[str]:
    """Return the short git commit hash of the package tree, or ``None``."""
    pkg_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _xclass_version() -> str:
    try:
        import xclass
        return getattr(xclass, "__version__", "unknown")
    except Exception:
        return "unknown"


def collect_provenance(timestamp: str) -> dict[str, Any]:
    """Gather provenance.  *timestamp* is supplied by the caller (the manifest
    is time-stamped once, at construction, by the stage code)."""
    return {
        "timestamp": timestamp,
        "xclass_version": _xclass_version(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {lib: _pkg_version(lib) for lib in _TRACKED_LIBS},
    }


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Structured record of one pipeline stage's actual behaviour."""

    stage: str
    provenance: dict[str, Any]
    resources: list[dict[str, Any]] = field(default_factory=list)
    sed_models: dict[str, Any] = field(default_factory=dict)
    caches: dict[str, Any] = field(default_factory=dict)
    catalog_shape: dict[str, Any] = field(default_factory=dict)
    fit_quality: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    # -- builders ---------------------------------------------------------
    def flag(self, msg: str) -> None:
        """Record a prominent alert (surfaced at the top of the summary)."""
        self.alerts.append(msg)

    def add_resource(self, name: str, status: str, *,
                     successes: Optional[int] = None,
                     non_detections: Optional[int] = None,
                     network_failures: Optional[int] = None,
                     detail: str = "") -> None:
        """Record an external resource's outcome.

        *status* is one of ``'ok'``, ``'cache'``, ``'fallback'``, ``'failed'``,
        ``'partial'``.  Success / genuine-non-detection / network-failure counts
        are reported separately (the distinction that was previously lost).
        """
        rec = {"name": name, "status": status, "detail": detail}
        if successes is not None:
            rec["successes"] = int(successes)
        if non_detections is not None:
            rec["non_detections"] = int(non_detections)
        if network_failures is not None:
            rec["network_failures"] = int(network_failures)
        self.resources.append(rec)
        if status == "failed":
            self.flag(f"RESOURCE FAILED: {name} — {detail or 'unavailable'}")
        elif status == "fallback":
            self.flag(f"FALLBACK IN USE: {name} — {detail or 'primary unavailable'}")
        if network_failures:
            self.flag(
                f"NETWORK FAILURES: {name} had {network_failures} failed queries "
                f"(not cached; a re-run will retry them)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provenance": self.provenance,
            "alerts": self.alerts,
            "resources": self.resources,
            "sed_models": self.sed_models,
            "caches": self.caches,
            "catalog_shape": self.catalog_shape,
            "fit_quality": self.fit_quality,
        }

    # -- output -----------------------------------------------------------
    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.provenance.get("timestamp", "unknown").replace(":", "").replace(" ", "_")
        path = out_dir / f"run_manifest_{self.stage}_{ts}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    def format_summary(self) -> str:
        """Human-readable summary; ALERTS first so degradation cannot be missed."""
        L: list[str] = []
        bar = "=" * 72
        L.append(bar)
        L.append(f"  XCLASS RUN MANIFEST — stage: {self.stage}")
        L.append(bar)

        # 1) ALERTS (always first, always visible)
        if self.alerts:
            L.append(f"  ⚠ ALERTS ({len(self.alerts)}) — the run degraded silently in these ways:")
            for a in self.alerts:
                L.append(f"      • {a}")
        else:
            L.append("  ✓ No alerts: no fallback fired, no resource failed, no class absent.")
        L.append("-" * 72)

        # 2) Provenance
        p = self.provenance
        L.append("  Provenance:")
        L.append(f"      timestamp     : {p.get('timestamp')}")
        L.append(f"      xclass version: {p.get('xclass_version')}   git: {p.get('git_commit') or 'n/a'}")
        L.append(f"      python        : {p.get('python')}")
        libs = p.get("libraries", {})
        lib_str = "  ".join(f"{k}={v}" for k, v in libs.items())
        L.append(f"      libraries     : {lib_str}")

        # 3) SED models
        if self.sed_models:
            L.append("-" * 72)
            L.append("  SED models actually used (per class):")
            for cls, info in self.sed_models.items():
                prim = info["primary"]
                mark = "  <-- FALLBACK FIRED" if info.get("fallback_fired") else ""
                L.append(f"      {cls:<8} primary={prim:<22} "
                         f"used_primary={info['n_primary']:>5}  "
                         f"fallback/other={info['n_fallback']:>5}  "
                         f"unfit={info['n_unfit']:>4}{mark}")
                fam = "  ".join(f"{k}:{v}" for k, v in info["families"].items())
                L.append(f"               families: {fam}")

        # 4) External resources
        if self.resources:
            L.append("-" * 72)
            L.append("  External resources:")
            for r in self.resources:
                counts = []
                for key, lab in (("successes", "ok"),
                                 ("non_detections", "no-counterpart"),
                                 ("network_failures", "net-fail")):
                    if key in r:
                        counts.append(f"{r[key]} {lab}")
                cstr = f"  [{', '.join(counts)}]" if counts else ""
                d = f"  ({r['detail']})" if r.get("detail") else ""
                L.append(f"      {r['name']:<20} {r['status']:<9}{cstr}{d}")

        # 5) Caches
        if self.caches:
            L.append("-" * 72)
            L.append("  Cache usage:")
            for k, v in self.caches.items():
                L.append(f"      {k}: {v}")

        # 6) Catalog shape
        if self.catalog_shape:
            L.append("-" * 72)
            L.append("  Catalog shape:")
            for k, v in self.catalog_shape.items():
                if k == "per_class":
                    L.append(f"      per-class: {v}")
                elif k == "absent_classes":
                    tag = "  ⚠" if v else ""
                    L.append(f"      absent classes: {v or 'none'}{tag}")
                else:
                    L.append(f"      {k}: {v}")

        # 7) Fit quality
        if self.fit_quality:
            L.append("-" * 72)
            L.append("  Fit quality — median chi2_reduced per class "
                     "(median is the published statistic; the mean is outlier-dominated):")
            for cls, m in self.fit_quality.items():
                L.append(f"      {cls:<8} median={m['median']:>12.1f}   "
                         f"(mean={m['mean']:.3e}, q90={m['q90']:.1f}, N={m['n']})")

        L.append(bar)
        return "\n".join(L)


# ---------------------------------------------------------------------------
# SED / catalog summarisers (shared by the live stage and offline gate checks)
# ---------------------------------------------------------------------------

def _canonical_primary(class_name: str) -> str:
    """Map a config primary-model name to the ``xclass_sed_family`` value it
    produces (``two_component_k5_disk`` -> ``two_component``)."""
    p = config.SED_MODEL_PRIMARY.get(class_name, "")
    if p.startswith("two_component"):
        return "two_component"
    return p


def summarize_sed_models(df: pd.DataFrame, class_col: str) -> dict[str, Any]:
    """Per-class SED family counts + primary/fallback accounting.

    A source is counted as *unfit* when its family is ``none`` (no usable bands
    or an SNR bypass row).  Any non-``none`` family that is not the intended
    primary counts as *fallback/other* and raises a per-class alert flag.
    """
    out: dict[str, Any] = {}
    if class_col not in df.columns or "xclass_sed_family" not in df.columns:
        return out
    for cls, g in df.groupby(class_col):
        primary = _canonical_primary(str(cls))
        fam = g["xclass_sed_family"].astype(str).value_counts()
        n_unfit = int(fam.get("none", 0))
        n_primary = int(fam.get(primary, 0)) if primary and primary != "none" else 0
        n_fallback = int(len(g) - n_primary - n_unfit)
        # SNR (primary 'none') never "falls back"; its bypass is expected.
        fallback_fired = bool(n_fallback > 0 and primary not in ("none", ""))
        out[str(cls)] = {
            "primary": primary or "n/a",
            "families": {str(k): int(v) for k, v in fam.items()},
            "n_primary": n_primary,
            "n_fallback": n_fallback,
            "n_unfit": n_unfit,
            "fallback_fired": fallback_fired,
        }
    return out


def summarize_fit_quality(df: pd.DataFrame, class_col: str) -> dict[str, Any]:
    """Per-class median (published statistic), mean and Q90 of chi2_reduced."""
    out: dict[str, Any] = {}
    if class_col not in df.columns or "xclass_fit_chi2red" not in df.columns:
        return out
    for cls, g in df.groupby(class_col):
        x = pd.to_numeric(g["xclass_fit_chi2red"], errors="coerce").dropna()
        if len(x) == 0:
            continue
        out[str(cls)] = {
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "q90": float(np.percentile(x, 90)),
            "n": int(len(x)),
        }
    return out


def summarize_catalog_shape(df: pd.DataFrame, class_col: str,
                            n_optical: Optional[int] = None) -> dict[str, Any]:
    """Source counts, per-class counts, and which expected classes are absent."""
    shape: dict[str, Any] = {"n_translated": int(len(df))}
    if n_optical is not None:
        shape["n_optical_baseline"] = int(n_optical)
    if class_col in df.columns:
        pc = df[class_col].value_counts()
        shape["per_class"] = {str(k): int(v) for k, v in pc.items()}
        expected = set(config.SED_MODEL_PRIMARY.keys())
        present = set(map(str, pc.index))
        shape["absent_classes"] = sorted(expected - present)
    return shape


def derive_snr_counts(df: pd.DataFrame, class_col: str) -> dict[str, int]:
    """Recover SNR HSC patched / no-counterpart counts from the catalog itself.

    Used when live per-source counters are unavailable (e.g. building a manifest
    offline from an existing catalog).  Network-failure count cannot be recovered
    post-hoc and is reported as 0 — a live run supplies the true value via
    ``snr_counts``.
    """
    empty = {"patched": 0, "no_counterpart": 0, "network_failures": 0}
    if class_col not in df.columns:
        return empty
    snr = df[df[class_col].astype(str) == "SNR"]
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    if len(snr) == 0 or not pred_cols:
        return empty
    has = snr[pred_cols].notna().any(axis=1)
    return {"patched": int(has.sum()),
            "no_counterpart": int((~has).sum()),
            "network_failures": 0}


def build_translate_manifest(
    df: pd.DataFrame,
    *,
    timestamp: str,
    class_col: Optional[str] = None,
    n_optical: Optional[int] = None,
    agn_composite_available: bool,
    snr_counts: Optional[dict[str, int]] = None,
    cache_hit: bool = False,
    force: bool = False,
    extra_resources: Optional[list[dict[str, Any]]] = None,
) -> RunManifest:
    """Assemble the translate-stage manifest from a translated catalog.

    Shared by the live ``stage_translate`` path and offline gate checks so the
    two cannot diverge.  *agn_composite_available* and *snr_counts* come from the
    live run; when omitted, SNR counts are derived from the catalog.
    """
    m = RunManifest(stage="translate", provenance=collect_provenance(timestamp))
    if class_col is None:
        class_col = next((c for c in ("Class", "class_label") if c in df.columns), "Class")

    m.sed_models = summarize_sed_models(df, class_col)
    m.fit_quality = summarize_fit_quality(df, class_col)
    m.catalog_shape = summarize_catalog_shape(df, class_col, n_optical=n_optical)

    for cls, info in m.sed_models.items():
        if info["fallback_fired"]:
            m.flag(f"SED FALLBACK: class {cls} used {info['n_fallback']} fallback/other "
                   f"fits instead of primary '{info['primary']}' "
                   f"(primary fired for {info['n_primary']}).")
    for cls in m.catalog_shape.get("absent_classes", []):
        m.flag(f"CLASS ABSENT: expected class {cls} has 0 sources in the catalog.")

    # AGN composite template — the item that would have caught the day-one bug.
    if agn_composite_available:
        m.add_resource("AGN composite (Vanden Berk)", "ok",
                       detail="template loaded; AGN fit with composite")
    else:
        m.add_resource("AGN composite (Vanden Berk)", "fallback",
                       detail="template unavailable -> AGN fit with power-law fallback")

    # HSC v3 SNR photometry (network failures counted separately).
    if snr_counts is None:
        snr_counts = derive_snr_counts(df, class_col)
    patched = int(snr_counts.get("patched", 0))
    noc = int(snr_counts.get("no_counterpart", 0))
    nf = int(snr_counts.get("network_failures", 0))
    total = patched + noc + nf
    status = "ok" if (nf == 0 and patched > 0) else ("partial" if patched > 0 else "failed")
    m.add_resource("HSC v3 (SNR photometry)", status,
                   successes=patched, non_detections=noc, network_failures=nf,
                   detail=f"{patched}/{total} SNRs patched with >=1 PHAT filter")

    m.caches["translate_cache"] = ("REUSED (input fingerprint matched)" if cache_hit
                                   else "written (fresh translation)")
    m.caches["force_retranslate"] = bool(force)

    for r in (extra_resources or []):
        m.add_resource(**r)
    return m

#!/usr/bin/env python3
"""Regenerate figures/sed_validation_quality.pdf — 2-panel single-column version.

Panel 1 (top):  log10(chi2_red) distribution by class, dashed cut at chi2_red = 10
Panel 2 (bottom): predicted ACS/F475W magnitude distribution by class
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRANSLATED_CATALOG = ROOT / "data" / "processed" / "translated_catalog.csv"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_ORDER = ["AGN", "LMXB", "HMXB", "CV", "LM-STAR", "HM-STAR"]
CLASS_COLORS = {
    "AGN":     "#1f77b4",
    "LMXB":    "#d62728",
    "HMXB":    "#ff7f0e",
    "CV":      "#9467bd",
    "LM-STAR": "#2ca02c",
    "HM-STAR": "#8c564b",
    "SNR":     "#e377c2",
}

CHI2_COL  = "xclass_fit_chi2red"
F475W_COL = "ACS_F475W_pred"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def main() -> None:
    print(f"Loading {TRANSLATED_CATALOG}")
    df = pd.read_csv(TRANSLATED_CATALOG, low_memory=False)
    print(f"  {len(df):,} sources")

    # ── Panel 1: log10(chi2_red) ──────────────────────────────────────────────
    has_chi2 = CHI2_COL in df.columns and df[CHI2_COL].notna().any()
    has_f475 = F475W_COL in df.columns and df[F475W_COL].notna().any()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.5))
    fig.subplots_adjust(hspace=0.38)

    # ── top: chi2_red distribution ────────────────────────────────────────────
    if has_chi2:
        # Focus on the physically meaningful range; SNRs skip SED fitting
        df_fit = df[df["Class"] != "SNR"]
        bins = np.linspace(-2, 4, 49)   # log10 range: chi2_red 0.01 → 10000

        for cls in CLASS_ORDER:
            sub = df_fit[df_fit["Class"] == cls][CHI2_COL].dropna()
            lsub = np.log10(sub.clip(1e-2, 1e4))
            if len(lsub) < 2:
                continue
            ax1.hist(lsub, bins=bins, histtype="step",
                     color=CLASS_COLORS[cls], lw=1.2, label=cls, density=True)

        ax1.axvline(0.0, color="k",   lw=0.8, ls="--", label=r"$\chi^2_{\rm red}=1$")
        ax1.axvline(1.0, color="0.5", lw=0.8, ls=":",  label="cut = 10")

        n_tot  = len(df_fit[CHI2_COL].dropna())
        n_good = (df_fit[CHI2_COL].dropna() < 10).sum()
        ax1.text(0.98, 0.97, f"{n_good}/{n_tot} pass $\\chi^2_{{\\rm red}}<10$",
                 transform=ax1.transAxes, fontsize=6.5,
                 va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

        ax1.set_xlabel(r"$\log_{10}(\chi^2_{\rm red})$", labelpad=2)
        ax1.set_ylabel("Density", labelpad=2)
        ax1.set_title(r"SED fit quality by class", pad=4)
        ax1.set_xlim(-2, 4)

        # Single 2-column legend (classes + reference lines) at upper left,
        # where the step-histogram density is lowest (log10 < -1.5)
        ax1.legend(ncol=2, fontsize=6.5,
                   frameon=True, framealpha=0.9, edgecolor="0.7",
                   loc="upper left",
                   handlelength=1.4, handletextpad=0.4, columnspacing=0.8,
                   borderpad=0.4)
    else:
        ax1.text(0.5, 0.5, "no $\\chi^2$ data", transform=ax1.transAxes,
                 ha="center", va="center")

    # ── bottom: F475W distribution ────────────────────────────────────────────
    if has_f475:
        bins = np.linspace(15, 35, 41)

        for cls in CLASS_ORDER:
            sub = df[df["Class"] == cls][F475W_COL].dropna()
            sub = sub[(sub > 15) & (sub < 35)]
            if len(sub) < 2:
                continue
            ax2.hist(sub, bins=bins, histtype="step",
                     color=CLASS_COLORS[cls], lw=1.2, label=cls, density=True)

        ax2.set_xlabel(r"$m_{\rm pred}$ (AB, F475W)", labelpad=2)
        ax2.set_ylabel("Density", labelpad=2)
        ax2.set_title("Predicted F475W distribution by class", pad=4)

        # Legend: place in upper left (magnitudes brightest at left, low density there)
        ax2.legend(ncol=2, fontsize=6.5,
                   frameon=True, framealpha=0.9, edgecolor="0.7",
                   loc="upper left",
                   handlelength=1.4, handletextpad=0.4, columnspacing=0.8,
                   borderpad=0.4)
    else:
        ax2.text(0.5, 0.5, "no F475W data", transform=ax2.transAxes,
                 ha="center", va="center")

    # ── shared styling ────────────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)

    out = FIGURES_DIR / "sed_validation_quality.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

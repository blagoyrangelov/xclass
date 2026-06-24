"""xclass.diagnostics — Visualisation and reporting.

All plotting functions for confusion matrices, SEDs, colour-magnitude
diagrams, ROC curves, and feature importance.  Uses the class colour
scheme from ``config.CLASS_COLORS`` consistently.

Functions
---------
plot_confusion_matrix
plot_feature_importance
plot_class_distributions
plot_color_magnitude_diagram
plot_xray_optical_ratio
plot_sed_fit
plot_roc_curves
generate_report
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xclass import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig, save_path: Optional[str | Path]) -> None:
    """Save and close figure when *save_path* is given; otherwise do nothing.

    Callers that want interactive display should call ``plt.show()``
    themselves after adding any final decorations (suptitle, tight_layout,
    etc.).  This makes the functions composable in notebooks and scripts.
    """
    import matplotlib.pyplot as plt

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved figure to %s", save_path)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    normalize: bool = True,
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str
        Class name labels for axes.
    normalize : bool
        If True, normalise by row (true class).
    save_path : str or Path, optional
        Save figure to this path instead of displaying.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=class_names,
                          normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""))
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """Bar chart of the top-N most important features.

    Parameters
    ----------
    model : fitted RF or XGBoost estimator
    feature_names : list of str
    top_n : int
        Number of top features to show.
    save_path : optional path
    """
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} feature importances")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_class_distributions(
    df: pd.DataFrame,
    class_col: str = "class_label",
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """Bar chart of source counts per class.

    Parameters
    ----------
    df : pd.DataFrame
    class_col : str
        Column holding class labels.
    save_path : optional path
    """
    import matplotlib.pyplot as plt

    counts = df[class_col].value_counts().reindex(
        [c for c in config.ALL_CLASSES if c in df[class_col].unique()]
    ).dropna()
    colors = [config.CLASS_COLORS.get(c, "#888") for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Source class distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_color_magnitude_diagram(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    class_col: str = "class_pred",
    save_path: Optional[str | Path] = None,
    *,
    mag_col: Optional[str] = None,
    color_col1: Optional[str] = None,
    color_col2: Optional[str] = None,
) -> "plt.Figure":
    """Scatter plot colour-magnitude diagram coloured by class.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str, optional
        Column for the x-axis (colour or magnitude).  Superseded by
        *color_col1* / *color_col2* when those are provided.
    y_col : str, optional
        Column for the y-axis.  Superseded by *mag_col* when provided.
    class_col : str
        Column with class labels for colouring.
    save_path : optional path
    mag_col : str, optional
        Magnitude column to use as the y-axis.
    color_col1, color_col2 : str, optional
        Two magnitude columns whose difference forms the x-axis colour.
    """
    import matplotlib.pyplot as plt

    # Resolve axis data: colour-magnitude interface takes priority
    if mag_col is not None:
        y_data = pd.to_numeric(df[mag_col], errors="coerce")
        y_label = mag_col
    elif y_col is not None:
        y_data = pd.to_numeric(df[y_col], errors="coerce")
        y_label = y_col
    else:
        raise ValueError("Provide either y_col or mag_col")

    if color_col1 is not None and color_col2 is not None:
        x_data = (
            pd.to_numeric(df[color_col1], errors="coerce")
            - pd.to_numeric(df[color_col2], errors="coerce")
        )
        short1 = color_col1.split("_")[-1]
        short2 = color_col2.split("_")[-1]
        x_label = f"{short1}−{short2}"
    elif x_col is not None:
        x_data = pd.to_numeric(df[x_col], errors="coerce")
        x_label = x_col
    else:
        raise ValueError("Provide either x_col or color_col1+color_col2")

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted = False
    for cls in config.ALL_CLASSES:
        if class_col not in df.columns:
            break
        mask = df[class_col] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            x_data[mask],
            y_data[mask],
            c=config.CLASS_COLORS.get(cls, "#888"),
            label=cls,
            s=15,
            alpha=0.7,
        )
        plotted = True

    if not plotted:
        ax.scatter(x_data, y_data, s=10, alpha=0.5, color="steelblue")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD: {y_label} vs {x_label}")
    if y_label.endswith("_pred") or "mag" in y_label.lower():
        ax.invert_yaxis()
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_xray_optical_ratio(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    class_col: Optional[str] = None,
) -> "plt.Figure":
    """Plot X-ray/optical flux ratio diagnostic.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain logFxFopt_B_F475W, logFxFopt_B_F814W, and class column.
    save_path : optional path
    class_col : str, optional
        Column to use for class colouring.  If None, auto-detects from
        ``["Class", "class_label", "class_pred"]``.
    """
    import matplotlib.pyplot as plt

    if class_col is None:
        class_col = next(
            (c for c in ["Class", "class_label", "class_pred"] if c in df.columns),
            None,
        )
    x_col = "logFxFopt_B_F475W"
    y_col = "logFxFopt_B_F814W"

    if x_col not in df.columns or y_col not in df.columns:
        log.warning("plot_xray_optical_ratio: missing required columns %s or %s", x_col, y_col)
        _fig, _ax = plt.subplots(figsize=(8, 6))
        _ax.text(0.5, 0.5, "X-ray/optical ratio columns not available",
                 ha="center", va="center", transform=_ax.transAxes, color="gray")
        _ax.set_title("X-ray/optical flux ratio (data not available)")
        return _fig

    fig, ax = plt.subplots(figsize=(8, 6))

    if class_col:
        for cls in config.ALL_CLASSES:
            mask = df[class_col] == cls
            if mask.sum() == 0:
                continue
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                c=config.CLASS_COLORS.get(cls, "#888"),
                label=cls,
                s=15,
                alpha=0.7,
            )
        ax.legend(markerscale=2, fontsize=8)
    else:
        ax.scatter(df[x_col], df[y_col], s=10, alpha=0.5)

    ax.set_xlabel("log(Fx_B / f_opt_F475W)")
    ax.set_ylabel("log(Fx_B / f_opt_F814W)")
    ax.set_title("X-ray/optical flux ratio diagnostic")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_sed_fit(
    row: pd.Series,
    filter_curves: dict,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot observed photometry and best-fit SED model.

    Shows:
    - Observed photometry as filled circles with error bars
    - Best-fit SED model as solid line (log-log scale)
    - Predicted HST filter fluxes as coloured rectangles spanning filter FWHM

    Labels: source class, SpType (if available), chi2_reduced, SED model type.

    Parameters
    ----------
    row : pd.Series
        One row from the translated catalog.
    filter_curves : dict
        Full filter curve dictionary.
    save_path : optional path
    """
    import matplotlib.pyplot as plt
    from xclass.photometry import abmag_to_fnu

    fig, ax = plt.subplots(figsize=(10, 5))

    # Survey photometry columns to plot
    _SURVEY_COLS = {
        "ps1_g": "ps1_g_err", "ps1_r": "ps1_r_err", "ps1_i": "ps1_i_err",
        "ps1_z": "ps1_z_err", "ps1_y": "ps1_y_err",
        "tmass_j": "tmass_j_err", "tmass_h": "tmass_h_err", "tmass_k": "tmass_k_err",
    }

    # Pivot wavelengths for known filters (Angstrom)
    _SURVEY_PIVOTS = {
        "ps1_g": 4810, "ps1_r": 6170, "ps1_i": 7520, "ps1_z": 8660, "ps1_y": 9620,
        "tmass_j": 12350, "tmass_h": 16620, "tmass_k": 21590,
    }

    # Plot observed photometry
    for col, err_col in _SURVEY_COLS.items():
        if col not in row.index:
            continue
        mag = float(row.get(col, np.nan))
        err = float(row.get(err_col, 0.1)) if err_col else 0.1
        if not np.isfinite(mag):
            continue
        pivot = _SURVEY_PIVOTS.get(col, 5500)
        fnu = abmag_to_fnu(mag)
        fnu_err = fnu * np.log(10) / 2.5 * err
        ax.errorbar(pivot, fnu, yerr=fnu_err, fmt="o", color="k", ms=6, capsize=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("f_ν (erg/s/cm²/Hz)")

    src_class = row.get("Class", row.get("class_label", "unknown"))
    sptype = row.get("SpType", "")
    chi2 = row.get("xclass_fit_chi2red", float("nan"))
    family = row.get("xclass_sed_family", "")

    title = f"Class: {src_class}"
    if sptype:
        title += f"  SpType: {sptype}"
    title += f"  SED: {family}  χ²_red: {chi2:.2f}" if np.isfinite(chi2) else f"  SED: {family}"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_roc_curves(
    y_true,
    y_proba: pd.DataFrame,
    class_names: list[str],
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """One-vs-rest ROC curves for all classes.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_proba : pd.DataFrame
        Class probability columns (output of ``classifier.predict``).
    class_names : list of str
    save_path : optional path
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=class_names)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(class_names):
        prob_col = f"prob_{i}"  # encoded probability column
        # Try class-name based column first
        for candidate in [f"prob_{cls}", f"prob_{i}", f"prob_{cls.lower()}"]:
            if candidate in y_proba.columns:
                prob_col = candidate
                break

        if prob_col not in y_proba.columns:
            continue

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[prob_col])
        roc_auc = auc(fpr, tpr)
        color = config.CLASS_COLORS.get(cls, "#888")
        ax.plot(fpr, tpr, lw=1.5, color=color, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def generate_report(
    metrics_dict: Optional[dict] = None,
    save_dir: Optional[str | Path] = None,
    *,
    metrics: Optional[dict] = None,
    class_counts: Optional[dict] = None,
    extra_lines: Optional[list[str]] = None,
) -> None:
    """Write a plain-text metrics report.

    Parameters
    ----------
    metrics_dict : dict, optional
        Metrics from Stage 1 and Stage 2 training (and CV).
        May also be passed as keyword argument ``metrics``.
    save_dir : str or Path
        Directory where ``report.txt`` will be written.
    metrics : dict, optional
        Alias for *metrics_dict* (keyword-only).
    class_counts : dict, optional
        Mapping of class name → count to include in the report.
    extra_lines : list of str, optional
        Additional free-text lines appended at the end of the report.
    """
    # Resolve metrics_dict from positional or keyword arg
    if metrics_dict is None:
        metrics_dict = metrics or {}

    if save_dir is None:
        raise ValueError("save_dir must be provided")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "report.txt"

    lines = ["xclass Classification Report", "=" * 60, ""]

    for stage, stage_metrics in metrics_dict.items():
        lines.append(f"--- {stage} ---")
        for key, val in stage_metrics.items():
            if key in ("confusion_matrix", "classification_report"):
                continue
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")
        lines.append("")

    if class_counts:
        lines.append("--- Class counts ---")
        for cls, cnt in class_counts.items():
            lines.append(f"  {cls}: {cnt}")
        lines.append("")

    if extra_lines:
        lines.append("--- Notes ---")
        lines.extend(extra_lines)
        lines.append("")

    report_path.write_text("\n".join(lines))
    log.info("Report written to %s", report_path)

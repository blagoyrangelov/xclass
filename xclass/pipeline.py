"""xclass.pipeline — single-source production pipeline (consolidated 2026-06).

Houses the production end-to-end logic (optical baseline N=11,374, no SMOTE,
balanced_subsample, 3-class Stage 2 = LMXB/HMXB/CV, seed 42, 600 trees) that was
previously split across scripts/eval_production.py and
scripts/eval_optical_baseline.py.  The functions below were RELOCATED VERBATIM in
behaviour from eval_production.py (constants, impute_median, make_rf, two_stage_cv,
train_and_save_production_model, compute_summary, calibration, figures, LaTeX tables);
only imports/namespacing changed.  Orchestration wrappers at the bottom
(build_optical_baseline / train_production / evaluate_production / apply_production)
are the importable entry points used by scripts/run_pipeline.py.

Reproduces the published numbers: LMXB F1 = 0.77, HMXB F1 = 0.95, balanced acc = 0.91.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from xclass import config
from xclass.features import build_feature_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_SUFFIX = "optical_v2"

RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",   # no SMOTE; class weight handles imbalance
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}

STAGE1_MAP = {
    "AGN": "AGN", "LM-STAR": "STAR", "HM-STAR": "STAR",
    "LMXB": "OTHER", "HMXB": "OTHER", "CV": "OTHER", "SNR": "SNR",
}
S1_CLASSES    = ["AGN", "OTHER", "SNR", "STAR"]      # alphabetical for LE
S2_CLASSES    = ["CV", "HMXB", "LMXB"]
FINAL_CLASSES = ["AGN", "CV", "HMXB", "LMXB", "SNR", "STAR"]
EVAL5         = ["AGN", "STAR", "SNR", "LMXB", "HMXB"]

CONF_THRESH = 0.5
N_SPLITS    = 5
N_CAL_BINS  = 10

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = config.MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── ApJ matplotlib style ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def impute_median(feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Global median imputation. Returns (X_imputed, medians_per_column)."""
    out = feat_df.copy()
    medians = []
    for col in out.columns:
        med = out[col].median()   # skipna=True; returns NaN only if all NaN
        if pd.isna(med):
            med = 0.0             # all-NaN column (e.g. Gaia missing entirely)
        if out[col].isna().any():
            out[col] = out[col].fillna(med)
        medians.append(med)
    return out.values.astype(float), np.array(medians, dtype=float)


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


# ── two-stage CV ──────────────────────────────────────────────────────────────

def two_stage_cv(df: pd.DataFrame) -> dict:
    """Stratified 5-fold two-stage CV.

    Returns dict with OOF arrays, per-fold feature importances,
    and Stage-1 OOF probability matrix (for multi-class Brier score).
    """
    feat_df, feat_names = build_feature_matrix(df)
    X, _ = impute_median(feat_df)

    labels7 = df["Class"].values
    labels1 = np.array([STAGE1_MAP[c] for c in labels7])
    labels6 = np.array(["STAR" if c in ("LM-STAR", "HM-STAR") else c for c in labels7])

    le1 = LabelEncoder().fit(S1_CLASSES)
    le2 = LabelEncoder().fit(S2_CLASSES)
    y1  = le1.transform(labels1)
    y2_full = np.array([c if c in S2_CLASSES else "___" for c in labels7])

    other_class_idx = le1.transform(["OTHER"])[0]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=config.RANDOM_STATE)

    s1_true_oof, s1_pred_oof, s1_conf_oof = [], [], []
    s1_probs_oof = np.zeros((len(df), len(S1_CLASSES)))   # full 4-class prob matrix
    fp_true_oof, fp_pred_oof, fp_conf_oof = [], [], []
    importances = []
    oof_indices = []

    for tr_idx, va_idx in skf.split(X, y1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y1_tr, y1_va = y1[tr_idx], y1[va_idx]

        # Stage 1
        s1 = make_rf()
        s1.fit(X_tr, y1_tr)
        importances.append(s1.feature_importances_)

        s1_prob_tr = s1.predict_proba(X_tr)
        s1_prob_va = s1.predict_proba(X_va)
        s1_pred_va = s1.predict(X_va)

        s1_true_oof.extend(le1.inverse_transform(y1_va))
        s1_pred_oof.extend(le1.inverse_transform(s1_pred_va))
        s1_conf_oof.extend(s1_prob_va.max(axis=1))
        s1_probs_oof[va_idx] = s1_prob_va
        oof_indices.extend(va_idx)

        # Stage 2 — trained on OTHER sources in training fold
        other_tr = y1_tr == other_class_idx
        X2_tr = np.hstack([X_tr[other_tr], s1_prob_tr[other_tr]])
        y2_tr = le2.transform(y2_full[tr_idx][other_tr])
        s2 = make_rf()
        s2.fit(X2_tr, y2_tr)

        # Full pipeline on val fold
        final_pred = le1.inverse_transform(s1_pred_va).copy()
        final_conf = s1_prob_va.max(axis=1).copy()

        other_va = s1_pred_va == other_class_idx
        if other_va.any():
            X2_va      = np.hstack([X_va[other_va], s1_prob_va[other_va]])
            s2_pred_va = le2.inverse_transform(s2.predict(X2_va))
            s2_prob_va = s2.predict_proba(X2_va).max(axis=1)
            final_pred[other_va] = s2_pred_va
            final_conf[other_va] = s2_prob_va

        fp_true_oof.extend(labels6[va_idx])
        fp_pred_oof.extend(final_pred)
        fp_conf_oof.extend(final_conf)

    return {
        "feat_names":   feat_names,
        "s1_true":      np.array(s1_true_oof),
        "s1_pred":      np.array(s1_pred_oof),
        "s1_conf":      np.array(s1_conf_oof,  dtype=float),
        "s1_probs":     s1_probs_oof,            # (N, 4) full probability matrix
        "s1_classes":   le1.classes_,
        "fp_true":      np.array(fp_true_oof),
        "fp_pred":      np.array(fp_pred_oof),
        "fp_conf":      np.array(fp_conf_oof,  dtype=float),
        "importances":  np.array(importances),
        "oof_indices":  np.array(oof_indices),
        "labels7":      labels7,
        "labels6":      labels6,
        "labels1":      labels1,
    }


# ── calibration metrics ───────────────────────────────────────────────────────

def calibration_metrics(conf: np.ndarray, is_correct: np.ndarray,
                        n_bins: int = N_CAL_BINS) -> dict:
    """ECE, MCE, Brier from (confidence, is_correct) arrays."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts, bin_los, bin_his = [], [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        n = mask.sum()
        if n == 0:
            continue
        bin_accs.append(is_correct[mask].mean())
        bin_confs.append(conf[mask].mean())
        bin_counts.append(n)
        bin_los.append(lo)
        bin_his.append(hi)

    bin_accs   = np.array(bin_accs)
    bin_confs  = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    weights    = bin_counts / bin_counts.sum()

    ece = float(np.sum(weights * np.abs(bin_accs - bin_confs)))
    mce = float(np.max(np.abs(bin_accs - bin_confs)))
    brier = float(np.mean((conf - is_correct.astype(float)) ** 2))

    return dict(ece=ece, mce=mce, brier=brier,
                bin_accs=bin_accs, bin_confs=bin_confs, bin_counts=bin_counts,
                bin_los=np.array(bin_los), bin_his=np.array(bin_his))


def s1_brier_multiclass(s1_probs: np.ndarray, s1_true: np.ndarray,
                         s1_classes: np.ndarray) -> float:
    """Full multi-class Brier score for Stage 1 (4-class) using OOF probability matrix."""
    le = LabelEncoder().fit(s1_classes)
    y_enc = le.transform(s1_true)
    n, k = s1_probs.shape
    y_onehot = np.zeros((n, k))
    y_onehot[np.arange(n), y_enc] = 1.0
    return float(np.mean(np.sum((s1_probs - y_onehot) ** 2, axis=1)))


def compute_summary(true, pred, conf) -> dict:
    mask_c = conf >= CONF_THRESH

    def _m(t, p):
        if len(t) == 0:
            return dict(acc=np.nan, bacc=np.nan, f1=np.nan, mcc=np.nan, n=0)
        return dict(
            acc=accuracy_score(t, p),
            bacc=balanced_accuracy_score(t, p),
            f1=f1_score(t, p, average="macro", zero_division=0),
            mcc=matthews_corrcoef(t, p),
            n=len(t),
        )

    all_m  = _m(true, pred)
    conf_m = _m(true[mask_c], pred[mask_c])
    cr = classification_report(true, pred, output_dict=True, zero_division=0)
    return {"all": all_m, "conf": conf_m, "cr": cr,
            "frac_confident": mask_c.mean()}


# ── figures ───────────────────────────────────────────────────────────────────

def fig_feature_importance(cv: dict, out: Path):
    imp   = cv["importances"].mean(axis=0)
    names = cv["feat_names"]
    order = np.argsort(imp)[::-1][:15]
    y_pos = np.arange(15)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.barh(y_pos, imp[order][::-1], color="#4878CF", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order[::-1]], fontsize=7)
    ax.set_xlabel("Mean Gini importance")
    ax.set_title("Stage 1 feature importance (top 15)")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_confusion_matrices(cv: dict, out: Path):
    s1_labels = S1_CLASSES
    fp_labels = FINAL_CLASSES

    s1_t = cv["s1_true"]; s1_p = cv["s1_pred"]
    fp_t = cv["fp_true"]; fp_p = cv["fp_pred"]

    valid_fp = np.isin(fp_t, fp_labels) & np.isin(fp_p, fp_labels)
    fp_t, fp_p = fp_t[valid_fp], fp_p[valid_fp]

    cm1 = confusion_matrix(s1_t, s1_p, labels=s1_labels, normalize="true")
    cm2 = confusion_matrix(fp_t, fp_p, labels=fp_labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    kw = dict(cmap="Blues", vmin=0, vmax=1)

    for ax, cm, labs, title in [
        (axes[0], cm1, s1_labels, "Stage 1 (4-class)"),
        (axes[1], cm2, fp_labels, "Full pipeline (6-class)"),
    ]:
        im = ax.imshow(cm, **kw)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        ax.tick_params(which="both", direction="in", top=True, right=True,
                       bottom=False, left=False)
        for i in range(len(labs)):
            for j in range(len(labs)):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_per_class_metrics(cv: dict, out: Path):
    cr = compute_summary(cv["fp_true"], cv["fp_pred"], cv["fp_conf"])["cr"]
    metrics = ["precision", "recall", "f1-score"]
    labels_disp = ["Prec.", "Rec.", "F1"]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    x = np.arange(len(EVAL5))
    width = 0.22

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for i, (m, label, color) in enumerate(zip(metrics, labels_disp, colors)):
        vals = [cr.get(c, {}).get(m, 0) for c in EVAL5]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL5, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics (full pipeline)")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_confidence_distribution(cv: dict, out: Path):
    fp_true = cv["fp_true"]
    fp_pred = cv["fp_pred"]
    fp_conf = cv["fp_conf"]
    correct   = fp_conf[fp_true == fp_pred]
    incorrect = fp_conf[fp_true != fp_pred]

    bins = np.linspace(0.4, 1, 13)   # 0.05-wide bins over the populated range
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.hist(correct,   bins=bins, alpha=0.7, color="#4878CF", label="Correct")
    ax.hist(incorrect, bins=bins, alpha=0.7, color="#D65F5F", label="Incorrect")
    ax.axvline(CONF_THRESH, color="k", linestyle="--", lw=1.0,
               label=f"$p={CONF_THRESH}$")
    ax.set_yscale("log")
    ax.set_xlabel("Max class probability (confidence)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Confidence distribution (full pipeline)")
    ax.legend(framealpha=0.9)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlim(0.4, 1.0)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_calibration_reliability(cv: dict, cal: dict, out: Path):
    """Reliability diagram with ECE annotation (full pipeline, max-prob confidence)."""
    bin_confs  = cal["bin_confs"]
    bin_accs   = cal["bin_accs"]
    bin_counts = cal["bin_counts"]
    ece        = cal["ece"]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Perfect calibration diagonal spanning the visible range only
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=0.8, label="Perfect calibration")

    # Reliability curve — symbol size proportional to bin count
    max_count = bin_counts.max()
    sizes = 20 + 80 * (bin_counts / max_count)
    ax.scatter(bin_confs, bin_accs, s=sizes, c="#4878CF",
               zorder=3, label="Classifier")
    ax.plot(bin_confs, bin_accs, color="#4878CF", lw=1.2, zorder=2)

    # Gap-fill bars (calibration gap)
    for bc, ba in zip(bin_confs, bin_accs):
        ax.plot([bc, bc], [bc, ba], color="#D65F5F", lw=1.5, alpha=0.7, zorder=1)

    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Reliability diagram (full pipeline)")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    ax.text(0.97, 0.05,
            f"ECE = {ece:.4f}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def fmt(v, d=3):
    return "\\nodata" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def print_table4(s1_m: dict, fp_m: dict):
    print(r"""
\begin{deluxetable*}{lcccccc}
\tablecaption{Cross-validation performance on the optical training baseline
  ($N = 11{,}374$). \textit{No SMOTE; class weight = balanced\_subsample.}
  ``All'' includes every source; ``Confident'' restricts
  to sources with maximum class probability $\geq 0.50$.
  \label{tab:cv_metrics}}
\tablehead{
  \colhead{Subset} &
  \colhead{$N$} &
  \colhead{Accuracy} &
  \colhead{Balanced Acc.} &
  \colhead{$F_1$ (macro)} &
  \colhead{MCC} &
  \colhead{Fraction}
}
\startdata""")

    rows = [
        (r"\noalign{\vskip2pt}\textit{Stage~1 (4-class: AGN/STAR/SNR/OTHER)}",
         None),
        (r"\quad All",
         (s1_m["all"]["n"], s1_m["all"]["acc"], s1_m["all"]["bacc"],
          s1_m["all"]["f1"], s1_m["all"]["mcc"], 1.0)),
        (r"\quad Confident ($p\geq0.5$)",
         (s1_m["conf"]["n"], s1_m["conf"]["acc"], s1_m["conf"]["bacc"],
          s1_m["conf"]["f1"], s1_m["conf"]["mcc"], s1_m["frac_confident"])),
        (r"\noalign{\vskip4pt}\textit{Full pipeline (6-class)}",
         None),
        (r"\quad All",
         (fp_m["all"]["n"], fp_m["all"]["acc"], fp_m["all"]["bacc"],
          fp_m["all"]["f1"], fp_m["all"]["mcc"], 1.0)),
        (r"\quad Confident ($p\geq0.5$)",
         (fp_m["conf"]["n"], fp_m["conf"]["acc"], fp_m["conf"]["bacc"],
          fp_m["conf"]["f1"], fp_m["conf"]["mcc"], fp_m["frac_confident"])),
    ]

    for label, data in rows:
        if data is None:
            print(f"  {label} \\\\")
        else:
            n, acc, bacc, f1, mcc, frac = data
            print(f"  {label} & {n} & {fmt(acc)} & {fmt(bacc)} & "
                  f"{fmt(f1)} & {fmt(mcc)} & {fmt(frac)} \\\\")

    print(r"""\enddata
\tablecomments{Metrics from 5-fold stratified cross-validation.
  Stage~1 predicts four broad classes; the full pipeline further
  resolves Stage~1 ``OTHER'' sources into LMXB, HMXB, and CV.
  No SMOTE oversampling; class imbalance handled by
  \texttt{balanced\_subsample} class weights.}
\end{deluxetable*}""")


def print_perclass_table(cv: dict):
    cr = compute_summary(cv["fp_true"], cv["fp_pred"], cv["fp_conf"])["cr"]
    print(r"""
\begin{deluxetable}{lccc}
\tablecaption{Per-class precision, recall, and $F_1$ from 5-fold CV
  (full pipeline, \textit{no SMOTE}).
  STAR~$=$~LM-STAR~$+$~HM-STAR; CV excluded.
  \label{tab:cv_perclass}}
\tablehead{
  \colhead{Class} &
  \colhead{Precision} &
  \colhead{Recall} &
  \colhead{$F_1$}
}
\startdata""")

    for cls in EVAL5:
        row = cr.get(cls, {})
        p  = row.get("precision", np.nan)
        r  = row.get("recall",    np.nan)
        f  = row.get("f1-score",  np.nan)
        ns = int(row.get("support", 0))
        print(f"  {cls} & {fmt(p)} & {fmt(r)} & {fmt(f)} \\\\  % support={ns}")

    print(r"""\enddata
\end{deluxetable}""")


# ── production model ──────────────────────────────────────────────────────────

def train_and_save_production_model(df: pd.DataFrame) -> None:
    """Train on full optical baseline; save Stage 1, Stage 2, encoders, medians."""
    print(f"  Building feature matrix…")
    feat_df, feat_names = build_feature_matrix(df)
    X, medians = impute_median(feat_df)

    labels7 = df["Class"].values
    labels1 = np.array([STAGE1_MAP[c] for c in labels7])

    le1 = LabelEncoder().fit(S1_CLASSES)
    le2 = LabelEncoder().fit(S2_CLASSES)
    y1  = le1.transform(labels1)
    y2_full = np.array([c if c in S2_CLASSES else "___" for c in labels7])

    other_idx = np.where(labels1 == "OTHER")[0]

    # Step 1: OOF Stage-1 probabilities for Stage-2 training (avoids leakage)
    print(f"  Computing OOF Stage-1 probs for Stage-2 training…")
    oof_s1_probs = np.zeros((len(df), len(S1_CLASSES)))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=config.RANDOM_STATE)
    for tr, va in skf.split(X, y1):
        s1_tmp = make_rf()
        s1_tmp.fit(X[tr], y1[tr])
        oof_s1_probs[va] = s1_tmp.predict_proba(X[va])

    # Step 2: Stage 2 on OTHER sources using OOF Stage-1 probs
    print(f"  Training Stage 2 ({len(other_idx)} OTHER sources)…")
    X2_other = np.hstack([X[other_idx], oof_s1_probs[other_idx]])
    y2_other = le2.transform(y2_full[other_idx])
    s2_prod = make_rf()
    s2_prod.fit(X2_other, y2_other)

    # Step 3: Final Stage 1 on all data
    print(f"  Training Stage 1 (all {len(df):,} sources)…")
    s1_prod = make_rf()
    s1_prod.fit(X, y1)

    # Step 4: Save everything
    suffix = MODEL_SUFFIX
    paths = {
        "stage1":    MODELS_DIR / f"stage1_rf_{suffix}.joblib",
        "stage2":    MODELS_DIR / f"stage2_rf_{suffix}.joblib",
        "le1":       MODELS_DIR / f"le1_rf_{suffix}.joblib",
        "le2":       MODELS_DIR / f"le2_rf_{suffix}.joblib",
        "feat":      MODELS_DIR / f"feat_names_{suffix}.joblib",
        "medians":   MODELS_DIR / f"impute_medians_{suffix}.npy",
        "X_matrix":  MODELS_DIR / f"feature_matrix_{suffix}.npy",
        "y_labels":  MODELS_DIR / f"feature_matrix_labels_{suffix}.npy",
    }

    joblib.dump(s1_prod,   paths["stage1"])
    joblib.dump(s2_prod,   paths["stage2"])
    joblib.dump(le1,       paths["le1"])
    joblib.dump(le2,       paths["le2"])
    joblib.dump(feat_names, paths["feat"])
    np.save(paths["medians"],  medians)
    np.save(paths["X_matrix"], X)
    np.save(paths["y_labels"], labels7)

    for key, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name}  ({size_kb:.0f} KB)")


# ── orchestration: single-source production entry points ───────────────────────
# Thin wrappers over the relocated production logic above. run_pipeline.py calls
# only these; no production logic lives in the CLI.

OPTICAL_CATALOG = config.PROCESSED_DIR / "translated_catalog_optical.csv"
FULL_CATALOG    = config.PROCESSED_DIR / "translated_catalog.csv"


def build_optical_baseline() -> pd.DataFrame:
    """Derive the optical-baseline catalog (>=1 non-NaN PHAT _pred column) from the
    full translated catalog and write translated_catalog_optical.csv.

    Relocated verbatim from eval_optical_baseline.main step 1 (the only unique
    production logic in that script). Produces N = 11,374.
    """
    pred_cols = [f"{f}_pred" for f in config.PHAT_FILTER_SET]
    df_full = pd.read_csv(FULL_CATALOG, low_memory=False)
    present = [c for c in pred_cols if c in df_full.columns]
    n_optical = df_full[present].notna().sum(axis=1)
    df_opt = df_full[n_optical >= 1].copy().reset_index(drop=True)
    df_opt.to_csv(OPTICAL_CATALOG, index=False)
    print(f"  optical baseline: N={len(df_opt):,} -> {OPTICAL_CATALOG}")
    return df_opt


def train_production(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train and save the production two-stage models (suffix '_optical_v2').

    Identical to eval_production.train_and_save_production_model; loads the optical
    baseline catalog if *df* is not supplied.
    """
    if df is None:
        if not OPTICAL_CATALOG.exists():
            raise FileNotFoundError(
                f"{OPTICAL_CATALOG} not found. Run build_optical_baseline() first "
                "(or `run_pipeline.py --stage build_optical`)."
            )
        df = pd.read_csv(OPTICAL_CATALOG, low_memory=False)
    train_and_save_production_model(df)
    return df


def evaluate_production(df: pd.DataFrame | None = None, make_figures: bool = True) -> dict:
    """Run the production 5-fold two-stage CV, compute summaries + calibration, and
    (optionally) regenerate the 5 production figures.  Returns the metric dicts.

    Mirrors eval_production.main's CV/metrics/figure sequence, behaviour identical.
    """
    if df is None:
        df = pd.read_csv(OPTICAL_CATALOG, low_memory=False)
    cv = two_stage_cv(df)
    s1_m = compute_summary(cv["s1_true"], cv["s1_pred"], cv["s1_conf"])
    fp_m = compute_summary(cv["fp_true"], cv["fp_pred"], cv["fp_conf"])

    is_correct_fp = (cv["fp_true"] == cv["fp_pred"]).astype(float)
    cal_fp = calibration_metrics(cv["fp_conf"], is_correct_fp)

    if make_figures:
        fig_feature_importance(cv, FIGURES_DIR / "stage1_feature_importance_top15.pdf")
        fig_confusion_matrices(cv, FIGURES_DIR / "stage1_stage2_cv_confusion.pdf")
        fig_per_class_metrics(cv, FIGURES_DIR / "stage2_cv_per_class_metrics.pdf")
        fig_confidence_distribution(cv, FIGURES_DIR / "stage2_cv_ct_distribution.pdf")
        fig_calibration_reliability(cv, cal_fp, FIGURES_DIR / "calibration_reliability.pdf")

    cr = fp_m["cr"]
    return {"cv": cv, "s1": s1_m, "fp": fp_m, "cal": cal_fp,
            "per_class": {c: cr.get(c, {}) for c in FINAL_CLASSES}}


def apply_production(target: str, filter_set: str = "PHAT") -> pd.DataFrame:
    """Apply the production 3-class models to a target footprint.

    Relocated verbatim from run_pipeline.stage_apply (orchestration over package
    functions). Uses the production '_rf_optical_v2' models.
    """
    import sys
    from xclass.query import query_csc_sources_in_polygon, query_hsc_for_chandra_sources
    from xclass.classifier import load_models, predict as clf_predict
    from xclass.diagnostics import plot_class_distributions, generate_report

    target_dir = config.TARGETS_DIR / target
    proc_dir = target_dir / "processed"
    plots_dir = target_dir / "plots"
    csc_csv = target_dir / "chandra_catalog.csv"
    if not csc_csv.exists():
        raise FileNotFoundError(f"CSC CSV not found: {csc_csv}")
    proc_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    polygon = config.PHAT_POLYGON_DEG
    phat_df = query_csc_sources_in_polygon(polygon, csc_csv, config.MIN_SIGNIFICANCE)

    hsc_best, hsc_all = query_hsc_for_chandra_sources(
        phat_df, search_radius_factor=1.0,
        cache_dir=config.QUERY_CACHE_DIR / f"hsc_{target.lower()}",
    )
    hsc_best.to_csv(proc_dir / "hsc_best_match.csv", index=False)
    hsc_all.to_csv(proc_dir / "hsc_all_candidates.csv", index=False)

    col_map = {
        "hsc_F275W_mag": "UVIS_F275W_pred", "hsc_F275W_err": "UVIS_F275W_pred_err",
        "hsc_F336W_mag": "UVIS_F336W_pred", "hsc_F336W_err": "UVIS_F336W_pred_err",
        "hsc_F475W_mag": "ACS_F475W_pred",  "hsc_F475W_err": "ACS_F475W_pred_err",
        "hsc_F814W_mag": "ACS_F814W_pred",  "hsc_F814W_err": "ACS_F814W_pred_err",
        "hsc_F110W_mag": "IR_F110W_pred",   "hsc_F110W_err": "IR_F110W_pred_err",
        "hsc_F160W_mag": "IR_F160W_pred",   "hsc_F160W_err": "IR_F160W_pred_err",
        "hsc_sep_normsep": "normsep_class_xray",
    }
    id_col = next((c for c in phat_df.columns
                   if c.lower() in ("name", "source_id", "xray_id")), "name")
    hsc_phot = hsc_best.rename(columns=col_map).rename(columns={"xray_id": id_col})
    merged = phat_df.merge(hsc_phot, on=id_col, how="left", suffixes=("", "_hsc"))

    csc_flux_map = {
        "flux_aper90_avg_s": "Fx_S", "flux_aper90_avg_m": "Fx_M",
        "flux_aper90_avg_h": "Fx_H", "flux_aper90_avg_b": "Fx_B",
    }
    merged = merged.rename(columns=csc_flux_map)
    merged.to_csv(proc_dir / "xray_hsc_merged.csv", index=False)

    feature_df, feature_names = build_feature_matrix(merged, ml_filter_set=filter_set)
    X = feature_df[feature_names].copy()
    for col in feature_names:
        X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Production 3-class optical-baseline models (suffix '_rf_optical_v2').
    suffix = "_rf_optical_v2"
    stage1, stage2, le, le2 = load_models(config.MODELS_DIR, suffix)
    pred_df = clf_predict(
        stage1, stage2, X, le,
        stage2_label_encoder=le2,
        stage1_other_class="OTHER",
    )

    out = pd.concat([
        phat_df[[id_col]].reset_index(drop=True),
        pred_df.reset_index(drop=True),
    ], axis=1)
    out.to_csv(proc_dir / "predictions.csv", index=False)

    fig = plot_class_distributions(out, class_col="class_pred")
    fig.savefig(plots_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    n_matched = int((hsc_best["hsc_match_status"] != "none").sum())
    generate_report(
        metrics={},
        class_counts=out["class_pred"].value_counts().to_dict(),
        save_dir=proc_dir,
        extra_lines=[
            f"Target: {target}",
            f"N Chandra sources (footprint, sig>={config.MIN_SIGNIFICANCE}): {len(phat_df)}",
            f"N with HSC match: {n_matched}",
            f"Model suffix: {suffix}",
        ],
    )
    return out

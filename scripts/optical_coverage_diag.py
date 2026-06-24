#!/usr/bin/env python3
"""Quick diagnostic: optical feature coverage and CV under exclusion conditions."""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xclass import config
from xclass.features import build_feature_matrix

RF_PARAMS = {**config.RF_PARAMS, "n_jobs": -1, "random_state": config.RANDOM_STATE}
PHAT_FILTERS = config.PHAT_FILTER_SET
PRED_COLS = [f"{f}_pred" for f in PHAT_FILTERS]

CLASS_ORDER = ["AGN", "LM-STAR", "HM-STAR", "SNR", "LMXB", "CV", "HMXB"]


def impute_median(X: pd.DataFrame) -> np.ndarray:
    out = X.copy()
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    return out.values


def run_cv(df: pd.DataFrame, tag: str, n_splits: int = 5) -> dict:
    feat, _ = build_feature_matrix(df)
    labels = df["Class"]
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    X_arr = impute_median(feat)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    accs, baccs, f1s, mccs = [], [], [], []
    oof_true, oof_pred = [], []

    for tr, va in skf.split(X_arr, y):
        m = RandomForestClassifier(**RF_PARAMS)
        m.fit(X_arr[tr], y[tr])
        p = m.predict(X_arr[va])
        accs.append(accuracy_score(y[va], p))
        baccs.append(balanced_accuracy_score(y[va], p))
        f1s.append(f1_score(y[va], p, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(y[va], p))
        oof_true.extend(y[va]); oof_pred.extend(p)

    oof_t = le.inverse_transform(np.array(oof_true))
    oof_p = le.inverse_transform(np.array(oof_pred))
    cr = classification_report(oof_t, oof_p, output_dict=True, zero_division=0)
    snr = cr.get("SNR", {})
    return {
        "tag": tag, "n": len(df),
        "acc": np.mean(accs), "bacc": np.mean(baccs),
        "f1": np.mean(f1s), "mcc": np.mean(mccs),
        "snr_p": snr.get("precision", np.nan),
        "snr_r": snr.get("recall", np.nan),
        "snr_f1": snr.get("f1-score", np.nan),
    }


def run_masquerade(df: pd.DataFrame, tag: str, n_runs: int = 5) -> dict:
    feat, _ = build_feature_matrix(df)
    labels = df["Class"]
    blank_cols = [c for c in feat.columns if c.startswith(("color_", "logFxFopt_"))]

    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    snr_le = list(le.classes_).index("SNR")
    non_snr_idx = np.where(labels.values != "SNR")[0]
    n_snr = int((labels == "SNR").sum())
    prior = n_snr / len(labels)

    rng = np.random.default_rng(42)
    snr_recalls, fake_rates = [], []

    for run in range(n_runs):
        blank_idx = rng.choice(non_snr_idx, size=n_snr, replace=False)
        fm = feat.copy()
        for col in blank_cols:
            fm.iloc[blank_idx, fm.columns.get_loc(col)] = np.nan
        X_arr = impute_median(fm)
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=config.RANDOM_STATE + run)
        fold_snr, fold_fake = [], []
        for tr, va in skf.split(X_arr, y):
            m = RandomForestClassifier(**RF_PARAMS)
            m.fit(X_arr[tr], y[tr])
            p = m.predict(X_arr[va])
            s = (y[va] == snr_le).sum()
            if s: fold_snr.append(((y[va] == snr_le) & (p == snr_le)).sum() / s)
            vb = np.intersect1d(va, blank_idx)
            if len(vb):
                fold_fake.append((p[np.searchsorted(va, vb)] == snr_le).mean())
        snr_recalls.append(np.mean(fold_snr))
        fake_rates.append(np.mean(fold_fake))

    return {
        "tag": tag, "n_snr": n_snr,
        "snr_recall": np.mean(snr_recalls), "snr_recall_std": np.std(snr_recalls),
        "fake_rate": np.mean(fake_rates), "fake_std": np.std(fake_rates),
        "prior": prior, "enrichment": np.mean(fake_rates) / prior,
    }


def main():
    df = pd.read_csv(config.PROCESSED_DIR / "translated_catalog.csv", low_memory=False)

    # ── 1. Coverage counts ────────────────────────────────────────────────────
    present = [c for c in PRED_COLS if c in df.columns]
    df["n_optical"] = df[present].notna().sum(axis=1)
    bins = {0: "0", 1: "1-2", 2: "1-2", 3: "3-5", 4: "3-5", 5: "3-5", 6: "6"}
    df["opt_bin"] = df["n_optical"].map(bins)

    print("=" * 72)
    print("1. OPTICAL COVERAGE — sources with N non-NaN PHAT _pred columns")
    print("=" * 72)

    all_bins = ["0", "1-2", "3-5", "6"]
    classes = [c for c in CLASS_ORDER if c in df["Class"].unique()]

    # Header
    hdr = f"{'Class':<10}" + "".join(f"{'n='+b:>8}" for b in all_bins) + f"{'Total':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cls in classes:
        sub = df[df["Class"] == cls]
        counts = sub["opt_bin"].value_counts()
        row = f"{cls:<10}"
        for b in all_bins:
            row += f"{counts.get(b, 0):>8}"
        row += f"{len(sub):>8}"
        print(row)
    print("-" * len(hdr))
    tot = df["opt_bin"].value_counts()
    row = f"{'TOTAL':<10}"
    for b in all_bins:
        row += f"{tot.get(b, 0):>8}"
    row += f"{len(df):>8}"
    print(row)

    # Identify the zero-optical subsets
    zero_snr_mask  = (df["Class"] == "SNR") & (df["n_optical"] == 0)
    zero_any_mask  = df["n_optical"] == 0
    n_zero_snr  = zero_snr_mask.sum()
    n_zero_any  = zero_any_mask.sum()

    print(f"\nSNRs with 0 optical columns: {n_zero_snr}")
    print(f"All classes with 0 optical columns: {n_zero_any}")
    print()
    for cls in classes:
        n = ((df["Class"] == cls) & (df["n_optical"] == 0)).sum()
        if n: print(f"  {cls}: {n} zero-optical sources")

    # ── 2. CV under three conditions ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("2. 5-FOLD CV UNDER THREE CONDITIONS")
    print("=" * 72)

    df_a = df.copy()
    df_b = df[~zero_snr_mask].copy().reset_index(drop=True)
    df_c = df[~zero_any_mask].copy().reset_index(drop=True)

    print(f"(a) Full fixed baseline:           {len(df_a):,} sources")
    print(f"(b) Exclude {n_zero_snr} zero-optical SNRs:  {len(df_b):,} sources")
    print(f"(c) Exclude all zero-optical:      {len(df_c):,} sources")
    print("\nRunning CV (a)…")
    ra = run_cv(df_a, "(a) Full")
    print("Running CV (b)…")
    rb = run_cv(df_b, f"(b) -SNR0")
    print("Running CV (c)…")
    rc = run_cv(df_c, f"(c) -All0")

    print()
    hdr2 = f"{'Condition':<22} {'N':>6} {'Acc':>7} {'BalAcc':>7} {'F1mac':>7} {'SNR-P':>7} {'SNR-R':>7} {'SNR-F1':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in [ra, rb, rc]:
        print(
            f"{r['tag']:<22} {r['n']:>6} {r['acc']:>7.4f} {r['bacc']:>7.4f}"
            f" {r['f1']:>7.4f} {r['snr_p']:>7.4f} {r['snr_r']:>7.4f} {r['snr_f1']:>7.4f}"
        )

    # ── 3. Masquerade under condition (b) ────────────────────────────────────
    print("\n" + "=" * 72)
    print("3. MASQUERADE TEST — condition (b): zero-optical SNRs excluded")
    print("=" * 72)
    print("Running masquerade (a) full…")
    ma = run_masquerade(df_a, "(a) Full")
    print("Running masquerade (b) -SNR0…")
    mb = run_masquerade(df_b, "(b) -SNR0")

    print()
    hdr3 = f"{'Condition':<22} {'N_SNR':>6} {'Fake→SNR':>10} {'±':>6} {'Prior':>7} {'Enrich':>8}"
    print(hdr3)
    print("-" * len(hdr3))
    for m in [ma, mb]:
        print(
            f"{m['tag']:<22} {m['n_snr']:>6}"
            f" {m['fake_rate']:>10.4f} {m['fake_std']:>6.4f}"
            f" {m['prior']:>7.4f} {m['enrichment']:>7.1f}×"
        )

    print("\n=== INTERPRETATION ===")
    delta_enrich = mb["enrichment"] - ma["enrichment"]
    print(f"Enrichment with zero-optical SNRs excluded: {mb['enrichment']:.1f}× "
          f"(Δ={delta_enrich:+.1f}× vs full {ma['enrichment']:.1f}×)")
    if mb["enrichment"] < 2.0:
        print("✓  Artefact effectively eliminated when all-NaN SNRs are removed.")
    elif mb["enrichment"] < ma["enrichment"] * 0.5:
        print("✓  Artefact substantially reduced (>50%) by removing zero-optical SNRs.")
    else:
        print("⚠  Enrichment persists even after removing zero-optical SNRs.")
        print("   The artefact is driven by partial-NaN SNRs as well as all-NaN ones.")


if __name__ == "__main__":
    main()

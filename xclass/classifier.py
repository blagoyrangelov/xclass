"""xclass.classifier — Two-stage ML classification pipeline.

Trains Random Forest or XGBoost models in a two-stage setup:
Stage 1 gives broad classifications; Stage 2 refines LMXB vs HMXB.

Functions
---------
train_stage1
compute_stage1_probabilities
build_stage2_features
train_stage2
predict
save_models
load_models
cross_validate
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


def _make_model(model_type: str, params: Optional[dict]):
    """Instantiate a classifier given model type and hyperparameters."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        p = {**config.RF_PARAMS, **(params or {})}
        return RandomForestClassifier(**p)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.warning("XGBoost not installed; falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**config.RF_PARAMS)
        p = {**config.XGBOOST_PARAMS, **(params or {})}
        # remove sklearn-incompatible keys
        p.pop("use_label_encoder", None)
        return XGBClassifier(**p)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")


def _compute_metrics(model, X, y_true) -> dict:
    """Compute classification metrics on a dataset."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        confusion_matrix,
        classification_report,
    )

    y_pred = model.predict(X)
    classes = model.classes_

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Stage 1 training
# ---------------------------------------------------------------------------


def train_stage1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "rf",
    params: Optional[dict] = None,
) -> tuple:
    """Train the Stage 1 (broad) classifier.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices.
    y_train, y_val : pd.Series
        Encoded class labels.
    model_type : str
        'rf' for Random Forest or 'xgboost'.
    params : dict, optional
        Override default model hyperparameters from ``config``.

    Returns
    -------
    model : fitted estimator
    metrics : dict
        Keys: accuracy, balanced_accuracy, f1_macro, matthews_corrcoef,
        per-class precision/recall/f1, confusion_matrix (normalised).
    """
    model = _make_model(model_type, params)
    log.info("train_stage1: fitting %s on %d samples", model_type, len(X_train))
    model.fit(X_train, y_train)

    metrics = _compute_metrics(model, X_val, y_val)
    log.info(
        "Stage 1 val: accuracy=%.3f  balanced_acc=%.3f  f1_macro=%.3f",
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1_macro"],
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Stage 1 probabilities
# ---------------------------------------------------------------------------


def compute_stage1_probabilities(model, X: pd.DataFrame) -> pd.DataFrame:
    """Compute class probability vectors from Stage 1 model.

    Parameters
    ----------
    model : fitted estimator
        Stage 1 classifier.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Columns: ``prob_{classname}`` for each class in ``model.classes_``.
    """
    proba = model.predict_proba(X)
    cols = [f"prob_{c}" for c in model.classes_]
    return pd.DataFrame(proba, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# Stage 2 feature construction
# ---------------------------------------------------------------------------


def build_stage2_features(
    X_original: pd.DataFrame,
    stage1_probs: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate original features with Stage 1 probabilities.

    Parameters
    ----------
    X_original : pd.DataFrame
        Original feature matrix.
    stage1_probs : pd.DataFrame
        Output of ``compute_stage1_probabilities``.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix for Stage 2 training.
    """
    # Both inputs share the same index (stage1_probs is computed from X_original),
    # so no reset_index needed here — preserving the index avoids misalignment.
    return pd.concat([X_original, stage1_probs], axis=1)


# ---------------------------------------------------------------------------
# Stage 2 training
# ---------------------------------------------------------------------------


def train_stage2(
    X2_train: pd.DataFrame,
    y_train: pd.Series,
    X2_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "rf",
    params: Optional[dict] = None,
) -> tuple:
    """Train the Stage 2 (LMXB vs HMXB) classifier.

    Same interface as ``train_stage1``.

    Returns
    -------
    model : fitted estimator
    metrics : dict
    """
    model = _make_model(model_type, params)
    log.info("train_stage2: fitting %s on %d samples", model_type, len(X2_train))
    model.fit(X2_train, y_train)

    metrics = _compute_metrics(model, X2_val, y_val)
    log.info(
        "Stage 2 val: accuracy=%.3f  balanced_acc=%.3f  f1_macro=%.3f",
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1_macro"],
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Full two-stage prediction
# ---------------------------------------------------------------------------


def predict(
    stage1_model,
    stage2_model,
    X,
    label_encoder,
    confidence_threshold: float = 0.5,
    stage2_label_encoder=None,
    stage1_other_class: str = "OTHER",
) -> pd.DataFrame:
    """Run full two-stage prediction on new data.

    Supports two architectures:

    **Symmetric** (``stage2_label_encoder=None``):
        Stage 1 classifies all sources; Stage 2 refines using the same
        label space.  Both stages share *label_encoder*.

    **Asymmetric** (``stage2_label_encoder`` provided):
        Stage 1 gives broad classes (e.g. AGN / STAR / OTHER / SNR).
        Sources predicted as *stage1_other_class* are re-classified by
        Stage 2 (e.g. LMXB / HMXB) using *stage2_label_encoder*.
        All other sources keep their Stage 1 prediction.

    Parameters
    ----------
    stage1_model, stage2_model : fitted estimators
    X : pd.DataFrame or np.ndarray
        Feature matrix.  Converted to DataFrame if an array is passed.
    label_encoder : LabelEncoder
        Stage 1 label encoder — maps integer labels to class strings.
    confidence_threshold : float
        Predictions below this max-probability are flagged as uncertain.
    stage2_label_encoder : LabelEncoder, optional
        Stage 2 label encoder (only needed for asymmetric two-stage).
    stage1_other_class : str
        Stage 1 class name that triggers Stage 2 refinement (default
        ``"OTHER"``).

    Returns
    -------
    pd.DataFrame
        Columns: class_pred, class_pred_stage1, confidence,
        classification_flag, and prob_{classname} for all Stage 1 classes.
    """
    # Ensure X is a DataFrame so index is available
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # -----------------------------------------------------------------
    # Stage 1
    # -----------------------------------------------------------------
    probs1 = compute_stage1_probabilities(stage1_model, X)
    stage1_pred_enc = stage1_model.predict(X)
    stage1_pred = np.array(label_encoder.inverse_transform(stage1_pred_enc), dtype=object)

    # -----------------------------------------------------------------
    # Stage 2
    # -----------------------------------------------------------------
    if stage2_model is None:
        final_pred = stage1_pred.copy()
        confidence = probs1.max(axis=1).values
    elif stage2_label_encoder is None:
        # Symmetric: Stage 2 runs on all sources, same label space
        X2 = build_stage2_features(X, probs1)
        probs2 = compute_stage1_probabilities(stage2_model, X2)
        stage2_pred_enc = stage2_model.predict(X2)
        stage2_pred = label_encoder.inverse_transform(stage2_pred_enc)
        final_pred = np.array(stage2_pred, dtype=object)
        confidence = probs2.max(axis=1).values
    else:
        # Asymmetric: Stage 2 only for Stage-1 "OTHER" sources
        X2_all = build_stage2_features(X, probs1)
        other_mask = stage1_pred == stage1_other_class

        final_pred = stage1_pred.copy()
        # .copy() so the array is writable: pandas .values can return a read-only
        # view, and OTHER-routed sources have their confidence overwritten below.
        confidence = probs1.max(axis=1).values.copy()

        if other_mask.any():
            X2_other = X2_all.loc[X.index[other_mask]]
            probs2_other = compute_stage1_probabilities(stage2_model, X2_other)
            stage2_enc = stage2_model.predict(X2_other)
            stage2_classes = np.array(
                stage2_label_encoder.inverse_transform(stage2_enc), dtype=object
            )
            final_pred[other_mask] = stage2_classes
            confidence[other_mask] = probs2_other.max(axis=1).values

    flag = np.where(confidence >= confidence_threshold, "classified", "uncertain")

    out = pd.DataFrame({
        "class_pred": final_pred,
        "class_pred_stage1": stage1_pred,
        "confidence": confidence,
        "classification_flag": flag,
    }, index=X.index)

    # Append Stage 1 probability columns (always present for all sources)
    out = pd.concat([out, probs1], axis=1)
    return out


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_models(
    stage1_model,
    stage2_model,
    label_encoder,
    model_dir: str | Path,
    suffix: str = "",
) -> None:
    """Save Stage 1, Stage 2, and LabelEncoder as joblib files.

    Parameters
    ----------
    model_dir : str or Path
        Directory for saving model files.
    suffix : str
        Optional suffix appended to filenames (e.g. '_rf' or '_xgboost').
    """
    import joblib

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(stage1_model, model_dir / f"stage1{suffix}.joblib")
    joblib.dump(stage2_model, model_dir / f"stage2{suffix}.joblib")
    joblib.dump(label_encoder, model_dir / f"label_encoder{suffix}.joblib")

    log.info("Saved models to %s (suffix='%s')", model_dir, suffix)


def load_models(
    model_dir: str | Path,
    suffix: str = "",
) -> tuple:
    """Load saved Stage 1, Stage 2, and LabelEncoder(s) from joblib files.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model files.
    suffix : str
        Suffix used when models were saved.

    Returns
    -------
    stage1_model, stage2_model, label_encoder, le2
        ``le2`` is the Stage 2 label encoder (from ``le2{suffix}.pkl`` or
        ``le2{suffix}.joblib``).  Returns ``None`` if no Stage 2 encoder
        file is found (symmetric two-stage or single-stage setup).
    """
    import joblib

    model_dir = Path(model_dir)
    stage1 = joblib.load(model_dir / f"stage1{suffix}.joblib")
    stage2 = joblib.load(model_dir / f"stage2{suffix}.joblib")
    # Stage-1 label encoder: notebook-era models save it as 'label_encoder{suffix}',
    # the production pipeline (xclass.pipeline.train_production) saves it as
    # 'le1{suffix}'. Accept either so the single apply path works with both.
    le_path = model_dir / f"label_encoder{suffix}.joblib"
    if not le_path.exists():
        alt = model_dir / f"le1{suffix}.joblib"
        if alt.exists():
            le_path = alt
    le = joblib.load(le_path)

    # Optional Stage 2 label encoder (asymmetric two-stage, saved by notebook 04b)
    le2 = None
    for le2_path in [
        model_dir / f"le2{suffix}.pkl",
        model_dir / f"le2{suffix}.joblib",
    ]:
        if le2_path.exists():
            le2 = joblib.load(le2_path)
            log.info("Loaded Stage 2 label encoder from %s", le2_path)
            break

    log.info("Loaded models from %s (suffix='%s')", model_dir, suffix)
    return stage1, stage2, le, le2


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_type: str = "rf",
) -> dict:
    """Run stratified k-fold cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Class labels.
    n_splits : int
        Number of CV folds.
    model_type : str
        'rf' or 'xgboost'.

    Returns
    -------
    dict
        Mean and standard deviation of all metrics across folds.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    all_acc, all_bacc, all_f1, all_mcc = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]

        model = _make_model(model_type, None)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_v)

        all_acc.append(accuracy_score(y_v, y_pred))
        all_bacc.append(balanced_accuracy_score(y_v, y_pred))
        all_f1.append(f1_score(y_v, y_pred, average="macro", zero_division=0))
        all_mcc.append(matthews_corrcoef(y_v, y_pred))

        log.debug("Fold %d: acc=%.3f f1_macro=%.3f", fold + 1, all_acc[-1], all_f1[-1])

    results = {
        "accuracy_mean": float(np.mean(all_acc)),
        "accuracy_std": float(np.std(all_acc)),
        "balanced_accuracy_mean": float(np.mean(all_bacc)),
        "balanced_accuracy_std": float(np.std(all_bacc)),
        "f1_macro_mean": float(np.mean(all_f1)),
        "f1_macro_std": float(np.std(all_f1)),
        "matthews_corrcoef_mean": float(np.mean(all_mcc)),
        "matthews_corrcoef_std": float(np.std(all_mcc)),
        "n_splits": n_splits,
    }

    log.info(
        "cross_validate (%d-fold): acc=%.3f±%.3f  f1=%.3f±%.3f",
        n_splits,
        results["accuracy_mean"], results["accuracy_std"],
        results["f1_macro_mean"], results["f1_macro_std"],
    )
    return results

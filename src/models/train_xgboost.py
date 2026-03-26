"""
train_xgboost.py
----------------
XGBoost training pipeline for supply-chain disruption prediction.

Two tasks are solved in this script:

  1. CLASSIFICATION  — predict binary `delayed` (0 / 1)
       Metric focus : AUC-ROC, F1-score (macro)
       Handles class imbalance via `scale_pos_weight`

  2. REGRESSION       — estimate `delay_hours` for shipments predicted as delayed
       Metric focus : RMSE, MAE, R²
       Training subset: only the `delayed == 1` rows

Persisted artefacts (saved to models/ directory):
  • xgb_classifier.json    — XGBoost binary classifier
  • xgb_regressor.json     — XGBoost delay-hours regressor
  • xgb_scaler.joblib      — StandardScaler for numerical features
  • xgb_feature_cols.json  — Ordered list of feature columns used at training

Usage:
  python -m src.models.train_xgboost
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.loader import SupplyChainLoader
from src.features.build_features import (
    build_feature_matrix, FEATURE_COLS, TARGET_CLF, TARGET_REG,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Config & Path Helpers
# =============================================================================

def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _model_dir(cfg: dict) -> Path:
    root = Path(__file__).parents[2]
    d    = root / cfg["data"]["models_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Data Loading & Preparation
# =============================================================================

def prepare_data(cfg: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    list[str],
]:
    """Load → merge → feature-engineer → time-split."""
    loader     = SupplyChainLoader()
    shipments  = loader.shipments()
    congestion = loader.port_congestion()

    df = build_feature_matrix(shipments, congestion)
    logger.info("Feature matrix shape: %s", df.shape)

    # Present feature columns (some may be absent when rolling/lag fills fail)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info("Using %d feature columns", len(feat_cols))

    # Time-based split
    df.sort_values("ship_date", inplace=True, errors="ignore")
    n        = len(df)
    val_cut  = int(n * (1.0 - cfg["data"]["split"]["val_size"] -
                            cfg["data"]["split"]["test_size"]))
    test_cut = int(n * (1.0 - cfg["data"]["split"]["test_size"]))

    train = df.iloc[:val_cut].reset_index(drop=True)
    val   = df.iloc[val_cut:test_cut].reset_index(drop=True)
    test  = df.iloc[test_cut:].reset_index(drop=True)

    logger.info("Split → train:%d  val:%d  test:%d", len(train), len(val), len(test))
    return train, val, test, feat_cols


# =============================================================================
# XGBoost Classifier
# =============================================================================

def train_classifier(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    feat_cols: list[str],
    cfg: dict,
) -> xgb.XGBClassifier:
    """
    Train XGBoost binary classifier for shipment delay prediction.

    scale_pos_weight = n_negative / n_positive  (handles class imbalance).
    Early stopping monitors validation AUC.
    """
    xgb_cfg = cfg["xgboost"]["classifier"]
    neg      = (train[TARGET_CLF] == 0).sum()
    pos      = (train[TARGET_CLF] == 1).sum()
    spw      = neg / pos if pos > 0 else 1.0
    logger.info("Class ratio  neg:pos = %d:%d  →  scale_pos_weight=%.2f", neg, pos, spw)

    model = xgb.XGBClassifier(
        n_estimators          = xgb_cfg["n_estimators"],
        max_depth             = xgb_cfg["max_depth"],
        learning_rate         = xgb_cfg["learning_rate"],
        subsample             = xgb_cfg["subsample"],
        colsample_bytree      = xgb_cfg["colsample_bytree"],
        min_child_weight      = xgb_cfg["min_child_weight"],
        gamma                 = xgb_cfg["gamma"],
        reg_alpha             = xgb_cfg["reg_alpha"],
        reg_lambda            = xgb_cfg["reg_lambda"],
        scale_pos_weight      = spw,
        eval_metric           = "auc",
        early_stopping_rounds = xgb_cfg["early_stopping_rounds"],
        use_label_encoder     = False,
        random_state          = 42,
        n_jobs                = -1,
    )

    X_tr, y_tr = train[feat_cols].values, train[TARGET_CLF].values
    X_vl, y_vl = val[feat_cols].values,   val[TARGET_CLF].values

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=50,
    )

    # Validation metrics
    prob_vl = model.predict_proba(X_vl)[:, 1]
    pred_vl = (prob_vl >= 0.50).astype(int)
    logger.info(
        "Classifier  │  AUC=%.4f  │  AP=%.4f  │  F1=%.4f",
        roc_auc_score(y_vl, prob_vl),
        average_precision_score(y_vl, prob_vl),
        f1_score(y_vl, pred_vl, average="macro"),
    )
    logger.info("\n%s", classification_report(y_vl, pred_vl))
    return model


# =============================================================================
# XGBoost Regressor
# =============================================================================

def train_regressor(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    feat_cols: list[str],
    cfg: dict,
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor to estimate `delay_hours` for delayed shipments.

    Training and validation are restricted to delayed==1 rows only.
    Target is log-transformed (log1p) for stability of the regression.
    """
    xgb_cfg = cfg["xgboost"]["regressor"]

    tr_d = train[train[TARGET_CLF] == 1].copy()
    vl_d = val[val[TARGET_CLF]     == 1].copy()
    logger.info("Regressor training on %d delayed shipments", len(tr_d))

    model = xgb.XGBRegressor(
        n_estimators          = xgb_cfg["n_estimators"],
        max_depth             = xgb_cfg["max_depth"],
        learning_rate         = xgb_cfg["learning_rate"],
        subsample             = xgb_cfg["subsample"],
        colsample_bytree      = xgb_cfg["colsample_bytree"],
        min_child_weight      = xgb_cfg["min_child_weight"],
        eval_metric           = "rmse",
        early_stopping_rounds = xgb_cfg["early_stopping_rounds"],
        random_state          = 42,
        n_jobs                = -1,
    )

    X_tr = tr_d[feat_cols].values
    y_tr = np.log1p(tr_d[TARGET_REG].values)       # log-transform target
    X_vl = vl_d[feat_cols].values
    y_vl_raw = vl_d[TARGET_REG].values

    model.fit(X_tr, y_tr, eval_set=[(X_vl, np.log1p(y_vl_raw))], verbose=50)

    # Back-transform and evaluate
    pred_vl = np.expm1(model.predict(X_vl))
    rmse    = np.sqrt(mean_squared_error(y_vl_raw, pred_vl))
    mae     = mean_absolute_error(y_vl_raw, pred_vl)
    r2      = r2_score(y_vl_raw, pred_vl)
    logger.info("Regressor │ RMSE=%.2f h │ MAE=%.2f h │ R²=%.4f", rmse, mae, r2)
    return model


# =============================================================================
# Persist Artefacts
# =============================================================================

def save_artefacts(
    clf:       xgb.XGBClassifier,
    reg:       xgb.XGBRegressor,
    feat_cols: list[str],
    model_dir: Path,
) -> None:
    clf.save_model(model_dir / "xgb_classifier.json")
    reg.save_model(model_dir / "xgb_regressor.json")
    with open(model_dir / "xgb_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)
    logger.info("Artefacts saved to %s", model_dir)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    cfg       = _load_config()
    model_dir = _model_dir(cfg)

    train, val, test, feat_cols = prepare_data(cfg)

    logger.info("─" * 55)
    logger.info("Training XGBoost Classifier …")
    clf = train_classifier(train, val, feat_cols, cfg)

    logger.info("─" * 55)
    logger.info("Training XGBoost Regressor …")
    reg = train_regressor(train, val, feat_cols, cfg)

    save_artefacts(clf, reg, feat_cols, model_dir)

    # Final hold-out test metrics
    logger.info("─" * 55)
    logger.info("Final Test Set Evaluation:")
    X_test = test[feat_cols].values
    y_test = test[TARGET_CLF].values

    prob   = clf.predict_proba(X_test)[:, 1]
    pred   = (prob >= 0.50).astype(int)
    logger.info(
        "TEST  │  AUC=%.4f  │  AP=%.4f  │  F1=%.4f",
        roc_auc_score(y_test, prob),
        average_precision_score(y_test, prob),
        f1_score(y_test, pred, average="macro"),
    )
    logger.info("\n%s", classification_report(y_test, pred))

    # Regression test (delayed only)
    test_d = test[test[TARGET_CLF] == 1].copy()
    X_td   = test_d[feat_cols].values
    pred_d = np.expm1(reg.predict(X_td))
    y_td   = test_d[TARGET_REG].values
    logger.info(
        "Regressor TEST │ RMSE=%.2f h │ MAE=%.2f h │ R²=%.4f",
        np.sqrt(mean_squared_error(y_td, pred_d)),
        mean_absolute_error(y_td, pred_d),
        r2_score(y_td, pred_d),
    )


if __name__ == "__main__":
    main()

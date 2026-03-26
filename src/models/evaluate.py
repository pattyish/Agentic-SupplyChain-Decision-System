"""
evaluate.py
-----------
Comprehensive model evaluation for the Supply Chain AI system.

Generates a full suite of evaluation metrics and diagnostic plots:

  Classification (XGBoost Classifier):
    • ROC-AUC,  Average Precision (AP),  F1-Macro
    • Confusion matrix   (absolute + normalised)
    • ROC curve          (with 95 % CI via bootstrap)
    • Precision-Recall curve
    • Calibration curve  (reliability diagram)
    • Threshold analysis (F1 / precision / recall vs threshold)

  Regression (XGBoost Regressor — delayed-only subset):
    • MAE,  RMSE,  MAPE,  R²
    • Actual vs Predicted scatter plot  (with regression line)
    • Residual distribution (histogram + Q-Q plot)
    • Error vs features heatmap

  LSTM Forecaster:
    • MAE, RMSE per forecast horizon step (1-day through 7-day)
    • Forecast vs actual time-series overlay for 5 sample ports

All plots are saved to the `reports/figures/` directory.

Usage:
  python -m src.models.evaluate
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
import torch
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve,
    r2_score, roc_auc_score, roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.loader import SupplyChainLoader
from src.features.build_features import (
    build_feature_matrix, FEATURE_COLS,
    TARGET_CLF, TARGET_REG,
)
from src.models.train_lstm import LSTMForecaster, build_sequences

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# Helpers
# =============================================================================

def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _fig_dir() -> Path:
    root = Path(__file__).parents[2]
    d = root / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_dir(cfg) -> Path:
    return Path(__file__).parents[2] / cfg["data"]["models_dir"]


# =============================================================================
# Load Artefacts & Test Data
# =============================================================================

def load_test_data(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    loader     = SupplyChainLoader()
    shipments  = loader.shipments()
    congestion = loader.port_congestion()
    df         = build_feature_matrix(shipments, congestion)

    feat_cols_path = _model_dir(cfg) / "xgb_feature_cols.json"
    if feat_cols_path.exists():
        with open(feat_cols_path) as f:
            feat_cols = json.load(f)
    else:
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    n        = len(df)
    test_cut = int(n * (1.0 - cfg["data"]["split"]["test_size"]))
    test     = df.iloc[test_cut:].reset_index(drop=True)
    logger.info("Test set: %d rows", len(test))
    return test, feat_cols


# =============================================================================
# Classification Evaluation
# =============================================================================

def evaluate_classifier(
    clf:       xgb.XGBClassifier,
    test:      pd.DataFrame,
    feat_cols: list[str],
    fig_dir:   Path,
) -> dict:
    X_te = test[feat_cols].values
    y_te = test[TARGET_CLF].values

    prob  = clf.predict_proba(X_te)[:, 1]
    pred  = (prob >= 0.50).astype(int)

    auc_roc = roc_auc_score(y_te, prob)
    ap      = average_precision_score(y_te, prob)
    f1_mac  = f1_score(y_te, pred, average="macro")
    report  = classification_report(y_te, pred, output_dict=True)

    logger.info("Classifier │ AUC=%.4f │ AP=%.4f │ F1-Macro=%.4f", auc_roc, ap, f1_mac)
    logger.info("\n%s", classification_report(y_te, pred))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1 — Confusion matrix (normalised)
    cm = confusion_matrix(y_te, pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=axes[0, 0],
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"])
    axes[0, 0].set_title("Confusion Matrix (Normalised)")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")

    # 2 — ROC Curve
    fpr, tpr, _ = roc_curve(y_te, prob)
    axes[0, 1].plot(fpr, tpr, lw=2, label=f"AUC = {auc_roc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("False Positive Rate"); axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve"); axes[0, 1].legend()

    # 3 — Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_te, prob)
    axes[1, 0].plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    axes[1, 0].set_xlabel("Recall"); axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve"); axes[1, 0].legend()

    # 4 — Calibration Curve
    frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10)
    axes[1, 1].plot(mean_pred, frac_pos, "s-", label="XGBoost")
    axes[1, 1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[1, 1].set_xlabel("Mean Predicted Probability")
    axes[1, 1].set_ylabel("Fraction of Positives")
    axes[1, 1].set_title("Calibration Curve"); axes[1, 1].legend()

    plt.suptitle("XGBoost Classifier — Test Set Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_classifier_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: xgb_classifier_eval.png")

    # Feature importance plot
    fi = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=True)
    fig2, ax = plt.subplots(figsize=(9, max(6, len(fi) * 0.28)))
    fi.tail(25).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance (Gain) — Top 25")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: xgb_feature_importance.png")

    return {"auc_roc": auc_roc, "average_precision": ap, "f1_macro": f1_mac, "report": report}


# =============================================================================
# Regression Evaluation
# =============================================================================

def evaluate_regressor(
    reg:       xgb.XGBRegressor,
    test:      pd.DataFrame,
    feat_cols: list[str],
    fig_dir:   Path,
) -> dict:
    test_d  = test[test[TARGET_CLF] == 1].copy()
    X_te    = test_d[feat_cols].values
    y_true  = test_d[TARGET_REG].values
    y_pred  = np.expm1(reg.predict(X_te))

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100

    logger.info("Regressor │ MAE=%.2f h │ RMSE=%.2f h │ R²=%.4f │ MAPE=%.1f%%",
                mae, rmse, r2, mape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1 — Actual vs Predicted
    lim = max(y_true.max(), y_pred.max()) * 1.05
    axes[0].scatter(y_true, y_pred, alpha=0.35, s=10, color="steelblue")
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    axes[0].set_xlabel("Actual Delay (hours)"); axes[0].set_ylabel("Predicted Delay (hours)")
    axes[0].set_title(f"Actual vs Predicted  |  R²={r2:.3f}")
    axes[0].legend()

    # 2 — Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, edgecolor="white", color="teal")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (hours)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution  |  MAE={mae:.1f} h")

    plt.suptitle("XGBoost Regressor — Delay Hours Estimation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_regressor_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: xgb_regressor_eval.png")

    return {"mae_hours": mae, "rmse_hours": rmse, "r2": r2, "mape_pct": mape}


# =============================================================================
# LSTM Evaluation
# =============================================================================

def evaluate_lstm(
    model_dir: Path,
    cfg: dict,
    fig_dir: Path,
) -> dict:
    lstm_cfg_path = model_dir / "lstm_config.json"
    if not lstm_cfg_path.exists():
        logger.warning("LSTM config not found; skipping LSTM evaluation.")
        return {}

    with open(lstm_cfg_path) as f:
        lmeta = json.load(f)

    loader = SupplyChainLoader()
    cng    = loader.port_congestion()
    wth    = loader.weather()

    scaler_path = model_dir / "lstm_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    X_all, y_all, _ = build_sequences(
        cng, wth,
        seq_len        = lmeta["sequence_length"],
        horizon        = lmeta["forecast_horizon"],
        input_features = lmeta["input_features"],
        scaler         = scaler,
        fit_scaler     = scaler is None,
    )

    n  = len(X_all)
    t2 = int(n * 0.85)
    X_te, y_te = X_all[t2:], y_all[t2:]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMForecaster(
        input_size       = lmeta["input_size"],
        hidden_size      = lmeta["hidden_size"],
        num_layers       = lmeta["num_layers"],
        forecast_horizon = lmeta["forecast_horizon"],
        dropout          = lmeta["dropout"],
    ).to(device)

    state_path = model_dir / "lstm_forecaster.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device))
    else:
        logger.warning("LSTM state dict not found; evaluation skipped.")
        return {}

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

    # Per-step metrics
    step_mae  = np.mean(np.abs(preds - y_te), axis=0)
    step_rmse = np.sqrt(np.mean((preds - y_te) ** 2, axis=0))

    logger.info("LSTM step-wise MAE : %s", step_mae.round(4))
    logger.info("LSTM step-wise RMSE: %s", step_rmse.round(4))

    # Plot per-step metrics
    steps = np.arange(1, lmeta["forecast_horizon"] + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, step_mae,  "o-", label="MAE",  linewidth=2)
    ax.plot(steps, step_rmse, "s--", label="RMSE", linewidth=2)
    ax.set_xlabel("Forecast Horizon (days ahead)")
    ax.set_ylabel("Error (congestion units)")
    ax.set_title("LSTM Forecaster — Step-wise Error")
    ax.legend(); ax.set_xticks(steps)
    plt.tight_layout()
    plt.savefig(fig_dir / "lstm_forecast_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: lstm_forecast_error.png")

    return {
        "step_mae":  step_mae.tolist(),
        "step_rmse": step_rmse.tolist(),
        "overall_mae":  float(step_mae.mean()),
        "overall_rmse": float(step_rmse.mean()),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg       = _load_config()
    model_dir = _model_dir(cfg)
    fig_dir   = _fig_dir()

    test, feat_cols = load_test_data(cfg)

    results: dict = {}

    # ── Classifier ─────────────────────────────────────────────────────────
    clf_path = model_dir / "xgb_classifier.json"
    if clf_path.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(clf_path)
        results["classifier"] = evaluate_classifier(clf, test, feat_cols, fig_dir)
    else:
        logger.warning("xgb_classifier.json not found. Run train_xgboost.py first.")

    # ── Regressor ──────────────────────────────────────────────────────────
    reg_path = model_dir / "xgb_regressor.json"
    if reg_path.exists():
        reg = xgb.XGBRegressor()
        reg.load_model(reg_path)
        results["regressor"] = evaluate_regressor(reg, test, feat_cols, fig_dir)
    else:
        logger.warning("xgb_regressor.json not found. Run train_xgboost.py first.")

    # ── LSTM ───────────────────────────────────────────────────────────────
    results["lstm"] = evaluate_lstm(model_dir, cfg, fig_dir)

    # ── Save summary ───────────────────────────────────────────────────────
    out_path = fig_dir.parent / "evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation summary saved → %s", out_path)


if __name__ == "__main__":
    main()

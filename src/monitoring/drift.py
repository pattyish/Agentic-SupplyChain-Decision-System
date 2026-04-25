"""
Drift and production monitoring utilities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.dropna().values
    cur = current.dropna().values
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(ref, quantiles))
    if len(breaks) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=breaks)
    cur_counts, _ = np.histogram(cur, bins=breaks)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_cols: list[str], cfg: dict) -> dict:
    drift_cfg = cfg.get("monitoring", {}).get("drift", {})
    warn = float(drift_cfg.get("psi_warn", 0.1))
    critical = float(drift_cfg.get("psi_critical", 0.25))

    features = {}
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        score = psi(reference_df[col], current_df[col])
        severity = "ok"
        if score >= critical:
            severity = "critical"
        elif score >= warn:
            severity = "warn"
        features[col] = {"psi": round(score, 6), "severity": severity}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_drift": features,
        "critical_features": [k for k, v in features.items() if v["severity"] == "critical"],
        "warn_features": [k for k, v in features.items() if v["severity"] == "warn"],
    }


def compute_prediction_quality(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    precision = float(((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1))
    recall = float(((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "alert_precision": precision,
        "alert_recall": recall,
    }


def save_monitoring_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

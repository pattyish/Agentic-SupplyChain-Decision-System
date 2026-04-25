"""
Run drift and prediction quality monitoring reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loader import SupplyChainLoader
from src.features.build_features import FEATURE_COLS, TARGET_CLF, build_feature_matrix
from src.monitoring.drift import compute_drift_report, compute_prediction_quality, save_monitoring_report


def _load_cfg() -> dict:
    return yaml.safe_load((Path(__file__).parents[2] / "configs" / "config.yaml").read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).parents[2]
    cfg = _load_cfg()
    model_dir = root / cfg["data"]["models_dir"]

    loader = SupplyChainLoader()
    shipments = loader.shipments()
    congestion = loader.port_congestion()
    df = build_feature_matrix(shipments, congestion)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    n = len(df)
    split = int(n * 0.70)
    baseline = df.iloc[:split].copy()
    current = df.iloc[split:].copy()

    drift_report = compute_drift_report(
        baseline,
        current,
        numeric_cols=[c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])],
        cfg=cfg,
    )

    quality = {}
    clf_path = model_dir / "xgb_classifier.json"
    cal_path = model_dir / "xgb_classifier_calibrator.joblib"
    if clf_path.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(clf_path)
        X_cur = current[feat_cols].values
        y_cur = current[TARGET_CLF].values
        if cal_path.exists():
            import joblib
            cal = joblib.load(cal_path)
            prob = cal.predict_proba(X_cur)[:, 1]
        else:
            prob = clf.predict_proba(X_cur)[:, 1]
        quality = compute_prediction_quality(y_cur, prob)

    slo_cfg = cfg.get("monitoring", {}).get("slo", {})
    slo_status = {
        "alert_precision_target": float(slo_cfg.get("min_alert_precision", 0.70)),
        "alert_precision_actual": float(quality.get("alert_precision", 0.0)),
        "meets_alert_precision": float(quality.get("alert_precision", 0.0)) >= float(slo_cfg.get("min_alert_precision", 0.70)),
    }

    report = {
        "drift": drift_report,
        "prediction_quality": quality,
        "slo_status": slo_status,
    }

    save_monitoring_report(report, root / "reports" / "monitoring_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

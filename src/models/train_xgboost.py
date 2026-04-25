"""
Enhanced XGBoost training pipeline with quality gates, calibration,
lineage tracking, and champion-challenger promotion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.contracts import run_data_quality_gate
from src.data.loader import SupplyChainLoader
from src.features.build_features import FEATURE_COLS, TARGET_CLF, TARGET_REG, build_feature_matrix
from src.mlops.tracking import mlflow_run, save_lineage_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _model_dir(cfg: dict) -> Path:
    root = Path(__file__).parents[2]
    d = root / cfg["data"]["models_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reports_dir() -> Path:
    d = Path(__file__).parents[2] / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    loader = SupplyChainLoader()
    shipments = loader.shipments()
    congestion = loader.port_congestion()
    suppliers = loader.suppliers()
    weather = loader.weather()

    if cfg.get("data_quality", {}).get("enabled", True):
        run_data_quality_gate(
            {
                "shipments": shipments,
                "congestion": congestion,
                "suppliers": suppliers,
                "weather": weather,
            },
            cfg,
            _reports_dir(),
        )

    df = build_feature_matrix(shipments, congestion)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    df.sort_values("ship_date", inplace=True, errors="ignore")
    n = len(df)
    val_cut = int(n * (1.0 - cfg["data"]["split"]["val_size"] - cfg["data"]["split"]["test_size"]))
    test_cut = int(n * (1.0 - cfg["data"]["split"]["test_size"]))

    train = df.iloc[:val_cut].reset_index(drop=True)
    val = df.iloc[val_cut:test_cut].reset_index(drop=True)
    test = df.iloc[test_cut:].reset_index(drop=True)
    logger.info("Prepared split train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test, feat_cols


def train_classifier(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feat_cols: list[str],
    cfg: dict,
) -> tuple[xgb.XGBClassifier, Optional[CalibratedClassifierCV]]:
    xgb_cfg = cfg["xgboost"]["classifier"]
    neg = int((train[TARGET_CLF] == 0).sum())
    pos = int((train[TARGET_CLF] == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        min_child_weight=xgb_cfg["min_child_weight"],
        gamma=xgb_cfg["gamma"],
        reg_alpha=xgb_cfg["reg_alpha"],
        reg_lambda=xgb_cfg["reg_lambda"],
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        random_state=42,
        n_jobs=-1,
    )

    X_tr, y_tr = train[feat_cols].values, train[TARGET_CLF].values
    X_vl, y_vl = val[feat_cols].values, val[TARGET_CLF].values
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    calibrator = None
    calib_cfg = cfg.get("mlops", {}).get("calibration", {})
    if calib_cfg.get("enabled", True):
        try:
            calibrator = CalibratedClassifierCV(model, method=calib_cfg.get("method", "sigmoid"), cv="prefit")
            calibrator.fit(X_vl, y_vl)
            logger.info("Calibrator fitted using method=%s", calib_cfg.get("method", "sigmoid"))
        except Exception as exc:
            logger.warning("Calibration skipped: %s", exc)

    prob_vl = calibrator.predict_proba(X_vl)[:, 1] if calibrator is not None else model.predict_proba(X_vl)[:, 1]
    pred_vl = (prob_vl >= 0.5).astype(int)

    logger.info(
        "Classifier validation AUC=%.4f AP=%.4f F1=%.4f",
        roc_auc_score(y_vl, prob_vl),
        average_precision_score(y_vl, prob_vl),
        f1_score(y_vl, pred_vl, average="macro"),
    )
    logger.info("\n%s", classification_report(y_vl, pred_vl))
    return model, calibrator


def train_regressor(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feat_cols: list[str],
    cfg: dict,
) -> xgb.XGBRegressor:
    xgb_cfg = cfg["xgboost"]["regressor"]
    tr_d = train[train[TARGET_CLF] == 1].copy()
    vl_d = val[val[TARGET_CLF] == 1].copy()

    model = xgb.XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        min_child_weight=xgb_cfg["min_child_weight"],
        eval_metric="rmse",
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        random_state=42,
        n_jobs=-1,
    )

    X_tr = tr_d[feat_cols].values
    y_tr = np.log1p(tr_d[TARGET_REG].values)
    X_vl = vl_d[feat_cols].values
    y_vl = np.log1p(vl_d[TARGET_REG].values)

    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return model


def compute_slice_metrics(
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    out: dict[str, dict] = {}
    pred = (y_prob >= threshold).astype(int)

    for col in ["transport_mode", "weather_severity"]:
        if col not in test.columns:
            continue
        out[col] = {}
        for value, grp in test.groupby(col):
            idx = grp.index.to_numpy()
            y_true = grp[TARGET_CLF].values
            y_hat = pred[idx]
            out[col][str(value)] = {
                "count": int(len(grp)),
                "f1_macro": float(f1_score(y_true, y_hat, average="macro")),
                "delay_rate": float(y_true.mean()),
            }

    if "supplier_risk" in test.columns:
        bins = pd.cut(test["supplier_risk"], bins=[-0.001, 0.33, 0.66, 1.0], labels=["low", "mid", "high"])
        out["supplier_risk_bucket"] = {}
        for label in ["low", "mid", "high"]:
            mask = bins == label
            grp = test[mask]
            if len(grp) == 0:
                continue
            idx = grp.index.to_numpy()
            y_true = grp[TARGET_CLF].values
            y_hat = pred[idx]
            out["supplier_risk_bucket"][label] = {
                "count": int(len(grp)),
                "f1_macro": float(f1_score(y_true, y_hat, average="macro")),
                "delay_rate": float(y_true.mean()),
            }

    return out


def _evaluate_classifier(test: pd.DataFrame, feat_cols: list[str], clf: xgb.XGBClassifier, calibrator: Optional[CalibratedClassifierCV]) -> dict:
    X_test = test[feat_cols].values
    y_test = test[TARGET_CLF].values
    prob = calibrator.predict_proba(X_test)[:, 1] if calibrator is not None else clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    return metrics | {"prob": prob}


def _evaluate_regressor(test: pd.DataFrame, feat_cols: list[str], reg: xgb.XGBRegressor) -> dict:
    test_d = test[test[TARGET_CLF] == 1].copy()
    if len(test_d) == 0:
        return {"rmse_hours": 0.0, "mae_hours": 0.0, "r2": 0.0}

    X_td = test_d[feat_cols].values
    pred_d = np.expm1(reg.predict(X_td))
    y_td = test_d[TARGET_REG].values
    return {
        "rmse_hours": float(np.sqrt(mean_squared_error(y_td, pred_d))),
        "mae_hours": float(mean_absolute_error(y_td, pred_d)),
        "r2": float(r2_score(y_td, pred_d)),
    }


def _promote_if_better(model_dir: Path, metrics: dict, cfg: dict) -> dict:
    ccfg = cfg.get("mlops", {}).get("champion_challenger", {})
    enabled = ccfg.get("enabled", True)
    min_gain = float(ccfg.get("promote_if_f1_gain_min", 0.005))

    champion_path = model_dir / "champion_metrics.json"
    previous_f1 = 0.0
    if champion_path.exists():
        try:
            previous_f1 = float(json.loads(champion_path.read_text(encoding="utf-8")).get("f1_macro", 0.0))
        except Exception:
            previous_f1 = 0.0

    new_f1 = float(metrics.get("f1_macro", 0.0))
    approved = (not enabled) or ((new_f1 - previous_f1) >= min_gain)
    decision = {
        "previous_f1": previous_f1,
        "new_f1": new_f1,
        "required_gain": min_gain,
        "approved": approved,
    }

    (model_dir / "promotion_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if approved:
        champion_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (model_dir / "production_alias.json").write_text(
            json.dumps(
                {
                    "classifier": "xgb_classifier.json",
                    "regressor": "xgb_regressor.json",
                    "calibrator": "xgb_classifier_calibrator.joblib",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return decision


def save_artefacts(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    calibrator: Optional[CalibratedClassifierCV],
    feat_cols: list[str],
    model_dir: Path,
) -> None:
    clf.save_model(model_dir / "xgb_classifier.json")
    reg.save_model(model_dir / "xgb_regressor.json")
    (model_dir / "xgb_feature_cols.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    if calibrator is not None:
        joblib.dump(calibrator, model_dir / "xgb_classifier_calibrator.joblib")


def main() -> None:
    cfg = _load_config()
    model_dir = _model_dir(cfg)

    train, val, test, feat_cols = prepare_data(cfg)
    lineage = save_lineage_manifest(model_dir, train, val, test, feat_cols)

    with mlflow_run(cfg, "train-xgboost") as mlf:
        clf, calibrator = train_classifier(train, val, feat_cols, cfg)
        reg = train_regressor(train, val, feat_cols, cfg)
        save_artefacts(clf, reg, calibrator, feat_cols, model_dir)

        clf_metrics = _evaluate_classifier(test, feat_cols, clf, calibrator)
        prob = clf_metrics.pop("prob")
        reg_metrics = _evaluate_regressor(test, feat_cols, reg)
        slice_metrics = compute_slice_metrics(test, prob)

        (model_dir / "slice_metrics.json").write_text(json.dumps(slice_metrics, indent=2), encoding="utf-8")
        promotion = _promote_if_better(model_dir, clf_metrics, cfg)

        logger.info("Classifier TEST metrics: %s", clf_metrics)
        logger.info("Regressor TEST metrics: %s", reg_metrics)
        logger.info("Promotion decision: %s", promotion)

        if hasattr(mlf, "log_metric"):
            for k, v in clf_metrics.items():
                mlf.log_metric(f"classifier_{k}", float(v))
            for k, v in reg_metrics.items():
                mlf.log_metric(f"regressor_{k}", float(v))
            mlf.log_artifact(str(lineage))
            mlf.log_artifact(str(model_dir / "slice_metrics.json"))
            mlf.log_artifact(str(model_dir / "promotion_decision.json"))

        summary = {
            "classifier": clf_metrics,
            "regressor": reg_metrics,
            "promotion": promotion,
        }
        (Path(__file__).parents[2] / "reports" / "training_summary_xgb.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

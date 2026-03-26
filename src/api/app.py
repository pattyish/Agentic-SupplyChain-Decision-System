"""
app.py
------
FastAPI REST service for the AI-Driven Supply Chain Predictive Monitoring System.

Endpoints:
  GET  /               — API description & version
  GET  /health         — Health check with model availability status
  POST /predict/delay  — Predict delay probability + hours for a single shipment
  POST /predict/batch  — Batch predictions (up to 500 shipments per request)
  GET  /anomalies      — Return recent anomalous shipments from stored data
  GET  /congestion     — Return current port congestion alerts
  GET  /metrics        — Summary statistics from the loaded test data

Security:
  • Pydantic schema validation on all inputs (prevents injection via field types)
  • Numeric bounds enforced with Field(ge=, le=) constraints
  • transport_mode is a Literal type (whitelist, not freeform string)
  • No raw SQL / OS commands executed anywhere

Run:
  uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Project imports ───────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import SupplyChainLoader
from src.anomaly.anomaly_detection import AnomalyDetector, congestion_alerts

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================

def _load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = _load_config()
MODEL_DIR = ROOT / CFG["data"]["models_dir"]

# =============================================================================
# Model Registry (loaded once at startup)
# =============================================================================

class ModelRegistry:
    clf:        Optional[xgb.XGBClassifier]  = None
    reg:        Optional[xgb.XGBRegressor]   = None
    feat_cols:  Optional[list[str]]          = None
    detector:   Optional[AnomalyDetector]    = None


_registry = ModelRegistry()


def _load_models() -> None:
    """Load trained models from the models/ directory at application startup."""
    clf_path  = MODEL_DIR / "xgb_classifier.json"
    reg_path  = MODEL_DIR / "xgb_regressor.json"
    feat_path = MODEL_DIR / "xgb_feature_cols.json"
    det_path  = MODEL_DIR / "anomaly_detector.joblib"

    if clf_path.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(clf_path)
        _registry.clf = clf
        logger.info("XGBoost classifier loaded.")
    else:
        logger.warning("No XGBoost classifier found at %s", clf_path)

    if reg_path.exists():
        reg = xgb.XGBRegressor()
        reg.load_model(reg_path)
        _registry.reg = reg
        logger.info("XGBoost regressor loaded.")
    else:
        logger.warning("No XGBoost regressor found at %s", reg_path)

    if feat_path.exists():
        with open(feat_path) as f:
            _registry.feat_cols = json.load(f)
        logger.info("Feature columns loaded (%d cols).", len(_registry.feat_cols))
    else:
        logger.warning("No feature columns file found at %s", feat_path)

    if det_path.exists():
        _registry.detector = AnomalyDetector.load(det_path)
        logger.info("Anomaly detector loaded.")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title       = "Supply Chain AI Monitoring API",
    description = (
        "REST service for predicting shipment delays, estimating disruption risk, "
        "and detecting logistics anomalies in real time."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    _load_models()


# =============================================================================
# Pydantic Schemas
# =============================================================================

class ShipmentInput(BaseModel):
    """Single shipment feature vector for delay prediction."""
    shipment_id:      Optional[str] = None
    weather_severity: Annotated[int,   Field(ge=0,   le=3)]
    traffic_level:    Annotated[int,   Field(ge=1,   le=5)]
    supplier_risk:    Annotated[float, Field(ge=0.0, le=1.0)]
    port_congestion:  Annotated[float, Field(ge=0.0, le=1.0)]
    distance_km:      Annotated[float, Field(gt=0)]
    transport_mode:   Literal["truck", "ship", "air"]

    class Config:
        json_schema_extra = {
            "example": {
                "shipment_id":      "SHP-00042",
                "weather_severity": 2,
                "traffic_level":    4,
                "supplier_risk":    0.35,
                "port_congestion":  0.72,
                "distance_km":      4500.0,
                "transport_mode":   "ship",
            }
        }


class DelayPrediction(BaseModel):
    """Delay prediction response."""
    shipment_id:          Optional[str]
    delayed:              bool
    delay_probability:    float
    estimated_delay_hours: float
    risk_score:           float
    risk_level:           Literal["low", "medium", "high", "critical"]
    recommendation:       str


class BatchInput(BaseModel):
    shipments: Annotated[List[ShipmentInput], Field(min_length=1, max_length=500)]


class BatchPrediction(BaseModel):
    predictions: List[DelayPrediction]
    summary: dict


# =============================================================================
# Prediction Logic
# =============================================================================

def _feature_vector(s: ShipmentInput) -> np.ndarray:
    """
    Build a dense feature vector from a ShipmentInput.

    Falls back to the minimal feature set if full feature engineering
    columns are not available (graceful degradation).
    """
    mode_ship  = int(s.transport_mode == "ship")
    mode_truck = int(s.transport_mode == "truck")
    dist_log   = float(np.log1p(s.distance_km))

    risk = (
        0.30 * (s.weather_severity / 3.0)
        + 0.25 * s.port_congestion
        + 0.30 * s.supplier_risk
        + 0.15 * (s.traffic_level / 5.0)
    )

    wx_c  = (s.weather_severity / 3.0) * s.port_congestion
    wx_s  = (s.weather_severity / 3.0) * s.supplier_risk
    cx_t  = s.port_congestion * (s.traffic_level / 5.0)

    base = {
        "weather_severity":           s.weather_severity,
        "traffic_level":              s.traffic_level,
        "supplier_risk":              s.supplier_risk,
        "port_congestion":            s.port_congestion,
        "distance_km":                s.distance_km,
        "mode_ship":                  mode_ship,
        "mode_truck":                 mode_truck,
        "feat_distance_log":          dist_log,
        "feat_weather_x_congestion":  wx_c,
        "feat_weather_x_supplier":    wx_s,
        "feat_congestion_x_traffic":  cx_t,
        "feat_total_risk_product":    wx_c * s.supplier_risk,
        "risk_score":                 risk,
        # Temporal features — use neutral defaults for API calls without date
        "day_of_week":  2,    # Wednesday (midweek neutral)
        "month":        6,    # June (low-season neutral)
        "quarter":      2,
        "is_weekend":   0,
        "month_sin":    float(np.sin(2 * np.pi * 6 / 12)),
        "month_cos":    float(np.cos(2 * np.pi * 6 / 12)),
        "dow_sin":      float(np.sin(2 * np.pi * 2 / 7)),
        "dow_cos":      float(np.cos(2 * np.pi * 2 / 7)),
        # Rolling / lag features — substitute current congestion as neutral estimate
        "cng_roll7_mean":  s.port_congestion,
        "cng_roll14_mean": s.port_congestion,
        "cng_roll30_mean": s.port_congestion,
        "cng_roll7_std":   0.05,
        "cng_roll14_std":  0.05,
        "cng_roll30_std":  0.05,
        "cng_lag1":        s.port_congestion,
        "cng_lag7":        s.port_congestion,
        "cng_lag14":       s.port_congestion,
    }

    feat_cols = _registry.feat_cols or list(base.keys())
    return np.array([base.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)


def _risk_level(score: float) -> str:
    if score >= 0.75:   return "critical"
    elif score >= 0.55: return "high"
    elif score >= 0.35: return "medium"
    return "low"


def _recommend(delayed: bool, risk_score: float, est_delay: float) -> str:
    if not delayed:
        return "Shipment is predicted to arrive on time. Continue routine monitoring."
    if risk_score >= 0.75:
        return (
            f"CRITICAL: High probability of delay (~{est_delay:.0f} h). "
            "Activate contingency plan immediately — engage backup supplier or expedite air freight."
        )
    if risk_score >= 0.55:
        return (
            f"HIGH RISK: Significant delay risk (~{est_delay:.0f} h). "
            "Notify stakeholders, monitor port congestion, consider alternative routing."
        )
    return (
        f"MODERATE RISK: Potential delay (~{est_delay:.0f} h). "
        "Increase monitoring frequency and prepare communication template."
    )


def _predict_single(s: ShipmentInput) -> DelayPrediction:
    if _registry.clf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available. Train the model first.",
        )

    X = _feature_vector(s)

    prob     = float(_registry.clf.predict_proba(X)[0, 1])
    delayed  = prob >= 0.50
    est_hrs  = 0.0

    if _registry.reg is not None and delayed:
        est_hrs = float(np.expm1(_registry.reg.predict(X)[0]))
    elif delayed:
        # Heuristic fallback when regressor absent
        est_hrs = prob * 80.0

    risk = (
        0.30 * (s.weather_severity / 3.0)
        + 0.25 * s.port_congestion
        + 0.30 * s.supplier_risk
        + 0.15 * (s.traffic_level / 5.0)
    )

    return DelayPrediction(
        shipment_id           = s.shipment_id,
        delayed               = delayed,
        delay_probability     = round(prob, 4),
        estimated_delay_hours = round(est_hrs, 2),
        risk_score            = round(risk, 4),
        risk_level            = _risk_level(risk),
        recommendation        = _recommend(delayed, risk, est_hrs),
    )


# =============================================================================
# Route Handlers
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Supply Chain AI Monitoring API",
        "version": "1.0.0",
        "docs":    "/docs",
        "models_loaded": {
            "classifier": _registry.clf is not None,
            "regressor":  _registry.reg is not None,
            "anomaly":    _registry.detector is not None,
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status":     "healthy",
        "classifier": _registry.clf is not None,
        "regressor":  _registry.reg is not None,
        "anomaly_detector": _registry.detector is not None,
    }


@app.post("/predict/delay", response_model=DelayPrediction, tags=["Prediction"])
async def predict_delay(shipment: ShipmentInput) -> DelayPrediction:
    """
    Predict whether a single shipment will be delayed and estimate delay hours.
    """
    return _predict_single(shipment)


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(payload: BatchInput) -> BatchPrediction:
    """
    Batch delay prediction for up to 500 shipments.
    Returns individual predictions and an aggregated summary.
    """
    predictions = [_predict_single(s) for s in payload.shipments]
    n_delayed   = sum(1 for p in predictions if p.delayed)
    avg_prob    = float(np.mean([p.delay_probability for p in predictions]))
    avg_hrs     = float(np.mean([p.estimated_delay_hours for p in predictions if p.delayed])) \
                  if n_delayed else 0.0

    return BatchPrediction(
        predictions=predictions,
        summary={
            "total":               len(predictions),
            "n_delayed":           n_delayed,
            "delay_rate_pct":      round(100 * n_delayed / max(len(predictions), 1), 2),
            "avg_delay_probability": round(avg_prob, 4),
            "avg_delay_hours_if_delayed": round(avg_hrs, 2),
        },
    )


@app.get("/anomalies", tags=["Monitoring"])
async def get_anomalies(top_n: int = 20):
    """
    Return the `top_n` most anomalous recent shipments from loaded data.
    Requires local data files to be present.
    """
    try:
        loader    = SupplyChainLoader()
        shipments = loader.shipments()
        if _registry.detector is not None:
            det = _registry.detector
        else:
            det = AnomalyDetector()
            det.fit(shipments)

        scored  = det.predict(shipments)
        anomaly = (
            scored[scored["is_anomaly"] == 1]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n)
        )

        cols = ["shipment_id", "ship_date", "transport_mode", "delayed",
                "delay_hours", "anomaly_score", "anomaly_level"]
        available_cols = [c for c in cols if c in anomaly.columns]
        return anomaly[available_cols].to_dict(orient="records")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(exc)}",
        )


@app.get("/congestion", tags=["Monitoring"])
async def get_congestion_alerts():
    """Return current port congestion alerts (elevated, high, critical)."""
    try:
        loader    = SupplyChainLoader()
        cng       = loader.port_congestion()
        alerts    = congestion_alerts(cng, CFG)
        cols = ["port_id", "location", "date", "congestion_level",
                "queue_time_hours", "alert_level"]
        available = [c for c in cols if c in alerts.columns]
        return alerts[available].to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch congestion data: {str(exc)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_summary_metrics():
    """
    Return high-level KPI metrics computed from the stored shipments dataset.
    """
    try:
        loader    = SupplyChainLoader()
        shipments = loader.shipments()
        delayed   = shipments[shipments["delayed"] == 1]

        return {
            "total_shipments":         len(shipments),
            "total_delayed":           int(shipments["delayed"].sum()),
            "delay_rate_pct":          round(shipments["delayed"].mean() * 100, 2),
            "avg_delay_hours":         round(delayed["delay_hours"].mean(), 2),
            "median_delay_hours":      round(delayed["delay_hours"].median(), 2),
            "max_delay_hours":         round(delayed["delay_hours"].max(), 2),
            "avg_supplier_risk":       round(shipments["supplier_risk"].mean(), 4),
            "avg_port_congestion":     round(shipments["port_congestion"].mean(), 4),
            "transport_mode_delay_rate": (
                shipments.groupby("transport_mode")["delayed"]
                .mean()
                .mul(100)
                .round(2)
                .to_dict()
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute metrics: {str(exc)}",
        )

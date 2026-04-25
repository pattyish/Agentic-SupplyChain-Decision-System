"""
FastAPI service for predictive monitoring and decision support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.anomaly.anomaly_detection import AnomalyDetector, congestion_alerts
from src.data.loader import SupplyChainLoader
from src.decision.intelligence import choose_best_action

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = _load_config()
MODEL_DIR = ROOT / CFG["data"]["models_dir"]


class ModelRegistry:
    clf: Optional[xgb.XGBClassifier] = None
    reg: Optional[xgb.XGBRegressor] = None
    calibrator = None
    feat_cols: Optional[list[str]] = None
    detector: Optional[AnomalyDetector] = None


_registry = ModelRegistry()

RATE_LIMIT_STATE: dict[str, list[float]] = defaultdict(list)
PREDICTION_CACHE: dict[str, tuple[float, dict]] = {}
BATCH_JOBS: dict[str, dict] = {}


def _cache_ttl() -> int:
    return int(CFG.get("api", {}).get("cache", {}).get("ttl_seconds", 300))


def _rate_limit_check(client_key: str) -> None:
    cfg = CFG.get("api", {}).get("rate_limit", {})
    rpm = int(cfg.get("requests_per_minute", 120))
    now = time.time()
    window_start = now - 60.0

    recent = [ts for ts in RATE_LIMIT_STATE[client_key] if ts >= window_start]
    if len(recent) >= rpm:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    recent.append(now)
    RATE_LIMIT_STATE[client_key] = recent


def _auth_enabled() -> bool:
    return bool(CFG.get("api", {}).get("security", {}).get("auth_enabled", False))


def _validate_api_key(request: Request) -> None:
    if not _auth_enabled():
        return
    sec_cfg = CFG.get("api", {}).get("security", {})
    header = sec_cfg.get("api_key_header", "X-API-Key")
    expected = sec_cfg.get("api_key", "")
    provided = request.headers.get(header)
    if not expected or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _load_models() -> None:
    clf_path = MODEL_DIR / "xgb_classifier.json"
    reg_path = MODEL_DIR / "xgb_regressor.json"
    feat_path = MODEL_DIR / "xgb_feature_cols.json"
    cal_path = MODEL_DIR / "xgb_classifier_calibrator.joblib"
    det_path = MODEL_DIR / "anomaly_detector.joblib"

    if clf_path.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(clf_path)
        _registry.clf = clf

    if reg_path.exists():
        reg = xgb.XGBRegressor()
        reg.load_model(reg_path)
        _registry.reg = reg

    if feat_path.exists():
        _registry.feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))

    if cal_path.exists():
        _registry.calibrator = joblib.load(cal_path)

    if det_path.exists():
        _registry.detector = AnomalyDetector.load(det_path)


app = FastAPI(
    title="Supply Chain AI Monitoring API",
    description="REST service for shipment-delay prediction, anomaly detection, and decision support.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_and_rate_limit(request: Request, call_next):
    client_key = request.client.host if request.client else "unknown"
    _rate_limit_check(client_key)

    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if CFG.get("api", {}).get("audit", {}).get("enabled", True):
        logger.info(
            "audit method=%s path=%s status=%s latency_ms=%.2f client=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            client_key,
        )
    return response


@app.on_event("startup")
async def startup_event() -> None:
    _load_models()


class ShipmentInput(BaseModel):
    shipment_id: Optional[str] = None
    weather_severity: Annotated[int, Field(ge=0, le=3)]
    traffic_level: Annotated[int, Field(ge=1, le=5)]
    supplier_risk: Annotated[float, Field(ge=0.0, le=1.0)]
    port_congestion: Annotated[float, Field(ge=0.0, le=1.0)]
    distance_km: Annotated[float, Field(gt=0)]
    transport_mode: Literal["truck", "ship", "air"]


class DelayPrediction(BaseModel):
    shipment_id: Optional[str]
    delayed: bool
    delay_probability: float
    estimated_delay_hours: float
    risk_score: float
    risk_level: Literal["low", "medium", "high", "critical"]
    recommendation: str
    action: str
    expected_delay_reduction_hours: float
    estimated_cost_avoided: float


class BatchInput(BaseModel):
    shipments: list[ShipmentInput] = Field(min_length=1, max_length=500)


class BatchPrediction(BaseModel):
    predictions: list[DelayPrediction]
    summary: dict


class AsyncBatchAccepted(BaseModel):
    job_id: str
    status: str


def _feature_vector(s: ShipmentInput) -> np.ndarray:
    mode_ship = int(s.transport_mode == "ship")
    mode_truck = int(s.transport_mode == "truck")
    risk = (
        0.30 * (s.weather_severity / 3.0)
        + 0.25 * s.port_congestion
        + 0.30 * s.supplier_risk
        + 0.15 * (s.traffic_level / 5.0)
    )

    base = {
        "weather_severity": s.weather_severity,
        "traffic_level": s.traffic_level,
        "supplier_risk": s.supplier_risk,
        "port_congestion": s.port_congestion,
        "distance_km": s.distance_km,
        "mode_ship": mode_ship,
        "mode_truck": mode_truck,
        "feat_distance_log": float(np.log1p(s.distance_km)),
        "feat_weather_x_congestion": (s.weather_severity / 3.0) * s.port_congestion,
        "feat_weather_x_supplier": (s.weather_severity / 3.0) * s.supplier_risk,
        "feat_congestion_x_traffic": s.port_congestion * (s.traffic_level / 5.0),
        "risk_score": risk,
        "day_of_week": 2,
        "month": 6,
        "quarter": 2,
        "is_weekend": 0,
        "month_sin": float(np.sin(2 * np.pi * 6 / 12)),
        "month_cos": float(np.cos(2 * np.pi * 6 / 12)),
        "dow_sin": float(np.sin(2 * np.pi * 2 / 7)),
        "dow_cos": float(np.cos(2 * np.pi * 2 / 7)),
        "cng_roll7_mean": s.port_congestion,
        "cng_roll14_mean": s.port_congestion,
        "cng_roll30_mean": s.port_congestion,
        "cng_roll7_std": 0.05,
        "cng_roll14_std": 0.05,
        "cng_roll30_std": 0.05,
        "cng_lag1": s.port_congestion,
        "cng_lag7": s.port_congestion,
        "cng_lag14": s.port_congestion,
    }

    feat_cols = _registry.feat_cols or list(base.keys())
    return np.array([base.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _cache_key(shipment: ShipmentInput) -> str:
    payload = shipment.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _predict_single(shipment: ShipmentInput) -> DelayPrediction:
    if _registry.clf is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Prediction model unavailable")

    key = _cache_key(shipment)
    now = time.time()
    ttl = _cache_ttl()
    if CFG.get("api", {}).get("cache", {}).get("enabled", True) and key in PREDICTION_CACHE:
        ts, cached = PREDICTION_CACHE[key]
        if now - ts <= ttl:
            return DelayPrediction(**cached)

    X = _feature_vector(shipment)
    if _registry.calibrator is not None:
        prob = float(_registry.calibrator.predict_proba(X)[0, 1])
    else:
        prob = float(_registry.clf.predict_proba(X)[0, 1])

    delayed = prob >= 0.5
    est_hrs = 0.0
    if _registry.reg is not None and delayed:
        est_hrs = float(np.expm1(_registry.reg.predict(X)[0]))
    elif delayed:
        est_hrs = prob * 80.0

    risk = (
        0.30 * (shipment.weather_severity / 3.0)
        + 0.25 * shipment.port_congestion
        + 0.30 * shipment.supplier_risk
        + 0.15 * (shipment.traffic_level / 5.0)
    )

    decision = choose_best_action(
        delay_probability=prob,
        estimated_delay_hours=est_hrs,
        risk_score=risk,
        transport_mode=shipment.transport_mode,
        cfg=CFG,
    )

    recommendation = (
        f"Recommended action: {decision.action}. "
        f"Estimated delay reduction {decision.expected_delay_reduction_hours:.1f}h."
    )

    result = DelayPrediction(
        shipment_id=shipment.shipment_id,
        delayed=delayed,
        delay_probability=round(prob, 4),
        estimated_delay_hours=round(est_hrs, 2),
        risk_score=round(risk, 4),
        risk_level=_risk_level(risk),
        recommendation=recommendation,
        action=decision.action,
        expected_delay_reduction_hours=decision.expected_delay_reduction_hours,
        estimated_cost_avoided=decision.estimated_cost_avoided,
    )

    if CFG.get("api", {}).get("cache", {}).get("enabled", True):
        PREDICTION_CACHE[key] = (now, result.model_dump())
    return result


def _run_batch_job(job_id: str, payload: BatchInput) -> None:
    try:
        predictions = [_predict_single(s) for s in payload.shipments]
        n_delayed = sum(1 for p in predictions if p.delayed)
        avg_prob = float(np.mean([p.delay_probability for p in predictions]))
        avg_hrs = float(np.mean([p.estimated_delay_hours for p in predictions if p.delayed])) if n_delayed else 0.0

        BATCH_JOBS[job_id] = {
            "status": "completed",
            "result": {
                "predictions": [p.model_dump() for p in predictions],
                "summary": {
                    "total": len(predictions),
                    "n_delayed": n_delayed,
                    "delay_rate_pct": round(100 * n_delayed / max(len(predictions), 1), 2),
                    "avg_delay_probability": round(avg_prob, 4),
                    "avg_delay_hours_if_delayed": round(avg_hrs, 2),
                },
            },
        }
    except Exception as exc:
        BATCH_JOBS[job_id] = {"status": "failed", "error": str(exc)}


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Supply Chain AI Monitoring API",
        "version": "1.1.0",
        "docs": "/docs",
        "models_loaded": {
            "classifier": _registry.clf is not None,
            "regressor": _registry.reg is not None,
            "calibrator": _registry.calibrator is not None,
            "anomaly": _registry.detector is not None,
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "healthy",
        "classifier": _registry.clf is not None,
        "regressor": _registry.reg is not None,
        "calibrator": _registry.calibrator is not None,
        "anomaly_detector": _registry.detector is not None,
    }


@app.post("/predict/delay", response_model=DelayPrediction, tags=["Prediction"])
async def predict_delay(shipment: ShipmentInput, request: Request) -> DelayPrediction:
    _validate_api_key(request)
    return _predict_single(shipment)


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(payload: BatchInput, request: Request) -> BatchPrediction:
    _validate_api_key(request)
    predictions = [_predict_single(s) for s in payload.shipments]
    n_delayed = sum(1 for p in predictions if p.delayed)
    avg_prob = float(np.mean([p.delay_probability for p in predictions]))
    avg_hrs = float(np.mean([p.estimated_delay_hours for p in predictions if p.delayed])) if n_delayed else 0.0

    return BatchPrediction(
        predictions=predictions,
        summary={
            "total": len(predictions),
            "n_delayed": n_delayed,
            "delay_rate_pct": round(100 * n_delayed / max(len(predictions), 1), 2),
            "avg_delay_probability": round(avg_prob, 4),
            "avg_delay_hours_if_delayed": round(avg_hrs, 2),
        },
    )


@app.post("/predict/batch/async", response_model=AsyncBatchAccepted, tags=["Prediction"])
async def predict_batch_async(payload: BatchInput, background_tasks: BackgroundTasks, request: Request) -> AsyncBatchAccepted:
    _validate_api_key(request)
    if not CFG.get("api", {}).get("async_batch", {}).get("enabled", True):
        raise HTTPException(status_code=400, detail="Async batch disabled")

    max_jobs = int(CFG.get("api", {}).get("async_batch", {}).get("max_jobs", 200))
    active = sum(1 for v in BATCH_JOBS.values() if v.get("status") in {"queued", "running"})
    if active >= max_jobs:
        raise HTTPException(status_code=429, detail="Too many queued jobs")

    job_id = str(uuid.uuid4())
    BATCH_JOBS[job_id] = {"status": "queued"}

    def _runner():
        BATCH_JOBS[job_id] = {"status": "running"}
        _run_batch_job(job_id, payload)

    background_tasks.add_task(_runner)
    return AsyncBatchAccepted(job_id=job_id, status="queued")


@app.get("/predict/batch/async/{job_id}", tags=["Prediction"])
async def get_async_batch_result(job_id: str, request: Request):
    _validate_api_key(request)
    if job_id not in BATCH_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return BATCH_JOBS[job_id]


@app.get("/anomalies", tags=["Monitoring"])
async def get_anomalies(top_n: int = 20):
    try:
        loader = SupplyChainLoader()
        shipments = loader.shipments()
        det = _registry.detector if _registry.detector is not None else AnomalyDetector()
        if _registry.detector is None:
            det.fit(shipments)

        scored = det.predict(shipments)
        anomaly = scored[scored["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(top_n)
        cols = ["shipment_id", "ship_date", "transport_mode", "delayed", "delay_hours", "anomaly_score", "anomaly_level"]
        available_cols = [c for c in cols if c in anomaly.columns]
        return anomaly[available_cols].to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {exc}")


@app.get("/congestion", tags=["Monitoring"])
async def get_congestion_alerts():
    try:
        loader = SupplyChainLoader()
        cng = loader.port_congestion()
        alerts = congestion_alerts(cng, CFG)
        cols = ["port_id", "location", "date", "congestion_level", "queue_time_hours", "alert_level"]
        return alerts[[c for c in cols if c in alerts.columns]].to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch congestion data: {exc}")


@app.get("/metrics", tags=["Monitoring"])
async def get_summary_metrics():
    try:
        loader = SupplyChainLoader()
        shipments = loader.shipments()
        delayed = shipments[shipments["delayed"] == 1]

        return {
            "total_shipments": len(shipments),
            "total_delayed": int(shipments["delayed"].sum()),
            "delay_rate_pct": round(shipments["delayed"].mean() * 100, 2),
            "avg_delay_hours": round(delayed["delay_hours"].mean(), 2),
            "median_delay_hours": round(delayed["delay_hours"].median(), 2),
            "max_delay_hours": round(delayed["delay_hours"].max(), 2),
            "avg_supplier_risk": round(shipments["supplier_risk"].mean(), 4),
            "avg_port_congestion": round(shipments["port_congestion"].mean(), 4),
            "transport_mode_delay_rate": shipments.groupby("transport_mode")["delayed"].mean().mul(100).round(2).to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {exc}")

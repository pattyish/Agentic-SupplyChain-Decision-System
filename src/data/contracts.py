"""
Data contracts and quality gates for supply chain datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
except Exception as exc:  # pragma: no cover
    pa = None
    Column = DataFrameSchema = Check = None


CONTRACTS: dict[str, Any] = {}

if pa is not None:
    CONTRACTS = {
        "shipments": DataFrameSchema(
            {
                "shipment_id": Column(int),
                "supplier_id": Column(int),
                "port_id": Column(int),
                "ship_date": Column(str),
                "expected_delivery_date": Column(str),
                "actual_delivery_date": Column(str),
                "distance_km": Column(float, checks=Check.gt(0)),
                "transport_mode": Column(str, checks=Check.isin(["truck", "ship", "air"])),
                "traffic_level": Column(int, checks=Check.in_range(1, 5)),
                "weather_severity": Column(int, checks=Check.in_range(0, 3)),
                "port_congestion": Column(float, checks=Check.in_range(0, 1)),
                "supplier_risk": Column(float, checks=Check.in_range(0, 1)),
                "delayed": Column(int, checks=Check.isin([0, 1])),
                "delay_hours": Column(float, checks=Check.ge(0)),
            },
            strict=False,
            coerce=True,
        ),
        "suppliers": DataFrameSchema(
            {
                "supplier_id": Column(int),
                "reliability_score": Column(float, checks=Check.in_range(0, 1)),
                "avg_lead_time_days": Column(float, checks=Check.gt(0)),
                "failure_rate": Column(float, checks=Check.in_range(0, 1)),
            },
            strict=False,
            coerce=True,
        ),
        "weather": DataFrameSchema(
            {
                "date": Column(str),
                "location": Column(str),
                "weather_severity": Column(int, checks=Check.in_range(0, 3)),
                "precipitation": Column(float, checks=Check.ge(0)),
                "wind_speed": Column(float, checks=Check.ge(0)),
                "storm_flag": Column(int, checks=Check.isin([0, 1])),
            },
            strict=False,
            coerce=True,
        ),
        "congestion": DataFrameSchema(
            {
                "port_id": Column(int),
                "location": Column(str),
                "date": Column(str),
                "congestion_level": Column(float, checks=Check.in_range(0, 1)),
                "queue_time_hours": Column(float, checks=Check.ge(0)),
            },
            strict=False,
            coerce=True,
        ),
    }


def _freshness_days(df: pd.DataFrame) -> float | None:
    for col in ["ship_date", "date", "timestamp"]:
        if col in df.columns and len(df):
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dates):
                return float((datetime.now(timezone.utc) - dates.max().to_pydatetime().replace(tzinfo=timezone.utc)).days)
    return None


def evaluate_dataset(name: str, df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    dq_cfg = cfg.get("data_quality", {})
    max_missing_ratio = float(dq_cfg.get("max_missing_ratio", 0.05))
    max_freshness_days = int(dq_cfg.get("max_allowed_freshness_days", 7))

    result: dict[str, Any] = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "contract_ok": True,
        "missing_ratio": float(df.isna().mean().mean()) if len(df.columns) else 0.0,
        "freshness_days": _freshness_days(df),
        "errors": [],
    }

    if pa is None:
        result["contract_ok"] = False
        result["errors"].append("pandera is not installed or failed to import")
    elif name in CONTRACTS:
        try:
            CONTRACTS[name].validate(df, lazy=True)
        except Exception as exc:
            result["contract_ok"] = False
            result["errors"].append(str(exc))

    if result["missing_ratio"] > max_missing_ratio:
        result["errors"].append(
            f"missing ratio {result['missing_ratio']:.4f} exceeds max {max_missing_ratio:.4f}"
        )

    if result["freshness_days"] is not None and result["freshness_days"] > max_freshness_days:
        result["errors"].append(
            f"freshness {result['freshness_days']:.1f} days exceeds max {max_freshness_days}"
        )

    return result


def run_data_quality_gate(datasets: dict[str, pd.DataFrame], cfg: dict, output_dir: Path) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "passed": True,
    }

    for name, df in datasets.items():
        ds_report = evaluate_dataset(name, df, cfg)
        report["datasets"][name] = ds_report
        if ds_report["errors"]:
            report["passed"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "data_quality_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if cfg.get("data_quality", {}).get("fail_on_error", True) and not report["passed"]:
        raise RuntimeError(f"Data quality gate failed. See {out_file}")

    return report

"""
loader.py
---------
Data loading, validation, and merging utilities for the
AI-Driven Supply Chain Predictive Monitoring System.

Responsibilities:
  • Load raw CSV datasets with schema validation
  • Build the merged analytical table (shipments + suppliers + weather + congestion)
  • Provide train / validation / test splits (time-based to prevent data leakage)
  • Expose typed DataFrames ready for feature engineering
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Schema definitions ────────────────────────────────────────────────────────
SHIPMENT_DTYPES: dict[str, str] = {
    "shipment_id":   "int64",
    "supplier_id":   "int64",
    "port_id":       "int64",
    "distance_km":   "float64",
    "traffic_level": "int64",
    "weather_severity": "int64",
    "port_congestion":  "float64",
    "supplier_risk":    "float64",
    "delayed":       "int64",
    "delay_hours":   "float64",
}

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "shipments":   ["shipment_id", "supplier_id", "port_id", "ship_date",
                    "expected_delivery_date", "actual_delivery_date",
                    "distance_km", "transport_mode", "traffic_level",
                    "weather_severity", "port_congestion", "supplier_risk",
                    "delayed", "delay_hours"],
    "suppliers":   ["supplier_id", "reliability_score", "avg_lead_time_days", "failure_rate"],
    "weather":     ["date", "location", "weather_severity", "precipitation", "wind_speed", "storm_flag"],
    "congestion":  ["port_id", "location", "date", "congestion_level", "queue_time_hours"],
    "disruptions": ["event_id", "shipment_id", "event_type", "cause", "severity",
                    "timestamp", "duration_hours"],
    "features":    ["shipment_id", "weather_severity", "traffic_level",
                    "supplier_risk", "congestion_level", "risk_score",
                    "delayed", "delay_hours"],
}


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _validate_schema(df: pd.DataFrame, name: str) -> None:
    """Raise ValueError if required columns are missing."""
    expected = REQUIRED_COLUMNS.get(name, [])
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


# =============================================================================
# Public API
# =============================================================================

class SupplyChainLoader:
    """
    Centralised data-access layer for the Supply Chain AI system.

    Parameters
    ----------
    data_dir : str | Path
        Path to the directory containing the raw CSV files.
    config_path : str, optional
        Path to config.yaml; defaults to configs/config.yaml.

    Example
    -------
    >>> loader = SupplyChainLoader("data/")
    >>> shipments = loader.shipments()
    >>> merged    = loader.merged()
    """

    def __init__(self, data_dir: Optional[str] = None, config_path: Optional[str] = None):
        self.cfg = _load_config(config_path)
        if data_dir is None:
            root = Path(__file__).parents[2]
            data_dir = root / self.cfg["data"]["raw_dir"]
        self.data_dir = Path(data_dir)
        logger.info("SupplyChainLoader initialised  →  %s", self.data_dir)

    # ── Individual loaders ────────────────────────────────────────────────────
    def shipments(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "shipments.csv", parse_dates=False)
        _validate_schema(df, "shipments")
        for col, dtype in SHIPMENT_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        for col in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
            df[col] = pd.to_datetime(df[col])
        logger.debug("Loaded shipments: %s rows", len(df))
        return df

    def suppliers(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "suppliers.csv")
        _validate_schema(df, "suppliers")
        return df

    def weather(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "weather.csv", parse_dates=["date"])
        _validate_schema(df, "weather")
        return df

    def port_congestion(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "port_congestion.csv", parse_dates=["date"])
        _validate_schema(df, "congestion")
        return df

    def disruptions(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "disruptions.csv", parse_dates=["timestamp"])
        _validate_schema(df, "disruptions")
        return df

    def features(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "features.csv")
        _validate_schema(df, "features")
        return df

    # ── Merged analytical table ───────────────────────────────────────────────
    def merged(self) -> pd.DataFrame:
        """
        Join shipments with supplier and congestion data to produce a single
        enriched DataFrame suitable for EDA and feature engineering.

        Joins performed:
          1. shipments ← suppliers on supplier_id   (LEFT JOIN)
          2. enriched  ← port_congestion on (port_id, ship_date)  (LEFT JOIN)
        """
        shp = self.shipments()
        sup = self.suppliers()
        cng = self.port_congestion()

        # Supplier enrichment
        df = shp.merge(
            sup.rename(columns={
                "reliability_score":  "sup_reliability",
                "avg_lead_time_days": "sup_lead_time",
                "failure_rate":       "sup_failure_rate",
            }),
            on="supplier_id", how="left",
        )

        # Daily port congestion merge (on port × date)
        cng_daily = (
            cng.groupby(["port_id", "date"])[["congestion_level", "queue_time_hours"]]
            .mean()
            .reset_index()
            .rename(columns={
                "congestion_level": "cng_level_daily",
                "queue_time_hours": "queue_hours_daily",
            })
        )
        cng_daily["date"] = pd.to_datetime(cng_daily["date"])
        df["ship_date_dt"] = pd.to_datetime(df["ship_date"])

        df = df.merge(
            cng_daily.rename(columns={"date": "ship_date_dt"}),
            on=["port_id", "ship_date_dt"], how="left",
        )
        df.drop(columns=["ship_date_dt"], inplace=True)

        logger.debug("Merged DataFrame shape: %s", df.shape)
        return df

    # ── Train / validation / test split ───────────────────────────────────────
    def time_split(
        self,
        df: Optional[pd.DataFrame] = None,
        val_size:  float = 0.15,
        test_size: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based split to prevent data leakage.

        The test set is the most-recent `test_size` fraction of dates,
        validation is the next `val_size` fraction, and training is the rest.

        Returns
        -------
        train, val, test : pd.DataFrame
        """
        if df is None:
            df = self.shipments()

        df = df.copy()
        df["_sort_date"] = pd.to_datetime(df["ship_date"])
        df.sort_values("_sort_date", inplace=True)
        df.drop(columns=["_sort_date"], inplace=True)

        n        = len(df)
        test_cut = int(n * (1.0 - test_size))
        val_cut  = int(n * (1.0 - test_size - val_size))

        train = df.iloc[:val_cut].reset_index(drop=True)
        val   = df.iloc[val_cut:test_cut].reset_index(drop=True)
        test  = df.iloc[test_cut:].reset_index(drop=True)

        logger.info(
            "Time split → train: %d  |  val: %d  |  test: %d",
            len(train), len(val), len(test),
        )
        return train, val, test


# ── Module-level convenience functions ─────────────────────────────────────────

def load_all(data_dir: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Return a dict of all datasets keyed by name."""
    loader = SupplyChainLoader(data_dir)
    return {
        "shipments":   loader.shipments(),
        "suppliers":   loader.suppliers(),
        "weather":     loader.weather(),
        "congestion":  loader.port_congestion(),
        "disruptions": loader.disruptions(),
        "features":    loader.features(),
        "merged":      loader.merged(),
    }

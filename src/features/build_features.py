"""
build_features.py
-----------------
Feature engineering pipeline for the Supply Chain AI system.

Transforms the raw merged DataFrame into a model-ready feature matrix by:
  1. Encoding categorical variables (transport_mode → one-hot)
  2. Extracting temporal features (day-of-week, month, quarter, is_weekend)
  3. Computing rolling-window statistics on port congestion (7 / 14 / 30 day)
  4. Computing lag features for congestion  (t−1, t−7, t−14)
  5. Creating interaction features capturing compound risk
  6. Computing a composite risk score
  7. Normalising numerical features (optional, for LSTM pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Individual Transform Functions
# =============================================================================

def encode_transport_mode(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode `transport_mode`; drop_first avoids dummy-variable trap."""
    dummies = pd.get_dummies(df["transport_mode"], prefix="mode", drop_first=drop_first)
    dummies = dummies.astype(int)
    return pd.concat([df.drop(columns=["transport_mode"]), dummies], axis=1)


def add_temporal_features(df: pd.DataFrame, date_col: str = "ship_date") -> pd.DataFrame:
    """
    Derive temporal features from the ship date:
      day_of_week (0=Mon … 6=Sun),  day_of_year,  month,  quarter,
      is_weekend,  is_month_end,  year,  week_of_year
    """
    dt = pd.to_datetime(df[date_col])
    df = df.copy()
    df["day_of_week"]   = dt.dt.dayofweek
    df["day_of_year"]   = dt.dt.dayofyear
    df["month"]         = dt.dt.month
    df["quarter"]       = dt.dt.quarter
    df["year"]          = dt.dt.year
    df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_end"]  = dt.dt.is_month_end.astype(int)
    # Seasonal sine/cosine encoding (preserves cyclical nature)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_port_rolling_features(
    shipments: pd.DataFrame,
    congestion: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling mean / std of port congestion over `windows` days and
    merge back onto the shipments DataFrame.

    Parameters
    ----------
    shipments   : DataFrame containing ship_date and port_id
    congestion  : Daily port congestion DataFrame (port_id, date, congestion_level)
    windows     : List of rolling window sizes in days; defaults to [7, 14, 30]
    """
    if windows is None:
        windows = [7, 14, 30]

    cng = congestion[["port_id", "date", "congestion_level"]].copy()
    cng["date"] = pd.to_datetime(cng["date"])
    cng.sort_values(["port_id", "date"], inplace=True)

    for w in windows:
        col_mean = f"cng_roll{w}_mean"
        col_std  = f"cng_roll{w}_std"
        cng[col_mean] = (
            cng.groupby("port_id")["congestion_level"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        cng[col_std] = (
            cng.groupby("port_id")["congestion_level"]
            .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0.0))
        )

    roll_cols = ["port_id", "date"] + [c for c in cng.columns if "roll" in c]
    ships_out = shipments.copy()
    ships_out["_ship_date_dt"] = pd.to_datetime(ships_out["ship_date"])

    ships_out = ships_out.merge(
        cng[roll_cols].rename(columns={"date": "_ship_date_dt"}),
        on=["port_id", "_ship_date_dt"],
        how="left",
    )
    ships_out.drop(columns=["_ship_date_dt"], inplace=True)
    return ships_out


def add_lag_features(
    shipments: pd.DataFrame,
    congestion: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged congestion values (t−lag) merged onto the shipments table.

    Useful for capturing historical congestion trend at time of shipment.
    """
    if lags is None:
        lags = [1, 7, 14]

    cng = congestion[["port_id", "date", "congestion_level"]].copy()
    cng["date"] = pd.to_datetime(cng["date"])
    cng.sort_values(["port_id", "date"], inplace=True)

    for lag in lags:
        cng[f"cng_lag{lag}"] = (
            cng.groupby("port_id")["congestion_level"].shift(lag).fillna(method="bfill")
        )

    lag_cols = ["port_id", "date"] + [c for c in cng.columns if "lag" in c]
    ships_out = shipments.copy()
    ships_out["_ship_date_dt"] = pd.to_datetime(ships_out["ship_date"])

    ships_out = ships_out.merge(
        cng[lag_cols].rename(columns={"date": "_ship_date_dt"}),
        on=["port_id", "_ship_date_dt"],
        how="left",
    )
    ships_out.drop(columns=["_ship_date_dt"], inplace=True)
    return ships_out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound risk interaction terms:
      weather_congestion  = weather_severity * port_congestion
      weather_supplier    = weather_severity * supplier_risk
      congestion_traffic  = port_congestion  * (traffic_level / 5)
      total_risk_product  = weather * congestion * supplier_risk
      distance_norm       = log1p(distance_km)
    """
    df = df.copy()
    w  = df["weather_severity"] / 3.0
    c  = df["port_congestion"].fillna(df.get("congestion_level", df["port_congestion"]))
    s  = df["supplier_risk"]
    t  = df["traffic_level"] / 5.0

    df["feat_weather_x_congestion"] = (w * c).round(4)
    df["feat_weather_x_supplier"]   = (w * s).round(4)
    df["feat_congestion_x_traffic"] = (c * t).round(4)
    df["feat_total_risk_product"]   = (w * c * s).round(4)
    df["feat_distance_log"]         = np.log1p(df["distance_km"]).round(4)
    return df


def compute_risk_score(
    df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Composite risk score (0–1) as a weighted linear combination:
      score = w_w*(weather/3) + w_c*congestion + w_s*supplier_risk + w_t*(traffic/5)

    Default weights taken from config.yaml: 0.30, 0.25, 0.30, 0.15.
    """
    if weights is None:
        cfg = _load_config()
        weights = cfg["features"]["risk_weights"]

    df = df.copy()
    df["risk_score"] = (
        weights.get("weather_severity", 0.30) * (df["weather_severity"] / 3.0)
        + weights.get("port_congestion",  0.25) * df["port_congestion"]
        + weights.get("supplier_risk",    0.30) * df["supplier_risk"]
        + weights.get("traffic_level",    0.15) * (df["traffic_level"] / 5.0)
    ).clip(0.0, 1.0).round(4)
    return df


def scale_features(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on train, apply to val and test.

    Returns scaled DataFrames and the fitted scaler (for inverse-transform).
    """
    scaler = StandardScaler()
    train  = train.copy()
    val    = val.copy()
    test   = test.copy()

    train[num_cols] = scaler.fit_transform(train[num_cols].astype(float))
    val[num_cols]   = scaler.transform(val[num_cols].astype(float))
    test[num_cols]  = scaler.transform(test[num_cols].astype(float))

    logger.info("StandardScaler fitted on %d training samples.", len(train))
    return train, val, test, scaler


# =============================================================================
# Full Pipeline
# =============================================================================

def build_feature_matrix(
    shipments:  pd.DataFrame,
    congestion: pd.DataFrame,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Steps applied (in order):
      1. Encode transport_mode (one-hot)
      2. Temporal features from ship_date
      3. Rolling congestion features (7/14/30 days)
      4. Lag congestion features (−1/−7/−14 days)
      5. Interaction features
      6. Composite risk score

    Parameters
    ----------
    shipments  : Raw shipments DataFrame (from SupplyChainLoader)
    congestion : Daily port congestion DataFrame
    config_path: Optional path to config.yaml

    Returns
    -------
    Feature-engineered DataFrame ready for model training.
    """
    cfg     = _load_config(config_path)
    windows = cfg["features"].get("rolling_windows", [7, 14, 30])
    lags    = cfg["features"].get("lag_periods",     [1, 7, 14])

    logger.info("Building feature matrix from %d shipments …", len(shipments))

    df = encode_transport_mode(shipments)
    df = add_temporal_features(df)
    df = add_port_rolling_features(df, congestion, windows=windows)
    df = add_lag_features(df, congestion, lags=lags)
    df = add_interaction_features(df)
    df = compute_risk_score(df)

    # Drop raw date columns (keep encoded temporal features)
    drop_cols = [c for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]
                 if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Fill any residual NaNs from rolling/lag at series start
    df.fillna(df.median(numeric_only=True), inplace=True)

    logger.info("Feature matrix ready: %s", df.shape)
    return df


FEATURE_COLS: list[str] = [
    "weather_severity", "traffic_level", "supplier_risk", "port_congestion",
    "distance_km", "day_of_week", "month", "quarter", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "feat_weather_x_congestion", "feat_weather_x_supplier",
    "feat_congestion_x_traffic", "feat_total_risk_product",
    "feat_distance_log", "risk_score",
    "mode_ship", "mode_truck",                   # one-hot (air dropped as reference)
    "cng_roll7_mean",  "cng_roll14_mean",  "cng_roll30_mean",
    "cng_roll7_std",   "cng_roll14_std",   "cng_roll30_std",
    "cng_lag1",        "cng_lag7",         "cng_lag14",
]

TARGET_CLF = "delayed"
TARGET_REG = "delay_hours"

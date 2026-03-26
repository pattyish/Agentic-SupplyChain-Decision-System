"""
anomaly_detection.py
--------------------
Multi-method anomaly detection pipeline for supply chain monitoring.

Three complementary detectors are combined into a unified anomaly score:

  1. Isolation Forest (global)
       Identifies globally anomalous points in feature space.
       Parameters: n_estimators=200, contamination from config.

  2. Local Outlier Factor (local density)
       Detects points in low-density regions relative to their neighbours.
       Effective at finding local clusters of unusual behaviour.

  3. Statistical Z-Score Detector
       Flags shipments whose individual feature values exceed ±3σ thresholds.
       Fast, interpretable, complementary to ML-based approaches.

Ensemble strategy:
  anomaly_score = 0.45 * iso_score + 0.35 * lof_score + 0.20 * z_score_norm

  anomaly_label = 1 if anomaly_score > threshold (default 0.55)

Port-level rule-based alerts:
  • CRITICAL : congestion_level > 0.80
  • HIGH     : congestion_level > 0.65
  • ELEVATED : delay_hours > 120 and supplier_risk > 0.50

Usage:
  from src.anomaly.anomaly_detection import AnomalyDetector
  detector = AnomalyDetector()
  detector.fit(features_df)
  results  = detector.predict(features_df)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ANOMALY_FEATURES = [
    "weather_severity", "traffic_level", "supplier_risk",
    "port_congestion", "delay_hours", "risk_score",
]


def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# AnomalyDetector
# =============================================================================

class AnomalyDetector:
    """
    Ensemble anomaly detector for supply chain shipment data.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. Defaults to project configs/config.yaml.
    contamination : float, optional
        Expected fraction of anomalies. Overrides config if provided.

    Attributes
    ----------
    iso_forest : IsolationForest
    lof        : LocalOutlierFactor
    scaler     : StandardScaler (fitted on training data)
    feature_means, feature_stds : ndarray  (for Z-score detector)
    """

    def __init__(
        self,
        config_path:   Optional[str]   = None,
        contamination: Optional[float] = None,
    ):
        self.cfg = _load_config(config_path)
        acfg     = self.cfg["anomaly"]
        cont     = contamination if contamination is not None else acfg["contamination"]

        self.iso_forest = IsolationForest(
            n_estimators = acfg["isolation_forest"]["n_estimators"],
            contamination = cont,
            max_samples   = acfg["isolation_forest"]["max_samples"],
            random_state  = acfg["isolation_forest"]["random_state"],
            n_jobs        = -1,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors   = acfg["lof"]["n_neighbors"],
            contamination = cont,
            novelty       = True,           # allows predict() on new data
            n_jobs        = -1,
        )
        self.scaler       = StandardScaler()
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds:  Optional[np.ndarray] = None
        self._is_fitted    = False

    # ── Internal utilities ────────────────────────────────────────────────────
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate the anomaly feature matrix."""
        available = [c for c in ANOMALY_FEATURES if c in df.columns]
        missing   = [c for c in ANOMALY_FEATURES if c not in df.columns]
        if missing:
            logger.warning("Missing anomaly features (will use zeros): %s", missing)
        X = df[available].copy()
        for m in missing:
            X[m] = 0.0
        return X[ANOMALY_FEATURES].values.astype(float)

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit all three detectors on the supplied DataFrame.

        Parameters
        ----------
        df : DataFrame with columns matching ANOMALY_FEATURES.
        """
        X = self._prepare_X(df)
        X_scaled = self.scaler.fit_transform(X)

        self.feature_means = X.mean(axis=0)
        self.feature_stds  = np.clip(X.std(axis=0), 1e-8, None)

        logger.info("Fitting Isolation Forest on %d samples …", len(X))
        self.iso_forest.fit(X_scaled)

        logger.info("Fitting Local Outlier Factor …")
        self.lof.fit(X_scaled)

        self._is_fitted = True
        logger.info("AnomalyDetector fitted successfully.")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        threshold: float = 0.55,
    ) -> pd.DataFrame:
        """
        Score and flag anomalous shipments.

        Parameters
        ----------
        df        : DataFrame with ANOMALY_FEATURES columns.
        threshold : Ensemble score threshold above which a record is anomalous.

        Returns
        -------
        DataFrame with additional columns:
          iso_score     — Isolation Forest normalised score (0=normal, 1=anomaly)
          lof_score     — LOF normalised score
          z_score_norm  — Z-score based normalised score
          anomaly_score — Weighted ensemble score
          is_anomaly    — Binary flag (1 = anomaly)
          anomaly_level — "normal" / "elevated" / "high" / "critical"
        """
        if not self._is_fitted:
            raise RuntimeError("Detector is not fitted. Call .fit() first.")

        X        = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)

        # ── 1. Isolation Forest score ─────────────────────────────────────────
        # decision_function: negative score → more anomalous
        iso_raw   = -self.iso_forest.decision_function(X_scaled)
        iso_score = self._minmax_norm(iso_raw)

        # ── 2. LOF score ──────────────────────────────────────────────────────
        lof_raw   = -self.lof.decision_function(X_scaled)
        lof_score = self._minmax_norm(lof_raw)

        # ── 3. Z-score based detector ─────────────────────────────────────────
        z_abs     = np.abs((X - self.feature_means) / self.feature_stds)
        z_max     = z_abs.max(axis=1)           # worst single-feature deviation
        z_mean    = z_abs.mean(axis=1)
        z_norm    = self._minmax_norm(z_max + 0.5 * z_mean)

        # ── Ensemble ─────────────────────────────────────────────────────────
        ensemble = 0.45 * iso_score + 0.35 * lof_score + 0.20 * z_norm

        # ── Rule-based port alerts ────────────────────────────────────────────
        cng    = df.get("port_congestion", pd.Series(0.0, index=df.index)).values
        dh     = df.get("delay_hours",     pd.Series(0.0, index=df.index)).values
        sr     = df.get("supplier_risk",   pd.Series(0.0, index=df.index)).values

        critical = (cng > 0.80) | (dh > 120)
        high     = (cng > 0.65) | ((dh > 72) & (sr > 0.50))
        elevated = (cng > 0.50) | (dh > 24)

        # Boost ensemble score for rule-triggered rows
        ensemble = np.where(critical, np.maximum(ensemble, 0.85), ensemble)
        ensemble = np.where(high,     np.maximum(ensemble, 0.70), ensemble)
        ensemble = np.where(elevated, np.maximum(ensemble, 0.58), ensemble)

        is_anomaly  = (ensemble > threshold).astype(int)

        level = np.where(
            ensemble > 0.80, "critical",
            np.where(ensemble > 0.65, "high",
            np.where(ensemble > 0.50, "elevated", "normal"))
        )

        out = df.copy()
        out["iso_score"]     = iso_score.round(4)
        out["lof_score"]     = lof_score.round(4)
        out["z_score_norm"]  = z_norm.round(4)
        out["anomaly_score"] = ensemble.round(4)
        out["is_anomaly"]    = is_anomaly
        out["anomaly_level"] = level

        n_anomalies = is_anomaly.sum()
        logger.info(
            "Anomaly detection complete: %d / %d flagged (%.1f %%)",
            n_anomalies, len(df), 100 * n_anomalies / max(len(df), 1),
        )
        return out

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("AnomalyDetector saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        det = joblib.load(path)
        logger.info("AnomalyDetector loaded from %s", path)
        return det


# =============================================================================
# Port-Level Congestion Alerts
# =============================================================================

def congestion_alerts(congestion: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Apply threshold-based alerts to the latest port congestion data.

    Returns a filtered DataFrame of ports currently exceeding alert thresholds,
    ordered by congestion_level descending.

    Alert levels:
      critical  : congestion_level > 0.80
      high      : congestion_level > 0.65
      elevated  : congestion_level > 0.50
    """
    if cfg is None:
        cfg = _load_config()

    thresholds = cfg["anomaly"]["thresholds"]
    crit = thresholds["congestion_alert"]      # 0.80

    latest = (
        congestion
        .sort_values("date", ascending=False)
        .groupby("port_id")
        .first()
        .reset_index()
    )

    def _level(val):
        if val >= crit:
            return "critical"
        elif val >= 0.65:
            return "high"
        elif val >= 0.50:
            return "elevated"
        return "normal"

    latest["alert_level"] = latest["congestion_level"].apply(_level)
    alerts = latest[latest["alert_level"] != "normal"].copy()
    alerts.sort_values("congestion_level", ascending=False, inplace=True)
    return alerts


# =============================================================================
# Convenience: Full Pipeline
# =============================================================================

def run_full_detection(
    shipments:  pd.DataFrame,
    congestion: pd.DataFrame,
    cfg:        Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete anomaly detection pipeline.

    Returns
    -------
    scored_shipments : shipments with anomaly columns appended
    port_alerts      : ports with elevated / high / critical congestion
    """
    if cfg is None:
        cfg = _load_config()

    detector = AnomalyDetector(contamination=cfg["anomaly"]["contamination"])
    detector.fit(shipments)
    scored = detector.predict(shipments)
    alerts = congestion_alerts(congestion, cfg)
    return scored, alerts

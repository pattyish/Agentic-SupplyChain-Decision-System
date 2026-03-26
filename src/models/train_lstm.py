"""
train_lstm.py
-------------
PyTorch LSTM training pipeline for port congestion forecasting.

Framing:
  Given the last `sequence_length` (default: 30) days of multivariate
  time-series observations at a port, predict the congestion level for
  the next `forecast_horizon` (default: 7) days.

  Input features per time-step: [congestion_level, queue_time_hours, weather_severity]
  → shape (batch, seq_len=30, n_features=3)

  Output: predicted congestion_level for the next 7 days
  → shape (batch, forecast_horizon=7)

Architecture:
  ┌────────────────────────────────────────────────────────────┐
  │  Input  (B, 30, 3)                                         │
  │     ↓                                                      │
  │  LSTM (hidden=128, layers=2, dropout=0.2, batch_first=True)│
  │     ↓  take last hidden state h[:, -1, :]                  │
  │  Dropout(0.2)                                              │
  │     ↓                                                      │
  │  Linear(128 → 7)                                           │
  │     ↓                                                      │
  │  Output  (B, 7)  — congestion levels, clipped to [0, 1]    │
  └────────────────────────────────────────────────────────────┘

Training:
  • Loss    : Huber (smooth L1) — robust to outliers
  • Optim   : AdamW + cosine annealing LR scheduler
  • Regularisation : dropout + weight decay
  • Early stopping : restores best val-loss checkpoint

Persisted artefacts (in models/):
  • lstm_forecaster.pt     — model state dict
  • lstm_scaler.joblib     — per-feature StandardScaler
  • lstm_config.json       — hyperparameters used

Usage:
  python -m src.models.train_lstm
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.loader import SupplyChainLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)


# =============================================================================
# Config Helpers
# =============================================================================

def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _model_dir(cfg: dict) -> Path:
    root = Path(__file__).parents[2]
    d = root / cfg["data"]["models_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# LSTM Model
# =============================================================================

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM encoder → fully-connected forecaster head.

    Parameters
    ----------
    input_size      : number of input features per timestep
    hidden_size     : LSTM hidden units
    num_layers      : stacked LSTM layers
    forecast_horizon: output steps to predict
    dropout         : applied between LSTM layers and before FC head
    """

    def __init__(
        self,
        input_size:       int = 3,
        hidden_size:      int = 128,
        num_layers:       int = 2,
        forecast_horizon: int = 7,
        dropout:          float = 0.20,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        (batch, forecast_horizon)  — predicted congestion levels
        """
        out, _ = self.lstm(x)                 # (B, T, H)
        last   = self.dropout(out[:, -1, :])  # (B, H) — last hidden state
        return torch.sigmoid(self.fc(last))   # (B, forecast_horizon) in (0,1)


# =============================================================================
# Sequence Dataset Builder
# =============================================================================

def build_sequences(
    cng:              pd.DataFrame,
    wth:              pd.DataFrame,
    seq_len:          int = 30,
    horizon:          int = 7,
    input_features:   list[str] | None = None,
    scaler:           StandardScaler | None = None,
    fit_scaler:       bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Build sliding-window sequences from the multi-port time-series.

    Each sample covers `seq_len` days of (congestion, queue_time, weather)
    at one port, and the target is the next `horizon` days of congestion.

    Parameters
    ----------
    cng           : port_congestion DataFrame
    wth           : weather DataFrame (aggregated over locations)
    seq_len       : number of history days
    horizon       : number of forecast days
    input_features: feature columns to use (default: config values)
    scaler        : pre-fitted scaler; if None and fit_scaler=True, new one fitted
    fit_scaler    : whether to fit a new scaler on this data

    Returns
    -------
    X : (n_samples, seq_len, n_features)
    y : (n_samples, horizon)
    scaler : fitted StandardScaler
    """
    if input_features is None:
        input_features = ["congestion_level", "queue_time_hours", "weather_severity"]

    # Average weather severity per day across all locations (global proxy)
    wth_daily = (
        wth.groupby("date")["weather_severity"]
        .mean()
        .reset_index()
        .rename(columns={"weather_severity": "weather_severity_avg"})
    )
    wth_daily["date"] = pd.to_datetime(wth_daily["date"])

    cng_sorted = cng.copy()
    cng_sorted["date"] = pd.to_datetime(cng_sorted["date"])
    cng_sorted = cng_sorted.merge(wth_daily, on="date", how="left")
    cng_sorted["weather_severity"] = cng_sorted["weather_severity_avg"].fillna(1.0)
    cng_sorted.sort_values(["port_id", "date"], inplace=True)

    X_list, y_list = [], []
    feat_cols = [c for c in input_features if c in cng_sorted.columns]

    for _, port_df in cng_sorted.groupby("port_id"):
        port_df = port_df.reset_index(drop=True)
        values  = port_df[feat_cols].values.astype(float)  # (T, F)
        target  = port_df["congestion_level"].values        # (T,)
        T       = len(port_df)

        for start in range(T - seq_len - horizon + 1):
            end   = start + seq_len
            X_list.append(values[start:end])
            y_list.append(target[end: end + horizon])

    X = np.stack(X_list)   # (N, seq_len, F)
    y = np.stack(y_list)   # (N, horizon)

    # Fit / apply scaler on X (reshape to 2D, scale, reshape back)
    N, S, F = X.shape
    X2d = X.reshape(-1, F)

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X2d = scaler.fit_transform(X2d)
    else:
        X2d = scaler.transform(X2d)

    X = X2d.reshape(N, S, F)
    logger.info("Sequences built: X=%s  y=%s", X.shape, y.shape)
    return X, y, scaler


# =============================================================================
# Training Loop
# =============================================================================

def train_lstm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
    cfg:  dict,
) -> tuple[LSTMForecaster, list[float], list[float]]:
    """
    Full training loop with early stopping and cosine-annealing LR.

    Returns trained model and (train_losses, val_losses) histories.
    """
    lcfg     = cfg["lstm"]
    n_feat   = X_tr.shape[2]
    model    = LSTMForecaster(
        input_size       = n_feat,
        hidden_size      = lcfg["hidden_size"],
        num_layers       = lcfg["num_layers"],
        forecast_horizon = lcfg["forecast_horizon"],
        dropout          = lcfg["dropout"],
    ).to(DEVICE)

    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lcfg["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=lcfg["epochs"], eta_min=1e-6,
    )
    criterion = nn.HuberLoss()

    # ── DataLoaders ──────────────────────────────────────────────────────────
    def _make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=lcfg["batch_size"], shuffle=shuffle, pin_memory=True)

    tr_loader = _make_loader(X_tr, y_tr, shuffle=True)
    vl_loader = _make_loader(X_vl, y_vl, shuffle=False)

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(1, lcfg["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred  = model(xb)
            loss  = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), lcfg["gradient_clip"])
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_loss += criterion(model(xb), yb).item() * len(xb)
        vl_loss /= len(vl_loader.dataset)

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optim.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d  │  train=%.5f  │  val=%.5f  │  lr=%.6f",
                epoch, lcfg["epochs"], tr_loss, vl_loss, lr_now,
            )

        # Early stopping
        if vl_loss < best_val_loss - 1e-6:
            best_val_loss  = vl_loss
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= lcfg["patience"]:
                logger.info("Early stopping at epoch %d (best val=%.5f)", epoch, best_val_loss)
                break

    model.load_state_dict(best_state)
    logger.info("Training complete. Best val loss: %.5f", best_val_loss)
    return model, train_losses, val_losses


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    cfg       = _load_config()
    lcfg      = cfg["lstm"]
    model_dir = _model_dir(cfg)

    loader = SupplyChainLoader()
    cng    = loader.port_congestion()
    wth    = loader.weather()

    # Build all sequences
    X_all, y_all, scaler = build_sequences(
        cng, wth,
        seq_len  = lcfg["sequence_length"],
        horizon  = lcfg["forecast_horizon"],
        input_features = lcfg["input_features"],
    )

    # Chronological split (70 / 15 / 15)
    n     = len(X_all)
    t1    = int(n * 0.70)
    t2    = int(n * 0.85)
    X_tr, y_tr = X_all[:t1], y_all[:t1]
    X_vl, y_vl = X_all[t1:t2], y_all[t1:t2]
    X_te, y_te = X_all[t2:], y_all[t2:]
    logger.info("Sequence split  train:%d  val:%d  test:%d", len(X_tr), len(X_vl), len(X_te))

    model, tr_hist, vl_hist = train_lstm(X_tr, y_tr, X_vl, y_vl, cfg)

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    mae    = np.mean(np.abs(preds - y_te))
    rmse   = np.sqrt(np.mean((preds - y_te) ** 2))
    logger.info("TEST  │  MAE=%.4f  │  RMSE=%.4f  (congestion units)", mae, rmse)

    # ── Persist ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), model_dir / "lstm_forecaster.pt")
    joblib.dump(scaler, model_dir / "lstm_scaler.joblib")

    lstm_meta = {
        "input_size":       X_tr.shape[2],
        "hidden_size":      lcfg["hidden_size"],
        "num_layers":       lcfg["num_layers"],
        "forecast_horizon": lcfg["forecast_horizon"],
        "dropout":          lcfg["dropout"],
        "sequence_length":  lcfg["sequence_length"],
        "input_features":   lcfg["input_features"],
        "test_mae":         float(mae),
        "test_rmse":        float(rmse),
    }
    with open(model_dir / "lstm_config.json", "w") as f:
        json.dump(lstm_meta, f, indent=2)

    logger.info("LSTM artefacts saved to %s", model_dir)


if __name__ == "__main__":
    main()

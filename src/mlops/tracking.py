"""
MLops helpers: optional MLflow integration and dataset lineage metadata.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _mlflow_client(cfg: dict):
    ml_cfg = cfg.get("mlops", {}).get("mlflow", {})
    if not ml_cfg.get("enabled", False):
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(ml_cfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(ml_cfg.get("experiment_name", "supply-chain-monitoring"))
        return mlflow
    except Exception:
        return None


@contextlib.contextmanager
def mlflow_run(cfg: dict, run_name: str):
    mlflow = _mlflow_client(cfg)
    if mlflow is None:
        yield _NullRun()
        return
    with mlflow.start_run(run_name=run_name):
        yield mlflow


def dataframe_hash(df: pd.DataFrame) -> str:
    content_hash = hashlib.sha256()
    content_hash.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    content_hash.update("|".join(df.columns).encode("utf-8"))
    return content_hash.hexdigest()


def save_lineage_manifest(
    model_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "train": {"rows": len(train_df), "hash": dataframe_hash(train_df)},
            "val": {"rows": len(val_df), "hash": dataframe_hash(val_df)},
            "test": {"rows": len(test_df), "hash": dataframe_hash(test_df)},
        },
        "feature_count": len(feat_cols),
        "feature_cols": feat_cols,
        "extra": extra or {},
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "lineage_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out

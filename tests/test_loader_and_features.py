from pathlib import Path


def test_loader_imports():
    from src.data.loader import SupplyChainLoader  # noqa: F401


def test_feature_pipeline_imports():
    from src.features.build_features import build_feature_matrix  # noqa: F401


def test_model_modules_import():
    from src.models.train_xgboost import train_classifier  # noqa: F401
    from src.models.train_lstm import LSTMForecaster  # noqa: F401

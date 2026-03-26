import pandas as pd
from pathlib import Path


def test_generated_csv_files_exist():
    root = Path(__file__).parents[1]
    data = root / "data"
    required = [
        "shipments.csv",
        "suppliers.csv",
        "weather.csv",
        "port_congestion.csv",
        "disruptions.csv",
        "features.csv",
    ]
    missing = [f for f in required if not (data / f).exists()]
    assert not missing, f"Missing generated files: {missing}"


def test_shipments_schema_minimum():
    root = Path(__file__).parents[1]
    shp_path = root / "data" / "shipments.csv"
    if not shp_path.exists():
        return

    shp = pd.read_csv(shp_path)
    expected_cols = {
        "shipment_id", "supplier_id", "port_id", "ship_date",
        "expected_delivery_date", "actual_delivery_date",
        "distance_km", "transport_mode", "traffic_level",
        "weather_severity", "port_congestion", "supplier_risk",
        "delayed", "delay_hours",
    }
    assert expected_cols.issubset(set(shp.columns))
    assert shp["delayed"].isin([0, 1]).all()
    assert (shp["traffic_level"].between(1, 5)).all()
    assert (shp["weather_severity"].between(0, 3)).all()

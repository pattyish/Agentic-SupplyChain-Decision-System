# AGENT.md

This file defines agent-specific guidance for contributors and coding agents working in this repository.

## Project Mission

Build a reproducible AI system to predict, monitor, and explain supply chain disruptions using synthetic and eventually real operational data.

## Agent Responsibilities

1. Keep pipelines deterministic when possible (`seed=42` by default).
2. Avoid leakage in modeling and evaluation (time-based splits only).
3. Maintain compatibility between generated dataset schemas and loader/model code.
4. Preserve API contract stability.
5. Add tests for any schema, feature, or API changes.

## Coding Standards

- Python 3.10+
- Use type hints for public functions
- Keep modules focused by domain (`data`, `features`, `models`, `anomaly`, `api`)
- Prefer explicit over implicit transformations
- Fail fast on schema inconsistencies

## Data Contracts

### Required CSV files in `data/`
- `shipments.csv`
- `suppliers.csv`
- `weather.csv`
- `port_congestion.csv`
- `disruptions.csv`
- `features.csv`

### Critical relationship keys
- `shipments.supplier_id` -> `suppliers.supplier_id`
- `shipments.port_id` -> `port_congestion.port_id`
- `disruptions.shipment_id` -> `shipments.shipment_id`

## Modeling Rules

1. Classification target: `delayed`
2. Regression target: `delay_hours`
3. Regressor trains only on delayed subset
4. Use temporal split for train/val/test
5. Log all primary metrics after training

## API Rules

- Validate all request payloads via Pydantic constraints
- Return structured and stable JSON responses
- Avoid breaking endpoint names without updating README and tests

## Dashboard Rules

- Keep charts focused on operational decisions
- Prioritize interpretability over decorative complexity
- Surface anomalies and congestion alerts prominently

## Testing Expectations

Any substantial change must include or update tests for:
- Data schema compatibility
- Feature generation behavior
- API import/health and request handling
- Basic pipeline smoke checks

## Command Reference

```bash
python data/generate_data.py
python -m src.models.train_xgboost
python -m src.models.train_lstm
python -m src.models.evaluate
uvicorn src.api.app:app --reload
streamlit run dashboard/streamlit_app.py
pytest -q
```

## Future Extension Guidelines

- Keep new external data sources behind loader abstractions
- Add config flags rather than hard-coded constants
- Maintain backward compatibility with existing CSV schemas where possible

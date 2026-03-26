"""
=============================================================================
generate_data.py
AI-Driven Predictive Monitoring System for Supply Chain Disruptions
=============================================================================

Research-Level Synthetic Data Generator
Author  : Research Data Science Team
Version : 1.0.0
Date    : March 2026

Statistical Framework (Data-Generating Process)
------------------------------------------------
1. SUPPLIER LAYER
   - Reliability scores ~ Beta(α=8, β=2) → right-skewed toward high reliability
   - Failure rate ~ Beta(1,5) × (1.5 - reliability) → anti-correlated noise
   - Lead time ~ Gamma(shape=5, scale=3) + 5 days → right-skewed positive times

2. ENVIRONMENTAL LAYER — Weather
   - Daily severity: raw(t) = base + A·cos(2πt/365 + φ) + AR₁(t)
   - AR₁(t) = 0.65·AR₁(t−1) + ε(t),  ε ~ N(0, 0.35)
   - Northern hemisphere: phase φ = π  → winter peak (Jan/Feb)
   - severity = round(clip(raw, 0, 3)) ∈ {0, 1, 2, 3}

3. PORT CONGESTION LAYER
   - Ornstein-Uhlenbeck:  X(t+1) = X(t) + θ(μ−X(t)) + σε + weekly_effect
   - θ = 0.12 (mean-reversion),  σ = 0.06 (volatility)
   - μ ~ Uniform(0.20, 0.65) per port  (long-run congestion mean)
   - Weekly effect: +0.04 on Monday, +0.03 on Friday

4. SHIPMENT DELAY MODEL
   - Binary delay:  logit P(delayed) = β·x  where:
     β = [intercept=-3.0, weather=+2.5, congestion=+2.0,
          supplier_risk=+3.0, traffic=+1.5, dist_norm=+0.4,
          air=−1.5, ship=+0.3] + N(0, 0.3) individual noise
   - Expected overall delay rate: ~28–33 %
   - delay_hours | delayed=1  ~ LogNormal(μ_d(x), σ=0.7)
     μ_d = 3.0 + 0.8(w/3) + 0.6c + 0.5(t/5) + 0.3s
   - delay_hours | delayed=0  ~ LogNormal(0.8, 0.4)  [small on-time variance]

5. DISRUPTION EVENTS
   - Hawkes-process approximation: # events ∝ Poisson(0.8 + delay_hours/72)
   - Event type sampled from weighted categorical (weights ∝ causal factors)
   - Duration ~ LogNormal(μ_sev, 0.6):
       low → μ=1.5,  medium → μ=2.5,  high → μ=3.5

References:
  Simchi-Levi et al. (2020). "Identifying Risks in the Supply Chain"
  Dolgui & Ivanov (2021). "Ripple Effect and Supply Chain Disruption Management"
=============================================================================
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
rng  = np.random.default_rng(SEED)

# ─── Global Constants ─────────────────────────────────────────────────────────
N_SHIPMENTS    = 10_000
N_SUPPLIERS    = 50
N_PORTS        = 20
START_DATE     = pd.Timestamp("2023-01-01")
END_DATE       = pd.Timestamp("2025-12-31")
DATE_RANGE     = pd.date_range(START_DATE, END_DATE, freq="D")
N_DAYS         = len(DATE_RANGE)

TRANSPORT_MODES = ["truck", "ship", "air"]
TRANSPORT_PROBS = [0.55, 0.30, 0.15]

LOCATIONS = [
    "Shanghai",        "Rotterdam",   "Singapore",        "Los_Angeles",
    "Hamburg",         "Antwerp",     "Dubai",            "Busan",
    "Hong_Kong",       "New_York",    "Felixstowe",       "Guangzhou",
    "Mumbai",          "Qingdao",     "Ningbo",           "Tianjin",
    "Kaohsiung",       "Tokyo",       "Tanjung_Pelepas",  "Port_Klang",
]

# Northern-hemisphere ports → winter severity peak
NORTHERN = {
    "Shanghai", "Rotterdam", "Hamburg", "Antwerp", "Los_Angeles",
    "New_York", "Felixstowe", "Guangzhou", "Qingdao", "Ningbo",
    "Tianjin", "Kaohsiung", "Tokyo", "Hong_Kong", "Busan",
}

EVENT_TYPES = [
    "weather_disruption", "port_congestion",
    "supplier_failure",   "transport_delay",
]

CAUSES: dict[str, list[str]] = {
    "weather_disruption": ["hurricane", "severe_storm", "flooding", "blizzard", "fog"],
    "port_congestion":    ["labor_strike", "equipment_failure", "high_demand", "inspection_backlog"],
    "supplier_failure":   ["bankruptcy", "production_halt", "quality_issue", "capacity_shortage"],
    "transport_delay":    ["road_closure", "vehicle_breakdown", "border_delay", "fuel_shortage"],
}

OUT = os.path.dirname(os.path.abspath(__file__))


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# =============================================================================
# 1.  SUPPLIERS
# =============================================================================
def make_suppliers() -> pd.DataFrame:
    """
    Generate supplier master catalogue.

    Reliability  ~ Beta(8, 2)            — skewed toward reliable suppliers
    failure_rate ~ Beta(1, 5) × (1.5−r)  — anti-correlated with reliability
    lead_time    ~ Gamma(5, 3) + 5 days  — right-skewed transit times
    """
    reliability  = rng.beta(8, 2, N_SUPPLIERS)
    noise        = rng.beta(1, 5, N_SUPPLIERS)
    failure_rate = np.clip(noise * (1.5 - reliability), 0.01, 0.60)
    lead_time    = np.clip(rng.gamma(5, 3, N_SUPPLIERS) + 5.0, 5.0, 60.0)

    return pd.DataFrame({
        "supplier_id":        np.arange(1, N_SUPPLIERS + 1),
        "reliability_score":  reliability.round(4),
        "avg_lead_time_days": lead_time.round(2),
        "failure_rate":       failure_rate.round(4),
    })


# =============================================================================
# 2.  WEATHER
# =============================================================================
def make_weather() -> pd.DataFrame:
    """
    Generate daily weather observations for all port locations.

    Model:  raw(t) = base_severity + A·cos(2πt/365 + phase) + AR₁(t)
      • Phase = π  for Northern-hemisphere ports (winter peak)
      • Phase = 0  for Southern-hemisphere ports  (summer peak)
    Derived: precipitation ~ Exp(5 + 8*(sev/3))
             wind_speed    ~ Exp(10 + 20*(sev/3))
             storm_flag    = 1 when sev==3, or sev==2 with 15 % probability
    """
    t    = np.arange(N_DAYS)
    rows = []

    for loc in LOCATIONS:
        phase    = np.pi if loc in NORTHERN else 0.0
        base     = rng.uniform(0.30, 0.80)
        amp      = rng.uniform(0.50, 1.00)
        seasonal = amp * np.cos(2 * np.pi * t / 365 + phase)

        # AR(1) noise
        ar  = np.zeros(N_DAYS)
        eps = rng.normal(0, 0.35, N_DAYS)
        ar[0] = rng.normal(0, 0.30)
        for i in range(1, N_DAYS):
            ar[i] = 0.65 * ar[i - 1] + eps[i]

        raw  = (base + seasonal + ar).clip(0.0, 3.0)
        sev  = np.round(raw).astype(int).clip(0, 3)
        prec = np.clip(rng.exponential(5.0  + 8.0  * (sev / 3.0)), 0.0, 80.0)
        wind = np.clip(rng.exponential(10.0 + 20.0 * (sev / 3.0)), 0.0, 120.0)
        storm = np.where(
            sev == 3, 1,
            np.where((sev == 2) & (rng.random(N_DAYS) < 0.15), 1, 0)
        )

        rows.append(pd.DataFrame({
            "date":             DATE_RANGE.date,
            "location":         loc,
            "weather_severity": sev,
            "precipitation":    prec.round(2),
            "wind_speed":       wind.round(2),
            "storm_flag":       storm.astype(int),
        }))

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 3.  PORT CONGESTION
# =============================================================================
def make_congestion() -> pd.DataFrame:
    """
    Generate daily port congestion via the Ornstein-Uhlenbeck process.

    Discretised OU:
      X(t+1) = X(t) + θ(μ − X(t)) + σ·ε(t) + weekly_effect(t)

    Parameters:  θ=0.12,  σ=0.06,  μ ~ Unif(0.20, 0.65) per port
    Weekly:       Monday +0.04,  Friday +0.03

    Queue time   ~ Exp(exp(3.5·X))  — exponentially grows with congestion
    """
    theta  = 0.12
    sigma  = 0.06
    mu_vec = rng.uniform(0.20, 0.65, N_PORTS)

    dow    = np.array([d.dayofweek for d in DATE_RANGE])
    weekly = np.where(dow == 0, 0.04, np.where(dow == 4, 0.03, 0.0))

    rows = []
    for p in range(N_PORTS):
        mu  = mu_vec[p]
        X   = np.empty(N_DAYS)
        X[0] = rng.uniform(0.10, 0.70)
        eps  = rng.normal(0, sigma, N_DAYS)

        for i in range(1, N_DAYS):
            X[i] = np.clip(
                X[i - 1] + theta * (mu - X[i - 1]) + eps[i] + weekly[i],
                0.0, 1.0,
            )

        qt = np.clip(rng.exponential(np.exp(3.5 * X)), 0.0, 200.0)

        rows.append(pd.DataFrame({
            "port_id":          p + 1,
            "location":         LOCATIONS[p],
            "date":             DATE_RANGE.date,
            "congestion_level": X.round(4),
            "queue_time_hours": qt.round(2),
        }))

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 4.  SHIPMENTS
# =============================================================================
def make_shipments(sup: pd.DataFrame, wth: pd.DataFrame, cng: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 10,000 shipments with environment-driven delay model.

    Delay probability (logistic DGP):
      logit P(delay) = −3.0
                     + 2.5 × (weather_severity / 3)
                     + 2.0 × congestion_level
                     + 3.0 × supplier_risk
                     + 1.5 × (traffic_level / 5)
                     + 0.4 × (distance_km / 8000)
                     − 1.5 × [mode == air]
                     + 0.3 × [mode == ship]
                     + N(0, 0.3)   individual variance

    Expected delay rate: ~28–33 % across all shipments.

    Transit-time model:
      expected_days = distance / speed_per_day + avg_lead_time + N(0,1.5)
      speed: truck=500 km/day, ship=350 km/day, air=2000 km/day
    """
    n = N_SHIPMENTS

    # ── Sampling base attributes ──────────────────────────────────────────────
    idx      = rng.integers(0, N_DAYS - 30, n)
    s_dates  = DATE_RANGE[idx]                              # ship dates
    sup_ids  = rng.integers(1, N_SUPPLIERS + 1, n)
    port_ids = rng.integers(1, N_PORTS + 1, n)
    modes    = rng.choice(TRANSPORT_MODES, n, p=TRANSPORT_PROBS)
    traffic  = rng.integers(1, 6, n)

    # Distance varies by transport mode
    dist_base  = {"truck": 300.0,  "ship": 1500.0, "air": 1000.0}
    dist_scale = {"truck": 800.0,  "ship": 6000.0, "air": 5000.0}
    speed_kd   = {"truck": 500.0,  "ship": 350.0,  "air": 2000.0}
    distances  = np.array([
        np.clip(rng.exponential(dist_scale[m]) + dist_base[m], 50.0, 20_000.0)
        for m in modes
    ])

    # ── Lookup tables ─────────────────────────────────────────────────────────
    sup_risk_map  = dict(zip(sup["supplier_id"], 1.0 - sup["reliability_score"]))
    lead_time_map = dict(zip(sup["supplier_id"], sup["avg_lead_time_days"]))
    port_loc      = {i + 1: LOCATIONS[i] for i in range(N_PORTS)}

    wth_dict = (
        wth.groupby(["date", "location"])["weather_severity"]
        .mean()
        .to_dict()
    )
    cng_dict = (
        cng.set_index(["date", "port_id"])["congestion_level"]
        .to_dict()
    )

    w_sev = np.array([
        wth_dict.get(
            (s_dates[i].date(), port_loc.get(int(port_ids[i]), LOCATIONS[0])),
            float(rng.integers(0, 3)),
        )
        for i in range(n)
    ], dtype=float)

    c_lvl = np.array([
        cng_dict.get(
            (s_dates[i].date(), int(port_ids[i])),
            float(rng.uniform(0.2, 0.6)),
        )
        for i in range(n)
    ], dtype=float)

    s_risk = np.array([sup_risk_map.get(int(sid), 0.20) for sid in sup_ids])
    l_time = np.array([lead_time_map.get(int(sid), 15.0) for sid in sup_ids])

    # ── Delay model ──────────────────────────────────────────────────────────
    air_m  = (modes == "air").astype(float)
    ship_m = (modes == "ship").astype(float)

    logit = (
        -3.00
        + 2.50 * (w_sev / 3.0)
        + 2.00 * c_lvl
        + 3.00 * s_risk
        + 1.50 * (traffic / 5.0)
        + 0.40 * (distances / 8000.0)
        - 1.50 * air_m
        + 0.30 * ship_m
        + rng.normal(0.0, 0.30, n)
    )
    p_delay = _sigmoid(logit)
    delayed = (rng.uniform(0.0, 1.0, n) < p_delay).astype(int)

    # ── Expected delivery ─────────────────────────────────────────────────────
    transit_days = distances / np.array([speed_kd[m] for m in modes])
    exp_days = np.clip(
        np.round(transit_days + l_time + rng.normal(0.0, 1.5, n)).astype(int),
        1, 180,
    )
    exp_delivery = s_dates + pd.to_timedelta(exp_days, unit="D")

    # ── Delay hours ───────────────────────────────────────────────────────────
    mu_d = (
        3.0
        + 0.8 * (w_sev / 3.0)
        + 0.6 * c_lvl
        + 0.5 * (traffic / 5.0)
        + 0.3 * s_risk
    )
    delay_hrs = np.where(
        delayed == 1,
        np.clip(rng.lognormal(mu_d, 0.70), 1.0,  500.0),
        np.clip(rng.lognormal(0.80, 0.40, n), 0.0, 24.0),
    )
    act_delivery = exp_delivery + pd.to_timedelta(
        np.clip(np.round(delay_hrs / 24).astype(int), 0, 60), unit="D"
    )

    return pd.DataFrame({
        "shipment_id":            np.arange(1, n + 1),
        "supplier_id":            sup_ids,
        "port_id":                port_ids,
        "ship_date":              s_dates.date,
        "expected_delivery_date": exp_delivery.date,
        "actual_delivery_date":   act_delivery.date,
        "distance_km":            distances.round(2),
        "transport_mode":         modes,
        "traffic_level":          traffic,
        "weather_severity":       w_sev.astype(int),
        "port_congestion":        c_lvl.round(4),
        "supplier_risk":          s_risk.round(4),
        "delayed":                delayed,
        "delay_hours":            delay_hrs.round(2),
    })


# =============================================================================
# 5.  DISRUPTIONS
# =============================================================================
def make_disruptions(shp: pd.DataFrame) -> pd.DataFrame:
    """
    Generate disruption event log for all delayed shipments.

    Hawkes-process approximation:
      n_events ~ Poisson(λ),  λ = 0.8 + min(delay_hours/72, 4.0)
      → short delays (< 24 h): ~1 event,  severe (> 72 h): ~2–4 events

    Event types are weighted proportionally to the causal risk factor
    observed for that shipment.

    Severity:
      delay_hours < 24  → 'low'
      24 ≤ delay < 72   → 'medium'
      delay ≥ 72        → 'high'

    Duration ~ LogNormal(μ_sev, σ=0.6):
      low=1.5,  medium=2.5,  high=3.5
    """
    dfm     = shp[shp["delayed"] == 1].copy()
    records = []
    eid     = 1

    for _, row in dfm.iterrows():
        sev_factor = min(float(row["delay_hours"]) / 72.0, 4.0)
        n_events   = max(1, int(rng.poisson(0.8 + sev_factor)))

        w = np.array([
            float(row["weather_severity"]) / 3.0 + 0.05,
            float(row["port_congestion"])         + 0.05,
            float(row["supplier_risk"])           + 0.05,
            float(row["traffic_level"]) / 5.0    + 0.05,
        ])
        w /= w.sum()

        t0      = pd.Timestamp(str(row["ship_date"]))
        offsets = np.sort(rng.uniform(0.0, max(float(row["delay_hours"]), 1.0), n_events))
        etypes  = rng.choice(EVENT_TYPES, n_events, p=w)

        for off, etype in zip(offsets, etypes):
            dh = float(row["delay_hours"])
            sev_str = "high" if dh >= 72 else "medium" if dh >= 24 else "low"
            mu_dur  = {"low": 1.5, "medium": 2.5, "high": 3.5}[sev_str]
            duration = float(np.clip(rng.lognormal(mu_dur, 0.60), 0.5, 200.0))

            records.append({
                "event_id":      eid,
                "shipment_id":   int(row["shipment_id"]),
                "event_type":    etype,
                "cause":         str(rng.choice(CAUSES[etype])),
                "severity":      sev_str,
                "timestamp":     (t0 + pd.Timedelta(hours=float(off))).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_hours": round(duration, 2),
            })
            eid += 1

    return pd.DataFrame(records)


# =============================================================================
# 6.  FEATURE MATRIX
# =============================================================================
def make_features(shp: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the ML-ready feature matrix with a composite risk score.

    risk_score = 0.30*(weather/3) + 0.25*congestion + 0.30*supplier_risk
               + 0.15*(traffic/5)

    Returns the minimal feature set for model training.
    """
    f = shp[[
        "shipment_id", "weather_severity", "traffic_level",
        "supplier_risk", "port_congestion", "delayed", "delay_hours",
    ]].copy()

    f["congestion_level"] = f["port_congestion"]
    f["risk_score"] = (
        0.30 * (f["weather_severity"] / 3.0)
        + 0.25 * f["port_congestion"]
        + 0.30 * f["supplier_risk"]
        + 0.15 * (f["traffic_level"] / 5.0)
    ).clip(0.0, 1.0).round(4)

    return f[[
        "shipment_id", "weather_severity", "traffic_level",
        "supplier_risk", "congestion_level", "risk_score",
        "delayed", "delay_hours",
    ]].round(4)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print("  AI-Driven Predictive Monitoring System for Supply Chain Disruptions")
    print(f"  Synthetic Data Generator  |  v1.0  |  seed={SEED}")
    print(f"{bar}\n")

    print("[1/6] Generating supplier catalogue …", end="  ", flush=True)
    sup = make_suppliers()
    sup.to_csv(os.path.join(OUT, "suppliers.csv"), index=False)
    print(f"{len(sup):>5} rows  →  suppliers.csv")

    print("[2/6] Generating weather data  (seasonal AR-1) …", end="  ", flush=True)
    wth = make_weather()
    wth.to_csv(os.path.join(OUT, "weather.csv"), index=False)
    print(f"{len(wth):>6,} rows  →  weather.csv")

    print("[3/6] Generating port congestion  (OU process) …", end="  ", flush=True)
    cng = make_congestion()
    cng.to_csv(os.path.join(OUT, "port_congestion.csv"), index=False)
    print(f"{len(cng):>6,} rows  →  port_congestion.csv")

    print("[4/6] Generating shipment records  (logistic DGP) …", end="  ", flush=True)
    shp = make_shipments(sup, wth, cng)
    shp.to_csv(os.path.join(OUT, "shipments.csv"), index=False)
    dr  = shp["delayed"].mean() * 100.0
    print(f"{len(shp):>6,} rows  →  shipments.csv   [delay rate: {dr:.1f} %]")

    print("[5/6] Generating disruption events  (Hawkes approx.) …", end="  ", flush=True)
    dis = make_disruptions(shp)
    dis.to_csv(os.path.join(OUT, "disruptions.csv"), index=False)
    print(f"{len(dis):>6,} rows  →  disruptions.csv")

    print("[6/6] Generating ML feature matrix …", end="  ", flush=True)
    fea = make_features(shp)
    fea.to_csv(os.path.join(OUT, "features.csv"), index=False)
    print(f"{len(fea):>6,} rows  →  features.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print(f"  {'Dataset':<24} {'Rows':>8}   Columns")
    print(f"  {'─'*24}  {'─'*8}   {'─'*20}")
    for df, name in [
        (sup, "suppliers.csv"),
        (wth, "weather.csv"),
        (cng, "port_congestion.csv"),
        (shp, "shipments.csv"),
        (dis, "disruptions.csv"),
        (fea, "features.csv"),
    ]:
        print(f"  {name:<24} {len(df):>8,}   {list(df.columns)}")
    print(f"\n  Overall delay rate : {dr:.1f} %")
    print(f"  Random seed        : {SEED}")
    print(f"  Date range         : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Output directory   : {OUT}")
    print(f"{'─' * 68}\n")


if __name__ == "__main__":
    main()

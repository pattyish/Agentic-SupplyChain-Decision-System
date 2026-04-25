"""
Decision intelligence: recommendation and simple optimization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionOutcome:
    action: str
    expected_delay_reduction_hours: float
    expected_cost_multiplier: float
    estimated_cost_avoided: float


def choose_best_action(
    delay_probability: float,
    estimated_delay_hours: float,
    risk_score: float,
    transport_mode: str,
    cfg: dict,
) -> DecisionOutcome:
    dcfg = cfg.get("decision", {})
    multipliers = dcfg.get("mode_cost_multiplier", {"ship": 1.0, "truck": 1.25, "air": 2.2})

    reroute_factor = float(dcfg.get("reroute_reduction_factor", 0.35))
    expedite_factor = float(dcfg.get("expedite_reduction_factor", 0.55))
    fallback_factor = float(dcfg.get("fallback_supplier_reduction_factor", 0.30))

    baseline_penalty = delay_probability * estimated_delay_hours * 100.0

    actions = {
        "monitor": (0.05, multipliers.get(transport_mode, 1.0)),
        "reroute": (reroute_factor, multipliers.get(transport_mode, 1.0) * 1.1),
        "fallback_supplier": (fallback_factor, multipliers.get(transport_mode, 1.0) * 1.08),
        "expedite": (expedite_factor, multipliers.get("air", 2.2)),
    }

    best = None
    best_total_cost = float("inf")
    for action, (reduction, cost_mult) in actions.items():
        reduced_delay = max(estimated_delay_hours * (1.0 - reduction), 0.0)
        total_cost = (delay_probability * reduced_delay * 100.0) + (50.0 * cost_mult)
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best = (action, reduction, cost_mult)

    action, reduction, cost_mult = best
    delay_reduced = estimated_delay_hours * reduction
    avoided = max(baseline_penalty - best_total_cost, 0.0)

    if risk_score < 0.35:
        action = "monitor"

    return DecisionOutcome(
        action=action,
        expected_delay_reduction_hours=round(delay_reduced, 2),
        expected_cost_multiplier=round(cost_mult, 3),
        estimated_cost_avoided=round(avoided, 2),
    )

"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


# ------------------------------------------------------------------- EXT-5
# Carbon-aware placement. Region choice moves $ and gCO2e at the same time, but
# not in the same direction — the cheapest grid is not always the cleanest.
REGION_LATENCY_MS = {          # RTT from the US-east user base, illustrative
    "us-east-1": 15,
    "us-east-wa": 60,
    "us-west-2": 70,
    "europe-north1": 110,
    "europe-central2": 125,
}


def region_table(kwh: float = 1.0) -> list:
    """Per-region $ and gCO2e for a given energy draw, plus the latency it costs."""
    rows = []
    for region, gco2_kwh in REGION_CARBON.items():
        price = REGION_PRICE_KWH.get(region, 0.12)
        rows.append({
            "region": region,
            "usd_per_kwh": price,
            "gco2_per_kwh": gco2_kwh,
            "energy_cost_usd": round(kwh * price, 2),
            "carbon_kg": round(kwh * gco2_kwh / 1000.0, 1),
            "latency_ms": REGION_LATENCY_MS.get(region, 0),
        })
    return sorted(rows, key=lambda r: r["region"])


def _minmax(values):
    lo, hi = min(values), max(values)
    span = hi - lo
    return lo, (span if span > 0 else 1.0)


def best_region(criterion: str = "carbon", weight_cost: float = 0.5) -> str:
    """Pick a region by 'cost', 'carbon' or 'balanced'.

    'balanced' min-max normalises both axes and takes a weighted sum, so the
    answer does not depend on the units — a company that has priced carbon
    internally should set `weight_cost` to whatever that price implies.
    """
    regions = list(REGION_CARBON)
    if criterion == "cost":
        return min(regions, key=lambda r: REGION_PRICE_KWH.get(r, 0.12))
    if criterion == "carbon":
        return min(regions, key=lambda r: REGION_CARBON[r])
    c_lo, c_span = _minmax([REGION_PRICE_KWH.get(r, 0.12) for r in regions])
    g_lo, g_span = _minmax([REGION_CARBON[r] for r in regions])

    def score(r):
        c = (REGION_PRICE_KWH.get(r, 0.12) - c_lo) / c_span
        g = (REGION_CARBON[r] - g_lo) / g_span
        return weight_cost * c + (1.0 - weight_cost) * g

    return min(regions, key=score)


def job_energy_kwh(watts: float, hours: float, num_gpus: int = 1, pue: float = 1.15) -> float:
    """Facility energy for a GPU job. PUE covers cooling and power delivery."""
    return (max(0.0, watts) * max(0.0, hours) * max(1, num_gpus) * max(1.0, pue)) / 1000.0

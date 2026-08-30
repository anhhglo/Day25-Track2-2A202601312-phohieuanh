"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Extension implemented here (Your Turn #1): a tier policy that prices reclaim risk
per GPU type, bills reserved capacity for the whole commitment, and chooses
between a 1-year and a 3-year term from the workload's survival odds.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30                    # billing month
SURVIVAL_PROB_3YR = 0.60     # confidence this workload still exists in 3 years


def legacy_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Policy v1, kept for comparison: duty = hours/24, spot for anything interruptible."""
    duty = max(0.0, hours_per_day) / 24.0
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= pricing.break_even_utilization(reserved_discount):
        return "reserved"
    return "on_demand"


def _cost_of(tier: str, c: dict, hpd: float, days: float, ngpu: int, gtype: str) -> float:
    """Price one tier under the honest cost model (reserved = pay for the month)."""
    used_hours = hpd * min(days, DAYS) * ngpu
    if tier == "spot":
        tax = pricing.spot_effective_multiplier(pricing.interrupt_rate_for(gtype))
        return used_hours * num(c["spot_hr"]) * tax
    if tier == "reserved":
        term = pricing.reserved_term_choice(num(c["reserved_1yr_hr"]), num(c["reserved_3yr_hr"]),
                                            SURVIVAL_PROB_3YR)
        rate = num(c["reserved_3yr_hr"]) if term["term"] == "3yr" else num(c["reserved_1yr_hr"])
        return 24.0 * DAYS * ngpu * rate
    return used_hours * num(c["on_demand_hr"])


def policy_grid(cat: dict) -> dict:
    """Where does policy v2 actually disagree with v1, and is it cheaper there?

    The fleet is only 8 jobs, so a like-for-like total hides the policy change.
    This sweeps GPU type x duty cycle x days-per-month x interruptible and prices
    both policies under the same cost model.
    """
    disagreements, legacy_total, new_total = [], 0.0, 0.0
    for gtype, c in cat.items():
        for duty in (0.25, 0.50, 0.75, 1.0):
            for days in (30, 15):
                for interruptible in (False, True):
                    hpd = duty * 24.0
                    t_old = legacy_tier(hpd, interruptible)
                    t_new = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, days=days)
                    c_old = _cost_of(t_old, c, hpd, days, 1, gtype)
                    c_new = _cost_of(t_new, c, hpd, days, 1, gtype)
                    legacy_total += c_old
                    new_total += c_new
                    if t_old != t_new:
                        disagreements.append({
                            "gpu_type": gtype, "duty": duty, "days": days,
                            "interruptible": interruptible, "v1": t_old, "v2": t_new,
                            "delta": round(c_old - c_new, 2),
                        })
    return {
        "cells": 7 * 4 * 2 * 2,
        "disagreements": disagreements,
        "legacy_total": round(legacy_total, 2),
        "new_total": round(new_total, 2),
        "delta": round(legacy_total - new_total, 2),
    }


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]

        plan = pricing.recommend_plan(
            hours_per_day=hpd, days=days, interruptible=interruptible,
            on_demand_hr=num(c["on_demand_hr"]), spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]), reserved_3yr_hr=num(c["reserved_3yr_hr"]),
            num_gpus=ngpu, gpu_type=gtype, month_days=DAYS, survival_prob_3yr=SURVIVAL_PROB_3YR,
        )
        on_demand_cost = plan["costs"]["on_demand"]
        opt_cost = plan["cost"]
        legacy_cost = _cost_of(legacy_tier(hpd, interruptible), c, hpd, days, ngpu, gtype)

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        legacy_monthly += legacy_cost
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "tier": plan["tier"], "term": plan["term"],
            "duty": plan["duty_cycle"], "interrupt_rate": plan["interrupt_rate"],
            "spot_tax": plan["spot_tax"], "used_hours": plan["used_hours"],
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
            "legacy": round(legacy_cost), "effective_hourly": plan["effective_hourly"],
            "why": plan["why"],
        })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    legacy_savings_pct = (on_demand_monthly - legacy_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0
    grid = policy_grid(cat)

    terms = {g: pricing.reserved_term_choice(num(c["reserved_1yr_hr"]), num(c["reserved_3yr_hr"]),
                                             SURVIVAL_PROB_3YR)
             for g, c in cat.items()}

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"reserved is billed 24h x {DAYS}d whether used or not — the comparison is against "
              f"MONTHLY duty, not hours/24\n")
        print(f"{'job':18}{'gpu':7}{'tier':10}{'term':6}{'duty':>7}{'reclaim':>9}{'on-demand':>12}{'optimized':>12}{'$/used-h':>10}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:10}{r['term']:6}{r['duty']:>7.0%}"
                  f"{r['interrupt_rate']:>9.1%}${r['on_demand']:>11,}${r['optimized']:>11,}{r['effective_hourly']:>10.3f}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- EXT-1: policy v1 vs v2 --")
        print(f"policy v1 (duty=h/24, spot for any interruptible): ${legacy_monthly:,.0f}/mo  ({legacy_savings_pct:.1f}% saved)")
        print(f"policy v2 (reclaim-rate + monthly duty + term)   : ${optimized_monthly:,.0f}/mo  ({savings_pct:.1f}% saved)")
        print(f"on THIS fleet the two policies agree on {sum(1 for r in recs if r['legacy'] == r['optimized'])}/{len(recs)} jobs — "
              f"the divergence needs a wider grid:")
        print(f"  grid sweep ({grid['cells']} cells: 7 GPU x 4 duty x 2 schedules x 2 interruptible)")
        print(f"  v1 disagrees with v2 in {len(grid['disagreements'])} cells, costing ${grid['delta']:,.0f}/mo extra")
        for d in grid["disagreements"][:8]:
            print(f"    {d['gpu_type']:7} duty={d['duty']:.0%} days={d['days']:>2} "
                  f"interruptible={str(d['interruptible']):5} v1={d['v1']:10} v2={d['v2']:10} saves ${d['delta']:,.0f}")
        if len(grid["disagreements"]) > 8:
            print(f"    ... {len(grid['disagreements']) - 8} more")

        print("\n-- EXT-1: 1-year vs 3-year commitment --")
        print(f"{'gpu':8}{'1yr $/h':>9}{'3yr $/h':>9}{'eff 3yr @' + f'{SURVIVAL_PROB_3YR:.0%}':>13}{'break-even survival':>21}{'pick':>7}")
        for g, t in sorted(terms.items()):
            print(f"{g:8}{t['rate_1yr']:>9}{t['rate_3yr']:>9}{t['effective_3yr_rate']:>13}"
                  f"{t['break_even_survival_prob']:>21.0%}{t['term']:>7}")

        print("\n-- why each job landed where it did --")
        for r in recs:
            print(f"  {r['job_id']:18} {r['tier']}{('/' + r['term']) if r['term'] else ''}: {r['why']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "legacy_monthly": round(legacy_monthly), "legacy_savings_pct": round(legacy_savings_pct, 1),
            "policy_grid": grid, "terms": terms}


if __name__ == "__main__":
    run()

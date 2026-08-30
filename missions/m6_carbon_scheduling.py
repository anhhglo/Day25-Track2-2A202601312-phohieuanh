"""M6 — Carbon-aware scheduling (Your Turn #5, deck §11).

Interruptible training jobs are the only ones free to move: they have no user
waiting on a round-trip, and they already survive being stopped. This mission
prices where they should run, on three different definitions of "optimal".

Run: python missions/m6_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability as sus

DAYS = 30
HOME_REGION = "us-east-1"
PUE = 1.15   # facility overhead: cooling + power delivery


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    movable, fixed_kwh, movable_kwh = [], 0.0, 0.0
    for j in jobs:
        c = cat[j["gpu_type"]]
        hours = num(j["hours_per_day"]) * min(num(j["days"]), DAYS)
        kwh = sus.job_energy_kwh(num(c["watts"]), hours, int(num(j["num_gpus"])), PUE)
        if bool(int(num(j["interruptible"]))):
            movable_kwh += kwh
            movable.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"], "kwh": round(kwh, 1)})
        else:
            fixed_kwh += kwh

    table = sus.region_table(movable_kwh)
    best_cost = sus.best_region("cost")
    best_carbon = sus.best_region("carbon")
    best_balanced = sus.best_region("balanced")

    home = next(r for r in table if r["region"] == HOME_REGION)
    for r in table:
        r["carbon_saved_kg"] = round(home["carbon_kg"] - r["carbon_kg"], 1)
        r["cost_saved_usd"] = round(home["energy_cost_usd"] - r["energy_cost_usd"], 2)
        r["latency_penalty_ms"] = r["latency_ms"] - home["latency_ms"]

    clean = next(r for r in table if r["region"] == best_carbon)
    balanced = next(r for r in table if r["region"] == best_balanced)

    per_job = []
    for m in movable:
        share = m["kwh"] / movable_kwh if movable_kwh else 0.0
        per_job.append({
            **m,
            "carbon_home_kg": round(m["kwh"] * sus.REGION_CARBON[HOME_REGION] / 1000.0, 1),
            "carbon_clean_kg": round(m["kwh"] * sus.REGION_CARBON[best_carbon] / 1000.0, 1),
            "carbon_saved_kg": round(share * clean["carbon_saved_kg"], 1),
        })

    if verbose:
        print("== M6 Carbon-aware Scheduling ==")
        print(f"movable (interruptible) load: {movable_kwh:,.0f} kWh/month across {len(movable)} jobs")
        print(f"pinned (user-facing) load   : {fixed_kwh:,.0f} kWh/month — latency-bound, stays in {HOME_REGION}\n")
        print(f"{'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'energy $':>10}{'carbon kg':>11}"
              f"{'kg saved':>10}{'$ saved':>9}{'latency':>9}")
        for r in sorted(table, key=lambda x: x["gco2_per_kwh"]):
            print(f"{r['region']:16}{r['usd_per_kwh']:>8}{r['gco2_per_kwh']:>10}{r['energy_cost_usd']:>10,.0f}"
                  f"{r['carbon_kg']:>11,.0f}{r['carbon_saved_kg']:>10,.0f}{r['cost_saved_usd']:>9,.0f}"
                  f"{r['latency_penalty_ms']:>+8}ms")

        print(f"\ncheapest power   : {best_cost}  (${next(r for r in table if r['region']==best_cost)['usd_per_kwh']}/kWh)")
        print(f"cleanest grid    : {best_carbon}  ({sus.REGION_CARBON[best_carbon]} gCO2/kWh)")
        print(f"balanced (50/50) : {best_balanced}  — {sus.REGION_CARBON[best_balanced]} gCO2/kWh at "
              f"${sus.REGION_PRICE_KWH[best_balanced]}/kWh, +{balanced['latency_penalty_ms']}ms")
        print(f"\nmove ALL movable load {HOME_REGION} -> {best_carbon}: "
              f"{clean['carbon_saved_kg']:,.0f} kgCO2e/month avoided "
              f"({clean['carbon_saved_kg'] / home['carbon_kg']:.0%} of that load's footprint), "
              f"energy bill {'+' if clean['cost_saved_usd'] < 0 else '-'}${abs(clean['cost_saved_usd']):,.0f}")
        print(f"\n{'job':18}{'gpu':7}{'kWh/mo':>9}{'kg @home':>10}{'kg @clean':>11}{'kg saved':>10}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['gpu_type']:7}{p['kwh']:>9,.0f}{p['carbon_home_kg']:>10,.0f}"
                  f"{p['carbon_clean_kg']:>11,.0f}{p['carbon_saved_kg']:>10,.0f}")
        print("\nNote: on a neocloud rental the power bill is already inside the $/GPU-hr, so the "
              "'$ saved' column is the physical energy component, not a second invoice. It is the "
              "carbon column that is the real, unbundled decision.")

    return {
        "movable_kwh": round(movable_kwh, 1), "fixed_kwh": round(fixed_kwh, 1),
        "regions": table, "per_job": per_job,
        "best_cost": best_cost, "best_carbon": best_carbon, "best_balanced": best_balanced,
        "carbon_saved_kg": clean["carbon_saved_kg"],
        "carbon_saved_pct": round(clean["carbon_saved_kg"] / home["carbon_kg"] * 100, 1) if home["carbon_kg"] else 0.0,
        "energy_cost_saved_usd": clean["cost_saved_usd"],
        "latency_penalty_ms": clean["latency_penalty_ms"],
    }


if __name__ == "__main__":
    run()

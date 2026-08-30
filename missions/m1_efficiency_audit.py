"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

Extension implemented here (Your Turn #2): MBU-driven right-sizing — pick the
replacement GPU from measured bandwidth + working set, not from $/GPU-hr.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
HEADROOM = 1.15               # size to measured demand + 15%, not to the spec sheet
MFU_OVERPROVISIONED = 0.30    # below this the rented FLOPs are going unused
MBU_OVERPROVISIONED = 0.50    # ...and so is the rented bandwidth


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "peak_bw": 0.0, "peak_tflops": 0.0, "peak_mem": 0.0, "workload": None})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["workload"] = r.get("workload")
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) >= 10:  # ignore idle hours when sizing
            a["peak_bw"] = max(a["peak_bw"], num(r["achieved_bw_tbs"]))
            a["peak_tflops"] = max(a["peak_tflops"], num(r["achieved_tflops"]))
            a["peak_mem"] = max(a["peak_mem"], num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        gtype = a["type"]
        ridge = metrics.arithmetic_intensity(num(cat[gtype]["peak_tflops_fp16"]), num(cat[gtype]["peak_bw_tbs"]))
        intensity = metrics.arithmetic_intensity(a["peak_tflops"], a["peak_bw"])
        summary.append({
            "gpu_id": gid, "gpu_type": gtype, "workload": a["workload"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "peak_bw_tbs": round(a["peak_bw"], 3),
            "peak_tflops": round(a["peak_tflops"], 1),
            "peak_mem_gb": round(a["peak_mem"], 1),
            "intensity": round(intensity, 1),
            "ridge": round(ridge, 1),
            "regime": metrics.roofline_regime(intensity, ridge),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # ---------------------------------------------------------------- EXT-2
    # Right-size on the two things that actually bind an inference GPU: the
    # bandwidth decode consumes and the VRAM the weights + KV cache occupy.
    # Gate first: only a GPU that is under-using BOTH its FLOPs and its HBM is
    # over-provisioned. A training H100 at MFU 0.42 is doing the work it was
    # rented for — moving it would be a migration, not a right-size.
    catalog_rows = list(cat.values())
    catalog_econ = sorted(
        [{
            "gpu_type": c["gpu_type"],
            "on_demand_hr": num(c["on_demand_hr"]),
            "hbm_gb": num(c["hbm_gb"]),
            "peak_bw_tbs": num(c["peak_bw_tbs"]),
            "usd_per_gb_vram": round(metrics.dollars_per_gb_vram(num(c["on_demand_hr"]), num(c["hbm_gb"])), 5),
            "usd_per_tbs": round(metrics.dollars_per_tbs(num(c["on_demand_hr"]), num(c["peak_bw_tbs"])), 4),
        } for c in catalog_rows],
        key=lambda x: x["usd_per_tbs"],
    )

    rightsize = []
    for s in summary:
        if s["mfu"] >= MFU_OVERPROVISIONED or s["mbu"] >= MBU_OVERPROVISIONED:
            continue
        cur = cat[s["gpu_type"]]
        cur_price = num(cur["on_demand_hr"])
        cands = metrics.rightsize_candidates(
            catalog_rows,
            required_bw_tbs=s["peak_bw_tbs"],
            required_vram_gb=s["peak_mem_gb"],
            required_tflops=s["peak_tflops"],
            current_on_demand_hr=cur_price,
            headroom=HEADROOM,
            exclude=s["gpu_type"],
        )
        if not cands:
            continue
        best = cands[0]
        need_bw = s["peak_bw_tbs"] * HEADROOM
        need_vram = s["peak_mem_gb"] * HEADROOM
        # Billable hours only: hours the GPU is already idle belong to the
        # "kill idle GPUs" lever, so charging them here would double-count.
        billable_hours = max(0.0, 24 - s["idle_hours"]) * DAYS
        monthly = best["hourly_saving"] * billable_hours
        rejected = []
        for c in catalog_rows:
            price = num(c["on_demand_hr"])
            if c["gpu_type"] in (s["gpu_type"], best["gpu_type"]) or price >= best["on_demand_hr"]:
                continue
            reasons = []
            if num(c["peak_bw_tbs"]) < need_bw:
                reasons.append(f"BW {num(c['peak_bw_tbs'])} < {need_bw:.2f} TB/s")
            if num(c["hbm_gb"]) < need_vram:
                reasons.append(f"VRAM {num(c['hbm_gb'])}GB < {need_vram:.0f}GB")
            if num(c["peak_tflops_fp16"]) < s["peak_tflops"] * HEADROOM:
                reasons.append(f"FP16 {num(c['peak_tflops_fp16']):.0f} < {s['peak_tflops']*HEADROOM:.0f} TFLOPs")
            if reasons:
                rejected.append(f"{c['gpu_type']} (${price}/h: " + ", ".join(reasons) + ")")
        rightsize.append({
            "gpu_id": s["gpu_id"], "from": s["gpu_type"], "to": best["gpu_type"],
            "regime": s["regime"], "mbu": s["mbu"], "mfu": s["mfu"],
            "need_bw_tbs": round(need_bw, 3), "need_vram_gb": round(need_vram, 1),
            "billable_hours": round(billable_hours),
            "from_hr": cur_price, "to_hr": best["on_demand_hr"],
            "from_usd_per_tbs": round(metrics.dollars_per_tbs(cur_price, num(cur["peak_bw_tbs"])), 4),
            "to_usd_per_tbs": best["usd_per_tbs"],
            "from_usd_per_gb": round(metrics.dollars_per_gb_vram(cur_price, num(cur["hbm_gb"])), 5),
            "to_usd_per_gb": best["usd_per_gb_vram"],
            "monthly_saving": round(monthly, 2),
            "rejected_cheaper": rejected,
        })
    rightsize_monthly = round(sum(r["monthly_saving"] for r in rightsize), 2)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}  {'regime':<14}{'FLOP/byte':>10}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}"
                  f"{s['idle_hours']:>8}  {s['regime']:<14}{s['intensity']:>10}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*DAYS:,.0f}/month")

        print("\n-- EXT-2: right-sizing by MBU (measured bandwidth + working set) --")
        print("catalog economics (cheapest bandwidth first):")
        print(f"  {'gpu':8}{'$/hr':>8}{'HBM GB':>8}{'TB/s':>7}{'$/GB-VRAM':>12}{'$/TB/s':>10}")
        for c in catalog_econ:
            print(f"  {c['gpu_type']:8}{c['on_demand_hr']:>8}{c['hbm_gb']:>8.0f}{c['peak_bw_tbs']:>7}"
                  f"{c['usd_per_gb_vram']:>12}{c['usd_per_tbs']:>10}")
        if rightsize:
            print(f"\n  {'GPU':13}{'now':6}{'->':4}{'fits':8}{'MFU':>6}{'MBU':>6}{'need BW':>9}"
                  f"{'need VRAM':>10}{'$/TB/s now':>11}{'$/TB/s new':>11}{'$/mo saved':>11}")
            for r in rightsize:
                print(f"  {r['gpu_id']:13}{r['from']:6}{'->':4}{r['to']:8}{r['mfu']:>6}{r['mbu']:>6}"
                      f"{r['need_bw_tbs']:>9}{r['need_vram_gb']:>10.0f}"
                      f"{r['from_usd_per_tbs']:>11}{r['to_usd_per_tbs']:>11}{r['monthly_saving']:>11,.0f}")
            print(f"  total if all right-sized: ${rightsize_monthly:,.0f}/month "
                  f"(billable hours only — idle hours belong to the idle lever)")
            print("  why not just the cheapest $/GPU-hr:")
            for r in rightsize:
                if r["rejected_cheaper"]:
                    print(f"    {r['gpu_id']}: rejected " + "; ".join(r["rejected_cheaper"]))
        else:
            print("  no GPU is under-using both its FLOPs and its bandwidth")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsize": rightsize, "rightsize_monthly": rightsize_monthly}


if __name__ == "__main__":
    run()

"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Extensions implemented here:
  * Your Turn #3 — cache_is_worth_it(): prompt caching is gated on the measured
    re-read rate per team, and the cache WRITE premium is charged honestly.
  * Your Turn #4 — reasoning budget: $ and Wh split by is_reasoning, plus the
    saving from capping reasoning traffic.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Cache tiers, Anthropic-style: a write is billed above the input rate, a read
# far below it. The 1-hour tier costs more to write but survives long gaps.
CACHE_TIERS = {
    "5min": {"ttl_s": 300, "write_multiplier": 1.25, "read_discount": 0.10},
    "1hour": {"ttl_s": 3600, "write_multiplier": 2.00, "read_discount": 0.10},
}
REASONING_CAP_TARGET = 0.10   # Your Turn #4: cap reasoning at 10% of requests


def _ts_seconds(ts: str) -> float:
    """'2026-06-15T13:07:00' -> seconds since midnight."""
    try:
        hh, mm, ss = ts.split("T")[1].split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return 0.0


def analyze_cache(rows) -> dict:
    """EXT-3 — measure how often a cached prefix is actually re-read, per team.

    A cache entry only pays off if it is read back before its TTL expires. We
    reconstruct that from the request timestamps: sort each team's cacheable
    requests, and every gap longer than the TTL forces a fresh (premium-priced)
    write. avg_reads = (requests - writes) / writes.
    """
    per_team = defaultdict(list)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            per_team[r["team"]].append(_ts_seconds(r["ts"]))

    out = {}
    for team, stamps in per_team.items():
        stamps.sort()
        entry = {"requests": len(stamps), "tiers": {}}
        for name, cfg in CACHE_TIERS.items():
            writes = 1 + sum(1 for a, b in zip(stamps, stamps[1:]) if (b - a) > cfg["ttl_s"])
            avg_reads = (len(stamps) - writes) / writes if writes else 0.0
            net = pricing.cache_net_multiplier(avg_reads, cfg["write_multiplier"], cfg["read_discount"])
            entry["tiers"][name] = {
                "writes": writes,
                "avg_reads": round(avg_reads, 1),
                "break_even_reads": round(pricing.cache_break_even_reads(
                    cfg["write_multiplier"], cfg["read_discount"]), 2),
                "worth_it": pricing.cache_is_worth_it(
                    avg_reads, cfg["write_multiplier"], cfg["read_discount"]),
                "net_multiplier": round(net, 4),
            }
        best = min(entry["tiers"], key=lambda t: entry["tiers"][t]["net_multiplier"])
        entry["best_tier"] = best
        entry["effective_discount"] = entry["tiers"][best]["net_multiplier"]
        entry["enabled"] = entry["tiers"][best]["worth_it"] and entry["effective_discount"] < 1.0
        if not entry["enabled"]:
            entry["effective_discount"] = 1.0   # pay full price, do not cache
        out[team] = entry
    return out


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache = analyze_cache(rows)

    base_cost = casc_cost = cache_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning = {
        True: {"requests": 0, "tokens": 0, "cost": 0.0, "wh": 0.0, "out_tokens": 0},
        False: {"requests": 0, "tokens": 0, "cost": 0.0, "wh": 0.0, "out_tokens": 0},
    }

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        tokens = inp + out
        total_tokens += tokens

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # LEVER 1 — cascade: 80% of traffic never needed the frontier model
        pin, pout = MODEL_PRICES[r["route_tier"]]
        casc_cost += pricing.request_cost(inp, out, pin, pout)

        # LEVER 2 — prompt caching, but only where the re-read rate clears the
        # write premium (EXT-3). Teams that fail the gate pay full input price.
        team_cache = cache.get(r["team"], {})
        eff_discount = team_cache.get("effective_discount", 1.0)
        c2 = pricing.request_cost(inp, out, pin, pout, cached_in=cached, cache_discount=eff_discount)
        cache_cost += c2

        # LEVER 3 — batch API on latency-tolerant traffic
        c3 = pricing.request_cost(inp, out, pin, pout, cached_in=cached,
                                  cache_discount=eff_discount, batch=is_batch)
        opt_cost += c3

        b = reasoning[is_reasoning]
        b["requests"] += 1
        b["tokens"] += tokens
        b["out_tokens"] += out
        b["cost"] += c3
        b["wh"] += sustainability.wh_per_query(tokens, is_reasoning=is_reasoning)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # ---- per-lever ledger, in $/1M-token (the unit the report is graded in) ----
    stages = [
        ("Baseline — toàn bộ chạy model lớn, không cache, không batch", base_cost),
        ("+ Cascade — định tuyến nhỏ/lớn", casc_cost),
        ("+ Prompt cache — có cổng theo tỷ lệ đọc lại", cache_cost),
        ("+ Batch API — traffic chịu được độ trễ", opt_cost),
    ]
    levers = []
    prev_cost, prev_pm = base_cost, base_pm
    for i, (name, cost) in enumerate(stages):
        pm = pricing.dollars_per_million(cost, total_tokens)
        levers.append({
            "stage": name,
            "daily_usd": round(cost, 2),
            "per_m": round(pm, 3),
            "marginal_daily": 0.0 if i == 0 else round(prev_cost - cost, 2),
            "marginal_monthly": 0.0 if i == 0 else round((prev_cost - cost) * 30, 2),
            "marginal_per_m": 0.0 if i == 0 else round(prev_pm - pm, 3),
            "cum_savings_pct": 0.0 if i == 0 else round((1 - cost / base_cost) * 100, 1),
        })
        prev_cost, prev_pm = cost, pm

    # ------------------------------------------------------------ EXT-4 ----
    rz, nz = reasoning[True], reasoning[False]
    n_req = rz["requests"] + nz["requests"]
    tot_cost = rz["cost"] + nz["cost"]
    tot_wh = rz["wh"] + nz["wh"]
    cost_per_reasoning = rz["cost"] / rz["requests"] if rz["requests"] else 0.0
    cost_per_normal = nz["cost"] / nz["requests"] if nz["requests"] else 0.0
    wh_per_reasoning = rz["wh"] / rz["requests"] if rz["requests"] else 0.0
    wh_per_normal = nz["wh"] / nz["requests"] if nz["requests"] else 0.0
    share_req = rz["requests"] / n_req if n_req else 0.0

    # Cap scenario: keep the top `target` share of requests on reasoning, demote
    # the rest to the normal path (same request, no extended thinking).
    target = min(REASONING_CAP_TARGET, share_req)
    keep = int(round(n_req * target))
    demoted = max(0, rz["requests"] - keep)
    cap = {
        "target_frac": REASONING_CAP_TARGET,
        "binding": share_req > REASONING_CAP_TARGET,
        "demoted_requests": demoted,
        "cost_saved_daily": round(demoted * (cost_per_reasoning - cost_per_normal), 2),
        "wh_saved_daily": round(demoted * (wh_per_reasoning - wh_per_normal), 1),
    }
    # A cap that does not bind teaches nothing, so also price a cap that does.
    half = int(rz["requests"] * 0.5)
    cap_half = {
        "label": "halve reasoning traffic",
        "demoted_requests": half,
        "cost_saved_daily": round(half * (cost_per_reasoning - cost_per_normal), 2),
        "cost_saved_monthly": round(half * (cost_per_reasoning - cost_per_normal) * 30, 2),
        "wh_saved_daily": round(half * (wh_per_reasoning - wh_per_normal), 1),
    }
    reasoning_out = {
        "share_requests": round(share_req, 4),
        "share_tokens": round(rz["tokens"] / (rz["tokens"] + nz["tokens"]), 4) if total_tokens else 0.0,
        "share_cost": round(rz["cost"] / tot_cost, 4) if tot_cost else 0.0,
        "share_energy": round(rz["wh"] / tot_wh, 4) if tot_wh else 0.0,
        "cost_per_request": {"reasoning": round(cost_per_reasoning, 5), "normal": round(cost_per_normal, 5)},
        "wh_per_request": {"reasoning": round(wh_per_reasoning, 2), "normal": round(wh_per_normal, 3)},
        "avg_output_tokens": {
            "reasoning": round(rz["out_tokens"] / rz["requests"], 0) if rz["requests"] else 0,
            "normal": round(nz["out_tokens"] / nz["requests"], 0) if nz["requests"] else 0,
        },
        "daily_wh": round(tot_wh, 1),
        "cap": cap, "cap_half": cap_half,
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- per-lever ledger ($/1M-token) --")
        print(f"{'stage':44}{'$/day':>10}{'$/1M-tok':>11}{'Δ$/1M':>9}{'Δ$/month':>11}{'cum %':>8}")
        for l in levers:
            print(f"{l['stage']:44}{l['daily_usd']:>10,.2f}{l['per_m']:>11.3f}"
                  f"{l['marginal_per_m']:>9.3f}{l['marginal_monthly']:>11,.0f}{l['cum_savings_pct']:>8}")

        print("\n-- EXT-3: is prompt caching worth it? --")
        be5 = pricing.cache_break_even_reads(**{k: v for k, v in CACHE_TIERS["5min"].items() if k != "ttl_s"})
        be1 = pricing.cache_break_even_reads(**{k: v for k, v in CACHE_TIERS["1hour"].items() if k != "ttl_s"})
        print(f"break-even reads per write: 5-min tier {be5:.2f}   1-hour tier {be1:.2f}")
        print(f"{'team':11}{'cacheable req':>14}{'writes(5m)':>12}{'reads/write':>13}"
              f"{'best tier':>11}{'net mult':>10}{'cache?':>8}")
        for team, c in sorted(cache.items()):
            t5 = c["tiers"]["5min"]
            print(f"{team:11}{c['requests']:>14}{t5['writes']:>12}{t5['avg_reads']:>13}"
                  f"{c['best_tier']:>11}{c['effective_discount']:>10.4f}{str(c['enabled']):>8}")

        print("\n-- EXT-4: reasoning budget --")
        ro = reasoning_out
        print(f"reasoning traffic : {ro['share_requests']:.1%} of requests, {ro['share_tokens']:.1%} of tokens, "
              f"{ro['share_cost']:.1%} of cost, {ro['share_energy']:.1%} of energy")
        print(f"per request       : ${ro['cost_per_request']['reasoning']:.5f} vs ${ro['cost_per_request']['normal']:.5f} "
              f"({ro['cost_per_request']['reasoning']/max(ro['cost_per_request']['normal'],1e-9):.1f}x)  |  "
              f"{ro['wh_per_request']['reasoning']:.2f} Wh vs {ro['wh_per_request']['normal']:.3f} Wh")
        print(f"avg output tokens : {ro['avg_output_tokens']['reasoning']:.0f} vs {ro['avg_output_tokens']['normal']:.0f}")
        print(f"cap at {REASONING_CAP_TARGET:.0%} of traffic: binding={cap['binding']}, "
              f"demote {cap['demoted_requests']} req -> ${cap['cost_saved_daily']:,.2f}/day, {cap['wh_saved_daily']:,.0f} Wh/day")
        print(f"halve reasoning   : demote {cap_half['demoted_requests']} req -> "
              f"${cap_half['cost_saved_monthly']:,.0f}/month, {cap_half['wh_saved_daily']:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "levers": levers, "cache": cache, "reasoning": reasoning_out,
    }


if __name__ == "__main__":
    run()

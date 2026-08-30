"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).

Student extensions (Your Turn):
  * EXT-1  interruption-rate-aware tier policy + 1yr-vs-3yr term economics
  * EXT-3  cache_is_worth_it() — prompt caching only pays above a read threshold
"""
from __future__ import annotations

# ---------------------------------------------------------------- EXT-1 data
# Spot interruption rate per GPU-hour. Scarcity drives reclaim: the newer and
# tighter the supply, the more often the provider takes the instance back.
# Illustrative June-2026 snapshot, same provenance class as the price catalog.
INTERRUPT_RATE_PER_HOUR = {
    "B200": 0.12,    # newest, most contended
    "H200": 0.10,
    "H100": 0.08,
    "A100": 0.05,
    "MI300X": 0.04,
    "A10G": 0.02,
    "L4": 0.015,     # abundant, rarely reclaimed
}
DEFAULT_INTERRUPT_RATE = 0.05


def interrupt_rate_for(gpu_type: str | None = None) -> float:
    """Per-hour spot interruption probability for a GPU type."""
    if not gpu_type:
        return DEFAULT_INTERRUPT_RATE
    return INTERRUPT_RATE_PER_HOUR.get(gpu_type, DEFAULT_INTERRUPT_RATE)


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# ------------------------------------------------------------------- EXT-1
def spot_effective_multiplier(
    interrupt_rate: float,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
) -> float:
    """Billed hours per hour of useful work on spot.

    Two taxes: the steady cost of writing checkpoints, and the compute redone
    after each reclaim. At H100's 8%/hr you pay 1.07x — the discount survives it,
    but on a hypothetical 60%/hr GPU it would not.
    """
    return 1.0 + max(0.0, ckpt_overhead_frac) + max(0.0, interrupt_rate) * max(0.0, rework_hours_per_interrupt)


def monthly_duty_cycle(hours_per_day: float, days: float = 30, month_days: float = 30) -> float:
    """Share of a *billing month* the workload actually occupies the GPU.

    The naive duty cycle (hours/24) overstates any job that does not run every
    day: a 20h/day job that runs 14 days a month occupies 39% of the month, not
    83%. A reserved instance is billed for all 720 hours either way.
    """
    if month_days <= 0:
        return 0.0
    used = max(0.0, hours_per_day) * max(0.0, min(days, month_days))
    return used / (24.0 * month_days)


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    days: float = 30,
    spot_discount: float = 0.40,
    max_rework_frac: float = 0.15,
) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility and reclaim risk.

    POLICY v2 (EXT-1). Two things the v1 policy got wrong:

      1. It sent every interruptible job to spot regardless of how often that GPU
         type is actually reclaimed. Spot only wins while the interruption tax
         (`spot_effective_multiplier`) stays smaller than the spot discount.
      2. It read the duty cycle as hours/24, ignoring how many days a month the
         job runs. Reserved capacity is billed 24x30 whether you use it or not,
         so the comparison must be against the *monthly* duty cycle.

    Returns one of 'spot' / 'reserved' / 'on_demand'. Use `recommend_plan()` when
    real catalog prices are available — it compares money instead of heuristics.
    """
    rate = interrupt_rate_for(gpu_type)
    duty = monthly_duty_cycle(hours_per_day, days)
    be = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        tax = spot_effective_multiplier(rate)
        rework_frac = rate * 0.5
        spot_still_cheaper = tax * (1.0 - spot_discount) < 1.0
        if spot_still_cheaper and rework_frac <= max_rework_frac:
            return "spot"
        # too hot to checkpoint around — fall through to the committed tiers
    if duty >= be:
        return "reserved"
    return "on_demand"


def reserved_term_choice(
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    survival_prob_3yr: float = 0.60,
    resale_recovery: float = 0.0,
) -> dict:
    """1-year vs 3-year commitment under the risk that the workload dies early.

    A 3-year rate is only the cheaper rate if you actually consume three years of
    it. Model: with probability `survival_prob_3yr` the workload lives the whole
    term; otherwise it dies on average halfway and the rest of the commitment is
    sunk (recoverable only at `resale_recovery` on the secondary market).

        useful_fraction f = p + (1-p) * (0.5 + 0.5 * resale_recovery)
        effective 3yr rate = r3 / f

    Break-even survival probability solves r3/f = r1, i.e. f* = r3/r1. Below that
    confidence, take the 1-year rate and keep the option to walk away.
    """
    r1, r3 = max(0.0, reserved_1yr_hr), max(0.0, reserved_3yr_hr)
    p = max(0.0, min(1.0, survival_prob_3yr))
    dead_fraction = 0.5 + 0.5 * max(0.0, min(1.0, resale_recovery))
    f = p + (1.0 - p) * dead_fraction
    eff_3yr = r3 / f if f > 0 else float("inf")

    f_star = (r3 / r1) if r1 > 0 else 1.0
    if dead_fraction >= 1.0:
        p_star = 0.0
    else:
        p_star = (f_star - dead_fraction) / (1.0 - dead_fraction)
    p_star = max(0.0, min(1.0, p_star))

    term = "3yr" if eff_3yr <= r1 else "1yr"
    return {
        "term": term,
        "rate_1yr": round(r1, 4),
        "rate_3yr": round(r3, 4),
        "effective_3yr_rate": round(eff_3yr, 4),
        "useful_fraction": round(f, 3),
        "break_even_survival_prob": round(p_star, 3),
    }


def recommend_plan(
    hours_per_day: float,
    days: float,
    interruptible: bool,
    on_demand_hr: float,
    spot_hr: float,
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    num_gpus: int = 1,
    gpu_type: str | None = None,
    month_days: float = 30,
    survival_prob_3yr: float = 0.60,
) -> dict:
    """Price every tier for one workload and return the cheapest, with its reason.

    Cost model, per calendar month:
      on_demand  used_hours x rate                 (pay per hour used)
      spot       used_hours x rate x interrupt tax (pay per hour used + rework)
      reserved   24 x month_days x rate            (pay for the WHOLE month, used or not)

    That asymmetry is the entire point: reserved is a rate cut bought with a
    utilization obligation, so a job that idles the capacity half the month is
    paying on-demand money for reserved capacity.
    """
    used_hours = max(0.0, hours_per_day) * max(0.0, min(days, month_days)) * max(1, num_gpus)
    committed_hours = 24.0 * month_days * max(1, num_gpus)
    rate = interrupt_rate_for(gpu_type)
    tax = spot_effective_multiplier(rate)

    term = reserved_term_choice(reserved_1yr_hr, reserved_3yr_hr, survival_prob_3yr)
    reserved_rate = reserved_3yr_hr if term["term"] == "3yr" else reserved_1yr_hr

    options = {
        "on_demand": used_hours * on_demand_hr,
        "spot": used_hours * spot_hr * tax if interruptible else float("inf"),
        "reserved": committed_hours * reserved_rate,
    }
    tier = min(options, key=options.get)
    duty = monthly_duty_cycle(hours_per_day, days, month_days)

    if tier == "spot":
        why = (f"interruptible, {gpu_type or 'gpu'} reclaim {rate:.1%}/h -> pay {tax:.2f}x "
               f"billed hours for 1x useful work, still under the spot discount")
    elif tier == "reserved":
        why = (f"monthly duty {duty:.0%} >= break-even, so the 24x{month_days:.0f}h "
               f"commitment ({term['term']}) amortises below on-demand")
    else:
        why = f"monthly duty only {duty:.0%} and not interruptible — nothing to commit to"

    return {
        "tier": tier,
        "term": term["term"] if tier == "reserved" else "",
        "used_hours": round(used_hours, 1),
        "duty_cycle": round(duty, 3),
        "interrupt_rate": rate,
        "spot_tax": round(tax, 3),
        "cost": round(options[tier], 2),
        "costs": {k: (round(v, 2) if v != float("inf") else None) for k, v in options.items()},
        "effective_hourly": round(options[tier] / used_hours, 4) if used_hours else 0.0,
        "break_even_survival_prob": term["break_even_survival_prob"],
        "why": why,
    }


def tier_matrix(catalog: dict, duty_cycles=(0.25, 0.50, 0.75, 1.0), month_days: float = 30) -> list:
    """Recommendation grid: GPU type x monthly duty cycle x interruptible.

    Printed by M3 so the policy is auditable as a table instead of a black box.
    """
    out = []
    for gtype, row in catalog.items():
        for duty in duty_cycles:
            hpd = duty * 24.0
            for interruptible in (False, True):
                plan = recommend_plan(
                    hours_per_day=hpd, days=month_days, interruptible=interruptible,
                    on_demand_hr=float(row["on_demand_hr"]), spot_hr=float(row["spot_hr"]),
                    reserved_1yr_hr=float(row["reserved_1yr_hr"]),
                    reserved_3yr_hr=float(row["reserved_3yr_hr"]),
                    gpu_type=gtype, month_days=month_days,
                )
                out.append({"gpu_type": gtype, "duty": duty, "interruptible": interruptible,
                            "tier": plan["tier"], "term": plan["term"],
                            "effective_hourly": plan["effective_hourly"]})
    return out


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ------------------------------------------------------------------- EXT-3
def cache_break_even_reads(write_multiplier: float = 1.25, read_discount: float = 0.10) -> float:
    """How many cache READS a written prefix needs before caching pays for itself.

    Writing a prefix into the cache is billed above the normal input rate
    (`write_multiplier`, ~1.25x on Anthropic-style pricing); every later read is
    billed at `read_discount` x the normal rate. So one write buys
    (write_multiplier - 1) of extra cost and each read returns (1 - read_discount):

        break-even reads = (write_multiplier - 1) / (1 - read_discount)

    A cache entry that is written and never re-read is strictly more expensive
    than not caching at all — which is why a 5-minute TTL on low-traffic routes
    quietly loses money.
    """
    gain_per_read = 1.0 - read_discount
    if gain_per_read <= 0:
        return float("inf")
    return max(0.0, write_multiplier - 1.0) / gain_per_read


def cache_is_worth_it(
    avg_reads: float,
    write_multiplier: float = 1.25,
    read_discount: float = 0.10,
    ttl_expiry_frac: float = 0.0,
) -> bool:
    """True when the average prefix is re-read often enough to beat the write premium.

    `ttl_expiry_frac` is the share of reads lost to TTL expiry (the prefix has to
    be re-written), which pushes the threshold up on bursty, low-QPS routes.
    """
    effective_reads = max(0.0, avg_reads) * (1.0 - max(0.0, min(1.0, ttl_expiry_frac)))
    return effective_reads >= cache_break_even_reads(write_multiplier, read_discount)


def cache_net_multiplier(
    avg_reads: float,
    write_multiplier: float = 1.25,
    read_discount: float = 0.10,
) -> float:
    """Cost of (1 write + avg_reads reads) relative to paying full price every time.

    < 1.0 means caching is cheaper. Lets M2 charge the *real* cache economics
    instead of assuming reads are free.
    """
    n = max(0.0, avg_reads)
    naive = 1.0 + n
    cached = write_multiplier + n * read_discount
    return cached / naive if naive > 0 else 1.0

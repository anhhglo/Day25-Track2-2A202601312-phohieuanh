"""Tests for the "Your Turn" extensions. The 15 lab tests are untouched.

Covers: EXT-1 tier policy + term economics, EXT-2 right-sizing, EXT-3 cache
break-even, EXT-4 reasoning split, EXT-5 region selection.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops import metrics, pricing, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling


# ------------------------------------------------------------------ EXT-1
def test_interrupt_rate_tracks_scarcity():
    assert pricing.interrupt_rate_for("B200") > pricing.interrupt_rate_for("H100")
    assert pricing.interrupt_rate_for("H100") > pricing.interrupt_rate_for("L4")
    assert pricing.interrupt_rate_for(None) == pricing.DEFAULT_INTERRUPT_RATE
    assert pricing.interrupt_rate_for("NOT-A-GPU") == pricing.DEFAULT_INTERRUPT_RATE


def test_spot_tax_grows_with_reclaim_rate():
    cheap = pricing.spot_effective_multiplier(0.015)   # L4
    dear = pricing.spot_effective_multiplier(0.12)     # B200
    assert 1.0 < cheap < dear
    # a GPU reclaimed twice an hour eats more than a 50% discount
    assert pricing.spot_effective_multiplier(2.5) * 0.5 > 1.0


def test_monthly_duty_cycle_counts_days_not_just_hours():
    # 20h/day but only 14 days a month is NOT an 83% duty cycle
    assert abs(pricing.monthly_duty_cycle(20, 14) - (20 * 14) / 720) < 1e-9
    assert abs(pricing.monthly_duty_cycle(24, 30) - 1.0) < 1e-9
    assert pricing.monthly_duty_cycle(20, 14) < pricing.monthly_duty_cycle(20, 30)


def test_policy_v2_refuses_to_commit_for_a_part_time_job():
    # v1 saw hours/24 = 75% and reserved it; v2 sees 37% of the month
    assert m3_purchasing.legacy_tier(18, False) == "reserved"
    assert pricing.recommend_tier(18, False, gpu_type="H100", days=15) == "on_demand"
    assert pricing.recommend_tier(18, False, gpu_type="H100", days=30) == "reserved"


def test_reserved_term_break_even_survival():
    h100 = pricing.reserved_term_choice(2.00, 1.40, survival_prob_3yr=0.60)
    assert h100["term"] == "3yr"
    assert abs(h100["break_even_survival_prob"] - 0.40) < 0.01
    # below that confidence the 1-year rate is the cheaper expected rate
    shaky = pricing.reserved_term_choice(2.00, 1.40, survival_prob_3yr=0.10)
    assert shaky["term"] == "1yr"
    assert shaky["effective_3yr_rate"] > shaky["rate_1yr"]


def test_recommend_plan_bills_reserved_for_the_whole_month():
    # 2h/day on-demand must beat a reserved commitment billed 24x30
    plan = pricing.recommend_plan(hours_per_day=2, days=30, interruptible=False,
                                  on_demand_hr=2.5, spot_hr=1.5,
                                  reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    assert plan["tier"] == "on_demand"
    assert plan["costs"]["reserved"] > plan["costs"]["on_demand"]
    # 24/7 flips it
    plan247 = pricing.recommend_plan(hours_per_day=24, days=30, interruptible=False,
                                     on_demand_hr=2.5, spot_hr=1.5,
                                     reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    assert plan247["tier"] == "reserved" and plan247["term"] == "3yr"


# ------------------------------------------------------------------ EXT-2
def test_rightsize_rejects_a_cheaper_gpu_that_cannot_do_the_work():
    catalog = [
        {"gpu_type": "L4", "on_demand_hr": 0.80, "hbm_gb": 24, "peak_bw_tbs": 0.30, "peak_tflops_fp16": 121},
        {"gpu_type": "A100", "on_demand_hr": 1.79, "hbm_gb": 80, "peak_bw_tbs": 2.00, "peak_tflops_fp16": 312},
    ]
    # needs 1 TB/s and 68GB (+15% headroom): L4 is cheaper but cannot serve it
    cands = metrics.rightsize_candidates(catalog, required_bw_tbs=1.0, required_vram_gb=68.0,
                                         required_tflops=200.0, current_on_demand_hr=2.50)
    assert [c["gpu_type"] for c in cands] == ["A100"]
    # a tiny workload can use the cheap card
    small = metrics.rightsize_candidates(catalog, required_bw_tbs=0.2, required_vram_gb=15.0,
                                         current_on_demand_hr=2.50)
    assert small[0]["gpu_type"] == "L4"


def test_bandwidth_price_can_invert_hourly_price():
    # MI300X costs more per hour than A100 but is far cheaper per TB/s
    assert metrics.dollars_per_tbs(1.95, 5.30) < metrics.dollars_per_tbs(1.79, 2.00)
    assert metrics.dollars_per_gb_vram(1.95, 192) < metrics.dollars_per_gb_vram(1.79, 80)
    assert metrics.dollars_per_gb_vram(1.0, 0) == 0.0


def test_m1_only_rightsizes_gpus_that_waste_both_flops_and_bandwidth():
    r1 = m1_efficiency_audit.run(verbose=False)
    moved = {r["gpu_id"] for r in r1["rightsize"]}
    by_id = {s["gpu_id"]: s for s in r1["summary"]}
    for gid in moved:
        assert by_id[gid]["mfu"] < m1_efficiency_audit.MFU_OVERPROVISIONED
        assert by_id[gid]["mbu"] < m1_efficiency_audit.MBU_OVERPROVISIONED
    assert "gpu-h100-0" not in moved          # MFU 0.42 is doing its job
    assert r1["rightsize_monthly"] > 0


# ------------------------------------------------------------------ EXT-3
def test_cache_break_even_reads():
    assert pricing.cache_break_even_reads(1.0, 0.10) == 0.0            # free write
    assert abs(pricing.cache_break_even_reads(1.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert abs(pricing.cache_break_even_reads(2.00, 0.10) - 1.0 / 0.9) < 1e-9
    # a pricier write needs more reads to pay for itself
    assert pricing.cache_break_even_reads(2.00) > pricing.cache_break_even_reads(1.25)


def test_cache_is_worth_it_gate():
    assert pricing.cache_is_worth_it(10) is True
    assert pricing.cache_is_worth_it(0.0, 1.25, 0.10) is False          # write-once, never read
    assert pricing.cache_is_worth_it(1.0, 2.00, 0.10) is False          # 1-hour tier needs >1.11
    assert pricing.cache_is_worth_it(2.0, 2.00, 0.10) is True
    # TTL expiry pushes the threshold up
    assert pricing.cache_is_worth_it(0.3, 1.25, 0.10) is True
    assert pricing.cache_is_worth_it(0.3, 1.25, 0.10, ttl_expiry_frac=0.5) is False


def test_cache_net_multiplier_beats_full_price_only_above_break_even():
    assert pricing.cache_net_multiplier(0.0, 1.25, 0.10) > 1.0
    assert pricing.cache_net_multiplier(100, 1.25, 0.10) < 0.15
    be = pricing.cache_break_even_reads(1.25, 0.10)
    assert abs(pricing.cache_net_multiplier(be, 1.25, 0.10) - 1.0) < 1e-9


def test_m2_charges_the_measured_cache_multiplier():
    r2 = m2_inference_levers.run(verbose=False)
    for team, c in r2["cache"].items():
        assert c["effective_discount"] >= 0.10      # never cheaper than a pure read
        if c["enabled"]:
            assert c["effective_discount"] < 1.0
    # the ledger stages must be monotonically cheaper
    per_m = [l["per_m"] for l in r2["levers"]]
    assert per_m == sorted(per_m, reverse=True)


# ------------------------------------------------------------------ EXT-4
def test_reasoning_is_a_minority_of_traffic_but_dominates_energy():
    rz = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rz["share_requests"] < 0.20
    assert rz["share_cost"] > rz["share_requests"]      # costs more than its share
    assert rz["share_energy"] > rz["share_cost"]        # and burns far more than that
    assert rz["wh_per_request"]["reasoning"] > 50 * rz["wh_per_request"]["normal"]


def test_reasoning_cap_only_saves_when_it_binds():
    rz = m2_inference_levers.run(verbose=False)["reasoning"]
    if not rz["cap"]["binding"]:
        assert rz["cap"]["demoted_requests"] == 0 and rz["cap"]["cost_saved_daily"] == 0.0
    assert rz["cap_half"]["wh_saved_daily"] > 0


# ------------------------------------------------------------------ EXT-5
def test_region_selection_depends_on_the_criterion():
    assert sustainability.best_region("carbon") == "europe-north1"
    assert sustainability.best_region("cost") == "us-east-wa"
    assert sustainability.best_region("balanced") in sustainability.REGION_CARBON
    # weighting purely on carbon must reproduce the carbon pick
    assert sustainability.best_region("balanced", weight_cost=0.0) == "europe-north1"
    assert sustainability.best_region("balanced", weight_cost=1.0) == "us-east-wa"


def test_only_interruptible_load_is_moved():
    r6 = m6_carbon_scheduling.run(verbose=False)
    assert {p["job_id"] for p in r6["per_job"]} == {
        "job-train-llm", "job-train-embed", "job-finetune", "job-dev-sandbox", "job-batch-eval"}
    assert r6["fixed_kwh"] > 0                      # user-facing load stays put
    assert r6["carbon_saved_kg"] > 0
    assert r6["latency_penalty_ms"] > 0             # the clean region is further away


def test_job_energy_includes_facility_overhead():
    raw = sustainability.job_energy_kwh(700, 10, 1, pue=1.0)
    with_pue = sustainability.job_energy_kwh(700, 10, 1, pue=1.15)
    assert abs(raw - 7.0) < 1e-9
    assert abs(with_pue - 8.05) < 1e-9

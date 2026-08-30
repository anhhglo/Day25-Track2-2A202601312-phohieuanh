"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


# ------------------------------------------------------------------- EXT-2
# Right-sizing by MBU. A GPU is the wrong size when you are renting FLOPs and
# HBM capacity you never touch — but the replacement still has to sustain the
# bandwidth and hold the working set, which is why $/GPU-hr alone picks wrong.
def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """$/hour per GB of HBM — the capacity price of the rental."""
    if hbm_gb <= 0:
        return 0.0
    return on_demand_hr / hbm_gb


def dollars_per_tbs(on_demand_hr: float, peak_bw_tbs: float) -> float:
    """$/hour per TB/s of memory bandwidth — the price of what decode actually consumes."""
    if peak_bw_tbs <= 0:
        return 0.0
    return on_demand_hr / peak_bw_tbs


def rightsize_candidates(
    catalog_rows,
    required_bw_tbs: float,
    required_vram_gb: float,
    required_tflops: float = 0.0,
    current_on_demand_hr: float = 0.0,
    headroom: float = 1.15,
    exclude: str | None = None,
):
    """Cheaper GPUs that still clear the measured bandwidth / VRAM / FLOPs demand.

    `catalog_rows` is an iterable of price-catalog dicts. Requirements are the
    *achieved* figures from telemetry times `headroom` — you size to the work the
    GPU is really doing, plus a margin, not to the spec sheet of what it is
    replacing. Returned cheapest-first.
    """
    need_bw = max(0.0, required_bw_tbs) * headroom
    need_vram = max(0.0, required_vram_gb) * headroom
    need_flops = max(0.0, required_tflops) * headroom
    out = []
    for r in catalog_rows:
        gtype = r.get("gpu_type")
        if exclude and gtype == exclude:
            continue
        price = float(r.get("on_demand_hr", 0) or 0)
        bw = float(r.get("peak_bw_tbs", 0) or 0)
        vram = float(r.get("hbm_gb", 0) or 0)
        flops = float(r.get("peak_tflops_fp16", 0) or 0)
        if bw < need_bw or vram < need_vram or flops < need_flops:
            continue
        if current_on_demand_hr and price >= current_on_demand_hr:
            continue
        out.append({
            "gpu_type": gtype,
            "on_demand_hr": price,
            "peak_bw_tbs": bw,
            "hbm_gb": vram,
            "peak_tflops_fp16": flops,
            "usd_per_gb_vram": round(dollars_per_gb_vram(price, vram), 5),
            "usd_per_tbs": round(dollars_per_tbs(price, bw), 4),
            "hourly_saving": round(current_on_demand_hr - price, 4) if current_on_demand_hr else 0.0,
        })
    return sorted(out, key=lambda x: x["on_demand_hr"])

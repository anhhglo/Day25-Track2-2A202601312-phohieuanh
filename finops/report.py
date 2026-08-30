"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_levers=None, sections=None, lead=None) -> str:
    """Return a markdown cost-optimization report.

    `unit_levers` is the per-lever inference ledger in $/1M-token (list of dicts
    from M2) — the unit the whole lab is graded in. `sections` is an ordered list
    of {"title", "body"} blocks appended after the numbers, so the analysis lives
    next to the figures it explains instead of in a separate document.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — Báo cáo tối ưu chi phí GPU",
        "",
        f"**Period / Kỳ báo cáo:** {period}  ",
        f"**Baseline spend / Chi phí gốc:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend / Chi phí sau tối ưu:** ${optimized_usd:,.0f}  ",
        f"**Projected savings / Tiết kiệm dự kiến:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
    ]
    if lead:
        lines += ["## Tóm tắt điều hành", "", lead["body"].rstrip(), ""]
    lines += [
        "## Tiết kiệm theo từng lever",
        "",
        "| Lever | Tiết kiệm (USD/tháng) | Tỷ trọng |",
        "|---|---:|---:|",
    ]
    total = sum(levers.values()) or 1.0
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total * 100:.0f}% |")
    lines.append(f"| **Tổng** | **${sum(levers.values()):,.0f}** | 100% |")

    if unit_levers:
        lines += [
            "",
            "## Đơn giá inference — $/1M-token theo từng lever",
            "",
            "| Giai đoạn | $/ngày | $/1M-token | Δ $/1M-token | Δ $/tháng | Tiết kiệm luỹ kế |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for l in unit_levers:
            delta_pm = f"−{l['marginal_per_m']:.3f}" if l["marginal_per_m"] else "—"
            delta_mo = f"${l['marginal_monthly']:,.0f}" if l["marginal_monthly"] else "—"
            cum = f"{l['cum_savings_pct']:.1f}%" if l["cum_savings_pct"] else "—"
            lines.append(f"| {l['stage']} | ${l['daily_usd']:,.2f} | **{l['per_m']:.3f}** | "
                         f"{delta_pm} | {delta_mo} | {cum} |")

    if sustainability:
        lines += [
            "",
            "## Tính bền vững (Sustainability)",
            "",
            f"- Năng lượng mỗi truy vấn: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon mỗi truy vấn: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Vùng tốt nhất (rẻ + sạch, cân bằng): {sustainability.get('best_region', 'n/a')}",
        ]
        for extra in sustainability.get("notes", []):
            lines.append(f"- {extra}")

    for s in sections or []:
        lines += ["", f"## {s['title']}", "", s["body"].rstrip()]

    lines += ["", "_Số liệu là snapshot tháng 6/2026; giá GPU đổi hàng tháng — phải chuẩn hoá lại trước khi áp dụng._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None,
                      optimized: float | None = None) -> str:
    """Write the savings waterfall PNG. Returns the path. No-op if matplotlib absent.

    A real waterfall: the baseline bar is knocked down one lever at a time, so the
    chart shows both the size of each lever and what is left after all of them.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    names = list(levers.keys())
    vals = [levers[n] for n in names]
    if baseline is None:
        baseline = sum(vals)
    if optimized is None:
        optimized = baseline - sum(vals)

    labels = ["Baseline"] + names + ["Optimized"]
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(0, baseline, color="#8c3b3b", width=0.62)
    ax.text(0, baseline, f"\\${baseline:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    running = baseline
    for i, (n, v) in enumerate(zip(names, vals), start=1):
        bottom = running - v
        ax.bar(i, v, bottom=bottom, color="#2e548a", width=0.62)
        ax.plot([i - 0.31 - 0.38, i - 0.31], [running, running], color="#999", lw=0.9, ls="--")
        ax.text(i, running, f"−\\${v:,.0f}", ha="center", va="bottom", fontsize=9, color="#2e548a")
        running = bottom

    ax.bar(len(names) + 1, optimized, color="#2e7d4f", width=0.62)
    ax.text(len(names) + 1, optimized, f"\\${optimized:,.0f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    pct = (baseline - optimized) / baseline * 100 if baseline else 0.0
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax.set_ylabel("Chi phí GPU (USD / tháng)")
    ax.set_title(f"Chi phí GPU NimbusAI: \\${baseline:,.0f} → \\${optimized:,.0f} mỗi tháng  (tiết kiệm {pct:.0f}%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

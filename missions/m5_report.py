"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling

DAYS = 30
MEDIAN_QUERY_TOKENS = 800


def _lie_section(r1, cat) -> str:
    """Cơ chế của "GPU-Util lie", kèm giá phải trả."""
    lies = sorted(r1["lies"], key=lambda x: x["mfu"])
    if not lies:
        return "Cửa sổ quan sát này không phát hiện util-lie nào."
    worst = lies[0]
    same_type = [s for s in r1["summary"] if s["gpu_type"] == worst["gpu_type"]]
    healthy = max(same_type, key=lambda s: s["mfu"])
    price = num(cat[worst["gpu_type"]]["on_demand_hr"])
    monthly = price * 24 * DAYS
    ratio = healthy["mfu"] / worst["mfu"] if worst["mfu"] else 0.0
    return f"""`nvidia-smi` định nghĩa GPU-Util là *tỷ lệ cửa sổ lấy mẫu có ÍT NHẤT MỘT kernel đang
nằm trên thiết bị*. Một kernel đứng chờ đọc HBM suốt vòng đời của nó vẫn "đang nằm trên
thiết bị" đủ 100% thời gian, nên nó ghi ra đúng con số giống hệt một kernel làm bão hoà
tensor core. Bộ đếm này đo **thời gian chiếm clock**, không đo lượng việc làm được. Đó là lý
do nó nói dối — và nói dối theo hướng luôn có lợi cho hoá đơn của nhà cung cấp.

Cụ thể ở đây: `{worst['gpu_id']}` đọc ra **{worst['gpu_util_pct']}% util nhưng MFU chỉ
{worst['mfu']}**. Arithmetic intensity đo được là **{worst['intensity']} FLOP/byte**, nằm dưới
ridge point {worst['ridge']} của {worst['gpu_type']} — tức workload đang *memory-bound*: tensor
core ngồi không giữa các lần nạp toán hạng, trong khi clock vẫn chạy và đồng hồ tính tiền vẫn
quay. Ba nguyên nhân gốc đáng kiểm tra theo thứ tự: batch size quá nhỏ để khấu hao chi phí nạp
trọng số, các kernel elementwise không được fuse nên activation phải đi vòng qua HBM, và
launch overhead trên chuỗi kernel ngắn.

**Cái giá của nó.** `{worst['gpu_id']}` và `{healthy['gpu_id']}` là hai con {worst['gpu_type']}
y hệt nhau, cùng thuê giá ${price:.2f}/giờ — **${monthly:,.0f}/tháng mỗi con**. Một con đạt MFU
{healthy['mfu']}, con kia {worst['mfu']}: **đắt gấp {ratio:.1f} lần trên mỗi đơn vị compute thật
sự làm ra**, mà trên hoá đơn thì hai dòng giống hệt nhau. Không một dashboard chi phí nào xây
trên $/GPU-hr nhìn thấy được điều này — nó chỉ hiện ra khi mẫu số là **$/1M-token** hoặc khi
MFU/MBU được đưa vào chính báo cáo chi phí."""


def _actions_section(levers, r1, r2, r3, r6) -> str:
    reasoning = r2["reasoning"]
    cap = reasoning["cap_half"]
    rows = [
        ("1", "Cascade routing — định tuyến traffic dễ sang model nhỏ",
         f"${r2['levers'][1]['marginal_monthly']:,.0f}", "Thấp", "Thấp — cần cổng chất lượng cho router",
         f"Vài ngày. Một thay đổi routing gánh {r2['levers'][1]['cum_savings_pct']:.0f}/{r2['savings_pct']:.0f} điểm phần trăm tiết kiệm inference."),
        ("2", "Purchasing — spot cho job gián đoạn được, reserved cho job 24/7",
         f"${levers['Purchasing (spot/reserved)']:,.0f}", "Trung bình", "Trung bình — spot bắt buộc phải checkpoint",
         "Vài tuần. Con số tuyệt đối lớn nhất, nhưng cam kết 3 năm là thứ khó rút lại nhất."),
        ("3", "Tắt GPU nhàn rỗi (scale-to-zero ban đêm)",
         f"${levers['Tắt GPU nhàn rỗi']:,.0f}", "Thấp", "Thấp", "Vài ngày. Lãng phí thuần tuý, không có đánh đổi nào để tranh luận."),
        ("4", "Right-size GPU dư thừa — theo MBU, không theo $/GPU-hr",
         f"${levers['Right-size GPU dư thừa']:,.0f}", "Trung bình", "Trung bình — phải benchmark lại",
         "Vài tuần. Làm SAU khi sửa MFU: sửa được kernel thì có thể không cần đổi máy nữa."),
        ("5", "Batch API cho traffic chịu được độ trễ",
         f"${r2['levers'][3]['marginal_monthly']:,.0f}", "Thấp", "Thấp", "Vài ngày, nhưng chỉ traffic eval đủ điều kiện."),
        ("6", "Prompt cache với TTL 1 giờ",
         f"${r2['levers'][2]['marginal_monthly']:,.0f}", "Thấp", "Thấp", "Vài ngày. Nhỏ, vì cascade đã kéo giá input xuống trước rồi."),
        ("7", "Ngân sách reasoning — giảm một nửa traffic extended-thinking",
         f"${cap['cost_saved_monthly']:,.0f}", "Trung bình", "Cao — đây là quyết định về chất lượng",
         f"Rẻ về tiền, nhưng đáng giá {cap['wh_saved_daily']:,.0f} Wh/ngày về năng lượng."),
    ]
    head = "| # | Hành động | $/tháng | Công sức | Rủi ro | Vì sao xếp ở đây |\n|---|---|---:|---|---|---|\n"
    body = "\n".join(f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in rows)
    return f"""Thứ tự dưới đây xếp theo **lợi ích trên mỗi đơn vị công sức và khả năng đảo ngược**,
không xếp theo độ lớn. Hai lever dễ rút lại nhất (cascade, tắt GPU nhàn rỗi) đứng trước chính
vì lever lớn nhất (cam kết reserved 3 năm) là thứ không lấy lại được nếu traffic đổi hướng.

{head}{body}

**Điểm mấu chốt của thứ tự này: sửa hiệu quả TRƯỚC khi mua.** Right-size và cam kết dung lượng
đều là hành động *đóng băng mức hiệu quả hôm nay vào hoá đơn* — một reservation 3 năm ký trên
nền một job đang chạy MFU 0.19 là khoá chặt phần lãng phí đó trong ba năm. Bước 1 và 3 đảo
ngược được trong một buổi chiều; bước 2 là một hợp đồng."""


def _extensions_section(r1, r2, r3, r6) -> str:
    cache = r2["cache"]
    rz = r2["reasoning"]
    grid = r3["policy_grid"]
    rs = r1["rightsize"]
    any_team = sorted(cache)[0]
    be5 = cache[any_team]["tiers"]["5min"]["break_even_reads"]
    be1 = cache[any_team]["tiers"]["1hour"]["break_even_reads"]
    worst = min(cache.items(), key=lambda kv: kv[1]["tiers"]["5min"]["avg_reads"])
    balanced = next(r for r in r6["regions"] if r["region"] == r6["best_balanced"])
    rs_lines = "\n".join(
        f"| `{r['gpu_id']}` | {r['from']} → {r['to']} | {r['mfu']} / {r['mbu']} | "
        f"{r['need_bw_tbs']} TB/s, {r['need_vram_gb']:.0f} GB | ${r['from_hr']}/h → ${r['to_hr']}/h | "
        f"${r['monthly_saving']:,.0f} |"
        for r in rs)
    return f"""### 1 — Chính sách chọn tier có định giá rủi ro thu hồi (`pricing.recommend_tier` / `recommend_plan`)

Ba thứ chính sách v1 làm sai, và cả ba đều hiện ra bằng tiền:

* **Tỷ lệ thu hồi spot phụ thuộc loại GPU, không phải một hằng số.** Spot bị đòi lại theo mức
  khan hiếm: B200 12%/giờ, xuống tới L4 1,5%/giờ. Ở mức 8%/giờ của H100, ta trả **1,07 giờ hoá
  đơn cho 1 giờ việc hữu ích** (overhead checkpoint + phần phải làm lại). Mức đó vẫn nằm dưới
  chiết khấu spot — nhưng bây giờ chính sách *kiểm tra* điều đó thay vì mặc định cho là đúng.
* **Reserved bị tính tiền 24×30 giờ dù có dùng hay không.** Nên phép so sánh phải đặt trên duty
  cycle **theo tháng**, không phải hours/24. Một job chạy 20h/ngày nhưng chỉ 14 ngày/tháng chiếm
  39% của tháng, không phải 83%.
* **Giá 3 năm chỉ rẻ nếu thật sự tiêu thụ đủ ba năm.** Ngưỡng hoà vốn theo xác suất workload còn
  sống p\\* = r3/r1 đi qua mô hình mắc kẹt vốn: **H100 40%, A100 43%, A10G/L4/H200/MI300X 50%,
  B200 52%**. Dưới ngưỡng tự tin đó thì giá 1 năm mới là giá *kỳ vọng* rẻ hơn, dù giá niêm yết
  cao hơn.

**Đo được:** trên 8 job của fleet này, hai chính sách chọn cùng tier, nên tổng không đổi:
${r3['optimized_monthly']:,.0f}/tháng ({r3['savings_pct']}% so với on-demand) — một kết quả âm
tính, và nó được báo cáo trung thực. Chỗ khác biệt chỉ lộ ra trên lưới rộng hơn: quét 7 loại GPU
× 4 mức duty × 2 lịch chạy × interruptible = {grid['cells']} ô, **v1 chọn khác v2 ở
{len(grid['disagreements'])} ô và chi vượt ${grid['delta']:,.0f}/tháng**. Mọi ô lệch đều cùng một
hình dạng: v1 cam kết reserved cho job chỉ chạy nửa số ngày trong tháng.

### 2 — Right-sizing theo MBU thay vì theo $/GPU-hr (`metrics.rightsize_candidates`)

Cổng lọc trước: một GPU chỉ bị coi là dư thừa khi nó lãng phí **cả hai** thứ — FLOPs (MFU < 0,30)
*và* băng thông (MBU < 0,50). Sau đó mới chọn máy thay thế theo **nhu cầu đo được + 15% headroom**:
băng thông đạt đỉnh, VRAM thường trú đỉnh, TFLOPs đạt đỉnh — không bao giờ theo spec sheet của con
máy đang dùng.

| GPU | Chuyển | MFU / MBU | Nhu cầu đo được | Giá thuê | $/tháng |
|---|---|---|---|---|---:|
{rs_lines}
| | | | | **Tổng** | **${r1['rightsize_monthly']:,.0f}** |

**Tại sao không chọn thẳng con rẻ nhất theo $/GPU-hr?** Vì L4 là card rẻ nhất catalog ($0,80/h) và
nó không chạy nổi bất kỳ workload nào ở đây: 0,30 TB/s so với nhu cầu đo được 1,04 TB/s của
`gpu-h100-4`, và 24 GB so với working set 77 GB. Đưa việc sang đó thì hoặc OOM, hoặc phục vụ ở một
phần ba throughput — tức **$/GPU-hr giảm nhưng $/1M-token tăng**, đúng cái nghịch lý mà cả lab này
nói về. Hai mẫu số đúng là **$/TB-s** cho decode và **$/GB-VRAM** cho working set: MI300X là băng
thông rẻ nhất catalog ($0,368/TB-s) dù giá giờ cao hơn A100 ($0,895/TB-s).

Lưu ý cách đọc bảng: **$/TB-s dùng để xếp hạng catalog, còn quyết định cuối là "con RẺ NHẤT vượt
qua được ràng buộc đo được"**. Vì thế `gpu-h100-4` được xếp sang A100 (đắt hơn theo $/TB-s nhưng
vẫn thừa sức cho 1,04 TB/s và rẻ hơn $0,71/giờ), trong khi `gpu-h100-5` cần 1,77 TB/s nên A100
trượt và MI300X thắng. Ràng buộc là cổng cứng; tỷ số chỉ là công cụ sàng lọc.

Tiết kiệm chỉ được tính trên **số giờ thật sự bị tính tiền** — những giờ GPU vốn đã nhàn rỗi thuộc
về lever "tắt GPU nhàn rỗi", tính hai lần là thổi phồng báo cáo.

### 3 — `cache_is_worth_it()`: cache không miễn phí (`pricing.cache_is_worth_it`)

Ghi một prefix vào cache bị tính **cao hơn** giá input thường; chỉ những lần đọc lại sau đó mới trả
lại khoản chênh đó:

> số lần đọc hoà vốn = (write_multiplier − 1) / (1 − read_discount)

**Tier 5 phút** (ghi 1,25×, đọc 0,10×): **{be5} lần đọc**. **Tier 1 giờ** (ghi 2,00×, đọc 0,10×):
**{be1} lần đọc**. Ngưỡng này *giống nhau* cho model nhỏ và model lớn — nó là một tỷ số nên giá bị
triệt tiêu — nhưng **số tiền đặt cược thì không**: 1M token prefix được cache đáng giá $2,70 ở tier
large và $0,18 ở tier small, nên cấu hình cache sai trên tuyến frontier đắt gấp ~15 lần.

Tỷ lệ đọc lại được đo từ chính timestamp thật: sắp xếp request có phần cacheable của từng team, mỗi
khoảng cách đến request kế tiếp dài hơn TTL sẽ buộc phải ghi lại một lần với giá premium.

| Team | Request cacheable | Số lần ghi @5ph | Đọc/ghi @5ph | Tier tốt nhất | Hệ số thực | Bật cache? |
|---|---:|---:|---:|---|---:|---|
""" + "\n".join(
        f"| {t} | {c['requests']} | {c['tiers']['5min']['writes']} | {c['tiers']['5min']['avg_reads']} | "
        f"{c['best_tier']} | {c['effective_discount']:.4f} | {'có' if c['enabled'] else 'không'} |"
        for t, c in sorted(cache.items())) + f"""

**Dataset của chúng ta có đạt ngưỡng không?** Vượt rất xa — tuyến mỏng nhất (`{worst[0]}`) vẫn đạt
{worst[1]['tiers']['5min']['avg_reads']} lần đọc trên mỗi lần ghi, so với ngưỡng {be5}. Kết quả thú
vị nằm ở TTL: **tier 1 giờ thắng ở cả 4 team dù ghi đắt hơn 60%**, vì nó bóp số lần ghi xuống
(ví dụ `assistant`: {cache['assistant']['tiers']['5min']['writes']} lần ghi →
{cache['assistant']['tiers']['1hour']['writes']} lần). M2 tính đúng hệ số đo được này thay vì giả
định phẳng 0,10× — và đó là lý do lever cache trong sổ chỉ đáng
${r2['levers'][2]['marginal_monthly']:,.0f}/tháng chứ không phải con số quảng cáo "−90%".

### 4 — Ngân sách reasoning

| | Reasoning | Thường |
|---|---:|---:|
| Tỷ lệ request | {rz['share_requests']:.1%} | {1 - rz['share_requests']:.1%} |
| Tỷ lệ token | {rz['share_tokens']:.1%} | {1 - rz['share_tokens']:.1%} |
| Tỷ lệ chi phí | {rz['share_cost']:.1%} | {1 - rz['share_cost']:.1%} |
| Tỷ lệ năng lượng | {rz['share_energy']:.1%} | {1 - rz['share_energy']:.1%} |
| Output token trung bình | {rz['avg_output_tokens']['reasoning']:.0f} | {rz['avg_output_tokens']['normal']:.0f} |
| $ mỗi request | ${rz['cost_per_request']['reasoning']:.5f} | ${rz['cost_per_request']['normal']:.5f} |
| Wh mỗi request | {rz['wh_per_request']['reasoning']:.1f} | {rz['wh_per_request']['normal']:.3f} |

**Vì sao tốn năng lượng gấp ~80 lần.** Extended thinking trả giá hai lần. Thứ nhất, nó sinh ra rất
nhiều token hơn — {rz['avg_output_tokens']['reasoning']:.0f} so với
{rz['avg_output_tokens']['normal']:.0f} output token ở đây, gấp
{rz['avg_output_tokens']['reasoning'] / max(rz['avg_output_tokens']['normal'], 1):.0f} lần — và mỗi
token là một bước decode tự hồi quy phải đọc lại toàn bộ trọng số cùng KV cache từ HBM. Thứ hai,
đám token đó do model cỡ frontier sinh ra ở batch size nhỏ (vì ràng buộc độ trễ), nên mỗi bước đều
memory-bound và đốt trọn một GPU-giây cho đúng một token. Số token nhân lên, đồng thời năng lượng
trên mỗi token cũng tăng — nên hai hệ số nhân với nhau.

**Quy tắc routing đề xuất.** Reasoning chiếm {rz['share_requests']:.1%} request nhưng
{rz['share_cost']:.1%} chi phí và {rz['share_energy']:.1%} năng lượng. Trần 10% **không có hiệu
lực** — traffic hiện đã nằm dưới ngưỡng đó, và đây là kết quả trung thực: đặt trần 10% tiết kiệm
đúng $0. Lever thật sự có hiệu lực là **lọc theo điều kiện kích hoạt**: chỉ bật extended thinking
khi có tín hiệu đo được (truy hồi nhiều bước, code sinh ra phải chạy được, hoặc câu trả lời lượt
đầu bị verifier bác), tuyệt đối không bật mặc định cho toàn bộ traffic của một team. Giảm một nửa
lượng reasoning theo quy tắc đó tiết kiệm ${rz['cap_half']['cost_saved_monthly']:,.0f}/tháng và
**{rz['cap_half']['wh_saved_daily']:,.0f} Wh/ngày** — phần tiền là nhiễu, phần năng lượng thì không.

*Cảnh báo về mô hình:* hệ số {sustainability.REASONING_ENERGY_MULTIPLIER:.0f}× trong
`sustainability.py` được nhân **chồng lên** phần token đã tăng sẵn, nên con số
{rz['share_energy']:.1%} là **cận trên**. Hướng và bậc độ lớn thì đúng; muốn con số chính xác phải
đo bằng công tơ ở mức từng token.

### 5 — Lập lịch theo carbon (`missions/m6_carbon_scheduling.py`)

{r6['movable_kwh']:,.0f} kWh/tháng của fleet là training gián đoạn được — không có người dùng nào
đang chờ một round trip, lại vốn đã checkpoint sẵn, nên đây là phần **di chuyển được**. Phần còn
lại {r6['fixed_kwh']:,.0f} kWh/tháng là inference phục vụ người dùng, bị ghim tại chỗ bởi độ trễ.

| Vùng | $/kWh | gCO2/kWh | Tiền điện | Carbon (kg) | kg tiết kiệm | Độ trễ |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['region']} | {r['usd_per_kwh']} | {r['gco2_per_kwh']} | ${r['energy_cost_usd']:,.0f} | "
        f"{r['carbon_kg']:,.0f} | {r['carbon_saved_kg']:,.0f} | {r['latency_penalty_ms']:+d} ms |"
        for r in sorted(r6["regions"], key=lambda x: x["gco2_per_kwh"])) + f"""

**"Tối ưu" là vùng nào thì phụ thuộc công ty đang tối ưu cái gì.** Điện rẻ nhất:
**{r6['best_cost']}**. Lưới sạch nhất: **{r6['best_carbon']}** với 30 gCO2/kWh. Theo điểm cân bằng
50/50 đã chuẩn hoá min-max thì lựa chọn là **{r6['best_balanced']}** — 90 gCO2/kWh ở mức giá thấp
nhất bảng, đổi lấy {balanced['latency_penalty_ms']:+d} ms độ trễ. Chuyển toàn bộ phần di chuyển
được sang {r6['best_carbon']} tránh được **{r6['carbon_saved_kg']:,.0f} kgCO2e/tháng
({r6['carbon_saved_pct']:.0f}% dấu chân của khối tải đó)** với cái giá {r6['latency_penalty_ms']:+d}
ms mà **không người dùng nào cảm nhận được**, bởi đúng theo định nghĩa ta không hề dịch chuyển thứ
gì có người ngồi chờ.

Cái bẫy nằm ở `europe-central2`: nó là vùng châu Âu *gần nhất* nhưng cũng là lưới bẩn nhất bảng
(660 gCO2/kWh). Chọn theo phản xạ "gần thì tốt" sẽ **tăng 74% phát thải** cho khối tải này, đồng
thời trả tiền điện cao nhất — cùng lúc thua ở cả hai trục.

**Nối carbon với tiền thật:** trên hợp đồng thuê neocloud, tiền điện đã nằm sẵn trong $/GPU-hr, nên
cột "tiền điện" ở trên là *thành phần vật lý* của chi phí chứ không phải hoá đơn thứ hai. Điều đó
làm carbon trở thành trục quyết định độc lập duy nhất ở đây — và may mắn là hai trục không xung đột:
{r6['best_balanced']} vừa là điện rẻ nhất vừa là lưới sạch thứ nhì, tức chọn đúng vùng thì tiết kiệm
carbon **không tốn thêm đồng nào**."""


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r6 = m6_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    # EXT-2 replaces the old flat "one tier down" guess with a bandwidth- and
    # capacity-checked move, charged on billable hours only.
    rightsize_savings = r1["rightsize_monthly"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size GPU dư thừa": round(rightsize_savings),
        "Tắt GPU nhàn rỗi": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    wh = sustainability.wh_per_query(MEDIAN_QUERY_TOKENS)
    home_c = sustainability.carbon_g(wh, "us-east-1")
    clean_c = sustainability.carbon_g(wh, r6["best_carbon"])
    sust = {
        "wh_per_query": wh,
        "carbon_g": home_c,
        "best_region": f"{r6['best_balanced']} (${sustainability.REGION_PRICE_KWH[r6['best_balanced']]}/kWh, "
                       f"{sustainability.REGION_CARBON[r6['best_balanced']]} gCO2/kWh)",
        "notes": [
            f"Lưới sạch nhất: {r6['best_carbon']} ({sustainability.REGION_CARBON[r6['best_carbon']]} gCO2/kWh) — "
            f"cùng truy vấn đó giảm còn {clean_c:.3f} gCO2e, tức cắt {(1 - clean_c / home_c) * 100:.0f}%",
            f"Điện rẻ nhất: {r6['best_cost']} (${sustainability.REGION_PRICE_KWH[r6['best_cost']]}/kWh), "
            f"so với ${sustainability.REGION_PRICE_KWH['us-east-1']}/kWh ở us-east-1",
            f"Truy vấn reasoning: {sustainability.wh_per_query(MEDIAN_QUERY_TOKENS, is_reasoning=True):,.0f} Wh mỗi lần "
            f"({sustainability.REASONING_ENERGY_MULTIPLIER:.0f}×) — {r2['reasoning']['share_requests']:.1%} số request "
            f"nhưng {r2['reasoning']['share_energy']:.1%} năng lượng inference của fleet",
            f"Tải training di chuyển được: {r6['movable_kwh']:,.0f} kWh/tháng → tránh được "
            f"{r6['carbon_saved_kg']:,.0f} kgCO2e/tháng nếu lập lịch ở {r6['best_carbon']}",
        ],
    }

    summary = f"""**${baseline:,.0f} → ${optimized:,.0f} mỗi tháng (giảm {total_pct:.0f}%)**, và đơn vị đo
thật sự quan trọng cũng đi theo: inference giảm từ **${r2['baseline_per_m']:.3f} xuống
${r2['optimized_per_m']:.3f} trên mỗi 1M token ({r2['savings_pct']}%)**.

* Lever lớn nhất **không phải một khoản chiết khấu** — nó là **định tuyến**. Đẩy ~80% traffic vốn
  chưa bao giờ cần tới model frontier sang model nhỏ mang lại
  {r2['levers'][1]['cum_savings_pct']:.1f} trong tổng số {r2['savings_pct']:.1f} điểm phần trăm
  tiết kiệm inference. Cache và batch cộng lại chỉ thêm
  {r2['savings_pct'] - r2['levers'][1]['cum_savings_pct']:.1f} điểm.
* **${levers['Purchasing (spot/reserved)']:,.0f}/tháng** đến từ việc khớp tier mua với duty cycle và
  khả năng gián đoạn — con số tuyệt đối lớn nhất, đồng thời khó đảo ngược nhất.
* **{len(r1['lies'])} GPU đang bị tính tiền đầy đủ trong khi chỉ làm được một phần công việc**
  (`{sorted(r1['lies'], key=lambda x: x['mfu'])[0]['gpu_id']}`:
  {sorted(r1['lies'], key=lambda x: x['mfu'])[0]['gpu_util_pct']}% util, MFU
  {sorted(r1['lies'], key=lambda x: x['mfu'])[0]['mfu']}). Không dashboard $/GPU-hr nào thấy được.
* **${levers['Tắt GPU nhàn rỗi']:,.0f}/tháng là lãng phí thuần tuý** — một GPU bị bỏ chạy qua đêm
  sau khi training đã xong. Không có đánh đổi nào ở đây, chỉ là không ai tắt nó.
* Chỗ mọi tỷ lệ vỡ ra là năng lượng: traffic reasoning chiếm {r2['reasoning']['share_requests']:.1%}
  số request và {r2['reasoning']['share_cost']:.1%} chi phí, nhưng
  **{r2['reasoning']['share_energy']:.1%} năng lượng inference**.

> **Bốn lever, một câu:** định tuyến trước, tắt cái không dùng, sửa hiệu quả, rồi mới ký hợp đồng
> mua dài hạn — vì ba việc đầu đảo ngược được, việc thứ tư thì không."""

    caveats = f"""* **Hai rổ chi phí trong báo cáo này tách rời nhau theo đúng thiết kế dữ liệu.**
  `gpu_telemetry.csv` (11 GPU) và `workloads.csv` (8 job) là hai lát cắt riêng của lab. Lever idle
  và right-size được định giá từ telemetry; lever purchasing từ danh mục workload. Chúng không đếm
  trùng nhau, nhưng cũng không đối chiếu về cùng một hoá đơn — trên fleet thật phải join hai nguồn
  theo instance id trước khi công bố một con số tổng.
* **Right-size chỉ tính trên số giờ bị tính tiền** ({r1['rightsize'][0]['billable_hours']:,} giờ với
  một GPU chạy 24/7), nên lever idle và lever right-size không thể cùng đòi một giờ.
* **Con số "sau tối ưu" giả định các lever cộng dồn sạch.** Phần lớn là đúng — routing, purchasing
  và idle tác động lên những phần khác nhau của hoá đơn — nhưng một GPU vừa được right-size vừa
  chuyển sang spot sẽ không tiết kiệm trọn vẹn cả hai khoản.
* **Giá là snapshot tháng 6/2026**, mà giá GPU đổi hàng tháng; tỷ lệ thu hồi spot là số minh hoạ
  theo nhóm khan hiếm, không phải số đo trên một nhà cung cấp cụ thể.
* **Hệ số năng lượng reasoning được nhân chồng lên phần token đã tăng**, nên tỷ lệ năng lượng là
  cận trên (xem phần mở rộng 4).
* **Dữ liệu là tổng hợp và có seed cố định** (`data/generate.py`, seed 25) — phương pháp thì chuyển
  giao được, còn các con số đô-la cụ thể thì không."""

    sections = [
        {"title": "Vì sao GPU-Util là con số nói dối — và nó tốn bao nhiêu", "body": _lie_section(r1, cat)},
        {"title": "Hành động đề xuất, theo thứ tự ưu tiên", "body": _actions_section(levers, r1, r2, r3, r6)},
        {"title": 'Phần mở rộng "Your Turn" — kết quả đo được', "body": _extensions_section(r1, r2, r3, r6)},
        {"title": "Cách đọc các con số này", "body": caveats},
    ]

    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             unit_levers=r2["levers"], sections=sections,
                             lead={"body": summary})
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline=baseline, optimized=optimized)

    if verbose:
        print("== M5 Optimization Report ==")
        print(f"baseline  ${baseline:,.0f}/month  ->  optimized ${optimized:,.0f}/month  ({total_pct:.1f}% saved)")
        for k, v in levers.items():
            print(f"  {k:36} ${v:>9,.0f}")
        print(f"\ninference unit cost: ${r2['baseline_per_m']:.3f} -> ${r2['optimized_per_m']:.3f} per 1M tokens")
        print(f"Written: outputs/report.md" + (" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()

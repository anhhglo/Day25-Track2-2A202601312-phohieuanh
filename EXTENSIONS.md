# Phần mở rộng "Your Turn" — bản đồ cho người chấm

Làm **cả 5/5** extension (rubric yêu cầu ≥2). Bảng dưới trỏ thẳng tới code, test và con số
đo được của từng phần. Toàn bộ chạy offline, không GPU / không API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python verify.py            # 11/11 checks passed
pytest -q                   # 33 passed (15 test gốc của lab + 18 test tự viết)
python missions/run_all.py  # M1 → M4, M6, rồi M5 sinh outputs/
```

Đầu ra: `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`,
`outputs/WRITEUP.md` (bài viết ngắn).

> **Không sửa file test gốc nào.** 15 test của lab giữ nguyên; test tự viết nằm riêng ở
> `tests/test_extensions.py`.

---

| # | Extension | Code | Test | Kết quả đo được |
|---|---|---|---|---|
| 1 | Cải thiện `recommend_tier()` | `finops/pricing.py`, `missions/m3_purchasing.py` | 6 test | v1 lệch v2 ở **21/112 ô**, chi vượt **$5.231/tháng** |
| 2 | Right-sizing theo MBU | `finops/metrics.py`, `missions/m1_efficiency_audit.py` | 3 test | **$1.063/tháng** trên 4 GPU |
| 3 | `cache_is_worth_it()` | `finops/pricing.py`, `missions/m2_inference_levers.py` | 4 test | ngưỡng **0,28 / 1,11 lần đọc**; tier 1 giờ thắng cả 4 team |
| 4 | Ngân sách reasoning | `missions/m2_inference_levers.py`, `missions/m5_report.py` | 2 test | 8,4% request → **16,4% tiền, 94% điện** |
| 5 | Lập lịch theo carbon | `finops/sustainability.py`, `missions/m6_carbon_scheduling.py` | 3 test | **720 kgCO2e/tháng** (92% dấu chân tải di chuyển được) |

Phân tích đầy đủ của cả 5 phần nằm trong `outputs/report.md`, mục
*Phần mở rộng "Your Turn" — kết quả đo được*.

---

## 1 — `recommend_tier()`: chính sách mua có định giá rủi ro

**`finops/pricing.py`** — `INTERRUPT_RATE_PER_HOUR`, `interrupt_rate_for()`,
`spot_effective_multiplier()`, `monthly_duty_cycle()`, `recommend_tier()` (v2),
`reserved_term_choice()`, `recommend_plan()`, `tier_matrix()`.
**`missions/m3_purchasing.py`** — `legacy_tier()` (giữ lại v1 để đối chứng), `policy_grid()`.

Ba yếu tố mới so với v1:

1. **Tỷ lệ thu hồi spot theo loại GPU** (B200 12%/h → L4 1,5%/h), quy ra "thuế gián đoạn":
   ở H100 phải trả 1,07 giờ hoá đơn cho 1 giờ việc hữu ích.
2. **Duty cycle theo THÁNG thay vì hours/24** — reserved bị tính 24×30 giờ dù dùng hay
   không, nên job 20h/ngày × 14 ngày chiếm 39% của tháng chứ không phải 83%.
3. **So sánh 1 năm vs 3 năm** qua xác suất workload còn sống: ngưỡng hoà vốn p\* = r3/r1
   đi qua mô hình mắc kẹt vốn → H100 40%, A100 43%, A10G/L4/H200/MI300X 50%, B200 52%.

**Savings thay đổi thế nào?** Trên 8 job của fleet, hai chính sách chọn *cùng* tier nên tổng
không đổi ($10.224/tháng, 38,2% so với on-demand) — báo cáo giữ nguyên kết quả âm tính này.
Chỗ khác biệt lộ ra khi quét lưới 7 GPU × 4 duty × 2 lịch × interruptible = 112 ô: **v1 chọn
khác ở 21 ô và chi vượt $5.231/tháng**, luôn cùng một dạng lỗi — cam kết reserved cho job
chỉ chạy nửa số ngày trong tháng.

## 2 — Right-sizing theo MBU

**`finops/metrics.py`** — `dollars_per_gb_vram()`, `dollars_per_tbs()`,
`rightsize_candidates()`. **`missions/m1_efficiency_audit.py`** — cổng lọc
`MFU_OVERPROVISIONED` / `MBU_OVERPROVISIONED`, bảng kinh tế catalog, danh sách máy bị loại
kèm lý do.

Chỉ right-size GPU lãng phí **cả** FLOPs (MFU < 0,30) **và** băng thông (MBU < 0,50) — một
H100 training ở MFU 0,42 đang làm đúng việc nó được thuê. Chọn máy thay theo nhu cầu **đo
được + 15% headroom** (băng thông đỉnh, VRAM đỉnh, TFLOPs đỉnh).

**Tại sao không chọn con rẻ nhất theo $/GPU-hr?** L4 rẻ nhất ($0,80/h) nhưng có 0,30 TB/s
so với nhu cầu 1,04 TB/s và 24 GB so với working set 77 GB → hoặc OOM, hoặc chạy ở 1/3
throughput, tức $/GPU-hr giảm mà $/1M-token tăng. Mẫu số đúng là **$/TB-s** cho decode và
**$/GB-VRAM** cho working set. Tiết kiệm chỉ tính trên giờ bị tính tiền để không đếm trùng
với lever "tắt GPU nhàn rỗi".

## 3 — `cache_is_worth_it()`

**`finops/pricing.py`** — `cache_break_even_reads()`, `cache_is_worth_it()`,
`cache_net_multiplier()`. **`missions/m2_inference_levers.py`** — `CACHE_TIERS`,
`analyze_cache()`, và cổng áp dụng trong vòng tính chi phí.

> số lần đọc hoà vốn = (write_multiplier − 1) / (1 − read_discount)

**Cần đọc lại bao nhiêu lần?** Tier 5 phút (ghi 1,25×) cần **0,28 lần**; tier 1 giờ
(ghi 2,00×) cần **1,11 lần**. Ngưỡng giống nhau cho model nhỏ và lớn vì nó là tỷ số — nhưng
số tiền đặt cược thì khác 15 lần.

**Dataset có đạt ngưỡng không?** Vượt xa. Đo từ timestamp thật (mỗi khoảng trống dài hơn TTL
buộc ghi lại): `assistant` 19,3 · `search` 11,1 · `rag` 7,2 · `eval` 4,3 lần đọc/lần ghi.
Kết quả đáng chú ý: **tier 1 giờ thắng cả 4 team dù ghi đắt hơn 60%**, vì nó bóp số lần ghi
của `assistant` từ 39 xuống 1. M2 tính đúng hệ số đo được thay vì giả định phẳng 0,10×.

## 4 — Ngân sách reasoning

**`missions/m2_inference_levers.py`** — tách $/Wh theo `is_reasoning`, kịch bản trần.
**`missions/m5_report.py`** — bảng so sánh + quy tắc routing trong báo cáo.

| | Reasoning | Thường |
|---|---:|---:|
| Request | 8,4% | 91,6% |
| Chi phí | 16,4% | 83,6% |
| Năng lượng | **94,0%** | 6,0% |
| Output token TB | 3.875 | 641 |

**Tại sao ~80× điện?** Trả giá hai lần: gấp 6 lần token, và mỗi token là một bước decode
memory-bound trên model frontier ở batch nhỏ. **Trần 10% không có hiệu lực** — traffic đã ở
dưới ngưỡng, đặt trần tiết kiệm $0; kết quả âm tính này được giữ nguyên. Lever có hiệu lực
là lọc theo điều kiện kích hoạt: giảm một nửa → $11/tháng và **14.734 Wh/ngày**.

## 5 — Lập lịch theo carbon

**`finops/sustainability.py`** — `REGION_LATENCY_MS`, `region_table()`, `best_region()`,
`job_energy_kwh()`. **`missions/m6_carbon_scheduling.py`** — mission mới, đã nối vào
`run_all.py` và vào `outputs/report.md`.

Chỉ 5 job `interruptible=1` (2.057 kWh/tháng) là di chuyển được; 1.918 kWh/tháng inference
bị ghim bởi độ trễ. Bảng đủ 5 vùng có $/kWh, gCO2/kWh, tiền điện thực, carbon thực, độ trễ.

**Vùng nào "tối ưu" thật sự?** Tuỳ công ty tối ưu cái gì: điện rẻ nhất `us-east-wa`, lưới
sạch nhất `europe-north1` (30 gCO2/kWh), cân bằng 50/50 chuẩn hoá min-max ra `us-east-wa`.
Chuyển toàn bộ tải di chuyển được sang `europe-north1` tránh **720 kgCO2e/tháng (92%)** với
+95 ms mà không người dùng nào cảm nhận, vì không có gì tương tác bị dịch chuyển. Bẫy:
`europe-central2` là vùng châu Âu *gần nhất* nhưng bẩn nhất (660 gCO2/kWh) — chọn theo phản
xạ "gần thì tốt" sẽ tăng 74% phát thải và trả tiền điện cao nhất.

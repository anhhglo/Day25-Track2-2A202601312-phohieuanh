# Hoá đơn GPU nói dối — và cách bắt được nó

*Bài viết ngắn nộp kèm Lab 25 — GPU FinOps · Track 2 · Day 25*
*Toàn bộ số liệu sinh từ `python missions/run_all.py`, dữ liệu seed cố định (seed 25), giá snapshot tháng 6/2026.*

---

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---:|---:|---:|
| Tổng chi phí GPU | **$18.005/tháng** | **$8.816/tháng** | **51,0%** |
| Đơn giá inference | **$6,488 /1M-token** | **$1,127 /1M-token** | **82,6%** |

Chia theo bốn lever: Purchasing $6.315 (69%) · Inference $1.211 (13%) · Right-size $1.063 (12%) ·
Tắt GPU nhàn rỗi $600 (7%).

## 2. Đòn bẩy nào đóng góp nhiều nhất, và tại sao

**Về tiền tuyệt đối, purchasing thắng ($6.315/tháng).** Nhưng con số dạy được nhiều hơn nằm ở
sổ chi tiết inference:

| Giai đoạn | $/1M-token | Tiết kiệm luỹ kế |
|---|---:|---:|
| Baseline (toàn bộ model lớn) | 6,488 | — |
| + Cascade (định tuyến nhỏ/lớn) | 1,523 | **76,5%** |
| + Prompt cache | 1,365 | 79,0% |
| + Batch API | 1,127 | 82,6% |

**Cascade một mình gánh 76,5 trong 82,6 điểm phần trăm**; cache và batch cộng lại chỉ thêm 6,1.
Tôi đã đoán ngược trước khi chạy số — cứ nghĩ batch (−50%) và cache (−90%) sẽ là hai lever chính.

Lý do rất đơn giản khi đã nhìn thấy: **chiết khấu tác động lên *giá*, định tuyến tác động lên
việc có phải trả cái giá đó hay không.** 80% traffic ở đây chưa bao giờ cần tới model frontier;
gửi nó tới đó rồi xin giảm 50% vẫn đắt hơn nhiều so với không gửi. Hệ quả về thứ tự làm việc:
**định tuyến trước, thương lượng chiết khấu sau.** Cũng vì cascade đã kéo giá input xuống trước,
lever cache còn lại chỉ đáng $36/tháng — 90% của một khoản đã nhỏ đi 15 lần thì vẫn là khoản nhỏ.

## 3. GPU-Util Lie: GPU nào, và mất bao nhiêu tiền

**`gpu-h100-4`: 98,2% GPU-Util nhưng MFU chỉ 0,194.** (Trình phát hiện bắt thêm `gpu-a10g-1`:
96,9% util, MFU 0,268.)

Cơ chế nằm ở định nghĩa: GPU-Util là *tỷ lệ cửa sổ lấy mẫu có ít nhất một kernel đang nằm trên
thiết bị*. Một kernel đứng chờ đọc HBM suốt vòng đời vẫn "đang nằm trên thiết bị" đủ 100% thời
gian, nên ghi ra con số giống hệt một kernel làm bão hoà tensor core. Bộ đếm đo **thời gian chiếm
clock**, không đo việc làm được. Arithmetic intensity đo được là 248 FLOP/byte, dưới ridge point
295 của H100 → workload memory-bound, tensor core ngồi không giữa các lần nạp toán hạng.

**Tác động tài chính:** `gpu-h100-4` và `gpu-h100-3` là hai con H100 giống hệt nhau, cùng thuê
$2,50/giờ, cùng **$1.800/tháng**. Một con đạt MFU 0,427, con kia 0,194 — **đắt gấp 2,2 lần trên
mỗi đơn vị compute thật sự làm ra**, mà trên hoá đơn hai dòng đó *y hệt nhau*. Chừng nào còn báo
cáo bằng $/GPU-hr, khác biệt 2,2 lần này vô hình theo đúng nghĩa toán học.

## 4. Năm phần mở rộng đã làm

**Ext 1 — `recommend_tier()` v2.** Thêm ba yếu tố: tỷ lệ thu hồi spot theo loại GPU (B200 12%/h →
L4 1,5%/h, quy ra "thuế gián đoạn" 1,07 giờ hoá đơn cho 1 giờ việc hữu ích ở H100); duty cycle
tính **theo tháng** vì reserved bị tính 24×30 giờ dù dùng hay không; và chọn 1yr vs 3yr qua ngưỡng
hoà vốn xác suất workload còn sống p\* = r3/r1 (H100 40%, A100 43%, B200 52%).
*Kết quả:* trên 8 job của fleet, hai chính sách chọn **cùng** tier → tổng không đổi. Tôi giữ nguyên
kết quả âm tính này. Khác biệt lộ ra trên lưới 112 ô: **v1 chọn sai ở 21 ô, chi vượt $5.231/tháng**,
luôn cùng một dạng — cam kết reserved cho job chỉ chạy nửa số ngày trong tháng.

**Ext 2 — Right-size theo MBU.** Chỉ đổi GPU khi lãng phí **cả** FLOPs (MFU<0,30) lẫn băng thông
(MBU<0,50), rồi chọn máy theo nhu cầu đo được +15% headroom. *Kết quả:* **$1.063/tháng** trên 4 GPU.

**Ext 3 — `cache_is_worth_it()`.** Ghi vào cache bị tính **cao hơn** giá input thường; ngưỡng hoà
vốn = (write_multiplier−1)/(1−read_discount) → **0,28 lần đọc** ở tier 5 phút, **1,11** ở tier 1 giờ.
*Kết quả:* đo từ timestamp thật, cả 4 team đều vượt xa (mỏng nhất là `eval` với 4,3 đọc/ghi). Nhưng
**tier 1 giờ thắng cả 4 team dù ghi đắt hơn 60%**, vì nó bóp số lần ghi của `assistant` từ 39 xuống 1.

**Ext 4 — Ngân sách reasoning.** *Kết quả:* 8,4% request → 16,4% tiền → **94% năng lượng**. Trần 10%
**không có hiệu lực** (traffic đã dưới ngưỡng, tiết kiệm đúng $0) — giữ nguyên kết quả âm tính này.

**Ext 5 — Lập lịch theo carbon.** Chỉ 5 job `interruptible=1` (2.057 kWh/tháng) là di chuyển được;
1.918 kWh/tháng inference bị ghim bởi độ trễ. *Kết quả:* chuyển sang `europe-north1` tránh
**720 kgCO2e/tháng (92%)** với +95 ms mà không người dùng nào cảm nhận. Bẫy: `europe-central2` là
vùng châu Âu *gần nhất* nhưng bẩn nhất (660 gCO2/kWh) — chọn theo phản xạ "gần thì tốt" tăng 74%
phát thải.

### Insight quan trọng nhất

**Ràng buộc kỹ thuật quyết định, tỷ số chỉ để sàng lọc.** L4 là card rẻ nhất catalog ($0,80/giờ) và
không chạy nổi workload nào cần thay máy: 0,30 TB/s so với nhu cầu 1,04 TB/s, 24 GB so với working
set 77 GB. Chọn nó thì **$/GPU-hr giảm còn $/1M-token tăng** — đúng nghịch lý cả lab này được xây để
dạy. Ngược lại MI300X đắt hơn A100 theo giờ nhưng là băng thông rẻ nhất catalog ($0,368 so với
$0,895 /TB-s). Quy trình đúng: **$/TB-s và $/GB-VRAM để xếp hạng, ràng buộc đo được để loại, rồi mới
lấy con rẻ nhất còn sống sót.**

## 5. Nếu tôi là FinOps lead của NimbusAI: ba hành động đầu tiên

**1. Đổi mẫu số của mọi báo cáo chi phí sang $/1M-token, và đưa MFU/MBU vào chính báo cáo đó.**
*Tuần 1, không tốn hạ tầng.* Đây phải là việc đầu tiên vì nó là điều kiện để nhìn thấy ba việc còn
lại. Chừng nào MFU nằm ở dashboard hạ tầng còn chi phí nằm ở dashboard tài chính thì `gpu-h100-4`
— 2,2 lần lãng phí trên một dòng hoá đơn bình thường — sẽ còn vô hình.

**2. Bật cascade routing, và tắt GPU nhàn rỗi.** *Tuần 1–2, $1.722/tháng.* Không phải vì lớn nhất,
mà vì **đảo ngược được trong một buổi chiều**. Cascade gánh 76,5% tiết kiệm inference; tắt GPU nhàn
rỗi là lãng phí thuần tuý không có đánh đổi nào để tranh luận. Hai việc rẻ nhất về mặt rủi ro nên
đi trước để lấy đà chính trị cho việc thứ ba.

**3. Sửa hiệu quả xong RỒI mới ký hợp đồng mua dài hạn.** *Tuần 3+, $6.315/tháng.* Purchasing là
con số lớn nhất nhưng tôi cố tình xếp cuối: một reservation 3 năm ký trên nền job đang chạy MFU
0,19 là **khoá chặt phần lãng phí đó trong ba năm**. Right-size và commit đều là hành động đóng
băng mức hiệu quả hôm nay vào hoá đơn. Và trước khi ký, dùng ngưỡng hoà vốn survival: dưới 40% tự
tin workload sống đủ 3 năm thì giá 1 năm mới là giá *kỳ vọng* rẻ hơn, dù giá niêm yết cao hơn.

> **Một câu tóm tắt cho ban lãnh đạo:** đổi mẫu số trước, làm việc rẻ và đảo ngược được trước, ký
> hợp đồng sau cùng — vì ba việc đầu sửa được, việc thứ tư thì không.

---

*Ghi chú về phương pháp: hai kết quả âm tính (trần reasoning 10% không có hiệu lực; hai chính sách
mua trùng nhau trên cả 8 job) được giữ nguyên trong báo cáo thay vì chỉnh ngưỡng cho ra số đẹp. Phần
caveat trong `outputs/report.md` nói rõ chỗ hai rổ chi phí — telemetry 11 GPU và catalogue 8 job —
không đối chiếu về cùng một hoá đơn.*

# Hoá đơn GPU nói dối, và bốn con số bắt được nó

*Bài viết ngắn nộp kèm Lab 25 — GPU FinOps · Track 2 · Day 25*

---

Chỗ tốn kém nhất trong hoá đơn GPU của NimbusAI không phải là giá thuê. Giá thuê thì ai
cũng nhìn thấy: $2,50/giờ cho một H100, nhân với số giờ, ra một con số ai cũng đọc được.
Chỗ tốn kém là **những thứ mà cách đo hiện tại về mặt cấu trúc không thể nhìn thấy** — và
sau khi chạy hết năm mission của lab này, tôi cho rằng đó mới là bài học đáng mang đi.

Con số cuối cùng: **$18.005 → $8.816 mỗi tháng, giảm 51%**. Nhưng con số làm tôi đổi cách
nghĩ không nằm ở đó.

## 1. `nvidia-smi` đo sai thứ, và nó sai theo hướng có lợi cho người bán

`gpu-h100-4` báo **98,2% GPU-Util**. Theo mọi dashboard hạ tầng thông thường, đó là một
con máy khoẻ mạnh, đang được khai thác tối đa, không có gì để tối ưu. MFU của nó là
**0,194**.

Nguyên nhân nằm ở định nghĩa: GPU-Util là *tỷ lệ cửa sổ lấy mẫu có ít nhất một kernel đang
nằm trên thiết bị*. Một kernel đứng chờ đọc HBM suốt vòng đời của nó vẫn "đang nằm trên
thiết bị" đủ 100% thời gian. Bộ đếm đo **thời gian chiếm clock**, không đo việc làm được.
Nó không hỏng — nó trả lời một câu hỏi khác với câu hỏi ta tưởng mình đang hỏi.

Cái giá thì rất cụ thể. `gpu-h100-4` và `gpu-h100-3` là hai con H100 giống hệt nhau, cùng
thuê $2,50/giờ, cùng **$1.800/tháng**. Một con đạt MFU 0,427, con kia 0,194 — **đắt gấp
2,2 lần trên mỗi đơn vị compute thật sự làm ra**. Trên hoá đơn, hai dòng đó *y hệt nhau*.

Không có mẹo tối ưu nào cứu được chuyện này, vì vấn đề không nằm ở mức tối ưu mà ở **mẫu
số**. Chừng nào còn báo cáo bằng $/GPU-hr, sự khác biệt 2,2 lần này là vô hình theo đúng
nghĩa toán học. Đổi mẫu số sang $/1M-token thì nó hiện ra ngay ở dòng đầu tiên.

## 2. Lever lớn nhất của inference không phải là chiết khấu

Đây là chỗ tôi đoán sai trước khi chạy số. Tôi nghĩ batch API (−50%) và prompt caching
(−90% trên phần cache) sẽ là hai lever chính. Sổ chi tiết theo $/1M-token nói ngược lại:

| Giai đoạn | $/1M-token | Tiết kiệm luỹ kế |
|---|---:|---:|
| Baseline (toàn bộ model lớn) | 6,488 | — |
| + Cascade (định tuyến nhỏ/lớn) | 1,523 | **76,5%** |
| + Prompt cache | 1,365 | 79,0% |
| + Batch API | 1,127 | 82,6% |

**Cascade một mình gánh 76,5 trong 82,6 điểm phần trăm.** Cache và batch cộng lại thêm
6,1 điểm. Lý do rất đơn giản khi đã nhìn thấy: chiết khấu tác động lên *giá*, còn định
tuyến tác động lên *việc có phải trả cái giá đó hay không*. 80% traffic ở đây chưa bao giờ
cần tới model frontier; gửi nó tới đó rồi xin giảm 50% vẫn đắt hơn nhiều so với không gửi.

Hệ quả về thứ tự làm việc: **định tuyến trước, thương lượng chiết khấu sau.** Chiết khấu là
thứ nhân với một con số; định tuyến là thứ quyết định con số ấy là bao nhiêu.

Cũng chính vì cascade đã kéo giá input xuống trước, lever cache còn lại chỉ đáng
**$36/tháng** — chứ không phải con số "−90%" trên tờ quảng cáo. Chiết khấu 90% trên một
khoản đã nhỏ đi 15 lần thì vẫn chỉ là 90% của một khoản nhỏ.

## 3. Cache không miễn phí, và ngưỡng hoà vốn là một phép chia

Phần mở rộng tôi thấy đắt giá nhất về mặt tư duy là `cache_is_worth_it()`. Ghi một prefix
vào cache bị tính **cao hơn** giá input thường (1,25× ở tier 5 phút, 2,00× ở tier 1 giờ);
chỉ những lần đọc lại mới trả khoản chênh đó về. Vậy:

> số lần đọc hoà vốn = (write_multiplier − 1) / (1 − read_discount)

Tier 5 phút cần **0,28 lần đọc**; tier 1 giờ cần **1,11 lần**. Một cache entry được ghi rồi
không bao giờ đọc lại thì **đắt hơn là không cache** — đó chính là cách một TTL 5 phút đặt
trên tuyến traffic thưa lặng lẽ làm mất tiền, trong khi mọi dashboard đều báo "cache đang
bật".

Đo trên dữ liệu thật của lab (dựng lại từ timestamp: mỗi khoảng trống dài hơn TTL buộc ghi
lại một lần), cả 4 team đều vượt ngưỡng rất xa — tuyến mỏng nhất là `eval` với 4,3 lần đọc
mỗi lần ghi. Nhưng kết quả thú vị là **tier 1 giờ thắng ở cả 4 team dù ghi đắt hơn 60%**,
vì nó bóp số lần ghi của team `assistant` từ 39 xuống còn 1. Ngưỡng hoà vốn cao hơn, nhưng
số lần phải trả ngưỡng đó giảm mạnh hơn.

## 4. Reasoning: 8% traffic, 16% tiền, 94% điện

Traffic reasoning chiếm **8,4% số request**, **16,4% chi phí** — và **94% năng lượng**
inference. Nó trả giá hai lần: sinh ra gấp 6 lần token (3.875 so với 641 output token), và
mỗi token đó là một bước decode memory-bound trên model cỡ frontier ở batch nhỏ. Hai hệ số
nhân với nhau.

Điều đáng nói: theo đề bài, trần 10% traffic **không có hiệu lực** — chúng tôi đã ở dưới
mức đó rồi, nên đặt trần tiết kiệm đúng $0. Tôi giữ nguyên kết quả âm tính này trong báo
cáo thay vì chỉnh ngưỡng cho ra một con số đẹp. Lever thật sự có hiệu lực là *lọc theo điều
kiện kích hoạt* — chỉ bật extended thinking khi có tín hiệu đo được — và giảm một nửa lượng
reasoning tiết kiệm $11/tháng cùng **14.734 Wh/ngày**. Phần tiền là nhiễu; phần điện thì
không. Đây là chỗ duy nhất trong cả lab mà **$ và Wh nói hai câu khác nhau**, và nếu chỉ
theo dõi $ thì sẽ không bao giờ thấy.

## 5. Ràng buộc kỹ thuật quyết định, tỷ số chỉ để sàng lọc

Bài học cuối đến từ phần right-sizing. L4 là card rẻ nhất catalog ($0,80/giờ). Nó không
chạy nổi bất kỳ workload nào trong danh sách cần thay máy: 0,30 TB/s so với nhu cầu đo được
1,04 TB/s, 24 GB so với working set 77 GB. Chọn nó thì **$/GPU-hr giảm còn $/1M-token
tăng** — đúng cái nghịch lý mà cả lab này được xây để dạy.

Ngược lại, MI300X đắt hơn A100 tính theo giờ ($1,95 so với $1,79) nhưng là **băng thông rẻ
nhất catalog** ($0,368/TB-s so với $0,895/TB-s). Với decode — vốn memory-bound — đó mới là
mẫu số đúng. Quy trình cuối cùng tôi dùng: **$/TB-s và $/GB-VRAM để xếp hạng, ràng buộc đo
được để loại, rồi mới lấy con rẻ nhất còn sống sót.** Tỷ số là công cụ sàng lọc; ràng buộc
là cổng cứng.

---

## Thứ tôi sẽ mang sang Milestone 2

1. **Không bao giờ báo cáo hạ tầng AI bằng $/GPU-hr.** Mẫu số phải là đơn vị công việc —
   $/1M-token, $/request, $/tài liệu xử lý. $/GPU-hr trả lời câu "ta đã thuê bao nhiêu",
   không trả lời câu "ta trả bao nhiêu cho mỗi thứ làm ra".
2. **Sửa hiệu quả trước khi ký hợp đồng.** Một reservation 3 năm ký trên nền job chạy MFU
   0,19 là khoá chặt phần lãng phí đó trong ba năm. Cascade và tắt GPU nhàn rỗi đảo ngược
   được trong một buổi chiều; cam kết dung lượng thì không.
3. **Đưa MFU/MBU vào chính báo cáo chi phí, không để riêng ở dashboard hạ tầng.** Chừng nào
   hai thứ đó còn nằm ở hai màn hình khác nhau thì `gpu-h100-4` sẽ còn vô hình.
4. **Báo cáo cả kết quả âm tính.** Trần reasoning 10% không tiết kiệm được gì; hai chính
   sách mua trùng nhau trên cả 8 job của fleet. Giữ nguyên những chỗ đó làm phần còn lại
   của báo cáo đáng tin hơn, chứ không kém đi.

*Toàn bộ số liệu sinh ra từ `python missions/run_all.py`; dữ liệu tổng hợp seed cố định
(seed 25). Giá là snapshot tháng 6/2026.*

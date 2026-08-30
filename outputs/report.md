# NimbusAI — Báo cáo tối ưu chi phí GPU

**Period / Kỳ báo cáo:** monthly  
**Baseline spend / Chi phí gốc:** $18,005  
**Optimized spend / Chi phí sau tối ưu:** $8,816  
**Projected savings / Tiết kiệm dự kiến:** $9,189  (**51%**)

## Tóm tắt điều hành

**$18,005 → $8,816 mỗi tháng (giảm 51%)**, và đơn vị đo
thật sự quan trọng cũng đi theo: inference giảm từ **$6.488 xuống
$1.127 trên mỗi 1M token (82.6%)**.

* Lever lớn nhất **không phải một khoản chiết khấu** — nó là **định tuyến**. Đẩy ~80% traffic vốn
  chưa bao giờ cần tới model frontier sang model nhỏ mang lại
  76.5 trong tổng số 82.6 điểm phần trăm
  tiết kiệm inference. Cache và batch cộng lại chỉ thêm
  6.1 điểm.
* **$6,315/tháng** đến từ việc khớp tier mua với duty cycle và
  khả năng gián đoạn — con số tuyệt đối lớn nhất, đồng thời khó đảo ngược nhất.
* **2 GPU đang bị tính tiền đầy đủ trong khi chỉ làm được một phần công việc**
  (`gpu-h100-4`:
  98.2% util, MFU
  0.194). Không dashboard $/GPU-hr nào thấy được.
* **$600/tháng là lãng phí thuần tuý** — một GPU bị bỏ chạy qua đêm
  sau khi training đã xong. Không có đánh đổi nào ở đây, chỉ là không ai tắt nó.
* Chỗ mọi tỷ lệ vỡ ra là năng lượng: traffic reasoning chiếm 8.4%
  số request và 16.4% chi phí, nhưng
  **94.0% năng lượng inference**.

> **Bốn lever, một câu:** định tuyến trước, tắt cái không dùng, sửa hiệu quả, rồi mới ký hợp đồng
> mua dài hạn — vì ba việc đầu đảo ngược được, việc thứ tư thì không.

## Tiết kiệm theo từng lever

| Lever | Tiết kiệm (USD/tháng) | Tỷ trọng |
|---|---:|---:|
| Inference (cascade/cache/batch) | $1,211 | 13% |
| Purchasing (spot/reserved) | $6,315 | 69% |
| Right-size GPU dư thừa | $1,063 | 12% |
| Tắt GPU nhàn rỗi | $600 | 7% |
| **Tổng** | **$9,189** | 100% |

## Đơn giá inference — $/1M-token theo từng lever

| Giai đoạn | $/ngày | $/1M-token | Δ $/1M-token | Δ $/tháng | Tiết kiệm luỹ kế |
|---|---:|---:|---:|---:|---:|
| Baseline — toàn bộ chạy model lớn, không cache, không batch | $48.87 | **6.488** | — | — | — |
| + Cascade — định tuyến nhỏ/lớn | $11.48 | **1.523** | −4.965 | $1,122 | 76.5% |
| + Prompt cache — có cổng theo tỷ lệ đọc lại | $10.28 | **1.365** | −0.158 | $36 | 79.0% |
| + Batch API — traffic chịu được độ trễ | $8.49 | **1.127** | −0.238 | $54 | 82.6% |

## Tính bền vững (Sustainability)

- Năng lượng mỗi truy vấn: 0.24 Wh
- Carbon mỗi truy vấn: 0.091 gCO2e
- Vùng tốt nhất (rẻ + sạch, cân bằng): us-east-wa ($0.055/kWh, 90 gCO2/kWh)
- Lưới sạch nhất: europe-north1 (30 gCO2/kWh) — cùng truy vấn đó giảm còn 0.007 gCO2e, tức cắt 92%
- Điện rẻ nhất: us-east-wa ($0.055/kWh), so với $0.12/kWh ở us-east-1
- Truy vấn reasoning: 19 Wh mỗi lần (80×) — 8.4% số request nhưng 94.0% năng lượng inference của fleet
- Tải training di chuyển được: 2,057 kWh/tháng → tránh được 720 kgCO2e/tháng nếu lập lịch ở europe-north1

## Vì sao GPU-Util là con số nói dối — và nó tốn bao nhiêu

`nvidia-smi` định nghĩa GPU-Util là *tỷ lệ cửa sổ lấy mẫu có ÍT NHẤT MỘT kernel đang
nằm trên thiết bị*. Một kernel đứng chờ đọc HBM suốt vòng đời của nó vẫn "đang nằm trên
thiết bị" đủ 100% thời gian, nên nó ghi ra đúng con số giống hệt một kernel làm bão hoà
tensor core. Bộ đếm này đo **thời gian chiếm clock**, không đo lượng việc làm được. Đó là lý
do nó nói dối — và nói dối theo hướng luôn có lợi cho hoá đơn của nhà cung cấp.

Cụ thể ở đây: `gpu-h100-4` đọc ra **98.2% util nhưng MFU chỉ
0.194**. Arithmetic intensity đo được là **248.3 FLOP/byte**, nằm dưới
ridge point 295.5 của H100 — tức workload đang *memory-bound*: tensor
core ngồi không giữa các lần nạp toán hạng, trong khi clock vẫn chạy và đồng hồ tính tiền vẫn
quay. Ba nguyên nhân gốc đáng kiểm tra theo thứ tự: batch size quá nhỏ để khấu hao chi phí nạp
trọng số, các kernel elementwise không được fuse nên activation phải đi vòng qua HBM, và
launch overhead trên chuỗi kernel ngắn.

**Cái giá của nó.** `gpu-h100-4` và `gpu-h100-3` là hai con H100
y hệt nhau, cùng thuê giá $2.50/giờ — **$1,800/tháng mỗi con**. Một con đạt MFU
0.427, con kia 0.194: **đắt gấp 2.2 lần trên mỗi đơn vị compute thật
sự làm ra**, mà trên hoá đơn thì hai dòng giống hệt nhau. Không một dashboard chi phí nào xây
trên $/GPU-hr nhìn thấy được điều này — nó chỉ hiện ra khi mẫu số là **$/1M-token** hoặc khi
MFU/MBU được đưa vào chính báo cáo chi phí.

## Hành động đề xuất, theo thứ tự ưu tiên

Thứ tự dưới đây xếp theo **lợi ích trên mỗi đơn vị công sức và khả năng đảo ngược**,
không xếp theo độ lớn. Hai lever dễ rút lại nhất (cascade, tắt GPU nhàn rỗi) đứng trước chính
vì lever lớn nhất (cam kết reserved 3 năm) là thứ không lấy lại được nếu traffic đổi hướng.

| # | Hành động | $/tháng | Công sức | Rủi ro | Vì sao xếp ở đây |
|---|---|---:|---|---|---|
| 1 | Cascade routing — định tuyến traffic dễ sang model nhỏ | $1,122 | Thấp | Thấp — cần cổng chất lượng cho router | Vài ngày. Một thay đổi routing gánh 76/83 điểm phần trăm tiết kiệm inference. |
| 2 | Purchasing — spot cho job gián đoạn được, reserved cho job 24/7 | $6,315 | Trung bình | Trung bình — spot bắt buộc phải checkpoint | Vài tuần. Con số tuyệt đối lớn nhất, nhưng cam kết 3 năm là thứ khó rút lại nhất. |
| 3 | Tắt GPU nhàn rỗi (scale-to-zero ban đêm) | $600 | Thấp | Thấp | Vài ngày. Lãng phí thuần tuý, không có đánh đổi nào để tranh luận. |
| 4 | Right-size GPU dư thừa — theo MBU, không theo $/GPU-hr | $1,063 | Trung bình | Trung bình — phải benchmark lại | Vài tuần. Làm SAU khi sửa MFU: sửa được kernel thì có thể không cần đổi máy nữa. |
| 5 | Batch API cho traffic chịu được độ trễ | $54 | Thấp | Thấp | Vài ngày, nhưng chỉ traffic eval đủ điều kiện. |
| 6 | Prompt cache với TTL 1 giờ | $36 | Thấp | Thấp | Vài ngày. Nhỏ, vì cascade đã kéo giá input xuống trước rồi. |
| 7 | Ngân sách reasoning — giảm một nửa traffic extended-thinking | $11 | Trung bình | Cao — đây là quyết định về chất lượng | Rẻ về tiền, nhưng đáng giá 14,734 Wh/ngày về năng lượng. |

**Điểm mấu chốt của thứ tự này: sửa hiệu quả TRƯỚC khi mua.** Right-size và cam kết dung lượng
đều là hành động *đóng băng mức hiệu quả hôm nay vào hoá đơn* — một reservation 3 năm ký trên
nền một job đang chạy MFU 0.19 là khoá chặt phần lãng phí đó trong ba năm. Bước 1 và 3 đảo
ngược được trong một buổi chiều; bước 2 là một hợp đồng.

## Phần mở rộng "Your Turn" — kết quả đo được

### 1 — Chính sách chọn tier có định giá rủi ro thu hồi (`pricing.recommend_tier` / `recommend_plan`)

Ba thứ chính sách v1 làm sai, và cả ba đều hiện ra bằng tiền:

* **Tỷ lệ thu hồi spot phụ thuộc loại GPU, không phải một hằng số.** Spot bị đòi lại theo mức
  khan hiếm: B200 12%/giờ, xuống tới L4 1,5%/giờ. Ở mức 8%/giờ của H100, ta trả **1,07 giờ hoá
  đơn cho 1 giờ việc hữu ích** (overhead checkpoint + phần phải làm lại). Mức đó vẫn nằm dưới
  chiết khấu spot — nhưng bây giờ chính sách *kiểm tra* điều đó thay vì mặc định cho là đúng.
* **Reserved bị tính tiền 24×30 giờ dù có dùng hay không.** Nên phép so sánh phải đặt trên duty
  cycle **theo tháng**, không phải hours/24. Một job chạy 20h/ngày nhưng chỉ 14 ngày/tháng chiếm
  39% của tháng, không phải 83%.
* **Giá 3 năm chỉ rẻ nếu thật sự tiêu thụ đủ ba năm.** Ngưỡng hoà vốn theo xác suất workload còn
  sống p\* = r3/r1 đi qua mô hình mắc kẹt vốn: **H100 40%, A100 43%, A10G/L4/H200/MI300X 50%,
  B200 52%**. Dưới ngưỡng tự tin đó thì giá 1 năm mới là giá *kỳ vọng* rẻ hơn, dù giá niêm yết
  cao hơn.

**Đo được:** trên 8 job của fleet này, hai chính sách chọn cùng tier, nên tổng không đổi:
$10,224/tháng (38.2% so với on-demand) — một kết quả âm
tính, và nó được báo cáo trung thực. Chỗ khác biệt chỉ lộ ra trên lưới rộng hơn: quét 7 loại GPU
× 4 mức duty × 2 lịch chạy × interruptible = 112 ô, **v1 chọn khác v2 ở
21 ô và chi vượt $5,231/tháng**. Mọi ô lệch đều cùng một
hình dạng: v1 cam kết reserved cho job chỉ chạy nửa số ngày trong tháng.

### 2 — Right-sizing theo MBU thay vì theo $/GPU-hr (`metrics.rightsize_candidates`)

Cổng lọc trước: một GPU chỉ bị coi là dư thừa khi nó lãng phí **cả hai** thứ — FLOPs (MFU < 0,30)
*và* băng thông (MBU < 0,50). Sau đó mới chọn máy thay thế theo **nhu cầu đo được + 15% headroom**:
băng thông đạt đỉnh, VRAM thường trú đỉnh, TFLOPs đạt đỉnh — không bao giờ theo spec sheet của con
máy đang dùng.

| GPU | Chuyển | MFU / MBU | Nhu cầu đo được | Giá thuê | $/tháng |
|---|---|---|---|---|---:|
| `gpu-h100-4` | H100 → A100 | 0.194 / 0.207 | 1.044 TB/s, 77 GB | $2.5/h → $1.79/h | $511 |
| `gpu-h100-5` | H100 → MI300X | 0.261 / 0.271 | 1.769 TB/s, 76 GB | $2.5/h → $1.95/h | $264 |
| `gpu-a10g-0` | A10G → L4 | 0.219 / 0.235 | 0.209 TB/s, 23 GB | $1.0/h → $0.8/h | $144 |
| `gpu-a10g-1` | A10G → L4 | 0.268 / 0.302 | 0.244 TB/s, 24 GB | $1.0/h → $0.8/h | $144 |
| | | | | **Tổng** | **$1,063** |

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

**Tier 5 phút** (ghi 1,25×, đọc 0,10×): **0.28 lần đọc**. **Tier 1 giờ** (ghi 2,00×, đọc 0,10×):
**1.11 lần đọc**. Ngưỡng này *giống nhau* cho model nhỏ và model lớn — nó là một tỷ số nên giá bị
triệt tiêu — nhưng **số tiền đặt cược thì không**: 1M token prefix được cache đáng giá $2,70 ở tier
large và $0,18 ở tier small, nên cấu hình cache sai trên tuyến frontier đắt gấp ~15 lần.

Tỷ lệ đọc lại được đo từ chính timestamp thật: sắp xếp request có phần cacheable của từng team, mỗi
khoảng cách đến request kế tiếp dài hơn TTL sẽ buộc phải ghi lại một lần với giá premium.

| Team | Request cacheable | Số lần ghi @5ph | Đọc/ghi @5ph | Tier tốt nhất | Hệ số thực | Bật cache? |
|---|---:|---:|---:|---|---:|---|
| assistant | 790 | 39 | 19.3 | 1hour | 0.1024 | có |
| eval | 415 | 79 | 4.3 | 1hour | 0.1046 | có |
| rag | 566 | 69 | 7.2 | 1hour | 0.1034 | có |
| search | 629 | 52 | 11.1 | 1hour | 0.1030 | có |

**Dataset của chúng ta có đạt ngưỡng không?** Vượt rất xa — tuyến mỏng nhất (`eval`) vẫn đạt
4.3 lần đọc trên mỗi lần ghi, so với ngưỡng 0.28. Kết quả thú
vị nằm ở TTL: **tier 1 giờ thắng ở cả 4 team dù ghi đắt hơn 60%**, vì nó bóp số lần ghi xuống
(ví dụ `assistant`: 39 lần ghi →
1 lần). M2 tính đúng hệ số đo được này thay vì giả
định phẳng 0,10× — và đó là lý do lever cache trong sổ chỉ đáng
$36/tháng chứ không phải con số quảng cáo "−90%".

### 4 — Ngân sách reasoning

| | Reasoning | Thường |
|---|---:|---:|
| Tỷ lệ request | 8.4% | 91.6% |
| Tỷ lệ token | 16.5% | 83.5% |
| Tỷ lệ chi phí | 16.4% | 83.5% |
| Tỷ lệ năng lượng | 94.0% | 6.0% |
| Output token trung bình | 3875 | 641 |
| $ mỗi request | $0.00695 | $0.00323 |
| Wh mỗi request | 148.2 | 0.858 |

**Vì sao tốn năng lượng gấp ~80 lần.** Extended thinking trả giá hai lần. Thứ nhất, nó sinh ra rất
nhiều token hơn — 3875 so với
641 output token ở đây, gấp
6 lần — và mỗi
token là một bước decode tự hồi quy phải đọc lại toàn bộ trọng số cùng KV cache từ HBM. Thứ hai,
đám token đó do model cỡ frontier sinh ra ở batch size nhỏ (vì ràng buộc độ trễ), nên mỗi bước đều
memory-bound và đốt trọn một GPU-giây cho đúng một token. Số token nhân lên, đồng thời năng lượng
trên mỗi token cũng tăng — nên hai hệ số nhân với nhau.

**Quy tắc routing đề xuất.** Reasoning chiếm 8.4% request nhưng
16.4% chi phí và 94.0% năng lượng. Trần 10% **không có hiệu
lực** — traffic hiện đã nằm dưới ngưỡng đó, và đây là kết quả trung thực: đặt trần 10% tiết kiệm
đúng $0. Lever thật sự có hiệu lực là **lọc theo điều kiện kích hoạt**: chỉ bật extended thinking
khi có tín hiệu đo được (truy hồi nhiều bước, code sinh ra phải chạy được, hoặc câu trả lời lượt
đầu bị verifier bác), tuyệt đối không bật mặc định cho toàn bộ traffic của một team. Giảm một nửa
lượng reasoning theo quy tắc đó tiết kiệm $11/tháng và
**14,734 Wh/ngày** — phần tiền là nhiễu, phần năng lượng thì không.

*Cảnh báo về mô hình:* hệ số 80× trong
`sustainability.py` được nhân **chồng lên** phần token đã tăng sẵn, nên con số
94.0% là **cận trên**. Hướng và bậc độ lớn thì đúng; muốn con số chính xác phải
đo bằng công tơ ở mức từng token.

### 5 — Lập lịch theo carbon (`missions/m6_carbon_scheduling.py`)

2,057 kWh/tháng của fleet là training gián đoạn được — không có người dùng nào
đang chờ một round trip, lại vốn đã checkpoint sẵn, nên đây là phần **di chuyển được**. Phần còn
lại 1,918 kWh/tháng là inference phục vụ người dùng, bị ghim tại chỗ bởi độ trễ.

| Vùng | $/kWh | gCO2/kWh | Tiền điện | Carbon (kg) | kg tiết kiệm | Độ trễ |
|---|---:|---:|---:|---:|---:|---:|
| europe-north1 | 0.09 | 30 | $185 | 62 | 720 | +95 ms |
| us-east-wa | 0.055 | 90 | $113 | 185 | 597 | +45 ms |
| us-west-2 | 0.07 | 120 | $144 | 247 | 535 | +55 ms |
| us-east-1 | 0.12 | 380 | $247 | 782 | 0 | +0 ms |
| europe-central2 | 0.18 | 660 | $370 | 1,358 | -576 | +110 ms |

**"Tối ưu" là vùng nào thì phụ thuộc công ty đang tối ưu cái gì.** Điện rẻ nhất:
**us-east-wa**. Lưới sạch nhất: **europe-north1** với 30 gCO2/kWh. Theo điểm cân bằng
50/50 đã chuẩn hoá min-max thì lựa chọn là **us-east-wa** — 90 gCO2/kWh ở mức giá thấp
nhất bảng, đổi lấy +45 ms độ trễ. Chuyển toàn bộ phần di chuyển
được sang europe-north1 tránh được **720 kgCO2e/tháng
(92% dấu chân của khối tải đó)** với cái giá +95
ms mà **không người dùng nào cảm nhận được**, bởi đúng theo định nghĩa ta không hề dịch chuyển thứ
gì có người ngồi chờ.

Cái bẫy nằm ở `europe-central2`: nó là vùng châu Âu *gần nhất* nhưng cũng là lưới bẩn nhất bảng
(660 gCO2/kWh). Chọn theo phản xạ "gần thì tốt" sẽ **tăng 74% phát thải** cho khối tải này, đồng
thời trả tiền điện cao nhất — cùng lúc thua ở cả hai trục.

**Nối carbon với tiền thật:** trên hợp đồng thuê neocloud, tiền điện đã nằm sẵn trong $/GPU-hr, nên
cột "tiền điện" ở trên là *thành phần vật lý* của chi phí chứ không phải hoá đơn thứ hai. Điều đó
làm carbon trở thành trục quyết định độc lập duy nhất ở đây — và may mắn là hai trục không xung đột:
us-east-wa vừa là điện rẻ nhất vừa là lưới sạch thứ nhì, tức chọn đúng vùng thì tiết kiệm
carbon **không tốn thêm đồng nào**.

## Cách đọc các con số này

* **Hai rổ chi phí trong báo cáo này tách rời nhau theo đúng thiết kế dữ liệu.**
  `gpu_telemetry.csv` (11 GPU) và `workloads.csv` (8 job) là hai lát cắt riêng của lab. Lever idle
  và right-size được định giá từ telemetry; lever purchasing từ danh mục workload. Chúng không đếm
  trùng nhau, nhưng cũng không đối chiếu về cùng một hoá đơn — trên fleet thật phải join hai nguồn
  theo instance id trước khi công bố một con số tổng.
* **Right-size chỉ tính trên số giờ bị tính tiền** (720 giờ với
  một GPU chạy 24/7), nên lever idle và lever right-size không thể cùng đòi một giờ.
* **Con số "sau tối ưu" giả định các lever cộng dồn sạch.** Phần lớn là đúng — routing, purchasing
  và idle tác động lên những phần khác nhau của hoá đơn — nhưng một GPU vừa được right-size vừa
  chuyển sang spot sẽ không tiết kiệm trọn vẹn cả hai khoản.
* **Giá là snapshot tháng 6/2026**, mà giá GPU đổi hàng tháng; tỷ lệ thu hồi spot là số minh hoạ
  theo nhóm khan hiếm, không phải số đo trên một nhà cung cấp cụ thể.
* **Hệ số năng lượng reasoning được nhân chồng lên phần token đã tăng**, nên tỷ lệ năng lượng là
  cận trên (xem phần mở rộng 4).
* **Dữ liệu là tổng hợp và có seed cố định** (`data/generate.py`, seed 25) — phương pháp thì chuyển
  giao được, còn các con số đô-la cụ thể thì không.

_Số liệu là snapshot tháng 6/2026; giá GPU đổi hàng tháng — phải chuẩn hoá lại trước khi áp dụng._
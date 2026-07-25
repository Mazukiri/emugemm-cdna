# Round 9 (2026-07-25) — Phase 1: giả thuyết bị bác bỏ, và Round 8 phải sửa

## ⚠️ ĐÍNH CHÍNH QUAN TRỌNG CHO `RESULTS_ROUND8.md`

Kết luận nổi bật của Round 8 — *"vượt K ≈ 76 500, bf16×3 chính xác hơn native fp32"* — **chỉ đúng khi so với
GEMM fp32 mặc định của rocBLAS**. Nó **KHÔNG đúng** khi fp32 được cộng dồn tốt hơn. Chi tiết bên dưới.

---

## Thí nghiệm (`flat_error.cpp`)

**Giả thuyết vào cuộc:** sai số bf16×3 gần phẳng theo K vì bị chặn bởi **lỗi biểu diễn** (2 lát bf16 ≈ 2⁻¹⁶),
nhưng chưa phẳng hoàn toàn vì bản thân nó cũng tích luỹ trong fp32. Chia K thành `c` chunk rồi **cộng các
chunk trong fp64** sẽ khử số hạng tích luỹ, giữ 4.45e-6 ở mọi K, biến giao điểm thành phát biểu sạch.

**Áp cho CẢ HAI phía** — đây là điều bắt buộc rút ra từ Round 8 ("tune một phía là tự lừa mình").
M=N=4096, chuẩn fp64 trên GPU, kết quả cuối làm tròn về fp32 để vẫn là một SGEMM thật.

### K = 262144

| c | fp32 err | ms | bf16×3 err | ms |
|---|---|---|---|---|
| 1 | 9.1619e-06 | 338.6 | **6.4167e-06** | **285.8** |
| 4 | 4.5794e-06 | 333.9 | 4.9905e-06 | 331.0 |
| 16 | 2.2906e-06 | 643.0 | 4.5661e-06 | 504.6 |
| 64 | 1.1457e-06 | 441.3 | 4.4615e-06 | 334.1 |
| 256 | **5.7339e-07** | 601.6 | 4.4397e-06 | 479.3 |

### K = 65536

| c | fp32 err | ms | bf16×3 err | ms |
|---|---|---|---|---|
| 1 | 4.5809e-06 | 83.3 | 4.9894e-06 | 70.3 |
| 4 | 2.2898e-06 | 144.8 | 4.5660e-06 | 111.3 |
| 16 | 1.1452e-06 | 121.3 | 4.4619e-06 | 83.9 |
| 64 | 5.7338e-07 | 135.7 | 4.4397e-06 | 98.4 |
| 256 | 2.8805e-07 | 134.9 | 4.4355e-06 | 197.6 |

Chạy lại hai lần: sai số trùng tới **5 chữ số**, thời gian lệch <1%.

---

## Ba kết quả

### 1. ✅ Sàn biểu diễn được xác nhận chính xác

bf16×3 hội tụ về **4.4355e-06** khi `c` tăng — trùng khít giá trị đo ở K=2048 (4.448e-06).
Nửa đầu giả thuyết đúng hoàn hảo: **bf16×3 CÓ một sàn cứng, không thể xuống dưới dù trả bao nhiêu thời gian.**

### 2. ❌ Nhưng fp32 KHÔNG có sàn — nên giao điểm biến mất

Đầu vào fp32 vốn đã chính xác trong fp32, nên nó **không có lỗi biểu diễn nào để bị chặn**. Chunking đẩy sai
số xuống mãi: 9.16e-06 → 5.73e-07 (giảm đúng `1/√c`). Ở c=256, **fp32 chính xác hơn bf16×3 7.7 lần**.

⇒ **Cho fp32 cùng cách cộng dồn thì nó thắng bf16×3 ở mọi mức chính xác dưới 4.4e-6.**
Nếu tôi chỉ chunk phía bf16×3, tôi đã báo cáo "đã làm sai số emulation phẳng, giao điểm đúng ở mọi K" —
**một kết luận hoàn toàn sai**. Đây đúng là lý do quy tắc "áp cho cả hai phía" tồn tại.

### 3. 🔍 Quy luật sạch: sai số chỉ phụ thuộc `K/c`

| | K=65536, c=1 | K=262144, c=4 |
|---|---|---|
| `K/c` | 65536 | 65536 |
| fp32 | 4.5809e-06 | 4.5794e-06 |
| bf16×3 | 4.9894e-06 | 4.9905e-06 |

Trùng tới 4 chữ số. **Sai số là hàm của độ dài chuỗi tích luỹ `K/c`, không phải của K.** Mô hình:

```
err_fp32(K,c)   = a·√(K/c)                          a ≈ 1.79e-8
err_bf16×3(K,c) = √( floor² + (a/2)²·(K/c) )        floor ≈ 4.4355e-6
```

Hệ số `a/2` cho bf16×3 (suy từ `√(4.9894e-6² − 4.4355e-6²) = 2.285e-6 ≈ 4.5809e-6 / 2`) khớp với phát hiện
Round 8: solution fp32 chính xác nhất của rocBLAS có sai số **đúng bằng một nửa** solution mặc định.
⇒ **kernel bf16 đang dùng cấu trúc cộng dồn tốt hơn kernel fp32 mặc định.** Nói cách khác, một phần đáng kể
của "chiến thắng" Round 8 đến từ **chiến lược cộng dồn của kernel, không phải từ định dạng số.**

---

## Phát biểu đúng, thay cho headline Round 8

Biên Pareto (thời gian, sai số) tại K=262144 — các điểm không bị chi phối:

| điểm | ms | err | |
|---|---|---|---|
| **bf16×3, c=1** | **285.8** | 6.42e-06 | nhanh nhất, không gì bằng |
| **bf16×3, c=64** | 334.1 | **4.46e-06** | tốt nhất ở mốc ~334 ms (fp32 c=4: 333.9 ms / 4.58e-06) |
| fp32, c=64 | 441.3 | 1.15e-06 | |
| fp32, c=256 | 601.6 | **5.73e-07** | chính xác nhất |

- **bf16×3 làm chủ đầu nhanh** và tầng chính xác ~4.5e-6: ở 286 ms không lựa chọn nào theo kịp, và ở 334 ms
  nó vẫn ngang hoặc hơn fp32 đã chunk.
- **fp32 làm chủ mọi thứ dưới 4.4e-6.** bf16×3 **vĩnh viễn** không xuống dưới sàn của nó.
- **Giao điểm của Round 8 chỉ đúng với fp32 mặc định của rocBLAS.** Với chunk + reduction fp64 — một kỹ thuật
  rẻ và tiêu chuẩn — fp32 đuổi kịp ở ~1.17× thời gian và vượt lên sau đó.

**Điều CÒN NGUYÊN từ Round 8:** bf16×3 vẫn **1.20–1.22× nhanh hơn native fp32 ở cùng tầng chính xác**
(Phase C), robust toàn dải mũ, và là điểm nhanh nhất trên biên Pareto. Cái mất là tuyên bố
*"chính xác hơn fp32"* ở dạng không điều kiện.

---

---

# Phase 2 — Bảng tune offline, và một đính chính nữa cho Round 8

`gen_tune_table.cpp`: 294 shape × 2 dtype, liệt kê + đo **mọi** solution của rocBLAS, hai giai đoạn
(chạy 1 lượt cho tất cả → giữ top 16 → đo lại 5 lượt), resume được, chia 8 shard trên 8 GCD.
48 shape bị bỏ qua bởi hàng rào ngân sách thời gian (được ghi log, không âm thầm bỏ).

| | gain trung bình | gain lớn nhất | default đã tối ưu (<1.02×) |
|---|---|---|---|
| **bf16** | **1.384×** | 3.773× tại 2048×32768×512 | **3%** / 287 shape |
| **fp32** | **1.349×** | 4.395× tại 1024×32768×16384 | **9%** / 253 shape |
| **fp16** | **1.304×** | 3.570× tại 2048×32768×512 | **10%** / 247 shape |

*(fp16 được thêm ở lần chạy thứ hai — xem mục "test bất biến" bên dưới: thiếu nó gây một vi phạm thật.
Logic resume khiến lần hai chỉ phải tính fp16, không chạy lại fp32/bf16.)*
**Tổng: 787 dòng.** Với cả ba kiểu, **chỉ 3–10% shape có solution mặc định nằm trong 2% của tối ưu.**

### ⚠️ Đính chính cho Round 8 mục C

Round 8 kết luận *"heuristic fp32 của rocBLAS thì ổn, chỉ heuristic bf16 mới hỏng"* — dựa trên **3 shape**.
Trên 253 shape thì **fp32 cũng tệ gần bằng bf16**. Ba shape đó tình cờ đều là shape "đẹp".

**Xác minh solo** (không tranh chấp GPU, để loại khả năng nhiễu do chạy 8 shard song song):

| shape | dtype | default | tốt nhất | gain solo | gain lưới |
|---|---|---|---|---|---|
| 1024×32768×16384 | fp32 | 133.0 ms = **8.26 TF** | 29.7 ms = **37.00 TF** | **4.48×** | 4.395 ✓ |
| 2048×32768×512 | bf16 | 3.11 ms = 22.11 TF | 0.81 ms = 84.46 TF | **3.82×** | 3.773 ✓ |
| 4096×4096×16384 | fp32 | 29.77 ms = 18.47 TF | 14.89 ms = 36.92 TF | **2.00×** | |
| 4096×4096×16384 | bf16 | 5.50 ms = 99.92 TF | 4.15 ms = 132.40 TF | 1.35× | |
| 12288×12288×12288 | fp32 | 100.00 ms = 37.11 TF | 100.04 ms = 37.09 TF | **1.00×** | ✓ (khớp Round 8) |

Solo khớp lưới tới ~2%. **Gain là thật.** Phát biểu đúng: heuristic của rocBLAS **tốt ở shape vuông lớn và
sụp 2–4.5× ở shape gầy hoặc K lớn**, cho *cả* fp32 lẫn bf16.

**Hệ quả ngược lại cho các phép đo trước:** `crossover_test` chạy solution **mặc định** ở M=N=4096 với K lớn —
đúng vùng default sụp — nên cột TFLOP/s của nó (dao động 19–35 TF) là do chọn kernel, không phải do lược đồ.
Cột đó đã được đánh dấu không dùng để trích dẫn. Phần **độ chính xác** không bị ảnh hưởng: Round 8 đã kiểm
rằng sai số của solution mặc định trùng với solution nhanh nhất (6.4817e-06 vs 6.4818e-06).

---

# Phase 3 — Thư viện `emugemm`

`emugemm/{emugemm.h, emugemm.cpp, emugemm_test.cpp}`. Mô hình hợp nhất, fit từ số đo Round 7–9:

```
err(scheme, K, c) = √( floor² + a²·(K/c) )
  fp32     floor 0          a 1.79e-8   tốc độ 1.00×
  bf16×3   floor 4.4355e-6  a 8.95e-9   1.21×   robust toàn dải
  fp16×3   floor 3.6e-7     a 8.95e-9   1.19×   cần dải fp16
  bf16×6   floor ~0         a 8.95e-9   0.57×
```

Mọi đường low-precision dùng chung `a = 8.95e-9` = **đúng một nửa** hằng số của kernel fp32 mặc định —
tức kernel bf16/fp16 cộng dồn tốt hơn. Đây là **đặc tính thư viện, không phải đặc tính của định dạng số**.

**Hai hệ quả rơi thẳng ra từ mô hình:**
1. **bf16×6 bị chi phối hoàn toàn, không bao giờ được chọn.** Nó tốn 1.75× thời gian native để đạt sai số mà
   `fp32 + chunk(4)` đạt ở ~1.0× (K=262144: 4.637e-6 @594 ms vs 4.579e-6 @334 ms).
2. **fp16×3 có sàn thấp hơn bf16×3 12×, ở cùng tốc độ** ⇒ khi dữ liệu nằm trong dải fp16 thì fp16×3 mới
   là lựa chọn đúng, không phải bf16×3.

### Bộ test bất biến tìm ra ba lỗi, tất cả đều thật

1. **Âm thầm giao thiếu.** Ở target 1e-7 không ứng viên nào đạt, nhưng planner trả về giá trị khởi tạo thay
   vì báo lỗi. Sửa: `TARGET UNREACHABLE`.
2. **Chậm hơn native ở shape nhỏ.** `hipMalloc`/`hipFree` buffer split ở **mỗi lời gọi** tốn hơn phần GEMM
   tiết kiệm được (2048×2048×4096: 1.39 ms vs 1.03 ms). Sửa: workspace bền + đưa chi phí quét/split vào mô
   hình (`overhead_seconds`). Shape nhỏ giờ đúng đắn rơi về native.
3. **Thiếu fp16 trong bảng tune.** Đường fp16 chạy solution mặc định — đúng chỗ Phase 2 vừa chứng minh
   default sụp 2–4×, nên nó thua native ở shape gầy (39.10 vs 32.22 ms) dù mô hình hứa 1.19×. Sửa: thêm
   fp16 vào bảng ⇒ cùng ca đó giờ chạy **28.31 ms = 1.14× nhanh hơn native**.

Và một lỗi trong **chính bất biến tôi viết**: *"không bao giờ chậm hơn native"* vô nghĩa khi target thấp hơn
mức native đạt được — ở đó native không phải lựa chọn hợp lệ, nên trả thêm thời gian là đúng.

**Kết quả cuối: 0 vi phạm trên 15 ca** (5 shape × 3 target), và **không có một vi phạm độ chính xác nào
ở bất kỳ lần chạy nào** — mô hình dự đoán 4.472e-6, đo được 4.455e-6.

**Độ tin của mô hình chi phí chunk đã bị hạ:** số đo **không đơn điệu** (c=16 → 643 ms nhưng c=64 → 441 ms)
vì kernel bị chọn lại theo độ sâu chunk. Đã thay công thức trơn bằng **bao hình bảo thủ**, và ghi trong code
rằng nó chỉ hiệu chuẩn ở **đúng một shape** — Phase 5 phải đo lại trước khi tin ở nơi khác.

---

---

# Phase 5 — Mô hình sai số qua các phân bố dữ liệu (`error_model.cpp`)

Mọi con số độ chính xác của Round 7–9 đều đo trên **N(0,1) độc lập** — trường hợp thân thiện nhất với fp32.
Phase 5 fit lại `err ~ a·K^p` cho từng phân bố. M=N=2048, chuẩn = DGEMM trên GPU.

| phân bố | `p` của fp32 | `p` của bf16×3 | @K=65536: bf16×3 vs fp32 |
|---|---|---|---|
| normal N(0,1) | 0.5249 | **0.0146** (phẳng) | 4.71e-6 vs 4.59e-6 — sát giao điểm |
| **positive U(0,1)** | 0.5506 | 0.2909 | **8.4e-7 vs 3.4e-6 → bf16×3 TỐT HƠN 4×** |
| lognormal 1e-3…1e3 | 0.5047 | 0.0173 | 4.34e-6 vs 3.71e-6 |
| **cancelling (gần trực giao)** | **0.1471** | 0.0055 | **1.65e-3 vs 4.66e-5 → bf16×3 TỆ HƠN 35×** |

### ❌ Dự đoán của tôi sai

Tôi vào Phase 5 với giả thuyết: *"dữ liệu tương quan làm lỗi tích luỹ fp32 tăng O(K) thay vì √K, nên giao
điểm đến sớm hơn — con số 76 500 là cận trên thận trọng."* **Sai hoàn toàn.** Trên dữ liệu triệt tiêu,
`p` của fp32 **giảm xuống 0.147**, không tăng.

Cái thực sự xảy ra: khi các hàng/cột gần trực giao, **‖C‖ sụp**, nên sai số *tương đối* của mọi lược đồ đều
phồng lên — nhưng **emulation phồng nặng hơn nhiều**, vì sai số biểu diễn của nó là tỉ lệ cố định của
|a|·|b| chứ không phải của ‖C‖. **Triệt tiêu hại emulation hơn hại fp32.**

### ⚠️ Giới hạn quan trọng nhất tìm được cả vòng

**Mô hình `err = √(floor² + a²K/c)` chỉ đúng cho dữ liệu điều kiện tốt.** Với bài toán có triệt tiêu mạnh,
bf16×3 sai số **1.65e-3** — gấp **35×** fp32 và gấp **370×** so với chính nó trên dữ liệu normal.
Một dispatcher tin vào mô hình sẽ chọn bf16×3 cho target 1e-5 và giao về sai số 1.65e-3.
⇒ **Phải ghi thành giả định của API, không được để ngầm.** Xem mục "Còn nợ".

### Mặt ngược lại, cũng đáng giá

Trên dữ liệu **toàn dương** (không triệt tiêu), bf16×3 **tốt hơn fp32 4×** ở K=65536, và `p` của nó chỉ 0.29
so với 0.55 của fp32. Vì ‖C‖ tăng như K trong khi sai số tăng như √K, sai số tương đối **giảm** theo K.
Đây là chế độ tốt nhất cho emulation, và nó phổ biến (ma trận không âm, tích chập, đếm, histogram).

---

# Phase 4 — Ứng dụng RandNLA: hai kết quả, một âm một dương

### (a) Householder QR — kết quả ÂM và nó tự giải thích

`randsvd.cpp`, m=262144 n=4096 rank=2048:

| | tổng ms | residual tương đối |
|---|---|---|
| native fp32 | 6192.10 | 1.21709548e-01 |
| emugemm target 1e-5 | 6085.08 (**1.018×**) | 1.21709793e-01 |

Chỉ nhanh hơn 1.8%. Lý do nằm ngay trong số liệu: tổng 6192 ms nhưng GEMM chỉ ~300 ms — **rocSOLVER
`geqrf`+`orgqr` trên panel 262144×2048 chạy ~1 TFLOP/s (3% peak) và ăn 5.9 giây.** Pipeline bị chặn bởi
trực giao hoá, nên tăng tốc GEMM 1.2× chỉ đổi được 0.8%.

**Kết luận đúng không phải "emulation vô dụng cho RandNLA"**, mà là: *emulation chỉ đáng giá khi pipeline
thực sự GEMM-bound; với panel tall-skinny điều đó đòi hỏi bỏ Householder QR.*

### (b) CholeskyQR2 — lựa chọn chuẩn trên GPU

`randsvd2.cpp` thay Householder bằng CholeskyQR2, biến 3/4 thao tác nặng thành GEMM với **K = m**:
`G = YᵀY` (K=m) → `chol(G)` (ℓ×ℓ, tí hon) → `Q = YR⁻¹` (trsm) → lặp một lần → `B = QᵀA` (K=m).

| | tổng ms | GEMM ms | residual tương đối | |
|---|---|---|---|---|
| native fp32 (**đã tune**) | 1312.15 | 598.43 | 1.2148000602e-01 | |
| emugemm target 1e-5 | 1101.87 | 566.38 | 1.2148003416e-01 | **1.191×**, residual giống tới 7 chữ số |
| emugemm target 1e-6 | 991.02 | 468.24 | 1.2171402813e-01 | **1.324×**, residual +0.19% |

**Đổi Householder → CholeskyQR2 làm cả pipeline nhanh 4.7×** (6192 → 1312 ms) và biến nó thành GEMM-bound
(598/1312 = 46%). Chỉ khi đó emulation mới có chỗ để giúp: **1.018× → 1.19–1.32×**.

> ⚠️ **Tôi suýt công bố con số bị thổi.** Bản đầu tiên cho đường "native" chạy `rocblas_gemm_ex` với
> `algo_standard` — **chưa tune** — trong khi `emugemm` tra bảng tune. Mà Phase 2 vừa đo được default sụp
> trung bình 1.35×. Đã sửa: baseline giờ đi qua `emugemm` với `force_scheme=NATIVE_FP32` nên **được cùng
> một lượt tra bảng**. Kết quả 1.204/1.305 → **1.191/1.324**. Lần này thay đổi nhỏ, nhưng đó là may mắn
> chứ không phải phương pháp.

**Kết luận Phase 4:** emulation đáng giá **chỉ khi pipeline thực sự GEMM-bound**. Với panel tall-skinny,
điều đó đòi hỏi bỏ Householder QR trước đã — và bước đó một mình đã đáng giá gấp bốn lần toàn bộ nghiên cứu
emulation này. Một kết luận khiêm tốn nhưng đúng: **hãy sửa nút cổ chai thật trước khi tối ưu GEMM.**

---

---

# Tổng kết Round 9

| phase | kết quả |
|---|---|
| 1 | Giả thuyết **bị bác bỏ**: cho fp32 cùng cách cộng dồn thì giao điểm biến mất. Sàn bf16×3 = **4.4355e-6** xác nhận chính xác. Quy luật `err = f(K/c)` khớp 4 chữ số. |
| 2 | Bảng tune **787 dòng, 3 kiểu**. Default chỉ tối ưu ở **3–10%** shape; gain trung bình **1.30–1.38×**, đỉnh **4.4×**. Sửa một kết luận Round 8. |
| 3 | Thư viện `emugemm` + dispatcher dựa trên mô hình đo được. Test bất biến tìm **3 lỗi thật**; kết thúc **0 vi phạm**. |
| 4 | RandNLA: Householder QR **1.018×** (QR-bound) → CholeskyQR2 **1.19–1.32×** (GEMM-bound). Baseline được tune công bằng. |
| 5 | Mô hình **chỉ đúng cho dữ liệu điều kiện tốt**. Dữ liệu triệt tiêu: bf16×3 tệ hơn fp32 **35×**. Dữ liệu toàn dương: bf16×3 tốt hơn **4×**. |

## Còn nợ (ghi rõ để không bị hiểu là đã làm)

1. **`emugemm` chưa an toàn cho bài toán điều kiện xấu.** Mô hình sai số giả định ‖C‖ ~ √K·‖a‖‖b‖. Với dữ
   liệu triệt tiêu mạnh nó sai **35×** theo hướng nguy hiểm (giao về sai số 1.65e-3 khi hứa 1e-5). Cần một
   trường trong `emu_request_t` để người gọi khai báo điều kiện, hoặc một ước lượng rẻ, **trước khi dùng thật**.
2. **`chunk_cost` mới hiệu chuẩn ở đúng một shape** và số đo gốc không đơn điệu. Hiện là bao hình bảo thủ.
3. **Bảng tune không chuyển được** giữa phiên bản ROCm hay arch khác (đã kiểm arch lúc nạp và từ chối nếu lệch).
4. **48 shape bị bỏ qua** bởi hàng rào ngân sách; các shape rất lớn dựa vào tra cứu shape gần nhất.
5. **Chưa thử MI300/MI355.** Toàn bộ hằng số ở đây là của gfx90a.

# Round 8 (2026-07-25) — giao điểm độ chính xác, và vá lỗ hổng shape

Tiếp nối `RESULTS_2026-07-25.md` (Round 7). Phần cứng: node 8× GCD gfx90a (MI250), mỗi GCD 104 CU / 68.7 GB
→ **549 GB tổng**. Container `ducmai-dev`, ROCm 7.2.3.
File mới: `mfma_peak2.cpp` (+fp64), `crossover_test.cpp`, `rocblas_tune.cpp`, `emu_tuned.cpp`, `node_sweep.sh`.

---

## A. Bảng kinh tế phần cứng đã hoàn tất — hướng emulate FP64 ĐÓNG

| lệnh | TFLOP/s | vs fp32 |
|---|---|---|
| **fp64** `mfma_16x16x4f64` | **41.7** | 0.98× |
| fp32 `mfma_16x16x4f32` | 42.5 | 1.00× |
| fp16 `mfma_16x16x16f16` | 169.4 | 3.99× |
| bf16 `mfma_16x16x16bf16_1k` | 169.2 | 3.98× |
| bf16 `mfma_16x16x8bf16` (legacy) | 85.1 | 2.00× |
| int8 `mfma_16x16x16i8` | 170.2 | 4.01× |

**CDNA2 có fp64 matrix full-rate (= fp32).** Hệ quả, đo được chứ không suy từ datasheet:

| lược đồ emulate fp64 | trần | |
|---|---|---|
| từ fp32, ≥3 lát | 0.34× | **THUA** |
| từ fp16, ~7 lát | 0.58× | **THUA** |
| Ozaki-int8, ~9 moduli | 0.45× | **THUA** |

⇒ Trên MI250 **không có đường nào** để emulate fp64 có lãi. Hướng này đóng lại bằng phép đo, giống cách
`mfma_peak.cpp` đóng hướng int8 ở Round 3.

---

## B. Giao điểm độ chính xác — ⚠️ ĐÃ BỊ SỬA BỞI ROUND 9, đọc `RESULTS_ROUND9.md` trước

> **Đính chính:** mục B dưới đây đúng **chỉ khi so với GEMM fp32 mặc định của rocBLAS**. Round 9 (Phase 1)
> cho fp32 cùng cách cộng dồn (chia K thành chunk, cộng chunk trong fp64) và **giao điểm biến mất**: fp32
> không có sàn biểu diễn nên sai số của nó giảm mãi theo `1/√c` (xuống 5.73e-07), trong khi bf16×3 chạm sàn
> cứng **4.4355e-06** và dừng. Cái còn nguyên: bf16×3 vẫn **1.20–1.22× nhanh hơn** ở cùng tầng chính xác và
> là **điểm nhanh nhất trên biên Pareto**. Cái mất: tuyên bố *"chính xác hơn fp32"* ở dạng vô điều kiện.

**Thiết lập:** M=N=4096, quét K = 2048 → 262144. **Chuẩn = rocBLAS DGEMM chạy ngay trên GPU**
(khả thi vì fp64 full-rate; sai số chuẩn ~6e-14, dưới thang đo 8 bậc).
**Chuẩn đã được kiểm ngược lại bằng vòng lặp fp64 trên CPU ở K=2048: khớp 1.355e-15.** ✓

| K | fp32 | **bf16×3** | bf16×6 | fp16×3 |
|---|---|---|---|---|
| 2 048 | 8.105e-07 | 4.448e-06 | 3.320e-07 | 5.413e-07 |
| 4 096 | 1.145e-06 | 4.460e-06 | 4.961e-07 | 6.782e-07 |
| 8 192 | 1.620e-06 | 4.494e-06 | 7.390e-07 | 8.877e-07 |
| 16 384 | 2.290e-06 | 4.566e-06 | 1.090e-06 | 1.200e-06 |
| 32 768 | 3.240e-06 | 4.712e-06 | 1.589e-06 | 1.660e-06 |
| 65 536 | 4.581e-06 | 4.989e-06 | 2.289e-06 | 2.318e-06 |
| **131 072** | 6.482e-06 | **5.511e-06** ⬅ | 3.273e-06 | 3.259e-06 |
| **262 144** | 9.162e-06 | **6.417e-06** ⬅ | 4.637e-06 | 4.596e-06 |

**Khớp lý thuyết:** fit `err_fp32 = 1.791e-08 · K^0.4999` qua 8 điểm — lý thuyết tích luỹ làm tròn dự đoán
đúng **p = 0.5**, khớp tới 4 chữ số. **Giao điểm đo được K ≈ 76 500** (dự đoán trước khi chạy: 62 000,
lệch 25% — hợp lý cho một ngoại suy từ 2 điểm).

> **⚠️ Cột tốc độ của `crossover_test` KHÔNG dùng để trích dẫn.** Nó chạy solution **mặc định** cho mọi biến
> thể, mà mục C dưới đây chứng minh solution mặc định có thể sai lệch tới 1.7×. Ở shape K-lớn này fp32 dao
> động 19–35 TF thuần tuý do chọn kernel. **Con số tốc độ chính thức là của `emu_tuned` (đã tune cả hai phía).**
> Phần độ chính xác **không** bị ảnh hưởng: sai số bf16 giống nhau qua cả 388 solution (spread 1.00×), và
> sai số fp32 mặc định trùng đúng với solution nhanh nhất.

**Vì sao bf16×3 là biến thể duy nhất phẳng:** sai số của nó bị chặn bởi **lỗi biểu diễn** (2 lát bf16 ⇒ ~2⁻¹⁶),
độc lập với K. fp32, bf16×6 và fp16×3 đều có lỗi biểu diễn **dưới** ngưỡng tích luỹ nên đều bị **lỗi tích luỹ
∝√K** chi phối. Ở K đủ lớn, thứ hằng số thắng thứ tăng dần.

### Kiểm confound (bắt buộc, đã ghi trong kế hoạch trước khi chạy)

Nghi vấn: fp32 và low-precision có thể dùng chiến lược tích luỹ khác nhau (split-K), khiến "chiến thắng"
đến từ chọn kernel chứ không từ định dạng số. Liệt kê **toàn bộ solution** của rocBLAS ở K=131072:

| | số solution | spread sai số | ghi chú |
|---|---|---|---|
| fp32 | 562 | **2.00×** | chính xác nhất 3.240e-06 **nhưng chỉ 4.07 TF** |
| bf16 | 388 | **1.00×** | mọi solution cho sai số y hệt |

- Sai số bf16 **không phụ thuộc** lựa chọn kernel ⇒ kết luận về bf16×3 vững.
- fp32 **có** một solution chính xác gấp đôi (3.24e-6), nhưng nó chạy **4.07 TF — chậm 9×**. Đối chiếu:
  fp64 **matrix peak đo được là 41.7 TF**; kể cả giả định hiệu suất thư viện thấp một cách bảo thủ (60%)
  thì DGEMM vẫn ~25 TF, tức **nhanh hơn 6×** solution fp32 đó *và* chính xác hơn ~9 bậc. (Chưa đo trực tiếp
  tốc độ DGEMM đạt được — đây là lập luận từ peak, đủ chắc vì biên tới 6×.) Solution fp32 "chính xác" đó
  **bị chi phối hoàn toàn**, không phải một lựa chọn thực tế.
- ⇒ So với fp32 mà **người dùng thật sự nhận được** (mặc định = nhanh nhất = 6.48e-6), giao điểm đứng vững.

**Phát biểu đúng của kết quả:** *ở tốc độ ngang hoặc nhanh hơn, bf16×3 chính xác hơn native fp32 khi
K ≳ 76 000; muốn fp32 chính xác hơn thì phải trả giá 9× tốc độ, mà ở mức đó fp64 native đã tốt hơn về mọi mặt.*

### Giới hạn của kết quả này (phải nói kèm)

- **Dữ liệu là N(0,1) độc lập.** Với dữ liệu đó lỗi tích luỹ của fp32 tăng theo **√K** vì các lỗi làm tròn
  triệt tiêu lẫn nhau như bước ngẫu nhiên. Với dữ liệu **cùng dấu / có tương quan**, lỗi tích luỹ tăng theo
  **O(K)** chứ không phải √K, nên **giao điểm sẽ đến SỚM HƠN**. Tức N(0,1) là trường hợp **thuận lợi cho fp32**
  — kết quả của ta là cận trên thận trọng, không phải cherry-pick.
- Sai số của bf16×3 phẳng vì bị chặn bởi lỗi **biểu diễn**, đại lượng này ổn định theo phân bố dữ liệu; nhưng
  chuẩn Frobenius chuẩn hoá theo ‖C‖ nên với dữ liệu có triệt tiêu mạnh ở đầu ra, cả hai đường đều dịch lên.
  **Chưa đo** với phân bố khác N(0,1) ở K lớn.
- K ≈ 76 000 là **rất lớn** cho GEMM đơn lẻ trong ML (hidden size thường ≤ 30k). Chế độ này thực tế với
  **HPC / tổng chập dài / tích luỹ theo batch**, không phải một lớp linear điển hình.

---

## C. Vá lỗ hổng shape — 0.71× thành 1.21×

Round 7 để lại một thất bại: bf16×3 chỉ đạt 0.74× ở shape MLP vì kernel bf16 mặc định của rocBLAS sụp
xuống 82 TF. `rocblas_gemm_ex_get_solutions` cho phép liệt kê và đo từng solution.

**Chẩn đoán (`rocblas_tune`, M=8192 N=28672 K=8192, bf16):**

| | ms | TFLOP/s |
|---|---|---|
| mặc định | 45.88 | 83.9 (tái lập đúng 82.0 của Round 7 ✓) |
| **tốt nhất / 388 solution** | 27.92 | **137.8 = 81.5% peak** |

Heuristic mặc định bỏ phí **1.64×**. Và ở cả shape vuông K=131072 nó cũng bỏ phí **1.58×** (76.9 → 121.3 TF)
⇒ đây là vấn đề **diện rộng của heuristic bf16 rocBLAS**, không riêng shape MLP.

**Xác minh đầu-cuối (`emu_tuned`) trên CẢ BA shape — baseline fp32 CŨNG được tune cho công bằng:**

| shape | fp32 mặc định | **fp32 tune (baseline)** | bf16×3 mặc định | **bf16×3 tune** | kết quả |
|---|---|---|---|---|---|
| M=8192 N=28672 K=8192 (MLP) | 21.57 | **37.52** | 26.48 (0.71×) | **45.33** | **1.21×** ✅ |
| M=16384 N=8192 K=8192 | 36.94 | **37.03** | 37.72 (1.02×) | **44.26** | **1.20×** ✅ |
| M=N=K=12288 (vuông) | 37.06 | **37.02** | 37.44 (1.01×) | **45.06** | **1.22×** ✅ |

Sai số không đổi khi tune (bf16×3: 4.495e-06 / 4.495e-06 / 4.530e-06).

**Hai kết luận từ bảng này:**
1. **bf16×3 = 1.20–1.22× trên MỌI shape đã thử** — đồng đều đến bất ngờ, một khi solution bf16 được tune.
   Lỗ hổng shape của Round 7 **hoàn toàn là lỗi chọn kernel**, không phải giới hạn kỹ thuật.
2. **Heuristic của rocBLAS kém ở bf16 nhưng tốt ở fp32.** Tune nâng bf16×3 lên 1.71× / 1.17× / 1.20×, còn
   fp32 hầu như không đổi ở hai shape vuông-ish (37.06→37.02; 36.94→37.03) và chỉ nhảy vọt ở shape MLP
   (21.57→37.52). ⇒ vấn đề nằm ở **đường bf16**, đúng hướng với mục E.

---

## D. Cả node — lợi thế có sống sót khi 8 GCD cùng chạy hết công suất không?

Câu hỏi thật: 8 GCD là **4 card hai-die dùng chung ngân sách điện**, và Round 7 đã đo được bf16×3 **tốn thêm
11% điện** so với fp32. Nên có lý do cụ thể để nghi lợi thế bị bào mòn khi cả bo mạch chạy hết.
`node_sweep.sh`, N=8192³, mỗi GCD một tiến trình độc lập, **cả hai phía đều tune**.

| | fp32 | bf16×3 |
|---|---|---|
| 1 GCD chạy một mình | 37.02 TF | 43.96 TF (**1.19×**) |
| **Tổng 8 GCD** | **278.4 TF** | **325.1 TF** (**1.17×**) |
| % so với tuyến tính | **94.0%** | **92.4%** |

**Lợi thế sống sót: 1.19× → 1.17×.** Cả hai mất 6–8% khi cả bo mạch chạy hết, và **bf16×3 mất nhiều hơn một
chút (92.4% vs 94.0%)** — khớp đúng với phép đo điện ở Round 7 (bf16×3 tốn thêm 11% điện, nên chạm trần công
suất sớm hơn). Hai phép đo độc lập chỉ về cùng một cơ chế.

**Cảnh báo về phương pháp:** mỗi tiến trình tự tune **trong lúc 7 tiến trình kia đang chạy**, nên phép đo
chọn solution bị nhiễu — thấy rõ ở độ tán của bf16×3 giữa các GCD (37.6–44.3 TF) so với fp32 (32.8–36.1 TF).
Con số tổng vẫn dùng được, nhưng số của **từng** GCD thì không nên trích riêng lẻ.

**Chưa làm:** trình diễn một GEMM đơn lẻ lớn nhất mà 549 GB cho phép. Phép quét này chỉ dùng ~3.3 GB/GCD;
bộ nhớ **không** phải yếu tố ràng buộc ở đây. Muốn làm cần một harness nhẹ hơn — `emu_tuned` phải đo ~950
solution nên ở N=32768 sẽ mất ~90 phút/GCD. Ghi nhận là **chưa đo**, không phải đã đo.

---

## E. hipBLASLt bf16 — giả thuyết của tôi ĐÃ BỊ BÁC BỎ, cơ chế vẫn chưa xác định

Vào Round 8 tôi mang theo giả thuyết: *hipBLASLt rơi về kernel non-MFMA cho bf16 trên gfx90a* (dựa trên
triệu chứng 25% peak + 10 algo, và câu trong tài liệu AMD *"Not all problem sizes may select MFMA-based
kernels"*). Đọc thẳng thư viện Tensile đã ship thì **giả thuyết sai**.

**Đã LOẠI TRỪ được hai cơ chế:**

1. **Không phải "rơi về non-MFMA".** Trích tên kernel thật từ `/opt/rocm/lib/hipblaslt/library/`:
   `Cijk_Ailk_Bljk_BBS_BH_..._MT64x16x32_MI16x16x1_...ISA90a...`
   Trong một file thư viện bf16 NN: **1912 kernel, cả 1912 đều có `_MI` (MFMA)**. Không có kernel nào không MFMA.
2. **Không phải "thiếu kernel tốt".** Phân bố macro-tile của bf16 tương đương fp16, có đủ tile lớn
   (`MT256x64x32`, `MT96x160x64`, `MT128x96x64`, …). bf16 **không** nghèo kernel.

**Cái vẫn đúng (tái lập được, đo nhiều lần):**

| | tốt nhất đo được | % của 169.1 TF |
|---|---|---|
| hipBLASLt bf16 (NN, gfx90a) | 41.8 TF | **25%** |
| rocBLAS bf16 (cùng phần cứng, cũng MFMA) | 137.8 TF | **81.5%** |
| hipBLASLt fp16 | 137.0 TF | 81% |

và hipBLASLt chỉ đưa ra **10 algo ứng viên** cho bf16 NN trong khi fp16/fp32 đều chạm trần 48 ta xin.

**Kết luận đúng mức:** khoảng cách là **thật, lớn (3.3×) và tái lập được**, và nó **không** do thiếu kernel
MFMA — nhưng **cơ chế chưa được xác định**. Nó nằm ở tầng chọn solution/heuristic của hipBLASLt cho bf16 NN
trên gfx90a, chỗ ta chưa quan sát được (`HIPBLASLT_LOG_LEVEL=4` chỉ in đường dẫn thư viện chứ không in tên
solution được chọn; container không có `hipblaslt-bench` lẫn `rocblas-gemm-tune`).

⇒ **Chưa nộp issue nào.** Có thể báo cáo *triệu chứng có thể tái lập* kèm phần đã loại trừ, nhưng
**không được tuyên bố nguyên nhân**. Việc dùng được ngay: **dùng rocBLAS cho bf16, hipBLASLt cho fp16.**

---

## Bài học phương pháp mới trong Round 8

1. **`| head` trên lệnh build giết compiler bằng SIGPIPE** — làm build "thất bại" mà không có thông báo lỗi
   nào. Mất một vòng chẩn đoán. Đừng pipe output của trình biên dịch qua `head`.
2. **Ghi tiến độ cho vòng lặp dài.** Bản `rocblas_tune` đầu tiên chỉ in ở cuối, chạy 12 phút không một dòng —
   không phân biệt được "đang chạy" với "treo".
3. **Tune cả hai phía.** Ở shape MLP, tune nâng fp32 từ 21.6 lên 37.5 TF. Nếu chỉ tune phía mình thì đã
   tự tặng một chiến thắng giả 2.1×.

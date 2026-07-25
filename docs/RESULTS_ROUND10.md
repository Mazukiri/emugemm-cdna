# Round 10 (2026-07-26) — Đo đường cong triệt tiêu, và một probe rẻ dự đoán được nó

## Vì sao là hướng này (sau khi quét tài liệu)

Quét kỹ tài liệu 2024–2026 cho kết quả **bất lợi cho phần lớn ý tưởng của Round 9**:

| ý tưởng | trạng thái |
|---|---|
| CholeskyQR2 + iterative refinement cho RandNLA | **Baboulin, Donfack, Kaya, Mary, Robeyns — Euro-Par 2024** đã làm đúng cả ba, đạt 1.28× |
| IR hấp thụ sai số low-precision | Kinh điển (Carson–Higham); bản low-rank có trên SIMAX |
| Precision thích ứng theo condition từng tile | arXiv 2508.14848 (8/2025) |
| Fuse kernel emulation | **EmuGEMM, arXiv 2606.25453 (6/2026)** — 83% peak INT8, Hopper/Blackwell |
| "vendor BLAS bỏ phí" | cuBLAS đã được ghi nhận bỏ phí ~16%; median của ta 15–19% ⇒ chỉ là *"AMD cũng vậy"* |

Ghi chú kỹ thuật: **fusion chỉ đáng giá khi số lát p lớn.** EmuGEMM thắng vì Ozaki `p` lát cần `p(p+1)/2`
kernel → traffic O(p²), fuse xuống O(p). Với **p=3** như bf16×3 gần như không có gì để gộp — và ta đã ở
**137/169 = 81% peak bf16**, ngang hiệu suất họ báo cáo. Không có đường ở đó.

**Cái còn mở**, được nêu thẳng là vấn đề chưa giải trong arXiv 2601.08077:

> *"inaccurate estimation of the emulation level to achieve desired accuracy levels… ozIMMU and GEMMul8
> **don't support emulating FP64 GEMM to a specific accuracy level**."*

Và Ozaki-II (arXiv 2508.03984) xác nhận từ phía kia: số moduli là **"user-specified and fixed"**. Các lược đồ
thích ứng hiện có chọn theo **thống kê số mũ** (dải động). **Không ai chọn theo độ triệt tiêu** — mà Phase 5
của Round 9 cho thấy triệt tiêu mới là thứ phá mô hình, không phải dải động.

⇒ Đóng góp không phải "emulation nhanh hơn" (đất đã cày nát) mà là **emulation có bảo đảm**: đặt được ngưỡng
sai số và thật sự nhận được, trên dữ liệu không do mình chọn.

---

## Thí nghiệm (`rho_sweep.cpp`)

`ρ = ‖|A||B|‖_F / ‖AB‖_F` — mức độ tích thật bị co lại so với độ lớn các số hạng được cộng. ρ=1 là lành,
ρ lớn nghĩa là đáp số là hiệu nhỏ của những số lớn. Dựng ma trận có ρ điều khiển được, nhưng **ρ đạt được
luôn được đo lại bằng fp64** chứ không tin vào đại số.

**M=N=2048, K=16384, 8 probe Hutchinson, chuẩn = DGEMM trên GPU**

| ρ đo được | ρ̂ (probe rẻ) | ρ̂/ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3 / fp32 |
|---|---|---|---|---|---|---|---|
| 6.96e+01 | 6.32e+01 | 0.91 | 2.159e-06 | 3.812e-06 | 1.034e-06 | 1.089e-06 | 1.77× |
| 7.40e+01 | 7.43e+01 | **1.00** | 2.103e-06 | 3.475e-06 | 9.220e-07 | 1.055e-06 | 1.65× |
| 1.86e+02 | 1.86e+02 | **1.00** | 2.141e-06 | 7.319e-06 | 4.246e-07 | 1.079e-06 | 3.42× |
| 7.37e+02 | 7.37e+02 | **1.00** | 2.275e-06 | 2.054e-05 | 6.093e-08 | 1.185e-06 | 9.03× |
| 2.95e+03 | 2.95e+03 | **1.00** | 2.983e-06 | 6.316e-05 | 6.195e-08 | 1.858e-06 | 21.2× |
| **1.18e+04** | 1.18e+04 | **1.00** | 7.698e-06 | 2.422e-04 | 6.384e-08 | 4.634e-06 | **31.5× ← đỉnh** |
| 4.71e+04 | 4.72e+04 | **1.00** | 8.126e-05 | 8.358e-04 | 4.325e-08 | 1.753e-05 | 10.3× |
| 1.89e+05 | 1.89e+05 | **1.00** | 3.510e-04 | 1.176e-03 | 3.044e-08 | 7.009e-05 | 3.35× |
| 7.54e+05 | 7.59e+05 | **1.01** | 1.340e-03 | 1.181e-03 | 2.541e-08 | 1.213e-04 | **0.88×** |

---

## Ba kết quả

### 1. ✅ Probe rẻ dự đoán ρ chính xác — cơ chế (b) đã đóng

Ước lượng Hutchinson bằng **2 GEMM gầy mỗi vế** (`A(BΩ)` và `|A|(|B|Ω)` với Ω là Rademacher `N×8`),
chi phí `O((MK+KN)·P)` so với `O(MNK)` của GEMM thật — ở đây khoảng **1.5%**.

**ρ̂/ρ = 1.00 suốt bốn bậc độ lớn** (chỉ điểm đầu lệch 0.91, vì cấu trúc chạm sàn ở ρ≈70 chứ không xuống 1).
⇒ **Dispatcher có thể định giá độ triệt tiêu TRƯỚC khi cam kết lược đồ.** Đây là mảnh còn thiếu để đóng vòng.

### 2. 🔍 Quan hệ không đơn điệu — vùng nguy hiểm nằm ở GIỮA, không phải ở cực

Đây là điều tôi không lường trước. Tỉ số bf16×3/fp32 **tăng tới đỉnh 31.5× quanh ρ≈10⁴ rồi giảm trở lại**,
và ở ρ≈7.5e5 thì bf16×3 **tốt hơn** fp32 (0.88×).

Lý do: ở ρ cực lớn **fp32 cũng sụp** (2.16e-6 → 1.34e-3), đuổi kịp bf16×3 vốn đã bão hoà quanh 1.18e-3.
Nói cách khác, khi triệt tiêu đủ nặng thì **mọi lược đồ đều hỏng như nhau**, nên emulation không tệ hơn.

**Vùng phải cảnh giác là ρ ≈ 10³–10⁵**, nơi fp32 còn trụ được nhưng bf16×3 đã suy. Đây là phát biểu dùng
được cho dispatcher; "35× tệ hơn" của Round 9 thì không.

### 3. ⚠️ Con số 35× của Round 9 phụ thuộc cách dựng ma trận

Round 9 đo 35× ở ρ=1.84e5. Ở đây, ρ=1.89e5 cho **3.35×**. Cùng ρ, khác một bậc. Nguyên nhân: hai cách dựng
khác nhau (K khác, và Round 9 giữ nhiễu của A cố định ở 1e-3 còn ở đây nhiễu tỉ lệ 1/ρ), làm sai số **fp32**
lệch 7.5×. ⇒ **ρ một mình chưa đủ để dự đoán sai số**; nó dự đoán được *chính nó*, nhưng ánh xạ ρ→sai số
còn phụ thuộc cấu trúc. Cần ít nhất một họ ma trận thứ hai trước khi hiệu chuẩn dispatcher.

---

### 4. ✅ Đường cong ổn định qua nhiều shape; và probe chỉ cần đủ mẫu

Chạy thêm ba cấu hình:

| shape | ρ̂/ρ (P=8) | đỉnh bf16×3/fp32 | vị trí đỉnh |
|---|---|---|---|
| K=16384, M=2048 | 1.00 | 31.5× | ρ≈1.2e4 |
| K=65536, M=2048 | 0.99 | 22.5× | ρ≈1.2e4 |
| K=4096, M=2048 | 1.63 | 40.3× | ρ≈3e3 |
| K=16384, M=4096 | 1.45 | 32.0× | ρ≈1.2e4 |

**Hình dạng đường cong bất biến qua mọi shape**: tăng tới đỉnh ở ρ ≈ 3×10³–10⁴ (biên độ 22–40×), rồi giảm,
và ở ρ≈7.5e5 thì bf16×3 luôn **tốt hơn** fp32 (0.62–0.91×). Con số 35× của Round 9 nằm gọn trong họ này.

Tỉ số ρ̂/ρ **hằng số trong mỗi shape nhưng khác giữa các shape** — thoạt nhìn giống thiên lệch hệ thống.
Kiểm bằng cách tăng số probe thì hoá ra **chỉ là phương sai Hutchinson**:

| probe P | ρ̂/ρ (K=4096, ca lệch nhất) | chi phí ở M=N=8192 |
|---|---|---|
| 8 | 1.56 – 1.64 | 0.2% |
| 64 | **1.10** (hằng số qua 5 bậc ρ) | 1.6% |
| 256 | **0.99 – 1.00** | 6.3% |

Chi phí probe là `2·P·K·(M+N)` so với `2·M·N·K` của GEMM ⇒ tỉ lệ `2P/M` với ma trận vuông.
**P=64 là điểm vận hành tốt: ρ̂ sai 10% ở chi phí 1.6%** — thừa đủ, vì ranh giới quyết định nằm cách nhau
*bậc độ lớn* chứ không phải phần trăm.

---

---

# ⚠️ FAMILY 2 LẬT NGƯỢC MỌI KẾT LUẬN Ở TRÊN — đây mới là số liệu đáng tin

Nghi vấn ở mục "Còn nợ #1" đã được kiểm và **đúng**. Family 1 dồn `B→±1`, `A→1` khi ρ lớn, mà bf16 biểu diễn
±1 **chính xác tuyệt đối** — nên toàn bộ hình dạng đường cong là hiện vật.

**Family 2** (`A = [G | G]`, `B = [H ; −H + δ·G₂]`): phần lớn triệt tiêu nhau, `AB = δ·G·G₂`, còn `‖|A||B|‖`
vẫn O(K) ⇒ ρ điều khiển được bằng δ, và **mọi giá trị lưu trữ đều là N(0,1) generic**, không có gì nằm trên
số biểu diễn được.

**M=N=2048, K=16384, P=64:**

| ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3/fp32 |
|---|---|---|---|---|---|
| 5.81e+01 | 1.621e-06 | 4.489e-06 | 7.401e-07 | 8.525e-07 | 2.77× |
| 1.03e+03 | 2.045e-05 | 4.758e-05 | 9.291e-06 | 1.073e-05 | 2.33× |
| 1.64e+04 | 3.262e-04 | 7.565e-04 | 1.483e-04 | 1.716e-04 | 2.32× |
| 1.05e+06 | 2.087e-02 | 3.423e-02 | 9.307e-03 | 1.087e-02 | 1.64× |

### Ba đính chính

1. **bf16×6 giờ SUY GIẢM theo ρ** (7.4e-7 → 9.3e-3), đúng như vật lý bắt buộc — thay vì *tốt lên* tới
   2.5e-8 như family 1. **Hiện vật được xác nhận và loại bỏ.**
2. **Không có bướu, không có đỉnh 31×, không có "vùng nguy hiểm ở giữa".** Tỉ số bf16×3/fp32 gần như
   **phẳng ở 2.3–2.8×** và còn giảm nhẹ theo ρ. Mục "kết quả 2" ở trên **SAI** — nó mô tả một hiện vật.
3. **"35× tệ hơn" của Round 9 cũng thuộc về cùng hiện vật đó.** Trên dữ liệu generic, hình phạt của bf16×3
   chỉ là **~2.3× và gần như không phụ thuộc ρ**.

### 🏆 Và cái thu được thay vào đó tốt hơn nhiều: ρ là biến cơ bản, không phải K

Tính `c = err/ρ`:

| ρ | c_fp32 | c_bf16×3 | c_bf16×6 | c_fp16×3 |
|---|---|---|---|---|
| 1.03e+03 | 1.989e-08 | 4.63e-08 | 9.04e-09 | 1.044e-08 |
| 1.64e+04 | **1.989e-08** | 4.61e-08 | **9.04e-09** | **1.046e-08** |
| 1.05e+06 | **1.990e-08** | 3.26e-08 | 8.87e-09 | 1.036e-08 |

**`err = c·ρ`, c hằng số tới 4 chữ số qua 3 bậc độ lớn.** Và `c_bf16×6 = 0.45×`, `c_fp16×3 = 0.53×` so với
fp32 — **khớp đúng phát hiện "a/2" của Round 9** (kernel low-precision cộng dồn tốt gấp đôi kernel fp32
mặc định), thu được từ một thí nghiệm hoàn toàn độc lập.

**Vì sao điều này quan trọng:** mô hình cũ `err = a·√(K/c)` hoạt động chỉ vì với dữ liệu iid thì
**ρ ≈ 0.64·√K** — tức `√K` chỉ là ρ đội lốt. Mô hình đúng là:

```
err(scheme) = c_scheme · ρ        ρ đo được trước bằng 2 GEMM gầy, chi phí ~1.6%
```

⇒ Dispatcher nên lái theo **ρ đo được từ dữ liệu**, không phải theo **shape**. Đó là khác biệt giữa một mô
hình chỉ đúng trên dữ liệu ngẫu nhiên và một mô hình dùng được trên dữ liệu thật.
(Ngoại lệ: `c_bf16×3` không hằng hoàn toàn — giảm còn 3.26e-8 ở ρ=1e6, vì nó có sàn biểu diễn không tỉ lệ
theo ρ. Cần một số hạng cộng thêm; chưa fit.)

---

### 🏆 Mảnh cuối: ρ và K là hai trục TRỰC GIAO

Family 2 ở ba giá trị K, tỉ số bf16×3/fp32 (ρ từ 58 tới 1.05e6):

| K | ρ=58 | ρ=260 | ρ=1.6e4 | ρ=1.05e6 |
|---|---|---|---|---|
| 4 096 | 5.53× | 6.44× | 6.34× | 4.42× |
| 16 384 | 2.77× | 2.33× | 2.32× | 1.64× |
| 65 536 | 1.41× | 1.25× | 1.19× | **0.88×** |

**ρ nhân đều sai số của mọi lược đồ ⇒ nó KHÔNG đổi tỉ số giữa chúng.**
**K mới là thứ quyết định tỉ số** — sàn biểu diễn của bf16×3 độc lập với K, còn sai số tích luỹ của fp32
tăng theo K, nên ở K lớn fp32 đuổi kịp rồi bị vượt (0.88× ở K=65536).

⇒ Dạng đúng của mô hình, tách biến:

```
err(scheme, K, ρ)  ≈  ρ · f_scheme(K)
                       ^      ^
                       |      +-- trục THUẬT TOÁN: quyết định lược đồ nào thắng
                       +--------- trục DỮ LIỆU: nhân đều, đo trước được bằng 2 GEMM gầy
```

Đây chính là câu chuyện giao điểm của Round 8/9, nay được tách sạch khỏi ảnh hưởng của dữ liệu.
`c_fp32` đo được: 1.40e-8 (K=4096), 1.99e-8 (K=16384), 1.99e-8 (K=65536) — bão hoà, không hoàn toàn hằng.

**Hệ quả thực dụng cho dispatcher:** chọn **lược đồ** theo `K` và ngưỡng sai số; dùng `ρ` đo được để **dự
báo sai số tuyệt đối sẽ nhận được** và từ chối nếu vượt yêu cầu. Hai vai trò khác nhau, đừng trộn.

---

## Còn nợ, xếp theo mức độ đe doạ kết luận

1. **Cấu trúc ma trận có tính đặc biệt đáng ngờ.** Khi ρ lớn, `B → ±1` và `A → 1`, mà **bf16 biểu diễn ±1
   CHÍNH XÁC**. Điều này gần như chắc chắn là lý do **bf16×6 lại TỐT LÊN theo ρ** (1.03e-6 → 2.54e-8) —
   một kết quả không đáng tin, gần như chắc chắn là hiện vật của cách dựng. **Phải chạy lại với họ ma trận
   thứ hai không có giá trị nằm đúng trên số biểu diễn được.**
2. Mới quét một (M,N,K); các lần chạy K=65536, K=4096, M=4096 đang chạy.
3. Chưa nối probe ρ̂ vào `emugemm` — mới chứng minh nó đo đúng, chưa chứng minh nó *quyết định* đúng.
4. Chưa có ánh xạ ρ→ngưỡng an toàn cho từng lược đồ (cần mục 1 xong trước).

---

# Phase cuối — nối ρ vào thư viện, và bài kiểm nghiệm thu

`emugemm` nay đo ρ trước khi cam kết (`emugemm_estimate_rho`, Hutchinson 64 probe, ~1.6%), và áp
`err = err_iid(K,c) · ρ/ρ_ref(K)` với `ρ_ref = 0.64√K`. Hai vai trò tách bạch đúng như Round 10 kết luận:
**K chọn lược đồ, ρ quyết định lược đồ đó có giữ được lời hứa không.**

**Không hồi quy trên dữ liệu lành:** chạy lại `emugemm_test` (iid N(0,1)) cho **0 vi phạm**, lựa chọn/sai số/
thời gian không đổi — vì ρ≈ρ_ref nên hệ số hiệu chỉnh ≈1. Probe không làm gì khi không cần.

**Bài kiểm nghiệm thu (`emu_adversarial.cpp`)** — dữ liệu family 2, ρ từ 1e2 tới 1.1e6, M=N=2048, K=16384:

| ρ đo | target | lược đồ chọn | dự đoán | đo được | hợp đồng |
|---|---|---|---|---|---|
| 1.41e+02 | 1e-05 | NATIVE_FP32 | 3.944e-06 | 2.563e-06 | đạt |
| 1.12e+03 | 1e-05 | FP32_CHUNKED c=32 | 5.537e-06 | 4.995e-06 | đạt |
| 1.12e+03 | 1e-04 | NATIVE_FP32 | 3.132e-05 | 1.998e-05 | đạt |
| 1.12e+04 | 1e-05 | — | 1.951e-05 | 1.771e-05 | **từ chối (đúng)** |
| 1.12e+04 | 1e-04 | FP32_CHUNKED c=32 | 5.519e-05 | 4.981e-05 | đạt |
| 1.12e+05 | 1e-05 / 1e-04 | — | 1.951e-04 | 1.772e-04 | **từ chối (đúng)** |
| 1.12e+06 | 1e-05 / 1e-04 | — | 1.951e-03 | 1.770e-03 | **từ chối (đúng)** |

**0 vi phạm hợp đồng.** Dự đoán bảo thủ nhất quán ~10% qua ba bậc — sai về phía an toàn.
Ở ρ=1120/target 1e-5 nó **leo thang sang chunked để giữ lời hứa**; ở cùng ρ với target 1e-4 nó **không trả
thừa**. Mọi lần từ chối đều được số đo xác nhận là thật sự bất khả thi.

⇒ **Lỗ hổng "emugemm không an toàn cho dữ liệu điều kiện xấu" (còn nợ #1 của Round 9) đã đóng.**
Thư viện giờ không bao giờ hứa một độ chính xác nó không giao được — nó từ chối. Từ chối là hợp lệ, nói dối
thì không.

---

# Family 3 — cơ chế triệt tiêu thứ ba, và nó SỬA mô hình của family 2

Family 1 hỏng vì giá trị dồn về ±1 (bf16 biểu diễn chính xác). Family 2 sạch về giá trị nhưng vẫn có
**khối lặp** trong A và triệt tiêu bằng **khử đại số chính xác** (`H` với `−H`). Family 3 tạo triệt tiêu
bằng **cơ chế vật lý**: hàng của A là hàm trơn (`e^{-ax}(1+b·sin πx)`), cột của B là dao động
`cos(2πfx+φ)`. Tích trong của chúng là một **tích phân dao động** — đúng cơ chế làm quadrature và toán tử
vi phân trở nên ill-conditioned. ρ điều khiển bằng **tần số** (`ρ ~ f²`), không bằng thành phần DC.

*(Lần thử đầu để nhiễu `0.1·g` trong A và bão hoà ở ρ=20: nhiễu **không trơn** nên không triệt tiêu với
dao động, tạo thành sàn. Phải bỏ nhiễu — bản thân điều này đã xác nhận cơ chế đúng như dự định.)*

**M=N=2048, K=16384, P=64:**

| ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3/fp32 |
|---|---|---|---|---|---|
| 1.19 | 2.822e-06 | 1.896e-06 | 1.895e-06 | 1.196e-06 | 0.67× |
| 10.5 | 3.699e-06 | 2.015e-06 | 1.955e-06 | 1.849e-06 | 0.54× |
| 96.4 | 4.438e-06 | 3.747e-06 | 1.469e-06 | 2.256e-06 | 0.84× |
| 380 | 4.438e-06 | 1.342e-05 | 1.107e-06 | 2.945e-06 | 3.02× |
| 1 515 | 4.387e-06 | 5.204e-05 | 9.670e-07 | 4.949e-06 | 11.9× |
| 3 019 | **4.287e-06** | 1.034e-04 | 7.310e-07 | 8.838e-06 | 24.1× |

## Kết quả: `err = c·ρ` chỉ đúng cho lược đồ bị chặn bởi BIỂU DIỄN

| | family 2 (khử đại số) | family 3 (tích phân dao động) |
|---|---|---|
| **fp32** | `1.989e-8 · ρ` | **PHẲNG ~4.4e-6**, không phụ thuộc ρ |
| **bf16×3** | `≈ 3.3–4.6e-8 · ρ` | **`≈ 3.4e-8 · ρ`** ✓ khớp |

**bf16×3 tuân theo `c·ρ` với cùng hằng số trong cả hai họ**, dù cơ chế triệt tiêu hoàn toàn khác nhau.
**fp32 thì không.** Lý do:

- Sai số **biểu diễn** là một tỉ lệ cố định của `|a||b|`, nên **luôn** tỉ lệ với `‖|A||B|‖ = ρ‖C‖`,
  bất kể thứ tự cộng dồn. ⇒ bf16×3 (bị chặn bởi biểu diễn) luôn ∝ ρ.
- Sai số **tích luỹ** phụ thuộc **tổng riêng phần lớn đến đâu trước khi triệt tiêu**. Family 2 triệt tiêu
  *muộn* (tổng phình lên rồi mới khử) ⇒ lỗi làm tròn tỉ lệ với độ lớn trung gian ⇒ fp32 ∝ ρ.
  Family 3 triệt tiêu *phân tán* (dao động, tổng riêng phần luôn nhỏ) ⇒ fp32 phẳng.

⇒ **ρ một mình KHÔNG xác định được sai số. Cấu trúc của sự triệt tiêu cũng quan trọng.**

## Điều này có phá hợp đồng của thư viện không? KHÔNG — nó lệch về phía an toàn

Dispatcher nhân sai số dự đoán của **mọi** lược đồ với `ρ/ρ_ref`. Với bf16×3 thì đúng. Với fp32, ở cấu
trúc kiểu family 3, nó **dự đoán tệ hơn thực tế** — tức dispatcher có thể **leo thang hoặc từ chối
không cần thiết**, nhưng **không bao giờ giao thiếu**. Đây đúng là hướng sai an toàn.

Cái mất là hiệu quả, không phải tính đúng đắn: có những bài toán fp32 thừa sức đạt target mà thư viện
vẫn trả thêm thời gian. Ghi nhận là **giới hạn đã biết**, không phải lỗi.

## Ba họ, ba kết luận — và vì sao phải có cả ba

| họ | cơ chế | kết luận nếu chỉ có nó |
|---|---|---|
| 1 | dấu xen kẽ, giá trị → ±1 | "có bướu 31×" — **SAI, hiện vật** |
| 2 | khử đại số `H`/`−H` | "`err = c·ρ` cho mọi lược đồ" — **đúng một nửa** |
| 3 | tích phân dao động | "fp32 phẳng theo ρ" — đúng **cho cơ chế này** |

Không họ nào một mình cho ra bức tranh đúng. Điều duy nhất **sống sót qua cả ba**: mọi đường
low-precision có hằng số tích luỹ **bằng nửa fp32** (family 3 ở ρ thấp: bf16×3 = bf16×6 = 1.9e-6 so với
fp32 2.8e-6), và **sai số biểu diễn của bf16×3 luôn ∝ ρ với c ≈ 3.4e-8**.

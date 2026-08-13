# Round 10 (2026-07-26) — Measuring the cancellation curve, and a cheap probe that predicts it

## Why this direction (after a literature scan)

A careful scan of the 2024–2026 literature is **unfavourable to most of Round 9's ideas**:

| idea | status |
|---|---|
| CholeskyQR2 + iterative refinement for RandNLA | **Baboulin, Donfack, Kaya, Mary, Robeyns — Euro-Par 2024** did all three, reaching 1.28× |
| IR absorbing low-precision error | classical (Carson–Higham); a low-rank variant exists in SIMAX |
| Per-tile precision adapted to condition number | arXiv 2508.14848 (Aug 2025) |
| Fusing emulation kernels | **EmuGEMM, arXiv 2606.25453 (Jun 2026)** — 83% of INT8 peak, Hopper/Blackwell |
| "vendor BLAS leaves performance on the table" | cuBLAS is already documented leaving ~16%; our median is 15–19%, so this is only *"AMD too"* |

A technical note: **fusion only pays when the slice count p is large.** EmuGEMM wins because an Ozaki
scheme with `p` slices needs `p(p+1)/2` kernels → O(p²) traffic, fused down to O(p). With **p=3**, as
in bf16×3, there is almost nothing to fuse — and we are already at **137/169 = 81% of bf16 peak**, on
par with the efficiency they report. There is no opening there.

**What remains open** is named directly as an unsolved problem in arXiv 2601.08077:

> *"inaccurate estimation of the emulation level to achieve desired accuracy levels… ozIMMU and
> GEMMul8 **don't support emulating FP64 GEMM to a specific accuracy level**."*

Ozaki-II (arXiv 2508.03984) confirms it from the other side: the modulus count is
**"user-specified and fixed"**. Existing adaptive schemes select on **exponent statistics** (dynamic
range). **Nobody selects on cancellation** — and Round 9's Phase 5 showed that cancellation, not
dynamic range, is what breaks the model.

⇒ The contribution is not "emulation is faster" (well-tilled ground) but **emulation with a
guarantee**: setting an error threshold and actually getting it, on data one did not choose.

---

## Experiment (`rho_sweep.cpp`)

`ρ = ‖|A||B|‖_F / ‖AB‖_F` — how far the true product shrinks relative to the magnitude of the terms
being summed. ρ=1 is benign; large ρ means the answer is a small difference of large numbers. The
matrices are constructed with controllable ρ, but **the achieved ρ is always re-measured in fp64**
rather than trusted from the algebra.

**M=N=2048, K=16384, 8 Hutchinson probes, reference = DGEMM on the GPU**

| measured ρ | ρ̂ (cheap probe) | ρ̂/ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3 / fp32 |
|---|---|---|---|---|---|---|---|
| 6.96e+01 | 6.32e+01 | 0.91 | 2.159e-06 | 3.812e-06 | 1.034e-06 | 1.089e-06 | 1.77× |
| 7.40e+01 | 7.43e+01 | **1.00** | 2.103e-06 | 3.475e-06 | 9.220e-07 | 1.055e-06 | 1.65× |
| 1.86e+02 | 1.86e+02 | **1.00** | 2.141e-06 | 7.319e-06 | 4.246e-07 | 1.079e-06 | 3.42× |
| 7.37e+02 | 7.37e+02 | **1.00** | 2.275e-06 | 2.054e-05 | 6.093e-08 | 1.185e-06 | 9.03× |
| 2.95e+03 | 2.95e+03 | **1.00** | 2.983e-06 | 6.316e-05 | 6.195e-08 | 1.858e-06 | 21.2× |
| **1.18e+04** | 1.18e+04 | **1.00** | 7.698e-06 | 2.422e-04 | 6.384e-08 | 4.634e-06 | **31.5× ← peak** |
| 4.71e+04 | 4.72e+04 | **1.00** | 8.126e-05 | 8.358e-04 | 4.325e-08 | 1.753e-05 | 10.3× |
| 1.89e+05 | 1.89e+05 | **1.00** | 3.510e-04 | 1.176e-03 | 3.044e-08 | 7.009e-05 | 3.35× |
| 7.54e+05 | 7.59e+05 | **1.01** | 1.340e-03 | 1.181e-03 | 2.541e-08 | 1.213e-04 | **0.88×** |

---

## Results

### 1. The cheap probe predicts ρ accurately

A Hutchinson estimate using **two skinny GEMMs per side** (`A(BΩ)` and `|A|(|B|Ω)` with Ω a Rademacher
`N×8`), costing `O((MK+KN)·P)` against `O(MNK)` for the real GEMM — about **1.5%** here.

**ρ̂/ρ = 1.00 across four orders of magnitude** (only the first point is off, at 0.91, because the
construction floors out at ρ≈70 rather than reaching 1).
⇒ **A dispatcher can price cancellation BEFORE committing to a scheme.** This was the missing piece.

### 2. A non-monotonic relationship — the dangerous region is in the MIDDLE, not at the extremes

This was not anticipated. The bf16×3/fp32 ratio **rises to a peak of 31.5× around ρ≈10⁴ and then falls
back**, and at ρ≈7.5e5 bf16×3 is **better** than fp32 (0.88×).

The reason: at extreme ρ **fp32 collapses too** (2.16e-6 → 1.34e-3), catching up with bf16×3, which
has already saturated around 1.18e-3. Put differently, once cancellation is severe enough **every
scheme fails equally**, so emulation is no worse.

**The region to watch is ρ ≈ 10³–10⁵**, where fp32 still holds up but bf16×3 has degraded. That is a
usable statement for a dispatcher; Round 9's "35× worse" is not.

*(Superseded below — see the family 2 section. This shape is an artifact of the matrix construction.)*

### 3. Round 9's 35× figure depends on how the matrices were built

Round 9 measured 35× at ρ=1.84e5. Here, ρ=1.89e5 gives **3.35×** — same ρ, an order of magnitude
apart. The cause is two different constructions (different K, and Round 9 held A's noise fixed at 1e-3
whereas here noise scales as 1/ρ), which moves the **fp32** error by 7.5×. ⇒ **ρ alone does not
predict error**; it predicts *itself*, but the ρ→error mapping also depends on structure. At least a
second matrix family is needed before calibrating a dispatcher.

### 4. The curve is stable across shapes; the probe just needs enough samples

Three further configurations:

| shape | ρ̂/ρ (P=8) | peak bf16×3/fp32 | peak location |
|---|---|---|---|
| K=16384, M=2048 | 1.00 | 31.5× | ρ≈1.2e4 |
| K=65536, M=2048 | 0.99 | 22.5× | ρ≈1.2e4 |
| K=4096, M=2048 | 1.63 | 40.3× | ρ≈3e3 |
| K=16384, M=4096 | 1.45 | 32.0× | ρ≈1.2e4 |

**The curve shape is invariant across shapes**: rising to a peak at ρ ≈ 3×10³–10⁴ (amplitude 22–40×),
then falling, with bf16×3 always **better** than fp32 at ρ≈7.5e5 (0.62–0.91×). Round 9's 35× sits
inside this family.

The ρ̂/ρ ratio is **constant within a shape but differs between shapes**, which initially looks like a
systematic bias. Increasing the probe count shows it is **only Hutchinson variance**:

| probes P | ρ̂/ρ (K=4096, worst case) | cost at M=N=8192 |
|---|---|---|
| 8 | 1.56 – 1.64 | 0.2% |
| 64 | **1.10** (constant across 5 orders of ρ) | 1.6% |
| 256 | **0.99 – 1.00** | 6.3% |

Probe cost is `2·P·K·(M+N)` against `2·M·N·K` for the GEMM ⇒ a ratio of `2P/M` for square matrices.
**P=64 is a good operating point: ρ̂ within 10% at 1.6% cost** — more than enough, since the decision
boundaries are *orders of magnitude* apart, not percentages.

---

# ⚠ Family 2 overturns the conclusions above — this is the data to trust

The suspicion recorded as open item #1 was checked and is **correct**. Family 1 drives `B→±1` and
`A→1` as ρ grows, and bf16 represents ±1 **exactly** — so the entire curve shape was an artifact.

**Family 2** (`A = [G | G]`, `B = [H ; −H + δ·G₂]`): most terms cancel, `AB = δ·G·G₂`, while
`‖|A||B|‖` stays O(K) ⇒ ρ is controlled by δ, and **every stored value is generic N(0,1)**, with
nothing landing on an exactly representable number.

**M=N=2048, K=16384, P=64:**

| ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3/fp32 |
|---|---|---|---|---|---|
| 5.81e+01 | 1.621e-06 | 4.489e-06 | 7.401e-07 | 8.525e-07 | 2.77× |
| 1.03e+03 | 2.045e-05 | 4.758e-05 | 9.291e-06 | 1.073e-05 | 2.33× |
| 1.64e+04 | 3.262e-04 | 7.565e-04 | 1.483e-04 | 1.716e-04 | 2.32× |
| 1.05e+06 | 2.087e-02 | 3.423e-02 | 9.307e-03 | 1.087e-02 | 1.64× |

### Three corrections

1. **bf16×6 now DEGRADES with ρ** (7.4e-7 → 9.3e-3), as physics requires — instead of *improving* to
   2.5e-8 as in family 1. The artifact is confirmed and removed.
2. **No hump, no 31× peak, no "dangerous middle region".** The bf16×3/fp32 ratio is nearly **flat at
   2.3–2.8×** and even decreases slightly with ρ. Result 2 above is **wrong** — it describes an
   artifact.
3. **Round 9's "35× worse" belongs to the same artifact.** On generic data bf16×3's penalty is only
   **~2.3× and almost independent of ρ**.

### What replaces it is better: ρ is the fundamental variable, not K

Computing `c = err/ρ`:

| ρ | c_fp32 | c_bf16×3 | c_bf16×6 | c_fp16×3 |
|---|---|---|---|---|
| 1.03e+03 | 1.989e-08 | 4.63e-08 | 9.04e-09 | 1.044e-08 |
| 1.64e+04 | **1.989e-08** | 4.61e-08 | **9.04e-09** | **1.046e-08** |
| 1.05e+06 | **1.990e-08** | 3.26e-08 | 8.87e-09 | 1.036e-08 |

**`err = c·ρ`, with c constant to four digits across three orders of magnitude.** And
`c_bf16×6 = 0.45×`, `c_fp16×3 = 0.53×` relative to fp32 — **matching Round 9's "a/2" finding** (the
low-precision kernels accumulate twice as well as the default fp32 kernel), recovered from a fully
independent experiment.

**Why this matters:** the old model `err = a·√(K/c)` worked only because for i.i.d. data
**ρ ≈ 0.64·√K** — `√K` was ρ in disguise. The correct model is:

```
err(scheme) = c_scheme · ρ        ρ measured up front by 2 skinny GEMMs, ~1.6% cost
```

⇒ A dispatcher should steer on **ρ measured from the data**, not on **shape**. That is the difference
between a model that holds on random data and one usable on real data.
(Exception: `c_bf16×3` is not perfectly constant — it falls to 3.26e-8 at ρ=1e6, because it has a
representation floor that does not scale with ρ. An additive term is needed; not yet fitted.)

### The last piece: ρ and K are ORTHOGONAL axes

Family 2 at three values of K, bf16×3/fp32 ratio (ρ from 58 to 1.05e6):

| K | ρ=58 | ρ=260 | ρ=1.6e4 | ρ=1.05e6 |
|---|---|---|---|---|
| 4 096 | 5.53× | 6.44× | 6.34× | 4.42× |
| 16 384 | 2.77× | 2.33× | 2.32× | 1.64× |
| 65 536 | 1.41× | 1.25× | 1.19× | **0.88×** |

**ρ scales every scheme's error equally, so it does NOT change the ratio between them.**
**K is what decides the ratio** — bf16×3's representation floor is independent of K, while fp32's
accumulation error grows with K, so at large K fp32 catches up and then passes it (0.88× at K=65536).

⇒ The correct separable form of the model:

```
err(scheme, K, ρ)  ≈  ρ · f_scheme(K)
                       ^      ^
                       |      +-- ALGORITHM axis: decides which scheme wins
                       +--------- DATA axis: scales everything, measurable up front
```

This is the crossover story of Rounds 8 and 9, now cleanly separated from the influence of the data.
Measured `c_fp32`: 1.40e-8 (K=4096), 1.99e-8 (K=16384), 1.99e-8 (K=65536) — saturating, not perfectly
constant.

**Practical consequence for the dispatcher:** choose the **scheme** from `K` and the error threshold;
use the measured `ρ` to **predict the absolute error that will be delivered** and refuse if it exceeds
the requirement. Two distinct roles; do not conflate them.

---

## Open items, ordered by how much they threaten the conclusions

1. **The matrix construction is suspiciously special.** At large ρ, `B → ±1` and `A → 1`, and **bf16
   represents ±1 exactly**. This is almost certainly why **bf16×6 improves with ρ** (1.03e-6 →
   2.54e-8) — an untrustworthy result, near-certainly an artifact of the construction. **Must be
   re-run with a second matrix family whose values do not land on exactly representable numbers.**
   *(Done — see the family 2 section above, which confirms the artifact.)*
2. Only one (M,N,K) swept so far; the K=65536, K=4096 and M=4096 runs are in progress.
3. The ρ̂ probe is not yet wired into `emugemm` — it has been shown to measure correctly, not to
   *decide* correctly.
4. No ρ→safe-threshold mapping per scheme yet (needs item 1 first).

---

# Final phase — wiring ρ into the library, and an acceptance test

`emugemm` now measures ρ before committing (`emugemm_estimate_rho`, 64 Hutchinson probes, ~1.6%) and
applies `err = err_iid(K,c) · ρ/ρ_ref(K)` with `ρ_ref = 0.64√K`. The two roles are separated exactly
as Round 10 concluded: **K chooses the scheme, ρ decides whether that scheme can keep its promise.**

**No regression on benign data:** re-running `emugemm_test` (i.i.d. N(0,1)) gives **0 violations**,
with selection, error and time unchanged — because ρ≈ρ_ref makes the correction factor ≈1. The probe
does nothing when nothing is needed.

**Acceptance test (`emu_adversarial.cpp`)** — family 2 data, ρ from 1e2 to 1.1e6, M=N=2048, K=16384:

| measured ρ | target | scheme chosen | predicted | measured | contract |
|---|---|---|---|---|---|
| 1.41e+02 | 1e-05 | NATIVE_FP32 | 3.944e-06 | 2.563e-06 | met |
| 1.12e+03 | 1e-05 | FP32_CHUNKED c=32 | 5.537e-06 | 4.995e-06 | met |
| 1.12e+03 | 1e-04 | NATIVE_FP32 | 3.132e-05 | 1.998e-05 | met |
| 1.12e+04 | 1e-05 | — | 1.951e-05 | 1.771e-05 | **refused (correctly)** |
| 1.12e+04 | 1e-04 | FP32_CHUNKED c=32 | 5.519e-05 | 4.981e-05 | met |
| 1.12e+05 | 1e-05 / 1e-04 | — | 1.951e-04 | 1.772e-04 | **refused (correctly)** |
| 1.12e+06 | 1e-05 / 1e-04 | — | 1.951e-03 | 1.770e-03 | **refused (correctly)** |

**0 contract violations.** The prediction is consistently ~10% conservative across three orders of
magnitude — erring safe. At ρ=1120 with target 1e-5 it **escalates to chunked to keep the promise**;
at the same ρ with target 1e-4 it does **not** overpay. Every refusal is confirmed by measurement to
be genuinely impossible.

⇒ **Round 9's open item #1 — "emugemm is not safe for ill-conditioned data" — is closed.** The library
never promises an accuracy it cannot deliver; it refuses. Refusing is valid, lying is not.

---

# Family 3 — a third cancellation mechanism, which corrects family 2's model

Family 1 failed because values collapse to ±1 (represented exactly in bf16). Family 2 is clean in its
values but still contains **repeated blocks** in A and cancels by **exact algebraic annihilation**
(`H` against `−H`). Family 3 creates cancellation by a **physical mechanism**: rows of A are smooth
functions (`e^{-ax}(1+b·sin πx)`), columns of B are oscillatory (`cos(2πfx+φ)`). Their inner product
is an **oscillatory integral** — the mechanism that makes quadrature and differential operators
ill-conditioned. ρ is controlled by **frequency** (`ρ ~ f²`), not by a DC component.

*(A first attempt kept `0.1·g` noise in A and saturated at ρ=20: the noise is **not smooth**, so it
does not cancel against oscillation and forms a floor. Removing it was necessary — which itself
confirms the mechanism works as intended.)*

**M=N=2048, K=16384, P=64:**

| ρ | fp32 | bf16×3 | bf16×6 | fp16×3 | bf16×3/fp32 |
|---|---|---|---|---|---|
| 1.19 | 2.822e-06 | 1.896e-06 | 1.895e-06 | 1.196e-06 | 0.67× |
| 10.5 | 3.699e-06 | 2.015e-06 | 1.955e-06 | 1.849e-06 | 0.54× |
| 96.4 | 4.438e-06 | 3.747e-06 | 1.469e-06 | 2.256e-06 | 0.84× |
| 380 | 4.438e-06 | 1.342e-05 | 1.107e-06 | 2.945e-06 | 3.02× |
| 1 515 | 4.387e-06 | 5.204e-05 | 9.670e-07 | 4.949e-06 | 11.9× |
| 3 019 | **4.287e-06** | 1.034e-04 | 7.310e-07 | 8.838e-06 | 24.1× |

## Result: `err = c·ρ` holds only for schemes bounded by REPRESENTATION

| | family 2 (algebraic annihilation) | family 3 (oscillatory integral) |
|---|---|---|
| **fp32** | `1.989e-8 · ρ` | **FLAT at ~4.4e-6**, independent of ρ |
| **bf16×3** | `≈ 3.3–4.6e-8 · ρ` | **`≈ 3.4e-8 · ρ`** — matches |

**bf16×3 follows `c·ρ` with the same constant in both families**, despite entirely different
cancellation mechanisms. **fp32 does not.** The reason:

- **Representation** error is a fixed fraction of `|a||b|`, so it is **always** proportional to
  `‖|A||B|‖ = ρ‖C‖`, regardless of summation order. ⇒ bf16×3, being representation-bound, is always
  ∝ ρ.
- **Accumulation** error depends on **how large the partial sums get before cancelling**. Family 2
  cancels *late* (the sum inflates, then annihilates) ⇒ rounding error scales with the intermediate
  magnitude ⇒ fp32 ∝ ρ. Family 3 cancels *diffusely* (oscillation keeps partial sums small) ⇒ fp32 is
  flat.

⇒ **ρ alone does not determine the error. The structure of the cancellation matters too.**

## Does this break the library's contract? No — it errs safe

The dispatcher multiplies **every** scheme's predicted error by `ρ/ρ_ref`. For bf16×3 that is correct.
For fp32 under family-3-like structure it **predicts worse than reality** — so the dispatcher may
**escalate or refuse unnecessarily**, but it will **never under-deliver**. That is the safe direction.

What is lost is efficiency, not correctness: there are problems where fp32 would comfortably hit the
target and the library still spends extra time. Recorded as a **known limitation**, not a bug.

## Three families, three conclusions — and why all three were necessary

| family | mechanism | conclusion if taken alone |
|---|---|---|
| 1 | alternating signs, values → ±1 | "there is a 31× hump" — **wrong, an artifact** |
| 2 | algebraic annihilation `H`/`−H` | "`err = c·ρ` for every scheme" — **half right** |
| 3 | oscillatory integral | "fp32 is flat in ρ" — correct **for this mechanism** |

No single family gives the right picture. What **survives all three**: every low-precision path has an
accumulation constant **half that of fp32** (family 3 at low ρ: bf16×3 = bf16×6 = 1.9e-6 against
fp32's 2.8e-6), and **bf16×3's representation error is always ∝ ρ with c ≈ 3.4e-8**.

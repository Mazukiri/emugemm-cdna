# Building

Requires ROCm (tested 7.2.3) and a CDNA GPU. Replace `gfx90a` with your arch.

```bash
ARCH=gfx90a
CXXFLAGS="-O3 -Wno-deprecated-declarations --offload-arch=$ARCH -Isrc"

# library + invariant suite
hipcc $CXXFLAGS bench/emugemm_test.cpp   src/emugemm.cpp -o emugemm_test   -lrocblas
# contract test on ill-conditioned data
hipcc $CXXFLAGS bench/emu_adversarial.cpp src/emugemm.cpp -o emu_adversarial -lrocblas
# randomized SVD application (needs rocSOLVER)
hipcc $CXXFLAGS bench/randsvd2.cpp        src/emugemm.cpp -o randsvd2 -lrocblas -lrocsolver

# standalone studies (no library dependency)
hipcc $CXXFLAGS bench/mfma_peak2.cpp  -o mfma_peak2
hipcc $CXXFLAGS bench/rho_sweep.cpp   -o rho_sweep   -lrocblas
hipcc $CXXFLAGS bench/flat_error.cpp  -o flat_error  -lrocblas
hipcc $CXXFLAGS bench/error_model.cpp -o error_model -lrocblas
hipcc $CXXFLAGS bench/gen_tune_table.cpp -o gen_tune_table -lrocblas

# re-measures the tuning table without the ordering bias, and spot-checks the published figures
hipcc $CXXFLAGS bench/audit_table.cpp    -o audit_table    -lrocblas
hipcc $CXXFLAGS bench/validate_issue.cpp -o validate_issue -lrocblas
```

The harnesses under `issue9985/src/` build the same way; `three_way2.cpp` additionally needs
`-lhipblaslt`.

## On new hardware, run this first

```bash
./mfma_peak2
```

It measures the raw fp64 / fp32 / fp16 / bf16 / int8 matrix-core rates. The whole economic argument
rests on the bf16:fp32 ratio `R`: the ceiling for a 3-product scheme is `R/3`. On MI250 `R = 3.98`, so
the ceiling is 1.33× and there is little headroom. On CDNA3/CDNA4 `R` is larger and the conclusions
may change qualitatively — including which schemes are worth using at all.

## Regenerating the tuning table

Solution indices are **not portable** across ROCm versions or architectures.

```bash
for i in $(seq 0 7); do
  HIP_VISIBLE_DEVICES=$i ./gen_tune_table data/tune_table.csv --shard $i/8 --budget 120 &
done; wait
```

Resumable: rerunning skips `(M,N,K,dtype)` rows already present.

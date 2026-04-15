# Blocked matvec kernel generator — experiment log

Experimental feature: tile the `K × K` complex matrix-vector multiply
inside each chunk's loop body into blocks of `T` output rows, emitting
explicit basic-block boundaries + `store volatile` accumulator retirement
to force LLVM to break the matvec's peak live-range pressure.

Activated by env var `CAST_BLOCK_GEMM_T=N` (T ∈ {4, 8, 16}, unset or 0 =
legacy). Only engages when `LK ≥ 4·T` (the empirical break-even — see
experiment 2 below).  Orthogonal to Option 3's lo-heavy load tiling —
the two compose cleanly, each attacking a different phase.

## Status: experimentally validated, opt-in

**Headline results (F64 W512, dense Haar unitary, `q=[0..6)`, sv=22q):**

| metric | T=0 | T=8 | delta |
|---|--:|--:|--:|
| compile total | 3877 ms | 3574 ms | **−8%** |
| exec @ 1 thr (compute-bound) | 90.2 ms | 68.7 ms | **−24%** |
| exec @ 4 thr | 24.9 ms | 19.8 ms | **−21%** |
| exec @ 32 thr (memory-bound) | 4.30 ms | 3.57 ms | **−17%** |

**Block-GEMM's main contribution is runtime, not compile-time.** The
initial hypothesis that blocking would primarily help LLVM's codegen
turned out wrong — `volatile` stores force the native regalloc to use a
tighter, L1-resident scratch buffer instead of LLVM's automatic spills,
and the resulting machine code is genuinely more efficient end-to-end.

## Implementation notes

- **Env var `CAST_BLOCK_GEMM_T`** controls tile size. Read once at
  `cast_cpu_generate_kernel_ir` entry, stashed in `KernelCodegen`.
  Values must be a power of two in [2, 64]; anything else silently
  reverts to legacy to avoid footguns.
- **Gate `LK ≥ 4·T`**: blocks only when the output partition count is
  large enough for blocking to pay off.  `LK=1` (hi-heavy) and `LK ≤ 2·T`
  (small lo-heavy) fall through to the legacy straight-line matvec.
- **Live-range break via `store volatile` / `load volatile`** through a
  per-kernel stack scratch buffer (one `alloca` in the entry block of
  size `LK × <s × scalar>`, one each for re and im).  Volatile memory
  ops cannot be reordered or elided, which is the only mechanism that
  reliably retires accumulators between blocks regardless of what
  SimplifyCFG does to the block BBs in O1.
- **Correctness** locked in by the existing `layout_coverage` tests
  (67 total). All pass under `CAST_BLOCK_GEMM_T ∈ {4, 8, 16}`.

## Measurement harness

`src/bin/bench_compile.rs` now supports runtime measurement as a
first-class mode:

```
--apply-budget SEC     # adaptive-time the kernel against a statevector
--sv-qubits N          # statevector size (default 26)
--threads a,b,c,...    # thread-count sweep (default 1,4,32)
```

Compile phase decomposition (ir / opt / cg) retained from Option 3 work.

---

## Experiment 1 — T sweep at k=6 all-lo F64 W512 dense (2026-04-15)

Goal: find the best tile size for the worst-case compile/runtime target.

| T | sv=20q compile | exec @ 1 thr | exec @ 4 thr | exec @ 32 thr |
|--:|--:|--:|--:|--:|
| 0  | 3875 ms | 23.2 ms | 6.85 ms | 1.63 ms |
| 4  | 3614 (−7%) | 20.3 (−12%) | 5.94 (−13%) | 1.46 (−10%) |
| **8**  | **3572 (−8%)** | **17.2 (−26%)** | **5.07 (−26%)** | **1.38 (−15%)** |
| 16 | 3673 (−5%) | 18.0 (−22%) | 5.36 (−22%) | 1.41 (−13%) |

Same relative pattern at sv=24q. **T=8 is the universal winner.** T=4
is close on runtime but less on compile; T=16 loses slightly to T=8
because the per-block BB control overhead grows with fewer, larger
blocks.

Raw: `benchmarks/block_gemm/exp1_k6_f64_w512_alllo.txt`

## Experiment 2 — k sweep with T=8 (initial gate `LK > T`)

Goal: find where the win applies vs where it regresses.

| k | compile T=0 | compile T=8 | exec@1 T=0 | exec@1 T=8 |
|--:|--:|--:|--:|--:|
| 3 (LK=8)  | 39.3 ms | 36.1 (−8%) | 8.40 ms | 8.63 (+3%, noise) |
| 4 (LK=16) | 134 ms | 127 (−5%) | 13.6 ms | **14.4 (+6%)** |
| 5 (LK=32) | 645 ms | 581 (−10%) | 35.2 ms | 34.5 (0%) |
| 6 (LK=64) | 3877 ms | 3608 (−7%) | **90.2 ms** | **68.0 (−25%)** |

**Finding:** runtime regression at k=4 (LK=16, only 2 blocks of T=8). The
volatile store overhead isn't amortised enough. Gate tightened to
`LK ≥ 4·T` → k=4 now falls through to legacy.

Raw: `benchmarks/block_gemm/exp2_k_sweep.txt`,
`benchmarks/block_gemm/exp2b_k_sweep_tightened.txt`

## Experiment 3 — density × layout at k=6

Goal: confirm block-GEMM's reach across realistic matrix content and
layout shapes.

| scenario | compile delta | exec @ 1 thr delta | exec @ 4 thr delta |
|---|--:|--:|--:|
| lo d=1.0  | **−7%**  | **−22%** | **−24%** |
| lo d=0.25 | **−12%** | −2%      | 0% |
| hi d=1.0  | 0%       | +2%      | +2% |
| hi d=0.25 | −2%      | −3%      | −4% |

**Conclusions:**
- Lo-heavy dense (our primary target) is the big win, both axes.
- Lo-heavy sparse: compile wins more than dense (InstCombine has less
  IR to grind through, so the structural savings show up faster).
  Runtime neutral because sparse matvec was already cheap.
- Hi-heavy: block-GEMM doesn't engage (LK=1 < 32), so neutral.
  Confirms the gate skips workloads that can't benefit.

Raw: `benchmarks/block_gemm/exp3_density_hi.txt`

## Experiment 4 — F32 W512 (the one Option 3 couldn't help)

Goal: test whether block-GEMM fills the gap Option 3 left on F32 W512
(where `s=16` made the tiled load path slower than legacy).

| metric | T=0 | T=8 | delta |
|---|--:|--:|--:|
| compile total | 4887 ms | 4326 ms | **−11.5%** |
| codegen       | 4146 ms | 3569 ms | −14% |
| exec @ 1 thr  | 38.5 ms | 37.4 ms | −3% |
| exec @ 4 thr  | 12.2 ms | 10.2 ms | **−16%** |
| exec @ 32 thr | 2.27 ms | 2.07 ms | **−9%** |

**Block-GEMM wins here.** It's structurally independent of the `s=16`
cross-lane-shuffle cost that blocked Option 3 — the volatile store/load
operations are simple ZMM moves, not shuffles. This closes the last
regression gap from Option 3's spec-matrix coverage.

Raw: `benchmarks/block_gemm/exp4_f32_w512.txt`

## Experiment 5 — end-to-end on qvc-30 (the big lo-heavy circuit)

Goal: check whether the per-kernel wins translate to full-circuit wall
time on a realistic fused workload.

| path | compile | exec |
|---|--:|--:|
| T=0 | 67.1 s | 57.3 s |
| T=8 | 67.6 s | 57.5 s |

**Neutral.** qvc-30's 112 fused gates are distributed across qubits 0–29,
so only a small fraction have `LK ≥ 32` (most lie above the first 3–5
qubit slots and span several positions outside the lo-bit range).
Block-GEMM is dormant on ~95% of the circuit's fusions and contributes
little to the aggregate.

This matches the gate's intent: block-GEMM targets a **specific shape**
(large lo-heavy output partitions on dense matrices), not circuit-wide
kernel time. Its value shows up in microbenchmarks and in circuits whose
fusion graph produces many of this shape.

Raw: `benchmarks/block_gemm/exp5_qvc30_e2e.txt` (via terminal log)

## Summary of applicability

| axis | when block-GEMM helps | when it doesn't |
|---|---|---|
| gate size | k ≥ 5 (LK ≥ 32) | k ≤ 4 (gated out) |
| matrix density | any — dense wins most | sparse compile-only win |
| layout | lo-heavy + mixed with ≥5 lo bits | hi-heavy (gated out) |
| precision × SIMD | all six configs, incl. F32 W512 | — |
| thread regime | wins across 1/4/32 threads | — |
| relationship to Option 3 | stacks orthogonally | — |
| real-circuit impact | visible when many fusions hit the trigger | qvc-30, hea-30 see ≤1% because of fusion mix |

## Reproduction

```sh
cargo build --release --bin bench_compile --lib

# Isolate block-GEMM effect at k=6 lo-heavy F64 W512:
for T in 0 8; do
  CAST_BLOCK_GEMM_T=$T target/release/bench_compile \
      --precision f64 --simd 512 --qubits 0,1,2,3,4,5 --reps 3 \
      --apply-budget 3 --sv-qubits 24 --threads 1,4,32
done

# Correctness: all 67 CPU tests pass under every T value:
CAST_BLOCK_GEMM_T=8 cargo test --release --lib cpu::
```

## Open questions / next iterations

1. **Hi-heavy gates don't benefit.** The matvec's K amps are live across
   all HK iterations regardless of how we block outputs. A rewrite that
   explicitly splits the HI loop with amp reloads could extend coverage
   — but each hi iteration already reads all amps exactly once, so it's
   unclear whether reload-on-demand would help register pressure without
   adding memory traffic. Worth a separate experiment.
2. **Threshold fine-tuning.** The current `LK ≥ 4·T` gate was set from
   the k-sweep; haven't tested finer-grained thresholds like `3·T` or
   density-conditioned gates. If a future workload shows regressions,
   revisit.
3. **Make this default (no env var).** Once we have enough confidence
   and the fusion-decision implications are understood, drop the gate
   and enable by default. Right now it's a straight-line win in its
   narrow applicability window, but the value of enabling it for
   everyone vs opt-in is a product decision, not a technical one.

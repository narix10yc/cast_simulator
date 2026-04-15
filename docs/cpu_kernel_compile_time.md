# CPU kernel compile time — where it goes, and why target-qubit placement matters

This note documents what we learned from a phase-decomposed microbenchmark of
the CPU kernel generator (`src/cpp/cpu/cpu_gen.cpp`, `src/cpp/cpu/cpu_jit.cpp`).
The harness is [`src/bin/bench_compile.rs`](../src/bin/bench_compile.rs); it
splits the per-kernel cost into three phases by calling the public APIs in
separate runs:

| phase | entry point | what runs |
|---|---|---|
| `ir` | `CpuKernelManager::generate` | IRBuilder emission in `cast_cpu_generate_kernel_ir` |
| `opt` | `generate_with_diagnostics(ir=true)` − `generate` | O1 pipeline (InstCombine, EarlyCSE, GVN, SimplifyCFG) |
| `codegen` | `emit_asm` − `generate_with_diagnostics(ir=true)` | SelectionDAG, regalloc, scheduling, LLJIT lookup |

All numbers below are median of 3 repetitions, one untimed warmup, on a 32-core
Threadripper 7970X running AVX-512. Matrix content is a Haar-random unitary
from `QuantumGate::random_unitary`. See `bench_compile --help` for flags.

## 1. Compile cost is dominated by native codegen and grows superlinearly with k

F64, W512, dense, `q=[0..k)`:

| k | ir (ms) | opt (ms) | cg (ms) | total (ms) | IR lines |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.1 |   1.2 |    3.6 |    4.9 |      63 |
| 2 | 0.1 |   2.3 |    7.0 |    9.4 |     163 |
| 3 | 0.5 |   6.8 |   28.3 |   35.6 |     555 |
| 4 | 1.2 |  22.4 |  112.4 |  136.1 |   2107 |
| 5 | 2.2 | 112.5 |  543.4 |  658.2 |   8283 |
| 6 | 8.3 | 635.9 | 3728.2 | **4372** |  32923 |

IR size scales as O(4^k) (the matvec dominates). Total compile grows ~6–7× per
step. **At k ≥ 4, native codegen is 80–85% of total time.** A single dense 6q
kernel costs ~4.4 s cold; a circuit with 15 of them (e.g. `qvc-30` size-only
fused with `max-size=6`) spends ~66 s in compile before execution starts.

## 2. Target-qubit placement changes compile time by ~45%

The generator's bit-layout code (`compute_bit_layout`, cpu_gen.cpp:64) splits
target qubits into three sets:

- `simd_bits` — non-target positions that form SIMD lanes (always `simd_s`
  entries, with `simd_s ∈ {1..4}` depending on precision × SIMD width)
- `lo_bits` — target qubits whose positions sit inside the SIMD lane region
- `hi_bits` — target qubits whose positions sit above the SIMD lane region

`sep_bit` is the first bit above the lane region, and the loop body emits
**one aligned load of width `vec_size = 2^sep_bit` doubles**. That width is
what feeds LLVM type-legalization.

At k=6, F64, W512, dense, comparing `q=[0,1,2,3,4,5]` (`lk=6, hk=0`) versus
`q=[24,25,26,27,28,29]` (`lk=0, hk=6`):

| metric | `q=[0..6)` (lo-heavy) | `q=[24..30)` (all-hi) |
|---|--:|--:|
| `total` compile (ms) | **4373** | **2958** |
| `opt` (ms) | 637 | 408 |
| `cg` (ms) | 3728 | 2541 |
| IR `load` ops | 4 | 67 |
| IR `store` ops | 1 | 64 |
| widest IR vector | **`<1024 × double>`** (259 uses) | `<16 × double>` (258 uses) |
| stack spill loads (asm) | **4468** | 1821 |
| stack spill stores (asm) | 393 | 127 |
| **stack frame** | **18 984 B** | **8 072 B** |

Same fmul/fadd counts (16 384 each, 12 224 FMAs after O1). The lo-heavy path
emits one `<1024 × double>` aligned load (8 KB) followed by 128 stride-128
shufflevector extractions from that single SSA value; LLVM has to legalize
that type into ZMM-width ops, and the regalloc sees ~128 simultaneously-live
amplitude vectors (2·LK = 2^(k+1)) — more than 4× the 32 ZMM registers —
producing the 18.6 KB stack frame.

The all-hi path instead emits 64 independent `<16 × double>` loads (one ZMM
each) and retires accumulators one at a time (`LK=1`), so no legalization is
needed and peak live state is ~66 vectors.

**Any combination where few "gaps" exist among the first `k + simd_s` positions
triggers the lo-heavy shape.** Examples that all produce the same
`<1024 × double>` load: `{0,1,2,3,4,5}`, `{0,1,2,3,4,6}`, `{0,1,2,3,4,7}`,
`{0,1,2,4,6,8}`. `sep_bit` only shrinks when enough non-target positions
below it exist to fill `simd_bits`.

## 3. Matrix sparsity is the biggest lever

`ImmValue` mode splats each matrix entry as a `ConstantFP` vector. Zero entries
get folded by InstCombine before native codegen runs, and ±1 entries fold into
sign flips. Measured at k=6, F64, W512, same qubits:

| qubits | density | cg (ms) | total (ms) | IR lines |
|---|--:|--:|--:|--:|
| `[0..6)`   | 1.0  | 3727 | **4373** | 32923 |
| `[0..6)`   | 0.5  | 1620 | 1885 | 17123 |
| `[0..6)`   | 0.25 |  729 |  845 | 8459 |
| `[24..30)` | 1.0  | 2541 | 2958 | 33053 |
| `[24..30)` | 0.25 |  572 |  677 | 8613 |

Placement alone: 1.48×. Sparsity alone: 5.2×. Combined: 6.5× (4373 ms →
677 ms). The effects compose — placement attacks codegen (spills), sparsity
shrinks IR before codegen sees it. Structural-zero circuits (`icmp`,
`qft-cp`) automatically benefit; dense variational circuits pay the full
per-kernel price.

## 4. SIMD width sweep

The legalize ratio (`vec_size / native_lanes = 2·LK`) is **independent of SIMD
width** for a given shape:

| precision | SIMD | `simd_s` | native lanes | `vec_size` (all-lo k=6) | ratio |
|---|---|--:|--:|--:|--:|
| F64 | W128 | 1 | 2  | 256  | 128 |
| F64 | W256 | 2 | 4  | 512  | 128 |
| F64 | W512 | 3 | 8  | 1024 | 128 |
| F32 | W128 | 2 | 4  | 512  | 128 |
| F32 | W256 | 3 | 8  | 1024 | 128 |
| F32 | W512 | 4 | 16 | 2048 | 128 |

Measured compile time at `q=[0..6)` dense F64 barely differs across widths
(W128 4130 ms, W256 3930 ms, W512 4314 ms). Any codegen-side fix benefits all
three widths similarly.

## 5. Takeaways

- **The cost model treats compile time as free**, but at k ≥ 5 cold compile
  is comparable to exec. This is why `qvc-30 size-only` at `max-size=6` takes
  66.6 s compile vs 57.0 s exec. `hw-adaptive` partially avoids it by
  rejecting unprofitable 6q fusions.
- **Placement is a lever the fusion pass doesn't use.** Two semantically
  equivalent fusions that differ only in which wires they touch can have
  compile times that differ by ~45%.
- **Sparsity-aware codegen is worth it twice over** — once at runtime,
  once at compile time.
- **Refactor opportunity.** In `emit_load_amplitudes` (cpu_gen.cpp:304) the
  lo-heavy case (`hi_bits.empty()`) can peel the `<vec_size × double>`
  mega-load into `vec_size / native_lanes` native-width aligned loads.
  The existing pdep-based split masks still apply, just per-ZMM-chunk
  instead of across the whole vector. Expected compile-time win of ~30%
  (matching the lo vs hi gap) and possibly a smaller exec win from fewer
  spill reloads. Correctness baseline is pinned by the `layout_coverage`
  module in `src/cpu/tests.rs`.

## Reproducing

```sh
cargo build --release --bin bench_compile

# k-sweep with phase decomposition (F64 W512, dense):
target/release/bench_compile --precision f64 --max-qubits 6 --reps 3

# Placement sweep at k=6:
for qs in "0,1,2,3,4,5" "24,25,26,27,28,29" "0,1,2,24,25,26"; do
    target/release/bench_compile --precision f64 --qubits $qs --reps 3
done

# Density sweep:
for d in 1.0 0.5 0.25 0.1; do
    target/release/bench_compile --precision f64 --max-qubits 6 --density $d
done

# Dump optimized IR and native asm for a configuration:
target/release/bench_compile --precision f64 --qubits 0,1,2,3,4,5 \
    --dump-ir /tmp/dump/run --dump-asm /tmp/dump/run
```

## Safety net for future codegen refactors

`src/cpu/tests.rs` contains a `layout_coverage` module (19 tests) that pins
the numerical output of the JIT for every bit-layout shape the generator can
produce, at every `{precision, SIMD}` combination, across content classes
(identity, diagonal, permutation, random dense, random sparse, zero),
`StackLoad` / `ImmValue` modes, multiple thread counts, and minimum-sized
statevectors. It runs in ~50 s via `cargo test --release --lib layout_coverage`
and must pass before any cpu_gen.cpp rewrite lands.

# CLI Tools

CAST ships three command-line binaries for hardware profiling and benchmarking.
All require `LLVM_CONFIG` to be set (see the README).

## profile_hw — Hardware Profiler

Measures the memory/compute crossover point by sweeping gate kernels across a
range of arithmetic intensities and fitting a two-segment piecewise roofline
model.

### Quick Start

```sh
# Profile CPU (all precisions, auto-detect statevector size)
cargo run --bin profile_hw --release

# Profile CPU + CUDA
cargo run --bin profile_hw --features cuda --release

# Save profiles to JSON for later use
cargo run --bin profile_hw --release -- --save-profiles profiles/

# CUDA only, F64, explicit 28-qubit statevector, 60s budget
cargo run --bin profile_hw --features cuda --release -- \
      --backend cuda --precision f64 -n 28 --budget 60 \
      --save-profiles profiles/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda,all}` | `all` | Which backend(s) to profile |
| `--precision {f32,f64,all}` | `all` | Which precision(s) to test |
| `-n, --n-qubits <N>` | auto | Statevector size; auto-detected from free memory if omitted |
| `--budget <SECS>` | `30` | Wall-time budget per profile run |
| `--threads <N>` | all cores | CPU thread count (also via `CAST_NUM_THREADS`) |
| `--save-profiles <DIR>` | — | Save each profile as `<backend>_<precision>.json` |

### Output

The profiler prints a progress bar during sweeping and reports:

- **Peak BW** (GiB/s) — measured memory bandwidth
- **Peak Compute** (GFLOPs/s) — measured compute throughput
- **BW slope** — GFLOPs/s per unit arithmetic intensity (memory-bound regime)
- **Crossover AI** — the arithmetic intensity where memory-bound meets compute-bound
- **R²** — goodness of fit for the roofline model

### Choosing Statevector Size

The profiling statevector should match the simulation workload size for
representative cache/TLB behavior.  For density-matrix simulation of n
physical qubits, the statevector has 2n virtual qubits — use `-n 2N`.

If omitted, profile_hw auto-detects available memory and picks the largest
feasible n (clamped to 30).

**Why size matters for profile accuracy:**  The roofline parameters —
especially effective memory bandwidth — depend on working set size because
of cache effects.  A 28-qubit statevector (4 GiB) can partially fit in GPU
L2 cache (modern GPUs have 48-96 MB L2), which raises effective bandwidth
for low-AI gates and shifts the crossover AI downward compared to a
30-qubit statevector (16 GiB) that streams entirely from HBM.

At large scales (30+ SV qubits), the statevector far exceeds any cache and
the roofline stabilizes — profiles at 30 and 32 qubits will agree closely.
But in the transition zone (24-30 qubits), cache effects can meaningfully
shift the crossover, so the profile qubit count should match the target
simulation size for best results.

### Saved Profiles

Profiles are saved as JSON and can be loaded by `bench` via
`--profile <path>` to skip re-profiling.

### Merging Multiple Runs

Because peak memory bandwidth has ±4–5% run-to-run variance on typical
GPUs, any single profile is noisy. Profile multiple times and merge for
a more robust consensus profile:

```sh
# Five independent runs
for i in 1 2 3 4 5; do
    cargo run --bin profile_hw --features cuda --release -- \
        --backend cuda --precision f64 -n 30 --budget 180 \
        --save-profiles tmp/profiles_runs/run${i}/
done

# Merge into one profile
cargo run --bin profile_hw --features cuda --release -- \
    --merge tmp/profiles_runs/run{1,2,3,4,5}/cuda_f64.json \
    --merge-out profiles/cuda_f64.json
```

---

## bench_compile — Per-Kernel Microbenchmark

Times **single-gate** CPU kernel JIT compilation, decomposed into three phases,
and optionally times the per-gate apply step on a real statevector. Useful for
isolating codegen cost from fusion / circuit-level effects.

### Phases measured

| Phase | What is timed | FFI entry |
|-------|---------------|-----------|
| **ir** | LLVM IRBuilder emission during `CpuKernelManager::generate` | `cast_cpu_generate_kernel_ir` |
| **opt** | O1 module pipeline | `cast_cpu_optimize_kernel_ir` |
| **codegen** | Native code emission + LLJIT symbol resolution | `cast_cpu_jit_compile_kernel` |

The bench uses fresh managers per rep:
1. Plain `generate()` → `ir`
2. `generate_with_diagnostics(ir=true)` → `ir + opt`
3. `generate() + emit_asm()` → `ir + opt + codegen`

Derived: `opt = (2) − (1)`, `codegen = (3) − (2)`.

With `--apply-budget > 0`, each row additionally compiles the kernel, applies
it to a statevector, and reports adaptively-sampled apply time per requested
thread count.

### Quick Start

```sh
# Default k-sweep (k=1..6) at native SIMD, F64, 3 reps per row.
cargo run --bin bench_compile --release

# Per-gate apply time on a 28-qubit statevector, 12 threads, 3 s budget
cargo run --bin bench_compile --release -- \
      --max-qubits 5 --apply-budget 3 --threads 12 --sv-qubits 28

# Explicit qubit placement (single row, no k-sweep)
cargo run --bin bench_compile --release -- --qubits 0,1,2,3,4

# Sparse matrix codegen sweep
cargo run --bin bench_compile --release -- --density 0.1

# Dump optimized IR and native asm for each row
cargo run --bin bench_compile --release -- \
      --max-qubits 5 --dump-ir /tmp/ir --dump-asm /tmp/asm

# Mega vs Tiled load-mode comparison (run twice, compare)
CAST_CPU_LOADMODE=mega  cargo run --bin bench_compile --release -- --apply-budget 3 --threads 12
CAST_CPU_LOADMODE=tiled cargo run --bin bench_compile --release -- --apply-budget 3 --threads 12
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-qubits <N>` | `6` | Largest gate size to sweep (`k=1..=N`). Ignored if `--qubits` given. |
| `--reps <N>` | `3` | Reps per row (one extra untimed warmup). Fresh manager each rep. |
| `--precision {f32,f64}` | `f64` | Scalar precision |
| `--simd {native,128,256,512}` | `native` | SIMD register width |
| `--qubits <LIST>` | — | Explicit target positions (e.g. `0,1,5`). Disables k-sweep. |
| `--density <F>` | `1.0` | Fraction of non-zero matrix entries. `<1.0` uses `random_sparse`. |
| `--force-dense` | off | Disable sparsity-aware codegen (`ztol=otol=0`) |
| `--dump-ir <PREFIX>` | — | Write optimized LLVM IR for each row to `<prefix>_<label>.ll` |
| `--dump-asm <PREFIX>` | — | Write native assembly for each row to `<prefix>_<label>.s` |
| `--apply-budget <S>` | `0` | Exec-phase time budget per thread count. `0` = skip apply timing. |
| `--sv-qubits <N>` | `26` | Statevector size for exec timing (26 → 16 MB F64, exceeds typical L3) |
| `--threads <LIST>` | `1,4,32` | Thread counts for the exec sweep (comma-separated) |

### Output

One row per gate shape. Columns: `row`, `ir (ms)`, `opt (ms)`, `cg (ms)`,
`total (ms)`, `ir lines`. When `--apply-budget > 0`, each row is followed by
`exec @ <n> thr  <mean>  ±<stddev>  (<iters> iters, cv=<cv>)` lines, one per
thread count.

### Notes

- The apply sweep is sensitive to SMT: on a 12-core / 24-thread machine, one
  thread per physical core (`--threads 12`) gives the most stable results.
- `--sv-qubits 26` uses 1 GiB (F64) which fits in DRAM and well exceeds L3 on
  typical workstation CPUs. Use `28` for 4 GiB if you want deeper DRAM
  exercise.
- `--max-qubits 6` still compiles correctly but the k=6 row takes 1–2 s of
  compile time on its own due to LLVM register spilling (see
  [`CAST_CPU_VEC_REGS`](#cast_cpu_vec_regs--matvec-strategy-threshold)).
  For fast iteration, pass `--max-qubits 5`.

---

## bench — Fusion Strategy Benchmark

Loads OpenQASM circuit files, applies each requested fusion strategy, then
times execution on the selected backend via `Simulator::bench`.
Compile and execute phases are reported separately; the execute phase is
sampled adaptively within `--bench-budget` wall seconds.

### Quick Start

```sh
# Default: all three fusion modes × one circuit, CUDA backend, cached profile
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      examples/journal_examples/qft-cx-30.qasm

# All 30-qubit circuits, default fusion modes
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      examples/journal_examples/*-30*.qasm

# CPU backend, only hardware-adaptive fusion
cargo run --bin bench --release -- \
      --backend cpu --fusion hw-adaptive \
      examples/journal_examples/qft-cx-30.qasm

# Two selected modes, comma-separated
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json \
      --fusion none,hw-adaptive \
      examples/journal_examples/mexp-17.qasm

# Sparse-vs-dense ablation: run twice with and without --force-dense,
# then compare rows (identical other args).
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json --fusion hw-adaptive \
      examples/journal_examples/mexp-17.qasm
cargo run --bin bench --features cuda --release -- \
      --profile profiles/cuda_f64.json --fusion hw-adaptive --force-dense \
      examples/journal_examples/mexp-17.qasm
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {cpu,cuda}` | `cuda` | Simulation backend |
| `--fusion <MODES>` | `none,size-only,hw-adaptive` | Fusion modes to benchmark (CSV or repeated flag) |
| `--profile <PATH>` | — | Cached HardwareProfile JSON (skips profiling) |
| `--profile-qubits <N>` | `30` | SV qubits for profiling |
| `--profile-budget <S>` | `20` | Profiling time budget (seconds) |
| `--max-size <N>` | `4` | Max gate size for `size-only` and `hw-adaptive` fusion |
| `--bench-budget <S>` | `5` | Exec-phase time budget per benchmark row (seconds) |
| `--force-dense` | off | Disable sparsity-aware codegen (ztol=0) |
| `--fusion-log` | off | Print per-decision fusion log for each hw-adaptive row |
| Files (positional) | required | One or more `.qasm` files |

### Fusion Modes

| Mode | Description |
|------|-------------|
| `none` | No fusion (`size_only(1)`, Phase 1 canonicalization only) |
| `size-only` | Size-gated fusion up to `--max-size` qubits |
| `hw-adaptive` | Roofline-guided fusion up to `--max-size` qubits |

### Output

One row per `(circuit, fusion mode)` pair. Columns:

- **Circuit** — QASM file stem
- **Fusion** — fusion mode label (`none`, `size-only`, `hw-adaptive`)
- **Gates** — post-fusion gate count
- **Depth** — post-fusion circuit depth (row count)
- **Compile** — cold-start wall time to generate and finalize all kernels
- **Exec time** — adaptively-sampled execution time (wall on CPU, summed
  CUDA event times on GPU), printed as `mean ± stddev (N iters)`

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `LLVM_CONFIG` | build.rs | **Required.** Path to `llvm-config` binary. |
| `CXX` | build.rs | C++17 compiler (default: `c++`) |
| `CUDA_PATH` | build.rs | CUDA toolkit root (auto-detected if unset) |
| `CAST_NUM_THREADS` | cpu::get_num_threads | CPU simulation threads (default: all cores) |
| `CAST_CPU_VEC_REGS` | cpu kernel codegen | Vector-register budget that gates Block vs Straight matvec (default: 32). See [CPU kernel generation](#cpu-kernel-generation-env-vars) |
| `CAST_CPU_LOADMODE` | cpu kernel codegen | Load/store strategy: `mega` (default) or `tiled`. See [CPU kernel generation](#cpu-kernel-generation-env-vars) and [cpu_kernel_loadmode.md](cpu_kernel_loadmode.md) |

### CPU kernel generation env vars

Both variables are read **once** on the first kernel generation and cached for the process lifetime, so setting them mid-process has no effect.

#### `CAST_CPU_VEC_REGS` — matvec strategy threshold

Controls how the codegen picks between two Phase-2 (matvec) lowerings based on register pressure. The gate's bit layout produces an output dimension `LK = 2^lk` (number of "lo" target qubits determines this). The generator then checks `LK >= vec_regs`:

| Condition | Matvec mode | IR shape |
|-----------|-------------|----------|
| `LK < vec_regs` | **Straight** (default path) | One SSA accumulator chain per output row; all `2·LK` vector accumulators stay live. Lowest overhead when it fits in the register file. |
| `LK >= vec_regs` | **Block** | Output rows tiled into groups of `T = max(2, vec_regs/4)`. Each tile retires to a stack scratch via `volatile` store/load, capping live registers at `~2T + O(1)`. Avoids heavy spilling when Straight would overflow the register file. |

**Default `vec_regs = 32`** (matches AVX-512, NEON, SVE). At this default, Block fires only when `lk >= 5` (i.e. `LK ∈ {32, 64, …}`) — so typical gates with ≤4 lo-qubits run in Straight mode.

**Typical uses:**
- `CAST_CPU_VEC_REGS=9999` — force Straight on every gate (A/B comparison)
- `CAST_CPU_VEC_REGS=8` — force Block earlier (any `lk >= 3`), useful when measuring tile-size trade-offs
- Any value `< 2` is ignored and the default (32) is used

Only `vec_regs` is configurable; `T` is always derived as `max(2, vec_regs/4)`.

#### `CAST_CPU_LOADMODE` — Phase-1 / Phase-3 memory strategy

Controls how amplitudes are gathered from the statevector (Phase 1) and written back (Phase 3). **Load and store are linked** — one flag switches both sides.

| Value | Phase 1 (Load) | Phase 3 (Store) |
|-------|----------------|-----------------|
| `mega` (default) | One wide aligned load per hi-combination, then `shufflevector` to split lanes across lo-partitions | Merge lo-partitions via shuffles, interleave re/im, one wide aligned store |
| `tiled` | Native-width chunk loads, then scalar extract/insert to gather amplitudes into K-vectors | Scatter K-vector into chunks via scalar insert, native-width chunk stores |

**Default: `mega`.** The right choice depends on ISA-specific shuffle-vs-scalar-gather costs (on x86 shuffles are fast; on some ARM variants chunked scalar gather wins). See [cpu_kernel_loadmode.md](cpu_kernel_loadmode.md) for benchmarking guidance.

Unrecognized values fall back to `mega`. Only the exact literal string `tiled` selects Tiled mode.

#### Interaction

The two flags are independent:
- `CAST_CPU_VEC_REGS` affects only Phase 2 (matvec); does not touch load/store IR
- `CAST_CPU_LOADMODE` affects only Phase 1 and Phase 3; does not touch matvec IR

All four combinations (Straight/Block × Mega/Tiled) are valid and used in practice.

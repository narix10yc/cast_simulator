# Build Notes

- Primary workflow is Cargo-based.
- The old standalone C++/CMake codebase has been removed.
- The Rust build still compiles the C++ bridge in `src/cpp/` through `build.rs`.
- `LLVM_CONFIG` selects the LLVM installation used for build flags and linking.
- The C++ compiler is `CXX` if set, otherwise plain `c++`.

# Developer Setup

- Each developer must point `LLVM_CONFIG` at a locally installed `llvm-config`.
- Example:

```sh
export LLVM_CONFIG=/path/to/llvm/bin/llvm-config
```

- For CUDA builds, that LLVM install must include the `NVPTX` target.
- If a developer wants a non-default C++ compiler for the bridge layer, set:

```sh
export CXX=/path/to/c++
```

# Quick Build

```sh
cargo check
cargo test
```

- If `LLVM_CONFIG` is set in the developer's shell startup files, source those first.
- Otherwise export `LLVM_CONFIG` explicitly for the current shell before running Cargo.

# LLVM Helper

- Local LLVM helper: `scripts/build_llvm.sh`
- Default mode builds release LLVM with `Native;NVPTX`
- Useful maintenance:
  - `scripts/build_llvm.sh <llvm-src-dir> --verify-targets`
  - `scripts/build_llvm.sh <llvm-src-dir> --clean`
  - `scripts/build_llvm.sh <llvm-src-dir> --with-clang-tools`

- Typical local build flow:

```sh
scripts/build_llvm.sh ~/llvm/${version}/llvm-project-${version}.src
export LLVM_CONFIG=~/llvm/${version}/release-install/bin/llvm-config
```

# Notes

- `build.rs` already handles the macOS/Homebrew `zstd` link path automatically, so developers should not need to add a manual `LIBRARY_PATH` workaround for that case.
- Editor settings such as VS Code `rust-analyzer`, `clangd`, or per-machine LLVM paths are local environment concerns and should not be treated as repository-wide defaults.

---

# Project Architecture

CAST is a quantum circuit simulator with a Rust front-end, LLVM-based JIT kernel generation, and optional CUDA execution. The C++ sources live in `src/cpp/cpu/` and `src/cpp/cuda/`; Rust wraps them via FFI defined in `build.rs`.

## Key source files

| File | Purpose |
|------|---------|
| `src/cpu/kernel.rs` | `CpuKernelManager` — LLVM JIT compilation + threaded dispatch |
| `src/cpu/statevector.rs` | `CPUStatevector` — SIMD-aligned interleaved re/im statevector |
| `src/cuda/kernel.rs` | `CudaKernelManager` — PTX/cubin generation, apply queue, LRU module cache, sync |
| `src/cuda/statevector.rs` | `CudaStatevector` — GPU device memory statevector |
| `src/fusion.rs` | Two-phase gate fusion optimizer |
| `src/cost_model.rs` | `HardwareProfile`, `FusionConfig`, roofline cost models |
| `src/profile.rs` | Adaptive roofline sweep profiler |
| `src/timing.rs` | `TimingStats`, `time_adaptive` — adaptive timing framework |
| `src/circuit.rs` | `CircuitGraph` — row-major gate matrix for structured fusion |
| `src/bin/bench_fusion.rs` | Benchmark: compare fusion strategies on CPU or CUDA |
| `src/bin/profile_hw.rs` | CLI: measure roofline crossover, save profiles as JSON |

---

# CPU Simulation Workflow

The CPU path compiles a unique LLVM JIT kernel per gate and dispatches it across a thread pool.

**Step-by-step:**
1. Create a `CPUStatevector::new(n_qubits, precision, simd_width)` — heap-allocated, SIMD-aligned.
2. Create a `CpuKernelManager::new()`.
3. For each gate, call `mgr.generate(&spec, &gate) → KernelId`:
   - Calls C++ to spin up a fresh LLVM `KernelGenerator` instance.
   - Generates LLVM IR for the gate (vectorized, SIMD shuffle/splat), applies O1 optimization, emits native JIT code.
   - Returns a `KernelId` (u64 handle).
4. Call `mgr.apply(kid, &mut sv, n_threads)`:
   - Splits `2^(n_qubits - n_gate_qubits - simd_s)` tasks across threads.
   - Each thread calls the JIT entry point with a counter range `[begin, end)`.
   - Implicit barrier on scope exit.

**Statevector memory layout:**
- Real and imaginary scalars are split and grouped in chunks of `2^simd_s`.
- Layout: `[re_0..re_{s-1}, im_0..im_{s-1}, re_s..re_{2s-1}, im_s..im_{2s-1}, ...]`
- `simd_s = log2(SIMD_bits / scalar_bits)` — e.g., 3 for f32+AVX2 (8 floats/register).
- Buffer aligned to SIMD vector width; JIT kernels use aligned loads.

**Spec presets:**
- `CPUKernelGenSpec::f64()` → F64, W256 SIMD, `ImmValue` matrix load, `ztol=1e-12`
- `CPUKernelGenSpec::f32()` → F32, W256 SIMD, `ImmValue` matrix load, `ztol=1e-6`

**Thread control:** Set `CAST_NUM_THREADS` env var or pass `n_threads` arg (0 = auto).

---

# CUDA Simulation Workflow

The CUDA path uses deferred execution: `apply()` queues requests and `sync()` does the actual GPU work.

**Step-by-step:**
1. Create a `CudaStatevector::new(n_qubits, CudaPrecision::F64)` — device memory allocated via cuMalloc.
2. Create a `CudaKernelManager::new()`.
3. For each gate, call `mgr.generate(&gate, spec) → CudaKernelId`:
   - C++ generates LLVM IR → O1 → NVPTX → PTX text (fully device-free).
   - If `cuda` feature enabled: also calls CUDA JIT linker (`nvJitLink`/driver) to compile PTX → cubin.
   - Returns a `CudaKernelId` (u64 handle).
4. Call `mgr.apply(kid, &mut sv)` — **non-blocking**, just appends to pending queue.
5. Call `mgr.sync() → SyncStats`:
   - Creates CUDA stream on first call.
   - For each queued launch: loads cubin via 2-slot LRU cache, records CUDA events, launches kernel.
   - `cast_cuda_stream_sync()` — blocks until GPU is idle.
   - Queries event elapsed time per kernel.
   - Returns `SyncStats` with per-kernel `KernelExecTime` and total wall time.

**LRU module cache:**
- `LRU_SIZE = 2` slots (CUmodule handles).
- Hit: bump tick. Miss: evict lowest-tick slot, `cuModuleUnload`, load new cubin.
- Stress-tested in `tests/circuit_execution_order.rs` (58–98 evictions per test).

**Statevector layout:** Interleaved complex scalars — `[re_0, im_0, re_1, im_1, ...]`. Upload/download via `sv.upload/download`.

---

# Performance Metrics and Timing

**Always use `time_adaptive` for fair benchmarks** — it avoids cold-cache bias via warmup and adapts sample count to a time budget.

```rust
// src/timing.rs
let stats: TimingStats = time_adaptive(|| { /* run kernel */ }, budget_s)?;
// stats.mean_s, stats.stddev_s, stats.cv, stats.min_s, stats.max_s, stats.n_iters
```

**GPU timing:** Use `time_adaptive_with(|| → Result<Duration>)` where the closure returns CUDA event elapsed time (not wall-clock). This is how `CpuKernelManager::time_adaptive` works, and how `SyncStats` per-kernel times are collected.

**Budget split:** 25% warmup + 75% measurement. Single probe first — if probe > 1.5× budget, returns 1 sample.

**Format:** `fmt_duration(secs)` → `"213 ms"`, `"6.18 µs"`, etc. (3 sig figs).

---

# Computation/Memory Balance and Roofline Model

Gate simulation is memory-bandwidth-bound at small gate sizes and compute-bound at large fused gates.

**Roofline model:**
```
GFLOPs/s = min(bw_slope × AI, peak_gflops)
crossover_AI = peak_gflops / bw_slope
```
- **AI (arithmetic intensity)** = FLOPs / bytes read+written for one kernel launch.
- Below `crossover_AI`: memory-bound — throughput ∝ AI, fusing gates is free.
- Above `crossover_AI`: compute-bound — fusing has diminishing returns.

**Per sweep point:**
```
gib_s = 2.0 * sv_bytes / mean_s / 2^30    # read + write
gflops_s = ai * n_elements * 2.0 / mean_s / 1e9
```

**Hardware adaptive fusion cost:**
```rust
cost = max(1.0, ai / crossover_ai)
```
Memory-bound gates cost 1.0 (always worth fusing); compute-bound cost scales with AI ratio.

---

# Hardware Profiling

**Profile measurement** (`src/profile.rs`):

```sh
cargo run --bin profile_hw -- \
  --backend all --precision all \
  --n-qubits 30 --budget 30 \
  --save-profiles ./profiles/
```
Saves `cpu_f32.json`, `cpu_f64.json`, `cuda_f32.json`, `cuda_f64.json`.

**Fitting procedure:**
1. Seed sweep at AI = 1, 2, 4, ..., 64.
2. Up to 5 refinement rounds near estimated crossover (probes at 0.5×, 1×, 1.5× crossover).
3. Fit: `bw_slope = median(GFLOPs/AI)` from lower-half points; `crossover = peak / bw_slope`.
4. Stop when R² ≥ 0.95 and ΔR² < 0.01. Warn if R² < 0.90.

**Programmatic API:**
```rust
let profile = profile::measure_cpu(&spec, n_qubits, budget_s)?;
let profile = profile::measure_cuda(&spec, n_qubits, budget_s)?;
profile.save("cpu_f64.json")?;
let profile = HardwareProfile::load("cpu_f64.json")?;
```

**Using profiles for fusion:**
```rust
let config = FusionConfig::hardware_adaptive(&profile, /*max_size=*/ 4);
fusion::optimize(&mut circuit_graph, &config);
```

---

# Gate Fusion

Fusion reduces kernel launch overhead by merging multiple gates into one larger gate.

**Two-phase algorithm (`src/fusion.rs`):**
1. **Size-2 canonicalization**: absorb 1-qubit gates into neighbors; merge adjacent 2-qubit gates.
2. **Agglomerative fusion** (sizes 3 → `max_size`):
   - Seed cluster at each gate. Sweep same row, then cross-row expansion.
   - Accept if `old_cost / new_cost - 1 ≥ benefit_margin`.
   - Repeat passes until no progress.

**Configs:**
- `FusionConfig::aggressive()` → `size_only(4)` — pure size limit, no AI check.
- `FusionConfig::hardware_adaptive(profile, max_size)` — roofline-aware, best for real hardware.

---

# Correctness Tests

- `tests/cpu_cuda_compare.rs` — CPU vs CUDA amplitude agreement, tolerance `1e-10`, gates: X/Y/Z/H/S/T/Rx/Ry/CX/CY/CZ/Swap/CCX/CSwap + Haar-random 1/2/3-qubit.
- `tests/circuit_execution_order.rs` — LRU cache stress: 10–20 qubit circuits, 60–100 Haar-random gates; tolerance `1e-9`.

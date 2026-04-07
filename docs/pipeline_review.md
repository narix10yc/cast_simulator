# Simulation Pipeline Review

Critical review of the CAST simulation pipeline, identifying design flaws and
improvement opportunities. Conducted 2026-04-04.

## Issues

### 1. Fixed-Spec KernelManager

Both `CpuKernelManager` and `CudaKernelManager` bind the generation spec at
construction time. The `Simulator` then bakes one spec + one manager. This
means:

- No mixed-precision simulation (e.g. F32 for cheap gates, F64 for sensitive
  ones).
- No per-gate SIMD width selection on CPU.
- No per-gate ztol/otol tuning (e.g. stricter tolerances for near-identity
  gates, looser for dense fused gates).

The current design forces 1:1 between precision and simulator instance, and
compiled kernels can't be shared across instances.

**Status:** deferred

### 2. CPU: One LLJIT Session Per Kernel

Each `generate_inner` call creates a fresh `KernelGenerator`, adds one kernel,
then calls `finish` to produce a JIT session. Every kernel gets its own LLJIT
session and its own compiled code pages. The C++ API supports batching (generate
N kernels, then finish once), but the Rust side never batches. Costs:

- Per-session overhead (LLJIT context setup/teardown, memory mapping).
- Missed opportunity for LLVM to share code pages across kernels.

**Status:** done — `generate` now adds kernels to a shared C++ generator;
`finalize()` batch-compiles them into a single LLJIT session. `apply`
auto-finalizes if needed. Shared `JitSessionHandle` via `Arc` keeps code pages
alive.

### 3. CUDA LRU Cache Hardcoded to 2 Slots

`LRU_SIZE = 2`. A circuit with 3+ distinct fused gate shapes will thrash the
module cache on every `sync` — loading and unloading CUmodules for each batch.
For a typical 30-qubit circuit with ~15 distinct kernels after fusion, this is a
significant hidden cost.

**Status:** done — `CudaKernelManager::with_lru_size(spec, n)` added; default
raised to 4. `loaded` changed from fixed-size array to `Vec`.

### 4. CPU `apply` Holds the Mutex During Computation

`CpuKernelManager::apply()` locks the manager mutex, looks up the
`KernelEntry`, then calls `apply_kernel` which spawns a thread scope and does
the actual computation — all while holding the lock. Consequences:

- No concurrent applies for different kernels / different statevectors.
- The lock is held for the entire duration of multi-threaded kernel execution.
- `generate` calls from other threads are blocked during every `apply`.

**Status:** done — `KernelEntry` wrapped in `Arc`, `apply` clones the entry and
releases the lock before kernel execution.

### 5. CPU Thread Spawning Per Apply

Every `apply_kernel` uses `std::thread::scope` to spawn `n_threads` worker
threads. For a 30-qubit circuit with 50+ gates after fusion, this means 50+
spawn/join cycles. Thread creation is ~10-50 us each on Linux; with 8-16 threads
that's 0.4-4 ms overhead per apply, which can dominate small-kernel execution
time. A persistent thread pool would amortize this.

**Status:** deferred

### 6. Fusion AI vs Codegen ztol Mismatch

The cost model's arithmetic intensity calculation uses a hardcoded `ztol =
1e-12`, but the actual kernel generator may use a different ztol from the spec
(e.g. `1e-6` for F32). A gate the cost model thinks is sparse (low AI →
memory-bound → "fuse it") might be treated as dense by the F32 kernel generator,
producing a slower kernel than predicted.

**Status:** done — `zero_tol` added to `FusionConfig` and
`HardwareAdaptiveCostModel::new()`. `hardware_adaptive*()` constructors now
require a `zero_tol` parameter. Fusion log uses `config.zero_tol` instead of
hardcoded `1e-12`.

### 7. Simulator Owns the Manager — No Kernel Reuse Across Runs

`Simulator<B>` owns `B::Mgr`. If you run two circuits that share common gate
structures (e.g. repeated H and CX), the second run recompiles everything. For
benchmarking or variational algorithms where the circuit structure is repeated,
making the manager `Arc`-shared or externally owned would allow cross-run kernel
reuse.

**Status:** deferred

### 8. No Pipelined Generation/Execution

The CPU path is fully serial: compile all gates, then execute all gates. The
CUDA path has apply/sync separation but still compiles all gates upfront. For
large circuits, a pipelined approach (execute already-compiled gates while
generating the next batch) could hide compilation latency.

**Status:** done (CUDA only) — `CudaKernelManager::execute_pipelined` runs a
compile-thread pool alongside a windowed launch loop on the GPU stream;
window slots own their own CUmodule and unload on drop.  Wired through
`Backend::execute_pipelined` so the simulator's non-trajectory path uses it
by default.  CPU still serializes (compile_all → execute_all), but with the
batched JIT session from #2 the upfront compile cost is much smaller.

### 9. CUDA Dedup Misses Structural Equivalence

CUDA dedup uses raw matrix bytes only. Two gates with the same sparsity pattern
but different non-zero values produce different PTX, but could potentially share
the same cubin structure with runtime-loaded matrix values (analogous to CPU
StackLoad mode). Currently CUDA has no StackLoad equivalent.

**Status:** deferred

### 10. Trajectory Ensemble Memory is Unchecked

At each noisy gate, `B::clone_sv` is called for ensemble branching. For
30-qubit F64, each clone is ~16 GiB. With `max_ensemble > 1`, there's no memory
check before cloning. The auto-detection path (`max_ensemble: None`) defaults to
`1`, so auto-detection isn't actually implemented.

**Status:** deferred

## Priority Summary

| Priority | Issue | Status |
|----------|-------|--------|
| High | CPU apply holds mutex during computation (#4) | done |
| High | LRU cache size = 2 (#3) | done |
| High | One LLJIT session per kernel (#2) | done |
| High | Fusion ztol mismatch (#6) | done |
| Medium | Manager not sharable (#7) | deferred — low effort, useful for parametric circuits |
| Medium | Thread spawn per apply (#5) | deferred — needs thread pool, measurable speedup |
| Medium | Ensemble memory unchecked (#10) | deferred — safety fix, prevents silent OOM |
| Low | Spec locked to manager (#1) | deferred — mixed-precision rarely needed |
| Medium | No pipeline overlap (#8) | done (CUDA via `execute_pipelined`) |

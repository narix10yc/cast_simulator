# CAST Architecture

CAST is a Rust quantum circuit simulator with LLVM-JIT CPU and optional CUDA
backends.  Circuits are represented as gate graphs, optimized by a two-phase
fusion pipeline, and executed via compiled kernels on statevectors.

## Module Map

```
src/
├── lib.rs                  Module root — re-exports CircuitGraph, GateId, QuantumCircuit
├── types/
│   ├── mod.rs              Re-exports: Complex, Rational, Precision, QuantumGate,
│   │                         QuantumCircuit, ComplexSquareMatrix
│   ├── gate.rs             QuantumGate — unitary + optional NoiseModel
│   ├── noise.rs            NoiseModel — probability-weighted Kraus operators
│   ├── circuit.rs          QuantumCircuit — user-facing gate sequence + measured qubits
│   ├── matrix.rs           ComplexSquareMatrix — dense row-major complex matrix
│   ├── rational.rs         Rational — exact i32/i32 fractions, auto-reduced
│   └── precision.rs        Precision enum (F32, F64)
├── circuit_graph.rs        CircuitGraph — 2-D gate grid (rows × qubits)
├── fusion.rs               Two-phase fusion optimizer
├── cost_model.rs           CostModel trait, FusionConfig, HardwareProfile
├── profile.rs              Adaptive roofline profiler (sweep + fit)
├── timing.rs               time_adaptive(), TimingStats, fmt_duration()
├── sysinfo.rs              cpu_free_memory_bytes(), max_feasible_n_qubits()
├── openqasm/
│   ├── mod.rs              Re-exports: parse_qasm, Angle, Gate, Circuit
│   ├── circuit.rs          Gate/Angle/Circuit types, QASM 2.0 serialization
│   └── parser.rs           Recursive-descent QASM parser
├── simulator/
│   ├── mod.rs              Simulator<B>, QuantumState<B>, Backend trait (Cpu/Cuda)
│   ├── trajectory.rs       Ensemble branching, pruning, TrajectoryResult
│   ├── measure.rs          Marginal probabilities, batch sampling
│   └── tests.rs            Simulator integration tests
├── cpu/
│   ├── mod.rs              Re-exports, get_num_threads(), native_simd_width()
│   ├── kernel.rs           CpuKernelManager — LLVM JIT generate/apply
│   ├── statevector.rs      CPUStatevector — SIMD-aware aligned memory
│   └── tests.rs            CPU backend unit tests
├── cuda/                   (behind `cuda` feature flag)
│   ├── mod.rs              Re-exports, device_sm(), cuda_free_memory_bytes()
│   ├── kernel.rs           CudaKernelManager — PTX gen, LRU module cache
│   ├── statevector.rs      CudaStatevector — GPU device memory, norm_squared, normalize
│   └── tests.rs            CUDA backend unit tests
├── cpp/
│   ├── cpu/                C++ FFI: LLVM IR generation + OrcJIT
│   ├── cuda/               C++ FFI: NVPTX IR gen + CUDA driver API
│   │   └── cuda_kernels.cu Device-side reduction/scale kernels (compiled by nvcc)
└── bin/
    ├── profile_hw.rs       CLI: roofline hardware profiler
    └── bench.rs            CLI: benchmark fusion strategies on QASM files
```

## Core Data Flow

### 1. Circuit Construction

Gates enter the system either programmatically or via OpenQASM parsing:

```
OpenQASM string ──parse_qasm()──► openqasm::Circuit ──from_qasm_circuit()──► CircuitGraph
QuantumGate values ──insert_gate()──────────────────────────────────────────► CircuitGraph
```

`CircuitGraph` is a 2-D grid of rows (time steps) × qubits.  Each gate
occupies one row and spans its target qubit slots.  `insert_gate()` uses
**left-pushing** semantics: a gate is placed in the earliest row where all its
qubits are free, preserving causal order.

### 2. Fusion Optimization

```
CircuitGraph ──fusion::optimize(&mut cg, &config)──► CircuitGraph (fewer, larger gates)
```

Two phases (see [fusion.md](fusion.md) for details):

- **Phase 1 — Size-2 canonicalization** (`apply_size_two_fusion`):
  absorb single-qubit gates into adjacent multi-qubit gates, then merge
  adjacent 2-qubit gate pairs on the same qubits.

- **Phase 2 — Agglomerative fusion** (`apply_gate_fusion`):
  iteratively merge gates across rows up to a size limit, guided by a cost
  model.

Noisy gates participate in fusion. Their Kraus operators are composed via
Cartesian product: `K_i^self · K_j^other`. In density-matrix mode, the cost
model uses `effective_n_qubits()` (2× physical) to account for superoperator size.

### 3. Kernel Compilation

Each gate in the optimized circuit is compiled to a native kernel:

**CPU path:**
```
QuantumGate ──generate()──► LLVM IR ──O1──► native code (OrcJIT) ──► KernelId
```

**CUDA path:**
```
QuantumGate ──generate()──► LLVM IR ──O1──► NVPTX PTX ──cubin JIT──► CudaKernelId
```

The C++ FFI layer under `src/cpp/` handles LLVM IR construction and JIT
compilation.  Rust owns the lifecycle via `CpuKernelManager` / `CudaKernelManager`.

### 4. Simulation Execution

**CPU:**
```rust
let mgr = CpuKernelManager::new(spec);
let kid = mgr.generate(&gate)?;         // LLVM IR → O1 → native JIT
mgr.apply(kid, &mut sv, n_threads)?;    // scoped thread pool, implicit barrier
```

**CUDA:**
```rust
let mgr = CudaKernelManager::new(spec);
let kid = mgr.generate(&gate)?;         // LLVM IR → PTX → cubin
mgr.apply(kid, &mut sv)?;               // non-blocking enqueue
let stats = mgr.sync()?;                // flush queue, launch, wait
```

Both managers deduplicate identical gates via content-based keys, avoiding
redundant LLVM compilations. The CUDA manager uses a 2-slot LRU module cache
for loaded CUmodules.

### 5. Statevector Layout

**CPU** — split real/imaginary, SIMD-grouped:

```
[re_0, re_1, ..., re_{2^s-1}, im_0, im_1, ..., im_{2^s-1}, re_{2^s}, ...]
```

where `s = log2(SIMD_register_width / scalar_bits)`.  For F64 + AVX2 (W256),
`s = 2`, so amplitudes are grouped in blocks of 4 reals followed by 4
imaginaries.  Memory is aligned to the SIMD vector width.

A kernel requires `n_sv_qubits >= n_gate_qubits + s` to have enough task bits
for the inner loop.

**CUDA** — interleaved real/imaginary:

```
[re_0, im_0, re_1, im_1, ...]
```

Standard complex interleaved layout on device memory.

## Density-Matrix Simulation

Noisy (open-system) simulation uses the density-matrix representation:

- An n-qubit density matrix ρ is vectorized as a 2n-qubit statevector with
  layout `sv[ket | (bra << n)] = ρ[ket, bra]`.

- A unitary gate U is lifted to the superoperator `S = U ⊗ conj(U)` acting on
  2n virtual qubits `[q₀, ..., q_{k-1}, q₀+n, ..., q_{k-1}+n]`.

- A noisy gate with Kraus operators `[(pᵢ, Kᵢ)]` computes the superoperator
  `S = Σ pᵢ · Kᵢ ⊗ conj(Kᵢ)` via `to_density_matrix_gate()`.

This allows reusing the existing statevector simulation engine for noisy
circuits — no separate density-matrix kernel is needed.

## Build System

`build.rs` compiles the C++ FFI layer into static archives:

1. Reads `LLVM_CONFIG` to get LLVM include/link flags.
2. Compiles `src/cpp/cpu/*.cpp` → `libcast_cpu_ffi.a` (always).
3. With `--features cuda`: compiles `src/cpp/cuda/*.cpp` → `libcast_cuda_ffi.a`,
   compiles `cuda_kernels.cu` via nvcc → `libcast_cuda_kernels.a`,
   links the CUDA driver and runtime libraries.
4. Links LLVM component libraries (`core`, `orcjit`/`nvptx`, `native`, `passes`).

Environment variables:
- `LLVM_CONFIG` — **required**, path to `llvm-config`
- `CXX` — optional, C++17 compiler (default: `c++`)
- `CUDA_PATH` — optional, CUDA toolkit root
- `CUDA_ARCH` — optional, target GPU architecture for nvcc (e.g. `120` for sm_120)
- `CAST_NUM_THREADS` — optional, CPU thread count for simulation (default: all cores)

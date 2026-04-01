# Simulation Workflow

This document describes how a quantum circuit simulation proceeds in CAST,
from circuit construction through kernel execution to result extraction.

## Overview

![CAST Simulation Workflow](simulation_workflow.png)

## Step 1: Build a Circuit

### Programmatic construction

```rust
use cast::types::{QuantumCircuit, QuantumGate};

let mut circuit = QuantumCircuit::new(4);  // 4 qubits
circuit.add(QuantumGate::h(0));
circuit.add(QuantumGate::cx(0, 1));
circuit.add(QuantumGate::depolarizing(0, 0.01));  // noisy gate
circuit.measure(&[0, 1]);  // which qubits to measure (used by trajectory mode)
```

`QuantumCircuit` (defined in `src/types/circuit.rs`) stores
`Vec<Arc<QuantumGate>>`, a qubit count, and a `measured_qubits` list. Each
`add()` validates that qubit indices are within bounds and wraps the gate in
an `Arc` (cheap reference-counted pointer — no deep copy when the circuit is
later consumed).

`measure()` specifies which qubits are measured at the end of the circuit.
This is used by trajectory mode for dead-gate elimination and measurement
sampling. For statevector/density-matrix modes it is ignored.

`eliminate_dead_gates()` returns a pruned copy of the circuit, removing gates
that cannot influence any measured qubit via backward liveness analysis.

### From OpenQASM

```rust
let circuit = QuantumCircuit::from_qasm(
    "OPENQASM 2.0; qreg q[4]; h q[0]; cx q[0],q[1];"
)?;
```

Internally calls `quantum_gate_from_qasm_gate()` to convert each
`openqasm::Gate` (enum of named gates: H, CX, RZ, etc.) to a `QuantumGate`
(matrix + qubit indices). No intermediate `CircuitGraph` is created.

### Gate representation

Each `QuantumGate` carries:

```rust
struct QuantumGate {
    matrix: ComplexSquareMatrix,  // 2^k × 2^k unitary
    qubits: Vec<u32>,            // target qubits, sorted ascending
    noise: Vec<(f64, ComplexSquareMatrix)>,  // noise branches (empty = noiseless)
}
```

Noise branches `[(p_i, U_i)]` represent a probability-weighted unitary
channel. Probabilities sum to 1.0. Each `U_i` has the same dimension as
`matrix`. Standard noise constructors:

- `QuantumGate::depolarizing(qubit, p)` → identity + `[(1-p, I), (p/3, X), (p/3, Y), (p/3, Z)]`
- `QuantumGate::h(0).with_depolarizing(0.01)` → H gate with attached noise

## Step 2: Create a Simulator

```rust
use cast::simulator::{Simulator, Cpu, Cuda, SimulationMode};
use cast::cost_model::FusionConfig;

// CPU, F64 precision, statevector mode (default)
let sim = Simulator::<Cpu>::f64();

// CUDA, F64, density-matrix mode with fusion
let sim = Simulator::<Cuda>::f64()
    .with_mode(SimulationMode::DensityMatrix)
    .with_fusion(FusionConfig::hardware_adaptive(&profile, 4));

// CPU, trajectory mode (measured qubits come from the circuit)
let sim = Simulator::<Cpu>::f64()
    .with_mode(SimulationMode::Trajectory {
        n_samples: 10_000,
        seed: Some(42),
        max_ensemble: Some(4),
    });
```

`Simulator<B>` is generic over a sealed `Backend` trait. `B` determines:

| | `Cpu` | `Cuda` |
|--|-------|--------|
| Statevector | `CPUStatevector` (SIMD-aligned, split re/im) | `CudaStatevector` (GPU device memory, interleaved) |
| Kernel manager | `CpuKernelManager` (LLVM OrcJIT) | `CudaKernelManager` (LLVM → PTX → cubin) |
| Spec | `CPUKernelGenSpec` (precision, SIMD width, tolerances) | `CudaKernelGenSpec` (precision, tolerances, SM version) |
| Apply semantics | Synchronous (threaded dispatch) | Asynchronous (enqueue + `sync()`) |

The Backend trait abstracts: `new_sv`, `init_sv`, `generate`, `apply`, `flush`,
`marginal_probabilities`, `clone_sv`. All dispatch is resolved at compile time
— no runtime enum matching.

## Step 3: Run the Simulation

```rust
let result = sim.run(&circuit)?;
```

### 3a. Build CircuitGraph (inside `run()`)

`QuantumCircuit` gates are inserted into a `CircuitGraph` — a 2D grid of
rows × qubits that represents the circuit schedule. Gates in the same row
act on disjoint qubits and can execute in any order.

### 3b. Apply fusion (inside `run()`, if configured)

`fusion::optimize()` runs two phases:

1. **Phase 1 — Size-2 canonicalization**: absorb 1-qubit gates into adjacent
   multi-qubit gates; merge adjacent 2-qubit gates on the same wires.

2. **Phase 2 — Agglomerative fusion**: iteratively merge gates across rows
   up to a size limit, accepting only fusions where the cost model predicts
   a benefit.

Noisy gates act as fusion barriers — they are never merged.

### 3c. Prepare gates (inside `run_graph()`)

The fused gate list is extracted from the CircuitGraph in row-major order.
Mode-specific transformation:

| Mode | Transform | SV qubits |
|------|-----------|-----------|
| StateVector | Validate no noisy gates; use as-is | n |
| DensityMatrix | Each gate → `to_density_matrix_gate(n)` (superoperator) | 2n |
| Trajectory | Use as-is; noisy gates handled per-trajectory | n |

For **DensityMatrix**, each gate is lifted to a superoperator:
- Noiseless gate U: `S = U ⊗ conj(U)` (4^k × 4^k matrix on 2k virtual qubits)
- Noisy gate `[(p_i, U_i)]`: `S = Σ p_i · (U_i·U) ⊗ conj(U_i·U)`

### 3d. Compile kernels

Each gate's matrix is compiled to a native kernel via the LLVM JIT pipeline:

**CPU path:**
```
QuantumGate.matrix → C++ FFI → LLVM IR (vectorized, SIMD) → O1 → OrcJIT → native function pointer
```

**CUDA path:**
```
QuantumGate.matrix → C++ FFI → LLVM IR → O1 → NVPTX PTX → cubin (driver JIT)
```

The kernel generator classifies each matrix entry:
- **Zero** (|entry| < ztol): skip entirely — no multiply, no memory access
- **±1** (|1 - |entry|| < otol): fold into sign flip — no multiply
- **General**: bake as immediate constant — one FMA

This sparsity-aware codegen is the key to CAST's performance advantage and
its favorable F32 precision properties (sparse gates accumulate zero rounding
error).

For **Trajectory** mode, noisy gate kernels are skipped. Instead, each noise
branch `U_i · matrix` is pre-composed and compiled separately. Sampling
distributions (`WeightedIndex`) are also pre-built.

### 3e. Execute

#### StateVector / DensityMatrix

```
allocate statevector (n or 2n qubits)
initialize to |0⟩ (or |0⟩⟨0|)
for each compiled kernel:
    B::apply(kernel_id, &mut sv)     // CPU: synchronous; CUDA: enqueued
B::flush()                           // CPU: no-op; CUDA: sync stream
```

#### Trajectory (Ensemble)

`Simulator::run()` first calls `circuit.eliminate_dead_gates()` to remove
gates that cannot influence the measured qubits, then proceeds with the
pruned circuit.

```
compile noise-branch kernels (deduplicated via matrix-bytes cache)
initialize ensemble = [single |0⟩ statevector, weight=1.0]

for each gate:
    if noiseless:
        apply kernel to every ensemble member
    if noisy:
        expand: each member × each noise branch → candidates
        sort candidates by weight descending
        keep top M (max_ensemble) candidates
        clone/move statevectors, apply branch kernel

B::flush()

sample measurement outcomes:
    for each ensemble member:
        compute marginal probabilities over measured_qubits
        batch-sample proportional to member weight
    aggregate into histogram
```

The key insight: noise branches store `(probability, unitary)`. The composed
operation `U_noise · U_gate` is always unitary, so applying it preserves
||ψ|| = 1 exactly. Ensemble pruning keeps the M highest-weight branches
at each noise gate, bounding approximation error by `1 − explored_weight`.

## Step 4: Inspect Results

```rust
let result = sim.run(&circuit)?;

// SimulationResult<B> fields:
result.state           // Option<QuantumState<B>> (None for trajectory mode)
result.timing          // TimingStats (execution time)
result.compile_time_s  // kernel JIT time
result.trajectory_data // Option<TrajectoryResult>
```

### QuantumState<B>

The state is either `Pure` (statevector) or `DensityMatrix` (vectorized ρ),
determined by the simulation mode:

| Mode | `state` | `is_pure()` |
|------|---------|-------------|
| StateVector | `Some(Pure)` | true |
| DensityMatrix | `Some(DensityMatrix)` | false |
| Trajectory | `None` | — |

### CPU inspection (QuantumState<Cpu>)

```rust
let state = result.state.unwrap();    // None for trajectory mode
state.n_qubits()      // u32 — physical qubit count
state.is_pure()        // bool
state.amplitudes()     // Vec<Complex> — full statevector
state.amp(idx)         // Complex — single amplitude
state.populations()    // Vec<f64> — |a_i|² or ρ[i,i]
state.trace()          // f64 — ||ψ||² or Tr(ρ)
```

### CUDA inspection (QuantumState<Cuda>)

```rust
let state = result.state.unwrap();
state.download_amplitudes()  // Result<Vec<(f64, f64)>>
state.populations()          // Result<Vec<f64>>
state.trace()                // Result<f64>
```

CUDA methods return `Result` because they involve GPU → host data transfer.
`trace()` on a pure state uses the GPU-side reduction kernel (92% peak
bandwidth on RTX 5090).

### Trajectory data

```rust
if let Some(traj) = result.trajectory_data {
    println!("samples: {}", traj.n_samples);
    println!("explored weight: {:.4}", traj.explored_weight);
    for branch in &traj.branches {
        println!("  weight={:.4}, path={:?}, samples={}",
            branch.weight, branch.noise_path, branch.n_samples);
    }
    for (outcome, &count) in &traj.histogram {
        println!("  |{outcome:b}⟩ → {count}");
    }
}
```

## Summary of Type Flow

```
QuantumGate          — gate unitary + qubits + noise branches
    │
QuantumCircuit       — ordered sequence of Arc<QuantumGate> + qubit count
    │                  + measured_qubits (user-facing, from code or QASM)
    │
    ├─ Simulator::run()
    │     │
    │  CircuitGraph   — 2D gate grid for scheduling + fusion (internal)
    │     │
    │  fusion::optimize()
    │     │
    │  gate list: Vec<Arc<QuantumGate>>
    │     │
    │  ├─ StateVector: validate, compile, execute
    │  ├─ DensityMatrix: lift to superoperators, compile, execute
    │  └─ Trajectory: compile base + noise branches, sample + execute
    │
SimulationResult<B>  — state + timing + compile time + trajectory data
    │
QuantumState<B>      — Pure(Sv) or DensityMatrix { sv, n_physical }
```

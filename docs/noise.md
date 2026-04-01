# Noisy Simulation

CAST supports noisy quantum simulation via probability-weighted unitary noise
branches embedded directly on `QuantumGate`. No separate noise channel type is
needed — noise is a property of the gate itself.

Three simulation modes handle noise differently:

- **StateVector**: errors if any gate has noise (pure unitary only)
- **DensityMatrix**: computes superoperator `S = Σ pᵢ · (Uᵢ·U) ⊗ conj(Uᵢ·U)` on 2n virtual qubits
- **Trajectory**: deterministic ensemble branching — expands all noise branches, prunes to top-M by weight, batch-samples measurements

## Noise Model

Each noisy gate stores `Vec<(f64, ComplexSquareMatrix)>` where each entry is
`(probability, unitary_operator)`. Probabilities must sum to 1.0.

For depolarizing noise with error probability `p`:
```
noise = [(1-p, I), (p/3, X), (p/3, Y), (p/3, Z)]
```

## Noise Constructors

### Standalone noise gates (identity + noise)

| Constructor | Parameters | Description |
|-------------|------------|-------------|
| `QuantumGate::depolarizing(q, p)` | p ∈ [0,1] | Symmetric depolarizing |
| `QuantumGate::bit_flip(q, p)` | p ∈ [0,1] | X error with probability p |
| `QuantumGate::phase_flip(q, p)` | p ∈ [0,1] | Z error with probability p |
| `QuantumGate::pauli_channel(q, px, py, pz)` | px+py+pz ≤ 1 | General Pauli channel |

### Attaching noise to an existing gate

```rust
// H gate with depolarizing noise
let noisy_h = QuantumGate::h(0).with_depolarizing(0.01);

// Generic noise attachment
let noisy_cx = QuantumGate::cx(0, 1).with_noise(vec![
    (0.99, ComplexSquareMatrix::eye(4)),
    (0.01, some_2q_unitary),
]);
```

## Density-Matrix Representation

An n-qubit density matrix ρ is stored as a statevector on 2n virtual qubits:

```
sv[ket_idx | (bra_idx << n)] = ρ[ket_idx, bra_idx]
```

### Lifting Gates

`gate.to_density_matrix_gate(n_total)` converts any gate to a superoperator:

- **Noiseless** gate `U`: `S = U ⊗ conj(U)`
- **Noisy** gate `[(pᵢ, Uᵢ)]`: `S = Σ pᵢ · (Uᵢ·U) ⊗ conj(Uᵢ·U)`

The `Simulator` handles this automatically in `DensityMatrix` mode.

## Worked Example

```rust
use cast::simulator::{Simulator, Cpu, SimulationMode};
use cast::types::{QuantumCircuit, QuantumGate};

// Build a noisy circuit.
let mut circuit = QuantumCircuit::new(2);
circuit.add(QuantumGate::h(0));
circuit.add(QuantumGate::depolarizing(0, 0.01));
circuit.add(QuantumGate::cx(0, 1));
circuit.add(QuantumGate::depolarizing(0, 0.01));
circuit.add(QuantumGate::depolarizing(1, 0.01));
circuit.measure(&[0, 1]);  // measure both qubits

// Density-matrix simulation.
let sim = Simulator::<Cpu>::f64()
    .with_mode(SimulationMode::DensityMatrix);
let result = sim.run(&circuit).unwrap();
let state = result.state.unwrap();
let trace = state.trace();       // should be ~1.0
let pops = state.populations();  // diagonal of ρ

// Trajectory simulation (same circuit, less memory).
// measured_qubits come from the circuit.
let sim = Simulator::<Cpu>::f64()
    .with_mode(SimulationMode::Trajectory {
        n_samples: 10_000,
        seed: Some(42),
        max_ensemble: Some(4),
    });
let result = sim.run(&circuit).unwrap();
let traj = result.trajectory_data.unwrap();
println!("explored weight: {:.4}", traj.explored_weight);
for (outcome, &count) in &traj.histogram {
    println!("  |{outcome:b}⟩ → {count}");
}
```

## Fusion with Noisy Circuits

Fusion skips noisy gates — they act as barriers. Fusion is applied
automatically when using `Simulator::run()` with `with_fusion()`:

```rust
let sim = Simulator::<Cpu>::f64()
    .with_mode(SimulationMode::DensityMatrix)
    .with_fusion(FusionConfig::size_only(4));
let result = sim.run(&circuit).unwrap();
// Noisy gates remain; only noiseless gates are fused.
```

## Extracting Results

`SimulationResult<B>` contains `state: Option<QuantumState<B>>` (None for
trajectory mode) and `trajectory_data: Option<TrajectoryResult>`:

```rust
// StateVector / DensityMatrix:
let state = result.state.unwrap();
state.n_qubits()       // u32
state.is_pure()        // true for StateVector, false for DM
state.populations()    // Vec<f64> — |aᵢ|² or ρ[i,i]
state.trace()          // f64 — ||ψ||² or Tr(ρ)

// Trajectory:
let traj = result.trajectory_data.unwrap();
traj.histogram         // HashMap<u64, u64> — bitstring → count
traj.n_samples         // u64
traj.branches          // Vec<ExploredBranch>
traj.explored_weight   // f64 (≤ 1.0)
```

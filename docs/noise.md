# Noisy Simulation

CAST supports noisy quantum simulation via a `NoiseModel` embedded directly
on `QuantumGate`.

Three simulation modes handle noise differently:

- **StateVector**: rejects noisy gates (pure unitary only)
- **DensityMatrix**: computes superoperator `S = Σ pᵢ · Kᵢ ⊗ conj(Kᵢ)` on 2n virtual qubits
- **Trajectory**: deterministic ensemble branching — expands all noise paths, prunes to top-M by weight, batch-samples measurements

## Noise Model

`NoiseModel` stores full Kraus operators `[(pᵢ, Kᵢ)]` where `Kᵢ = Vᵢ · U`
(noise-only operator `Vᵢ` pre-composed with the gate's base unitary `U`).
Probabilities must sum to 1.0.

Constructors like `with_noise()` accept noise-only operators and pre-compose
them internally. For depolarizing noise with error probability `p`:
```
with_noise([(1-p, I), (p/3, X), (p/3, Y), (p/3, Z)])
→ stores [(1-p, U), (p/3, X·U), (p/3, Y·U), (p/3, Z·U)]
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
- **Noisy** gate with Kraus ops `[(pᵢ, Kᵢ)]`: `S = Σ pᵢ · Kᵢ ⊗ conj(Kᵢ)`

`Simulator::simulate(&graph, Representation::DensityMatrix)` handles this
lifting automatically.

## Worked Example

```rust
use cast::simulator::{Simulator, Cpu, Representation, TrajectoryOpts};
use cast::types::{QuantumCircuit, QuantumGate};
use cast::CircuitGraph;

// Build a noisy circuit.
let mut circuit = QuantumCircuit::new(2);
circuit.add(QuantumGate::h(0));
circuit.add(QuantumGate::depolarizing(0, 0.01));
circuit.add(QuantumGate::cx(0, 1));
circuit.add(QuantumGate::depolarizing(0, 0.01));
circuit.add(QuantumGate::depolarizing(1, 0.01));

let graph = CircuitGraph::from_circuit(&circuit);
let sim = Simulator::<Cpu>::f64();

// Density-matrix simulation.
let state = sim.simulate(&graph, Representation::DensityMatrix).unwrap();
let trace = state.trace();       // should be ~1.0
let pops = state.populations();  // diagonal of ρ

// Trajectory simulation (same circuit, less memory).
let opts = TrajectoryOpts {
    measured_qubits: vec![0, 1],
    n_samples: 10_000,
    seed: Some(42),
    max_ensemble: Some(4),
};
let traj = sim.sample_trajectory(&graph, &opts).unwrap();
println!("explored weight: {:.4}", traj.explored_weight);
for (outcome, &count) in &traj.histogram {
    println!("  |{outcome:b}⟩ → {count}");
}
```

## Fusion with Noisy Circuits

Noisy gates participate in fusion. When two noisy gates are fused, their
Kraus operators are composed via Cartesian product: each `Kᵢ · Kⱼ` pair
becomes a branch in the fused gate's noise model.

Apply fusion as a separate step before calling the simulator:

```rust
use cast::cost_model::FusionConfig;
use cast::fusion;

let mut graph = CircuitGraph::from_circuit(&circuit);
fusion::optimize(&mut graph, &FusionConfig::size_only(4));
let state = sim.simulate(&graph, Representation::DensityMatrix).unwrap();
```

## Extracting Results

Each `Simulator` method returns a concrete result type — no `Option` fields
or mode-dependent unwrapping.

```rust
// Statevector or density matrix: QuantumState<B>
let state = sim.simulate(&graph, Representation::DensityMatrix).unwrap();
state.n_qubits();    // u32
state.is_pure();     // true for StateVector, false for DensityMatrix
state.populations(); // Vec<f64> — |aᵢ|² or ρ[i,i]
state.trace();       // f64 — ||ψ||² or Tr(ρ)

// Trajectory: TrajectoryResult
let traj = sim.sample_trajectory(&graph, &opts).unwrap();
traj.histogram;       // HashMap<u64, u64> — bitstring → count
traj.n_samples;       // u64
traj.branches;        // Vec<ExploredBranch>
traj.explored_weight; // f64 (≤ 1.0)
```

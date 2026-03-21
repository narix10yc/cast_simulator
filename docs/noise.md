# Noisy Simulation

CAST supports open-system (noisy) quantum simulation using the density-matrix
formalism.  Noise channels are specified as Kraus operators and executed on the
existing statevector engine — no separate density-matrix kernel is needed.

## Noise Channels

All channels are constructed via `KrausChannel` and converted to gates with
`.to_gate()`.

### Single-Qubit Channels

| Constructor | Parameters | Description |
|-------------|------------|-------------|
| `depolarizing(q, p)` | p ∈ [0,1] | Symmetric depolarizing: ε(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ) |
| `bit_flip(q, p)` | p ∈ [0,1] | X error with probability p |
| `phase_flip(q, p)` | p ∈ [0,1] | Z error with probability p |
| `pauli_channel(q, px, py, pz)` | px+py+pz ≤ 1 | General Pauli channel |
| `amplitude_damping(q, γ)` | γ ∈ [0,1] | Spontaneous emission (\|1⟩ → \|0⟩ with prob γ) |
| `phase_damping(q, λ)` | λ ∈ [0,1] | Pure dephasing (coherences decay, populations preserved) |
| `generalized_amplitude_damping(q, p, γ)` | p,γ ∈ [0,1] | Thermalization at finite temperature |

### Multi-Qubit Channels

| Constructor | Parameters | Description |
|-------------|------------|-------------|
| `symmetric_depolarizing(qubits, p)` | sorted qubits, p ∈ [0,1] | n-qubit depolarizing (all 4^n − 1 Paulis) |

### From Existing Gates

| Constructor | Description |
|-------------|-------------|
| `from_gate(gate)` | Wrap a unitary gate as a single-Kraus channel (noiseless) |

### Validation

```rust
let ch = KrausChannel::depolarizing(0, 0.1);
assert!(ch.check_cptp(1e-12));  // verify trace-preserving condition
```

## Density-Matrix Representation

An n-qubit density matrix ρ is stored as a statevector on 2n virtual qubits:

```
sv[ket_idx | (bra_idx << n)] = ρ[ket_idx, bra_idx]
```

This flattens the n×n matrix into a vector that the statevector engine can
process directly.

## Lifting Gates to Density-Matrix Form

```rust
let dm_gate = gate.to_density_matrix_gate(n_total);
```

- **Unitary gate U** on k physical qubits: computes the superoperator
  `S = U ⊗ conj(U)` (a 4^k × 4^k matrix) acting on 2k virtual qubits
  `[q₀, ..., q_{k-1}, q₀+n, ..., q_{k-1}+n]`.

- **Channel gate**: reuses its pre-computed superoperator matrix with the same
  virtual-qubit mapping.

The resulting gate is a plain (non-channel) `QuantumGate` on 2k virtual qubits,
applied to the 2n-qubit density-matrix statevector using the standard kernel.

## Worked Example

```rust
use cast::types::{QuantumGate, KrausChannel};

let n_phys = 4;

// Build a noisy circuit.
let mut gates = Vec::new();
gates.push(QuantumGate::h(0));
gates.push(KrausChannel::depolarizing(0, 0.01).to_gate());
gates.push(QuantumGate::cx(0, 1));
gates.push(KrausChannel::depolarizing(0, 0.01).to_gate());
gates.push(KrausChannel::depolarizing(1, 0.01).to_gate());

// Lift to density-matrix gates.
let dm_gates: Vec<QuantumGate> = gates
    .iter()
    .map(|g| g.to_density_matrix_gate(n_phys as usize))
    .collect();

// Execute on a 2n-qubit statevector initialized to |0...0⟩⟨0...0|.
// (The statevector has 2^(2n) amplitudes, with only index 0 set to 1.)
```

## Fusion with Noisy Circuits

Fusion skips channel gates — they are preserved unchanged:

```rust
use cast::{fusion, cost_model::FusionConfig, CircuitGraph};

let mut graph = CircuitGraph::new();
for gate in &gates {
    graph.insert_gate(gate.clone());
}
fusion::optimize(&mut graph, &FusionConfig::size_only(4));
// Channel gates remain; only unitary gates are fused.
```

For density-matrix circuits, all gates have even virtual-qubit counts.
Use even fusion size limits (4, 6) and cap at 6 to avoid excessive JIT
pressure.

## Extracting Results

From the density-matrix statevector:

```rust
// Trace: Tr(ρ) = Σᵢ sv[i | (i << n)].re
let trace: f64 = (0..dim).map(|i| sv[i | (i << n)].re).sum();

// Diagonal: ρ[i,i] = sv[i | (i << n)].re  (populations/probabilities)
let populations: Vec<f64> = (0..dim).map(|i| sv[i | (i << n)].re).collect();

// Off-diagonal: ρ[i,j] = sv[i | (j << n)]  (coherences)
```

For a CPTP channel, `Tr(ρ) = 1` is preserved exactly (up to floating-point
precision).

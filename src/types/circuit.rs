//! User-facing quantum circuit: a flat sequence of [`QuantumGate`]s.

use std::sync::Arc;

use anyhow::Result;

use crate::types::QuantumGate;

/// A sequence of quantum gates forming a circuit.
///
/// This is the primary user-facing circuit type. Build circuits programmatically
/// or parse from OpenQASM, then convert to a
/// [`CircuitGraph`](crate::CircuitGraph) via
/// [`CircuitGraph::from_circuit`](crate::CircuitGraph::from_circuit) and pass
/// to a [`Simulator`](crate::simulator::Simulator) method.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    gates: Vec<Arc<QuantumGate>>,
    n_qubits: u32,
    /// Qubits to measure at the end of the circuit. Empty means "measure
    /// nothing" — dead-gate elimination will remove all gates.
    measured_qubits: Vec<u32>,
}

impl QuantumCircuit {
    /// Create an empty circuit on `n_qubits` qubits.
    pub fn new(n_qubits: u32) -> Self {
        Self {
            gates: Vec::new(),
            n_qubits,
            measured_qubits: Vec::new(),
        }
    }

    /// Add a gate to the circuit. Returns `&mut Self` for chaining.
    ///
    /// # Panics
    /// Panics if any qubit index in the gate is ≥ `n_qubits`.
    pub fn add(&mut self, gate: QuantumGate) -> &mut Self {
        for &q in gate.qubits() {
            assert!(
                q < self.n_qubits,
                "qubit {q} out of range for {}-qubit circuit",
                self.n_qubits
            );
        }
        self.gates.push(Arc::new(gate));
        self
    }

    /// Parse an OpenQASM 2.0 string into a circuit.
    pub fn from_qasm(qasm: &str) -> Result<Self> {
        let parsed = crate::openqasm::parse_qasm(qasm)?;
        Ok(Self::from_openqasm(&parsed))
    }

    /// Convert from a parsed OpenQASM circuit.
    pub fn from_openqasm(qasm_circuit: &crate::openqasm::Circuit) -> Self {
        use crate::circuit_graph::quantum_gate_from_qasm_gate;
        let n_qubits = qasm_circuit.required_qreg_size();
        let gates: Vec<Arc<QuantumGate>> = qasm_circuit
            .gates
            .iter()
            .map(|g| Arc::new(quantum_gate_from_qasm_gate(g)))
            .collect();
        Self {
            gates,
            n_qubits,
            measured_qubits: Vec::new(),
        }
    }

    /// Number of qubits in the circuit.
    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    /// Number of gates in the circuit.
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Whether the circuit has no gates.
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// The gates in circuit order.
    pub fn gates(&self) -> &[Arc<QuantumGate>] {
        &self.gates
    }

    /// Mark qubits to measure at the end of the circuit. Returns `&mut Self`
    /// for chaining.
    ///
    /// # Panics
    /// Panics if any qubit index is ≥ `n_qubits`.
    pub fn measure(&mut self, qubits: &[u32]) -> &mut Self {
        for &q in qubits {
            assert!(
                q < self.n_qubits,
                "measured qubit {q} out of range for {}-qubit circuit",
                self.n_qubits
            );
        }
        self.measured_qubits = qubits.to_vec();
        self
    }

    /// Which qubits are measured at the end of the circuit.
    pub fn measured_qubits(&self) -> &[u32] {
        &self.measured_qubits
    }

    /// Return a new circuit with gates that cannot influence any measured qubit
    /// removed. Performs backward liveness analysis: starting from
    /// `measured_qubits`, any gate touching a live qubit makes all its qubits
    /// live; gates touching only dead qubits are eliminated.
    pub fn eliminate_dead_gates(&self) -> Self {
        let mut live = 0u64;
        for &q in &self.measured_qubits {
            live |= 1u64 << q;
        }

        let mut is_live: Vec<bool> = vec![false; self.gates.len()];
        for (i, gate) in self.gates.iter().enumerate().rev() {
            let gate_mask: u64 = gate.qubits().iter().fold(0u64, |m, &q| m | (1u64 << q));
            if live & gate_mask != 0 {
                is_live[i] = true;
                live |= gate_mask;
            }
        }

        let gates = self
            .gates
            .iter()
            .zip(is_live.iter())
            .filter(|(_, &l)| l)
            .map(|(g, _)| Arc::clone(g))
            .collect();

        Self {
            gates,
            n_qubits: self.n_qubits,
            measured_qubits: self.measured_qubits.clone(),
        }
    }

    /// Build an `n_qubits`-qubit QFT circuit.
    ///
    /// When `swap` is true, trailing SWAP gates are appended to reverse the
    /// qubit order (standard QFT convention).
    pub fn qft(n_qubits: u32, swap: bool) -> Self {
        let n = n_qubits;
        let mut circuit = Self::new(n);
        for i in 0..n {
            circuit.add(QuantumGate::h(i));
            for j in (i + 1)..n {
                let k = j - i;
                circuit.add(QuantumGate::cp(
                    std::f64::consts::PI / (1u64 << k) as f64,
                    j,
                    i,
                ));
            }
        }
        if swap {
            for i in 0..n / 2 {
                circuit.add(QuantumGate::swap(i, n - 1 - i));
            }
        }
        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::{Cpu, Representation, Simulator};
    use crate::CircuitGraph;
    #[test]
    #[should_panic(expected = "out of range")]
    fn add_rejects_out_of_range_qubit() {
        let mut c = QuantumCircuit::new(2);
        c.add(QuantumGate::h(2));
    }

    #[test]
    fn qft_gate_count() {
        // n-qubit QFT without swaps: n H gates + n(n-1)/2 CP gates
        // with swaps: + floor(n/2) SWAP gates
        let c3 = QuantumCircuit::qft(3, false);
        assert_eq!(c3.n_qubits(), 3);
        assert_eq!(c3.len(), 3 + 3); // 3 H + 3 CP

        let c3s = QuantumCircuit::qft(3, true);
        assert_eq!(c3s.len(), 3 + 3 + 1); // + 1 SWAP

        let c4 = QuantumCircuit::qft(4, true);
        assert_eq!(c4.len(), 4 + 6 + 2); // 4 H + 6 CP + 2 SWAP
    }

    #[test]
    fn qft_on_computational_basis_produces_uniform() {
        // QFT|0⟩ = uniform superposition = H⊗n|0⟩
        // All 2^n amplitudes should have magnitude 1/√(2^n).
        let n = 8u32;
        let circuit = QuantumCircuit::qft(n, true);
        let graph = CircuitGraph::from_circuit(&circuit);
        let sim = Simulator::<Cpu>::f64();
        let pops = sim
            .simulate(&graph, Representation::StateVector)
            .unwrap()
            .populations();
        let expected = 1.0 / (1u64 << n) as f64;
        for (i, &p) in pops.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-10,
                "population[{i}] = {p}, expected {expected}"
            );
        }
    }

    #[test]
    fn qft_swap_permutes_amplitudes() {
        // QFT|k⟩ has uniform magnitudes but different phases.
        // The trailing swaps bit-reverse the amplitude indices.
        // Verify: amps_swap[j] == amps_noswap[bit_reverse(j)].
        let n = 8u32;
        let sim = Simulator::<Cpu>::f64();

        let mut c_swap = QuantumCircuit::new(n);
        c_swap.add(QuantumGate::x(0)); // |1⟩
        for g in QuantumCircuit::qft(n, true).gates() {
            c_swap.add((**g).clone());
        }

        let mut c_noswap = QuantumCircuit::new(n);
        c_noswap.add(QuantumGate::x(0));
        for g in QuantumCircuit::qft(n, false).gates() {
            c_noswap.add((**g).clone());
        }

        let g_swap = CircuitGraph::from_circuit(&c_swap);
        let g_noswap = CircuitGraph::from_circuit(&c_noswap);
        let amps_swap = sim
            .simulate(&g_swap, Representation::StateVector)
            .unwrap()
            .amplitudes();
        let amps_noswap = sim
            .simulate(&g_noswap, Representation::StateVector)
            .unwrap()
            .amplitudes();

        let bit_rev = |x: usize, bits: u32| -> usize {
            let mut r = 0;
            for b in 0..bits {
                if x & (1 << b) != 0 {
                    r |= 1 << (bits - 1 - b);
                }
            }
            r
        };

        for j in 0..(1 << n) {
            let rev = bit_rev(j, n);
            let diff = (amps_swap[j] - amps_noswap[rev]).norm();
            assert!(
                diff < 1e-10,
                "amps_swap[{j}] != amps_noswap[{rev}]: diff = {diff}"
            );
        }
    }

    #[test]
    fn double_qft_is_identity() {
        // QFT followed by inverse QFT should return to |0⟩.
        // QFT^{-1} = SWAP · QFT† but for the standard QFT with swaps,
        // (QFT_swap)^2 on |0⟩ maps back to a computational basis state.
        // Easier check: QFT(no-swap) twice = bit-reversal permutation,
        // so QFT(no-swap)^2 on |0...0⟩ = |0...0⟩ (since 0 reversed is 0).
        let n = 8u32;
        let qft_gates = QuantumCircuit::qft(n, false);

        let mut circuit = QuantumCircuit::new(n);
        // Apply QFT twice (without swaps)
        for g in qft_gates.gates() {
            circuit.add((**g).clone());
        }
        for g in qft_gates.gates() {
            circuit.add((**g).clone());
        }

        let graph = CircuitGraph::from_circuit(&circuit);
        let sim = Simulator::<Cpu>::f64();
        let pops = sim
            .simulate(&graph, Representation::StateVector)
            .unwrap()
            .populations();

        // QFT(no-swap)^2 is bit-reversal. |00...0⟩ reversed = |00...0⟩,
        // so population should be concentrated on |0⟩.
        assert!(
            pops[0] > 0.99,
            "expected |0⟩ dominant after double QFT, got pop[0] = {}",
            pops[0],
        );
    }

    // ── Dead gate elimination tests ────────────────────────────────────────

    #[test]
    fn dce_eliminates_independent_qubits() {
        // H(0), H(2), CX(2,3). Measure [0]. H(2) and CX(2,3) are dead.
        let mut c = QuantumCircuit::new(4);
        c.add(QuantumGate::h(0));
        c.add(QuantumGate::h(2));
        c.add(QuantumGate::cx(2, 3));
        c.measure(&[0]);
        let pruned = c.eliminate_dead_gates();
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned.gates()[0].qubits(), &[0]);
    }

    #[test]
    fn dce_keeps_entangled_chain() {
        // H(0), CX(0,1), H(2). Measure [1].
        // q1 live → CX(0,1) touches q1 → q0 becomes live → H(0) kept.
        let mut c = QuantumCircuit::new(4);
        c.add(QuantumGate::h(0));
        c.add(QuantumGate::cx(0, 1));
        c.add(QuantumGate::h(2));
        c.measure(&[1]);
        let pruned = c.eliminate_dead_gates();
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned.gates()[0].qubits(), &[0]); // H(0)
        assert_eq!(pruned.gates()[1].qubits(), &[0, 1]); // CX(0,1)
    }

    #[test]
    fn dce_eliminates_noisy_gates() {
        // H(0), depolarizing(2), measure [0]. Noise on q2 is dead.
        let mut c = QuantumCircuit::new(4);
        c.add(QuantumGate::h(0));
        c.add(QuantumGate::depolarizing(2, 0.01));
        c.measure(&[0]);
        let pruned = c.eliminate_dead_gates();
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned.gates()[0].qubits(), &[0]);
    }

    #[test]
    fn dce_empty_measured_qubits_removes_all() {
        let mut c = QuantumCircuit::new(4);
        c.add(QuantumGate::h(0));
        c.add(QuantumGate::cx(0, 1));
        // no measure() call — measured_qubits is empty
        let pruned = c.eliminate_dead_gates();
        assert!(pruned.is_empty());
    }

    #[test]
    fn dce_preserves_measured_qubits() {
        let mut c = QuantumCircuit::new(4);
        c.add(QuantumGate::h(0));
        c.measure(&[0]);
        let pruned = c.eliminate_dead_gates();
        assert_eq!(pruned.measured_qubits(), &[0]);
    }
}

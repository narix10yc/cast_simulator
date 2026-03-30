//! User-facing quantum circuit: a flat sequence of [`QuantumGate`]s.

use std::sync::Arc;

use anyhow::Result;

use crate::types::QuantumGate;

/// A sequence of quantum gates forming a circuit.
///
/// This is the primary user-facing circuit type. Build circuits programmatically
/// or parse from OpenQASM, then pass to [`Simulator::run`]. Internally converted
/// to a [`CircuitGraph`] for scheduling and fusion before simulation.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    gates: Vec<Arc<QuantumGate>>,
    n_qubits: u32,
}

impl QuantumCircuit {
    /// Create an empty circuit on `n_qubits` qubits.
    pub fn new(n_qubits: u32) -> Self {
        Self {
            gates: Vec::new(),
            n_qubits,
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
        Self { gates, n_qubits }
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

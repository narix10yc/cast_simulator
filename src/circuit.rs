//! Quantum circuit graph built from a sequence of [`QuantumGate`]s.
//!
//! [`CircuitGraph`] arranges gates into a 2-D grid of *rows × qubits*.
//! Each row holds at most one gate per qubit; gates that touch non-overlapping
//! qubits are packed into the same row. The grid can be constructed directly
//! or loaded from an [`openqasm::Circuit`] via [`CircuitGraph::from_qasm_circuit`].

use std::f64::consts::PI;
use std::ops::{Index, IndexMut};
use std::collections::{HashMap, HashSet};

use crate::{
    openqasm,
    types::{QuantumGate, Rational},
};

/// Stable, dense index into [`CircuitGraph::gates`].
pub type GateId = usize;

/// One row of the circuit grid: a slot per qubit, containing the [`GateId`] of
/// the gate occupying that qubit in this row, or `None` if the qubit is idle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitRow {
    cells: Vec<Option<GateId>>,
}

impl CircuitRow {
    pub fn new(width: usize) -> Self {
        Self {
            cells: vec![None; width],
        }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn resize(&mut self, new_len: usize, value: Option<GateId>) {
        self.cells.resize(new_len, value);
    }

    pub fn get(&self, index: usize) -> Option<&Option<GateId>> {
        self.cells.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Option<GateId>> {
        self.cells.iter()
    }

    pub fn gate_id_at(&self, qubit: usize) -> Option<GateId> {
        self.get(qubit).copied().flatten()
    }

    pub fn is_vacant(&self, qubits: &[u32]) -> bool {
        qubits.iter().all(|&qubit| {
            let q = qubit as usize;
            q >= self.len() || self[q].is_none()
        })
    }

    pub fn is_vacent(&self, qubits: &[u32]) -> bool {
        self.is_vacant(qubits)
    }

    pub fn place_gate(&mut self, gate_id: GateId, qubits: &[u32]) {
        assert!(
            self.is_vacant(qubits),
            "row is not vacant for gate placement"
        );
        for &qubit in qubits {
            self[qubit as usize] = Some(gate_id);
        }
    }

    pub fn clear_gate(&mut self, qubits: &[u32]) {
        for &qubit in qubits {
            self[qubit as usize] = None;
        }
    }

    pub fn gate_ids(&self) -> Vec<GateId> {
        let mut out = Vec::new();
        for gate_id in self.iter().flatten().copied() {
            if !out.contains(&gate_id) {
                out.push(gate_id);
            }
        }
        out
    }
}

impl Index<usize> for CircuitRow {
    type Output = Option<GateId>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cells[index]
    }
}

impl IndexMut<usize> for CircuitRow {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

/// A quantum circuit represented as a 2-D gate grid.
///
/// The grid has dimensions `n_rows × n_qubits`. Each [`CircuitRow`] is a
/// `Vec<Option<GateId>>` indexed by qubit. When a multi-qubit gate occupies
/// qubits `q0..qk` in row `r`, every one of those qubit slots stores the same
/// [`GateId`], so you can look up any participant qubit's gate in O(1).
///
/// Gates are pushed left: [`insert_gate`](CircuitGraph::insert_gate) places
/// each gate in the earliest existing row where all of its target qubits are
/// free, creating a new row only if none qualifies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitGraph {
    n_qubits: usize,
    rows: Vec<CircuitRow>,
    gates: Vec<Option<QuantumGate>>,
}

impl CircuitGraph {
    /// Creates an empty circuit with no qubits and no rows.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a [`CircuitGraph`] from an OpenQASM [`Circuit`](openqasm::Circuit)
    /// by converting each gate and inserting it in order.
    pub fn from_qasm_circuit(circuit: &openqasm::Circuit) -> Self {
        let mut graph = Self::new();
        for gate in &circuit.gates {
            graph.insert_gate(quantum_gate_from_qasm_gate(gate));
        }
        graph
    }

    /// Number of qubit wires tracked by the graph.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of rows (time steps) in the circuit grid.
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// All rows of the circuit grid, in order.
    pub fn rows(&self) -> &[CircuitRow] {
        &self.rows
    }

    /// All gate slots, indexed by [`GateId`]. Entries are `Some` for live gates
    /// and `None` for removed ones (removal is not yet implemented but the slot
    /// type leaves room for it).
    pub fn gates(&self) -> &[Option<QuantumGate>] {
        &self.gates
    }

    /// Returns `true` if the graph has no rows and no gates.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.gates.is_empty()
    }

    pub fn check_consistency(&self) -> Result<(), String> {
        for (row_idx, row) in self.rows.iter().enumerate() {
            if row.len() != self.n_qubits {
                return Err(format!(
                    "row {row_idx} has width {}, expected {}",
                    row.len(),
                    self.n_qubits
                ));
            }
        }

        let mut seen_gate_rows: HashMap<GateId, usize> = HashMap::new();
        let mut seen_gate_ids: HashSet<GateId> = HashSet::new();

        for (row_idx, row) in self.rows.iter().enumerate() {
            for gate_id in row.gate_ids() {
                let Some(gate) = self.gate(gate_id) else {
                    return Err(format!("row {row_idx} references dead or invalid gate id {gate_id}"));
                };

                if let Some(previous_row) = seen_gate_rows.insert(gate_id, row_idx) {
                    if previous_row != row_idx {
                        return Err(format!(
                            "gate id {gate_id} appears on multiple rows: {previous_row} and {row_idx}"
                        ));
                    }
                }

                for &qubit in gate.qubits() {
                    let q = qubit as usize;
                    if q >= self.n_qubits {
                        return Err(format!(
                            "gate id {gate_id} targets qubit {q} outside graph width {}",
                            self.n_qubits
                        ));
                    }
                    if row.gate_id_at(q) != Some(gate_id) {
                        return Err(format!(
                            "gate id {gate_id} is missing from row {row_idx} at qubit {q}"
                        ));
                    }
                }

                for qubit in 0..self.n_qubits {
                    if row.gate_id_at(qubit) == Some(gate_id)
                        && !gate.qubits().contains(&(qubit as u32))
                    {
                        return Err(format!(
                            "row {row_idx} contains gate id {gate_id} at unrelated qubit {qubit}"
                        ));
                    }
                }

                seen_gate_ids.insert(gate_id);
            }
        }

        for (gate_id, gate) in self.gates.iter().enumerate() {
            match gate {
                Some(_) if !seen_gate_ids.contains(&gate_id) => {
                    return Err(format!("live gate id {gate_id} is not placed in any row"));
                }
                None if seen_gate_ids.contains(&gate_id) => {
                    return Err(format!("dead gate id {gate_id} is still referenced by a row"));
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Returns a reference to the gate with the given [`GateId`], or `None` if
    /// the id is out of range or the slot has been vacated.
    pub fn gate(&self, gate_id: GateId) -> Option<&QuantumGate> {
        self.gates.get(gate_id).and_then(|gate| gate.as_ref())
    }

    /// Returns the [`GateId`] at `(row, qubit)`, or `None` if the cell is empty
    /// or the coordinates are out of range.
    pub fn gate_id_at(&self, row: usize, qubit: usize) -> Option<GateId> {
        self.rows
            .get(row)
            .and_then(|row_data| row_data.gate_id_at(qubit))
    }

    /// Grows every row to `required` qubit slots if the graph currently has
    /// fewer than `required` qubits. Does nothing if already wide enough.
    pub fn ensure_n_qubits(&mut self, required: usize) {
        if required <= self.n_qubits {
            return;
        }
        for row in &mut self.rows {
            row.resize(required, None);
        }
        self.n_qubits = required;
    }

    /// Appends a new, fully-empty row and returns its index.
    pub fn append_empty_row(&mut self) -> usize {
        self.rows.push(CircuitRow::new(self.n_qubits));
        self.rows.len() - 1
    }

    /// Returns `true` if every qubit in `qubits` is free (no gate assigned) in
    /// the given row. Qubit indices beyond the current grid width are also
    /// considered free.
    pub fn is_row_vacant(&self, row: usize, qubits: &[u32]) -> bool {
        self.rows[row].is_vacant(qubits)
    }

    /// Inserts `gate` into the graph and returns its [`GateId`].
    ///
    /// The gate is placed in the earliest existing row where all of its target
    /// qubits are free. If no such row exists, a new row is appended. The
    /// graph's qubit count is expanded if the gate addresses qubits beyond the
    /// current width.
    pub fn insert_gate(&mut self, gate: QuantumGate) -> GateId {
        let required_qubits = gate
            .qubits()
            .iter()
            .copied()
            .max()
            .map(|q| q as usize + 1)
            .unwrap_or(0);
        self.ensure_n_qubits(required_qubits);
        let row_index = self.find_or_create_row(&gate);
        let qubits = gate.qubits().to_vec();

        let gate_id = self.gates.len();
        self.gates.push(Some(gate));
        self.rows[row_index].place_gate(gate_id, &qubits);
        gate_id
    }

    pub(crate) fn insert_gate_at_row(&mut self, row_index: usize, gate: QuantumGate) -> GateId {
        assert!(row_index < self.rows.len(), "row index out of bounds");
        let required_qubits = gate
            .qubits()
            .iter()
            .copied()
            .max()
            .map(|q| q as usize + 1)
            .unwrap_or(0);
        self.ensure_n_qubits(required_qubits);
        assert!(
            self.is_row_vacant(row_index, gate.qubits()),
            "target row is not vacant for gate insertion"
        );

        let qubits = gate.qubits().to_vec();
        let gate_id = self.gates.len();
        self.gates.push(Some(gate));
        self.rows[row_index].place_gate(gate_id, &qubits);
        gate_id
    }

    pub(crate) fn remove_gate_at(&mut self, row: usize, qubit: usize) -> Option<QuantumGate> {
        let gate_id = self.gate_id_at(row, qubit)?;
        let qubits = self.gate(gate_id)?.qubits().to_vec();
        self.rows[row].clear_gate(&qubits);
        self.gates[gate_id].take()
    }

    pub(crate) fn squeeze(&mut self) {
        let live_gate_ids = self.live_gate_ids_in_row_order();
        let mut new_rows: Vec<CircuitRow> = Vec::new();

        for gate_id in live_gate_ids {
            let gate = self
                .gate(gate_id)
                .expect("live gate id should refer to an existing gate");

            let mut row_index = None;
            for idx in 0..new_rows.len() {
                if new_rows[idx].is_vacant(gate.qubits()) {
                    row_index = Some(idx);
                    break;
                }
            }

            let idx = row_index.unwrap_or_else(|| {
                new_rows.push(CircuitRow::new(self.n_qubits));
                new_rows.len() - 1
            });

            new_rows[idx].place_gate(gate_id, gate.qubits());
        }

        self.rows = new_rows;
    }

    /// Returns the index of the first row where `gate` can be placed, creating
    /// a new empty row if every existing row has a conflict.
    fn find_or_create_row(&mut self, gate: &QuantumGate) -> usize {
        for row_index in 0..self.rows.len() {
            if self.is_row_vacant(row_index, gate.qubits()) {
                return row_index;
            }
        }
        self.append_empty_row()
    }

    fn live_gate_ids_in_row_order(&self) -> Vec<GateId> {
        let mut out = Vec::new();
        for row in &self.rows {
            out.extend(row.gate_ids());
        }
        out
    }
}

/// Converts an OpenQASM [`Gate`](openqasm::Gate) to a [`QuantumGate`].
///
/// Angle parameters are resolved to `f64` radians before constructing the gate.
/// Multi-qubit gate operands may arrive in any order; [`QuantumGate::new`]
/// canonicalises qubit order internally.
fn quantum_gate_from_qasm_gate(gate: &openqasm::Gate) -> QuantumGate {
    match gate {
        openqasm::Gate::X(q) => QuantumGate::x(*q),
        openqasm::Gate::Y(q) => QuantumGate::y(*q),
        openqasm::Gate::Z(q) => QuantumGate::z(*q),
        openqasm::Gate::H(q) => QuantumGate::h(*q),
        openqasm::Gate::S(q) => QuantumGate::s(*q),
        openqasm::Gate::T(q) => QuantumGate::t(*q),
        openqasm::Gate::RX(theta, q) => QuantumGate::rx(angle_to_f64(theta), *q),
        openqasm::Gate::RY(theta, q) => QuantumGate::ry(angle_to_f64(theta), *q),
        openqasm::Gate::RZ(theta, q) => QuantumGate::rz(angle_to_f64(theta), *q),
        openqasm::Gate::U3(theta, phi, lambda, q) => QuantumGate::u3(
            angle_to_f64(theta),
            angle_to_f64(phi),
            angle_to_f64(lambda),
            *q,
        ),
        openqasm::Gate::CX(ctrl, targ) => QuantumGate::cx(*ctrl, *targ),
        openqasm::Gate::CZ(ctrl, targ) => QuantumGate::cz(*ctrl, *targ),
        openqasm::Gate::SWAP(q0, q1) => QuantumGate::swap(*q0, *q1),
        openqasm::Gate::CCX(ctrl0, ctrl1, targ) => QuantumGate::ccx(*ctrl0, *ctrl1, *targ),
    }
}

/// Converts an OpenQASM [`Angle`](openqasm::Angle) to radians as `f64`.
fn angle_to_f64(angle: &openqasm::Angle) -> f64 {
    match angle {
        openqasm::Angle::Number(value) => *value,
        openqasm::Angle::RationalPi(r) => rational_pi_to_f64(*r),
    }
}

/// Converts a rational multiple of π to radians: `r × π`.
fn rational_pi_to_f64(r: Rational) -> f64 {
    r.to_f64() * PI
}

#[cfg(test)]
mod tests {
    use super::{CircuitGraph, CircuitRow};
    use crate::{
        openqasm::{parse_qasm, Angle, Circuit, Gate},
        types::QuantumGate,
    };

    #[test]
    fn new_graph_starts_empty() {
        let graph = CircuitGraph::new();

        assert_eq!(graph.n_qubits(), 0);
        assert_eq!(graph.n_rows(), 0);
        assert!(graph.rows().is_empty());
        assert!(graph.gates().is_empty());
        assert!(graph.is_empty());
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(CircuitGraph::default(), CircuitGraph::new());
    }

    #[test]
    fn check_consistency_accepts_valid_graph() {
        let mut graph = CircuitGraph::new();
        graph.insert_gate(QuantumGate::x(0));
        graph.insert_gate(QuantumGate::cx(0, 1));

        assert!(graph.check_consistency().is_ok());
    }

    #[test]
    fn check_consistency_rejects_live_gate_missing_from_rows() {
        let mut graph = CircuitGraph::new();
        graph.insert_gate(QuantumGate::x(0));
        graph.gates[0] = Some(QuantumGate::x(0));
        graph.rows[0].clear_gate(&[0]);

        let err = graph.check_consistency().unwrap_err();
        assert!(err.contains("not placed in any row"));
    }

    #[test]
    fn check_consistency_rejects_dead_gate_referenced_by_row() {
        let mut graph = CircuitGraph::new();
        graph.insert_gate(QuantumGate::x(0));
        graph.gates[0] = None;

        let err = graph.check_consistency().unwrap_err();
        assert!(err.contains("dead or invalid gate id"));
    }

    #[test]
    fn row_place_clear_and_gate_ids_work() {
        let mut row = CircuitRow::new(4);
        row.place_gate(7, &[0, 2]);
        row.place_gate(8, &[1]);

        assert_eq!(row.gate_id_at(0), Some(7));
        assert_eq!(row.gate_id_at(2), Some(7));
        assert_eq!(row.gate_ids(), vec![7, 8]);

        row.clear_gate(&[0, 2]);
        assert_eq!(row.gate_id_at(0), None);
        assert_eq!(row.gate_id_at(2), None);
        assert_eq!(row.gate_ids(), vec![8]);
    }

    #[test]
    fn insert_gate_reuses_first_vacant_row() {
        let mut graph = CircuitGraph::new();
        graph.insert_gate(QuantumGate::x(0));
        graph.insert_gate(QuantumGate::x(1));
        graph.insert_gate(QuantumGate::cx(0, 1));

        assert_eq!(graph.n_rows(), 2);
        assert_eq!(graph.gate_id_at(0, 0), Some(0));
        assert_eq!(graph.gate_id_at(0, 1), Some(1));
        assert_eq!(graph.gate_id_at(1, 0), Some(2));
        assert_eq!(graph.gate_id_at(1, 1), Some(2));
    }

    #[test]
    fn from_qasm_circuit_builds_packed_rows() {
        let circuit = Circuit {
            gates: vec![Gate::X(0), Gate::H(1), Gate::CX(0, 1)],
        };

        let graph = CircuitGraph::from_qasm_circuit(&circuit);

        assert_eq!(graph.n_qubits(), 2);
        assert_eq!(graph.n_rows(), 2);
        assert_eq!(graph.gate(0), Some(&QuantumGate::x(0)));
        assert_eq!(graph.gate(1), Some(&QuantumGate::h(1)));
        assert_eq!(graph.gate(2), Some(&QuantumGate::cx(0, 1)));
    }

    #[test]
    fn from_qasm_circuit_handles_unsorted_multi_qubit_operands() {
        let circuit = Circuit {
            gates: vec![Gate::CX(2, 0)],
        };

        let graph = CircuitGraph::from_qasm_circuit(&circuit);
        let gate = graph.gate(0).expect("gate should exist");

        assert_eq!(graph.n_qubits(), 3);
        assert_eq!(gate.qubits(), &[0, 2]);
    }

    #[test]
    fn from_qasm_circuit_converts_parametric_gates() {
        let circuit = Circuit {
            gates: vec![Gate::RZ(Angle::Number(0.25), 1)],
        };

        let graph = CircuitGraph::from_qasm_circuit(&circuit);

        assert_eq!(graph.gate(0), Some(&QuantumGate::rz(0.25, 1)));
    }

    #[test]
    fn from_parsed_qasm_round_trips_into_graph() {
        let circuit =
            parse_qasm("OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];").expect("parse succeeds");

        let graph = CircuitGraph::from_qasm_circuit(&circuit);

        assert_eq!(graph.n_qubits(), 2);
        assert_eq!(graph.n_rows(), 2);
        assert_eq!(graph.gate(0), Some(&QuantumGate::h(0)));
        assert_eq!(graph.gate(1), Some(&QuantumGate::cx(0, 1)));
    }
}

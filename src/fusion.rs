use crate::CircuitGraph;

pub fn apply_size_two_fusion(cg: &mut CircuitGraph) {
    let n_absorbed = absorb_single_qubit_gates(cg);
    if n_absorbed > 0 {
        cg.squeeze();
    }

    let n_two_qubit_fused = fuse_adjacent_two_qubit_gates(cg);
    if n_two_qubit_fused > 0 {
        cg.squeeze();
    }
}

fn absorb_single_qubit_gates(cg: &mut CircuitGraph) -> usize {
    let mut n_fused = 0;
    let n_rows = cg.n_rows();
    let n_qubits = cg.n_qubits();

    for row in 0..n_rows {
        for qubit in 0..n_qubits {
            if absorb_single_qubit_gate(cg, row, qubit) {
                n_fused += 1;
            }
        }
    }

    n_fused
}

fn absorb_single_qubit_gate(cg: &mut CircuitGraph, row: usize, qubit: usize) -> bool {
    let gate_id = match cg.gate_id_at(row, qubit) {
        Some(gate_id) => gate_id,
        None => return false,
    };

    let gate = match cg.gate(gate_id) {
        Some(gate) if gate.n_qubits() == 1 => gate.clone(),
        _ => return false,
    };

    for next_row in row + 1..cg.n_rows() {
        let Some(next_gate_id) = cg.gate_id_at(next_row, qubit) else {
            continue;
        };
        let Some(next_gate) = cg.gate(next_gate_id) else {
            continue;
        };
        let fused = next_gate.matmul(&gate);
        cg.remove_gate_at(row, qubit);
        cg.remove_gate_at(next_row, qubit);
        cg.insert_gate_at_row(next_row, fused);
        return true;
    }

    for prev_row in (0..row).rev() {
        let Some(prev_gate_id) = cg.gate_id_at(prev_row, qubit) else {
            continue;
        };
        let Some(prev_gate) = cg.gate(prev_gate_id) else {
            continue;
        };
        let fused = gate.matmul(prev_gate);
        cg.remove_gate_at(row, qubit);
        cg.remove_gate_at(prev_row, qubit);
        cg.insert_gate_at_row(prev_row, fused);
        return true;
    }

    false
}

fn fuse_adjacent_two_qubit_gates(cg: &mut CircuitGraph) -> usize {
    let mut n_fused = 0;
    if cg.n_rows() < 2 {
        return n_fused;
    }

    let mut row = 0;
    while row + 1 < cg.n_rows() {
        for qubit in 0..cg.n_qubits() {
            if fuse_adjacent_two_qubit_gate(cg, row, qubit) {
                n_fused += 1;
            }
        }
        row += 1;
    }

    n_fused
}

fn fuse_adjacent_two_qubit_gate(cg: &mut CircuitGraph, row: usize, qubit: usize) -> bool {
    let next_row = row + 1;
    let Some(left_gate_id) = cg.gate_id_at(row, qubit) else {
        return false;
    };
    let Some(left_gate) = cg.gate(left_gate_id) else {
        return false;
    };
    if left_gate.n_qubits() != 2 || left_gate.qubits()[0] as usize != qubit {
        return false;
    }

    let Some(right_gate_id) = cg.gate_id_at(next_row, qubit) else {
        return false;
    };
    let Some(right_gate) = cg.gate(right_gate_id) else {
        return false;
    };
    if right_gate.n_qubits() != 2 || right_gate.qubits() != left_gate.qubits() {
        return false;
    }

    let fused = right_gate.matmul(left_gate);
    cg.remove_gate_at(row, qubit);
    cg.remove_gate_at(next_row, qubit);
    cg.insert_gate_at_row(next_row, fused);
    true
}

#[cfg(test)]
mod tests {
    use super::apply_size_two_fusion;
    use crate::{types::QuantumGate, CircuitGraph};

    #[test]
    fn absorbs_single_qubit_gate_into_next_row() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::x(0));
        cg.insert_gate(QuantumGate::cx(0, 1));

        apply_size_two_fusion(&mut cg);

        assert_eq!(cg.n_rows(), 1);
        assert_eq!(
            cg.gate(cg.gate_id_at(0, 0).unwrap()),
            Some(&QuantumGate::cx(0, 1).matmul(&QuantumGate::x(0)))
        );
        assert_eq!(cg.gate_id_at(0, 0), cg.gate_id_at(0, 1));
    }

    #[test]
    fn absorbs_single_qubit_gate_into_previous_row() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::x(0));

        apply_size_two_fusion(&mut cg);

        assert_eq!(cg.n_rows(), 1);
        assert_eq!(
            cg.gate(cg.gate_id_at(0, 0).unwrap()),
            Some(&QuantumGate::x(0).matmul(&QuantumGate::cx(0, 1)))
        );
        assert_eq!(cg.gate_id_at(0, 0), cg.gate_id_at(0, 1));
    }

    #[test]
    fn leaves_isolated_single_qubit_gate_alone() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::x(0));

        apply_size_two_fusion(&mut cg);

        assert_eq!(cg.n_rows(), 1);
        assert_eq!(cg.gate(0), Some(&QuantumGate::x(0)));
    }

    #[test]
    fn fuses_adjacent_two_qubit_gates_on_same_targets() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cz(0, 1));

        apply_size_two_fusion(&mut cg);

        assert_eq!(cg.n_rows(), 1);
        let fused_gate_id = cg.gate_id_at(0, 0).unwrap();
        assert_eq!(cg.gate_id_at(0, 0), cg.gate_id_at(0, 1));
        assert_eq!(
            cg.gate(fused_gate_id),
            Some(&QuantumGate::cz(0, 1).matmul(&QuantumGate::cx(0, 1)))
        );
    }

    #[test]
    fn does_not_fuse_adjacent_two_qubit_gates_on_different_targets() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));

        apply_size_two_fusion(&mut cg);

        assert_eq!(cg.n_rows(), 2);
        assert_eq!(cg.gate(0), Some(&QuantumGate::cx(0, 1)));
        assert_eq!(cg.gate(1), Some(&QuantumGate::cx(1, 2)));
    }
}

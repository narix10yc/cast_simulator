use super::*;
use crate::types::QuantumCircuit;

const TEST_QUBITS: u32 = 6;

/// Build a [`CircuitGraph`] from a flat gate list. The `_measured` parameter
/// is retained for symmetry with trajectory tests but is not stored on the
/// graph — trajectory tests pass measured qubits explicitly via
/// [`TrajectoryOpts`].
fn test_graph(gates: Vec<QuantumGate>) -> CircuitGraph {
    let mut circuit = QuantumCircuit::new(TEST_QUBITS);
    for g in gates {
        circuit.add(g);
    }
    CircuitGraph::from_circuit(&circuit)
}

// ── Statevector / density matrix ─────────────────────────────────────────

#[test]
fn statevector_h_gate() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0)]);
    let state = sim.simulate(&graph, Representation::StateVector).unwrap();
    assert!(state.is_pure());
    let pops = state.populations();
    assert!((pops[0] - 0.5).abs() < 1e-10);
    assert!((pops[1] - 0.5).abs() < 1e-10);
}

#[test]
fn statevector_rejects_noisy_gates() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);
    assert!(sim.simulate(&graph, Representation::StateVector).is_err());
}

#[test]
fn density_matrix_trace_preserved() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![
        QuantumGate::h(0),
        QuantumGate::cx(0, 1),
        QuantumGate::depolarizing(0, 0.05),
    ]);
    let state = sim.simulate(&graph, Representation::DensityMatrix).unwrap();
    assert!(!state.is_pure());
    let trace = state.trace();
    assert!(
        (trace - 1.0).abs() < 1e-10,
        "trace should be 1.0, got {trace}"
    );
}

#[test]
fn from_qasm() {
    let circuit =
        QuantumCircuit::from_qasm("OPENQASM 2.0; qreg q[6]; h q[0]; cx q[0],q[1]; cx q[4],q[5];")
            .unwrap();
    assert_eq!(circuit.n_qubits(), 6);
    assert_eq!(circuit.len(), 3);

    let sim = Simulator::<Cpu>::f64();
    let graph = CircuitGraph::from_circuit(&circuit);
    let state = sim.simulate(&graph, Representation::StateVector).unwrap();
    assert!(state.is_pure());
}

// ── Trajectory / ensemble ────────────────────────────────────────────────
//
// Dead-gate elimination is the caller's responsibility under the new API;
// these tests either have no dead gates or call `eliminate_dead_gates`
// explicitly before building the graph.

fn traj_opts(measured: &[u32], n_samples: u64, max_ensemble: usize) -> TrajectoryOpts {
    TrajectoryOpts {
        measured_qubits: measured.to_vec(),
        n_samples,
        seed: Some(42),
        max_ensemble: Some(max_ensemble),
    }
}

#[test]
fn trajectory_h_gate_histogram() {
    // H(0), no noise. Measure qubit 0. Expect ~50/50.
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0)]);
    let traj = sim
        .sample_trajectory(&graph, &traj_opts(&[0], 10000, 1))
        .unwrap();
    assert_eq!(traj.n_samples, 10000);
    assert_eq!(traj.histogram.values().sum::<u64>(), 10000);
    let count_0 = traj.histogram.get(&0).copied().unwrap_or(0);
    let count_1 = traj.histogram.get(&1).copied().unwrap_or(0);
    assert!(
        (count_0 as f64 - 5000.0).abs() < 500.0,
        "expected ~5000 for |0⟩, got {count_0}"
    );
    assert!(
        (count_1 as f64 - 5000.0).abs() < 500.0,
        "expected ~5000 for |1⟩, got {count_1}"
    );
    assert!((traj.explored_weight - 1.0).abs() < 1e-10);
}

#[test]
fn trajectory_deterministic_branch() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);
    let traj = sim
        .sample_trajectory(&graph, &traj_opts(&[0], 100, 1))
        .unwrap();
    assert_eq!(traj.branches.len(), 1);
    assert_eq!(traj.branches[0].noise_path, vec![0]);
    assert!((traj.branches[0].weight - 0.9).abs() < 1e-10);
    assert!((traj.explored_weight - 0.9).abs() < 1e-10);
}

#[test]
fn ensemble_m4_explores_more_weight() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);

    let t1 = sim
        .sample_trajectory(&graph, &traj_opts(&[0], 1000, 1))
        .unwrap();
    let t4 = sim
        .sample_trajectory(&graph, &traj_opts(&[0], 1000, 4))
        .unwrap();

    assert_eq!(t1.branches.len(), 1);
    assert_eq!(t4.branches.len(), 4);
    assert!(
        t4.explored_weight > t1.explored_weight,
        "M=4 ({:.4}) should explore more than M=1 ({:.4})",
        t4.explored_weight,
        t1.explored_weight,
    );
    assert!(
        (t4.explored_weight - 1.0).abs() < 1e-10,
        "all branches explored: weight should be 1.0, got {}",
        t4.explored_weight,
    );
}

#[test]
fn ensemble_matches_dm_populations() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![
        QuantumGate::h(0),
        QuantumGate::depolarizing(0, 0.1),
        QuantumGate::cx(0, 1),
    ]);

    let dm_pops = sim
        .simulate(&graph, Representation::DensityMatrix)
        .unwrap()
        .populations();

    let opts = TrajectoryOpts {
        measured_qubits: vec![0, 1],
        n_samples: 100_000,
        seed: Some(99),
        max_ensemble: Some(64),
    };
    let traj = sim.sample_trajectory(&graph, &opts).unwrap();

    assert!(
        (traj.explored_weight - 1.0).abs() < 1e-10,
        "expected full weight coverage, got {}",
        traj.explored_weight,
    );

    let total = traj.n_samples as f64;
    for (outcome, &expected_pop) in dm_pops.iter().enumerate() {
        let count = traj.histogram.get(&(outcome as u64)).copied().unwrap_or(0);
        let freq = count as f64 / total;
        assert!(
            (freq - expected_pop).abs() < 0.02,
            "outcome {outcome}: freq={freq:.4}, expected={expected_pop:.4}"
        );
    }
}

#[test]
fn ensemble_multi_noise_pruning() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![
        QuantumGate::h(0),
        QuantumGate::depolarizing(0, 0.1),
        QuantumGate::cx(0, 1),
        QuantumGate::depolarizing(1, 0.1),
    ]);

    let traj = sim
        .sample_trajectory(&graph, &traj_opts(&[0, 1], 10000, 2))
        .unwrap();

    assert_eq!(traj.branches.len(), 2);
    assert!(traj.branches[0].weight >= traj.branches[1].weight);
    assert!(
        traj.explored_weight < 1.0,
        "with pruning, explored_weight should be < 1.0, got {}",
        traj.explored_weight,
    );
    assert!(
        traj.explored_weight > 0.7,
        "explored_weight too low: {}",
        traj.explored_weight,
    );
    assert_eq!(traj.histogram.values().sum::<u64>(), 10000);
}

#[test]
fn ensemble_with_dce_partial_measure() {
    // Explicitly call eliminate_dead_gates to drop the dead H(2)/CX(2,3) that
    // cannot influence the measured qubit 0.
    let mut circuit = QuantumCircuit::new(TEST_QUBITS);
    circuit.add(QuantumGate::h(0));
    circuit.add(QuantumGate::depolarizing(0, 0.1));
    circuit.add(QuantumGate::h(2)); // dead
    circuit.add(QuantumGate::cx(2, 3)); // dead
    circuit.measure(&[0]);
    let circuit = circuit.eliminate_dead_gates();
    let graph = CircuitGraph::from_circuit(&circuit);

    let sim = Simulator::<Cpu>::f64();
    let traj = sim
        .sample_trajectory(&graph, &traj_opts(&[0], 5000, 4))
        .unwrap();

    assert_eq!(traj.branches.len(), 4);
    assert!((traj.explored_weight - 1.0).abs() < 1e-10);

    let total = traj.n_samples as f64;
    let count_0 = traj.histogram.get(&0).copied().unwrap_or(0);
    let count_1 = traj.histogram.get(&1).copied().unwrap_or(0);
    assert!(
        (count_0 as f64 / total - 0.5).abs() < 0.05,
        "expected ~50% for |0⟩, got {:.1}%",
        100.0 * count_0 as f64 / total,
    );
    assert!(
        (count_1 as f64 / total - 0.5).abs() < 0.05,
        "expected ~50% for |1⟩, got {:.1}%",
        100.0 * count_1 as f64 / total,
    );
}

#[test]
fn ensemble_noiseless_circuit() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![QuantumGate::h(0), QuantumGate::cx(0, 1)]);

    let traj = sim
        .sample_trajectory(&graph, &traj_opts(&[0, 1], 1000, 10))
        .unwrap();

    assert_eq!(traj.branches.len(), 1);
    assert!((traj.branches[0].weight - 1.0).abs() < 1e-10);
    assert!((traj.explored_weight - 1.0).abs() < 1e-10);
    assert_eq!(traj.histogram.len(), 2);
    assert!(traj.histogram.contains_key(&0)); // |00⟩
    assert!(traj.histogram.contains_key(&3)); // |11⟩
}

#[test]
fn ensemble_multi_noise_matches_dm() {
    let sim = Simulator::<Cpu>::f64();
    let graph = test_graph(vec![
        QuantumGate::h(0),
        QuantumGate::depolarizing(0, 0.05),
        QuantumGate::cx(0, 1),
        QuantumGate::depolarizing(0, 0.05),
        QuantumGate::depolarizing(1, 0.05),
    ]);

    let dm_pops = sim
        .simulate(&graph, Representation::DensityMatrix)
        .unwrap()
        .populations();

    let opts = TrajectoryOpts {
        measured_qubits: vec![0, 1],
        n_samples: 200_000,
        seed: Some(123),
        max_ensemble: Some(64),
    };
    let traj = sim.sample_trajectory(&graph, &opts).unwrap();

    assert!(
        (traj.explored_weight - 1.0).abs() < 1e-10,
        "all branches should be explored, got {}",
        traj.explored_weight,
    );

    let total = traj.n_samples as f64;
    for (outcome, &expected) in dm_pops.iter().enumerate() {
        let count = traj.histogram.get(&(outcome as u64)).copied().unwrap_or(0);
        let freq = count as f64 / total;
        assert!(
            (freq - expected).abs() < 0.01,
            "outcome {outcome}: freq={freq:.4}, dm={expected:.4}"
        );
    }
}

// ── CUDA ─────────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn cuda_norm_squared_and_normalize() {
    use crate::cuda::{CudaPrecision, CudaStatevector};

    let mut sv = CudaStatevector::new(4, CudaPrecision::F64).unwrap();
    sv.zero().unwrap();
    let ns = sv.norm_squared().unwrap();
    assert!((ns - 1.0).abs() < 1e-12);

    let n = 1usize << 4;
    let mut data = vec![(0.0, 0.0); n];
    data[0] = (3.0, 0.0);
    data[1] = (4.0, 0.0);
    sv.upload(&data).unwrap();
    let ns = sv.norm_squared().unwrap();
    assert!((ns - 25.0).abs() < 1e-10);

    sv.normalize().unwrap();
    let ns = sv.norm_squared().unwrap();
    assert!((ns - 1.0).abs() < 1e-10);
}

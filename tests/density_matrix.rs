//! Small integration tests for density-matrix (noisy) simulation.
//!
//! Each test uses 1–2 physical qubits so the DM statevector is at most 4 virtual
//! qubits (16 amplitudes). Tests verify trace preservation, known analytic
//! populations, and that noisy gates interact correctly with fusion.

use std::sync::Arc;

use cast::{
    cost_model::FusionConfig,
    cpu::{CPUKernelGenSpec, CPUStatevector, CpuKernelManager, MatrixLoadMode, SimdWidth},
    fusion,
    types::{Complex, Precision, QuantumGate},
    CircuitGraph,
};

const TOL: f64 = 1e-10;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cpu_spec() -> CPUKernelGenSpec {
    CPUKernelGenSpec {
        precision: Precision::F64,
        simd_width: SimdWidth::W128,
        mode: MatrixLoadMode::ImmValue,
        ztol: 1e-12,
        otol: 1e-12,
    }
}

/// Applies gates to a DM statevector and returns the result.
/// `n_phys` is the number of physical qubits; the SV has `2*n_phys` virtual qubits.
fn run_dm(gates: &[QuantumGate], n_phys: u32) -> Vec<(f64, f64)> {
    let n_sv = 2 * n_phys;

    // DM initial state: |0…0⟩⟨0…0| → only element 0 is 1.
    let spec = cpu_spec();
    let mgr = CpuKernelManager::new(spec);
    let mut sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    sv.set_amp(0, Complex::new(1.0, 0.0));

    for gate in gates {
        let kid = mgr
            .generate(&Arc::new(gate.clone()))
            .unwrap_or_else(|e| panic!("generate: {e}"));
        mgr.apply(kid, &mut sv, 1)
            .unwrap_or_else(|e| panic!("apply: {e}"));
    }

    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

/// Tr(ρ) from a vectorized DM statevector.
fn dm_trace(sv: &[(f64, f64)], n_phys: u32) -> f64 {
    let n = n_phys as usize;
    let dim = 1usize << n;
    (0..dim).map(|i| sv[i | (i << n)].0).sum()
}

/// Diagonal entries ρ[i,i] from a vectorized DM statevector.
fn dm_diagonal(sv: &[(f64, f64)], n_phys: u32) -> Vec<f64> {
    let n = n_phys as usize;
    let dim = 1usize << n;
    (0..dim).map(|i| sv[i | (i << n)].0).collect()
}

fn to_dm_gates(gates: &[QuantumGate], n_phys: u32) -> Vec<QuantumGate> {
    gates
        .iter()
        .map(|g| g.to_density_matrix_gate(n_phys as usize))
        .collect()
}

fn to_graph(gates: &[QuantumGate]) -> CircuitGraph {
    let mut cg = CircuitGraph::new();
    for g in gates {
        cg.insert_gate(Arc::new(g.clone()));
    }
    cg
}

// ── Depolarizing channel ─────────────────────────────────────────────────────

#[test]
fn depolarizing_on_zero_state_trace_and_populations() {
    // ε(|0⟩⟨0|) with depolarizing(p):
    //   ρ[0,0] = 1 − 2p/3,  ρ[1,1] = 2p/3,  off-diag = 0.
    let p = 0.15;
    let n_phys: u32 = 2; // pad to 2 physical qubits so DM SV has 4 virtual qubits

    let channel_gate = QuantumGate::depolarizing(0, p);
    let dm_gates = to_dm_gates(&[channel_gate], n_phys);
    let result = run_dm(&dm_gates, n_phys);

    let trace = dm_trace(&result, n_phys);
    assert!((trace - 1.0).abs() < TOL, "trace = {trace}");

    let diag = dm_diagonal(&result, n_phys);
    // qubit 0 populations: sum over qubit 1 states
    let p00 = diag[0b00] + diag[0b10]; // qubit0=0
    let p01 = diag[0b01] + diag[0b11]; // qubit0=1
    assert!(
        (p00 - (1.0 - 2.0 * p / 3.0)).abs() < TOL,
        "P(q0=0) = {p00}, expected {}",
        1.0 - 2.0 * p / 3.0
    );
    assert!(
        (p01 - 2.0 * p / 3.0).abs() < TOL,
        "P(q0=1) = {p01}, expected {}",
        2.0 * p / 3.0
    );
}

#[test]
fn depolarizing_p0_is_identity() {
    // p=0 channel should leave |0⟩⟨0| unchanged.
    let n_phys: u32 = 2;
    let dm_gates = to_dm_gates(&[QuantumGate::depolarizing(0, 0.0)], n_phys);
    let result = run_dm(&dm_gates, n_phys);

    let diag = dm_diagonal(&result, n_phys);
    assert!((diag[0] - 1.0).abs() < TOL, "ρ[0,0] = {}", diag[0]);
    for (i, &d) in diag.iter().enumerate().skip(1) {
        assert!(d.abs() < TOL, "ρ[{i},{i}] = {d}");
    }
}

// ── Noiseless DM agrees with pure statevector ────────────────────────────────

#[test]
fn noiseless_dm_diagonal_matches_sv_probabilities() {
    // Apply H(0) to |00⟩.  SV: |00⟩ → (|00⟩+|01⟩)/√2.
    // Probabilities: P(00)=P(01)=0.5.
    // DM diagonal should match.
    let n_phys: u32 = 2;

    // Pure SV simulation.
    let spec = cpu_spec();
    let mgr = CpuKernelManager::new(spec);
    let mut sv = CPUStatevector::new(n_phys, spec.precision, spec.simd_width);
    sv.set_amp(0, Complex::new(1.0, 0.0));
    let kid = mgr.generate(&Arc::new(QuantumGate::h(0))).unwrap();
    mgr.apply(kid, &mut sv, 1).unwrap();
    let probs: Vec<f64> = sv
        .amplitudes()
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();

    // DM simulation.
    let dm_gates = to_dm_gates(&[QuantumGate::h(0)], n_phys);
    let dm_result = run_dm(&dm_gates, n_phys);
    let diag = dm_diagonal(&dm_result, n_phys);

    for (i, (&p_sv, &p_dm)) in probs.iter().zip(diag.iter()).enumerate() {
        assert!(
            (p_sv - p_dm).abs() < TOL,
            "state |{i:0>2b}⟩: sv_prob={p_sv:.6e}, dm_diag={p_dm:.6e}"
        );
    }
}

// ── Fusion + channels ────────────────────────────────────────────────────────

#[test]
fn noisy_gates_fuse_and_preserve_trace() {
    // Noisy gates participate in fusion; the fused circuit must still
    // preserve DM trace and agree with the unfused result.
    let n_phys: u32 = 3;
    let gates: Vec<QuantumGate> = vec![
        QuantumGate::h(0),
        QuantumGate::depolarizing(0, 0.1),
        QuantumGate::cx(0, 1),
        QuantumGate::depolarizing(0, 0.1),
        QuantumGate::depolarizing(1, 0.1),
    ];

    // Fusion should reduce gate count (noisy gates are absorbed).
    let mut cg = to_graph(&gates);
    let config = FusionConfig::size_only(3);
    fusion::optimize(&mut cg, &config);
    let after = cg.gates_in_row_order();
    assert!(
        after.len() < gates.len(),
        "expected fusion to reduce gate count"
    );

    // Unfused DM.
    let dm_unfused = to_dm_gates(&gates, n_phys);
    let result_unfused = run_dm(&dm_unfused, n_phys);
    let trace_unfused = dm_trace(&result_unfused, n_phys);
    assert!(
        (trace_unfused - 1.0).abs() < TOL,
        "unfused trace = {trace_unfused}"
    );

    // Fused DM.
    let fused: Vec<QuantumGate> = after.iter().map(|a| a.as_ref().clone()).collect();
    let dm_fused = to_dm_gates(&fused, n_phys);
    let result_fused = run_dm(&dm_fused, n_phys);
    let trace_fused = dm_trace(&result_fused, n_phys);
    assert!(
        (trace_fused - 1.0).abs() < TOL,
        "fused trace = {trace_fused}"
    );

    // Results must agree.
    for (i, (&(ru, iu), &(rf, ifu))) in result_unfused.iter().zip(result_fused.iter()).enumerate() {
        let diff = ((ru - rf).powi(2) + (iu - ifu).powi(2)).sqrt();
        assert!(
            diff < TOL,
            "dm[{i}]: unfused=({ru:.4e},{iu:.4e}) fused=({rf:.4e},{ifu:.4e}) diff={diff:.2e}"
        );
    }
}

#[test]
fn fused_noisy_circuit_preserves_trace() {
    // After fusion, a noisy circuit must still preserve trace.
    // Use n_phys=3 so the fused 2-qubit DM gate (4 virtual qubits) fits in
    // the 6-qubit SV with room for simd_s=1.
    let n_phys: u32 = 3;
    let gates: Vec<QuantumGate> = vec![
        QuantumGate::h(0),
        QuantumGate::depolarizing(0, 0.1),
        QuantumGate::cx(0, 1),
        QuantumGate::depolarizing(0, 0.05),
        QuantumGate::depolarizing(1, 0.05),
        QuantumGate::h(1),
        QuantumGate::depolarizing(1, 0.1),
    ];

    // Unfused DM result.
    let dm_unfused = to_dm_gates(&gates, n_phys);
    let result_unfused = run_dm(&dm_unfused, n_phys);
    let trace_unfused = dm_trace(&result_unfused, n_phys);
    assert!(
        (trace_unfused - 1.0).abs() < TOL,
        "unfused trace = {trace_unfused}"
    );

    // Fuse the unitary gates, then lift to DM.
    let mut cg = to_graph(&gates);
    fusion::optimize(&mut cg, &FusionConfig::size_only(3));
    let fused: Vec<QuantumGate> = cg
        .gates_in_row_order()
        .into_iter()
        .map(|arc| arc.as_ref().clone())
        .collect();
    let dm_fused = to_dm_gates(&fused, n_phys);
    let result_fused = run_dm(&dm_fused, n_phys);
    let trace_fused = dm_trace(&result_fused, n_phys);
    assert!(
        (trace_fused - 1.0).abs() < TOL,
        "fused trace = {trace_fused}"
    );

    // Both results should agree (fusion only merges unitary gates).
    for (i, (&(ru, iu), &(rf, ifu))) in result_unfused.iter().zip(result_fused.iter()).enumerate() {
        let diff = ((ru - rf).powi(2) + (iu - ifu).powi(2)).sqrt();
        assert!(
            diff < TOL,
            "dm[{i}]: unfused=({ru:.4e},{iu:.4e}) fused=({rf:.4e},{ifu:.4e}) diff={diff:.2e}"
        );
    }
}

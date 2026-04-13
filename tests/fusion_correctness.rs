//! End-to-end simulation correctness tests.
//!
//! Verifies that measurement populations are invariant under:
//! - Fusion mode (no fusion, size-only 2/3/4)
//! - CPU SIMD width (W128, W256, W512)
//! - CPU thread count (1, 2, 4)
//! - Backend (CPU vs CUDA)
//!
//! Population vectors are compared with a per-element absolute tolerance
//! of [`TOL`].  CPU tests run unconditionally; CUDA tests require
//! `--features cuda`.

use cast::cost_model::FusionConfig;
use cast::cpu::{CPUKernelGenSpec, MatrixLoadMode, SimdWidth};
use cast::fusion;
use cast::simulator::{Cpu, Representation, Simulator};
use cast::types::{Precision, QuantumCircuit, QuantumGate};
use cast::CircuitGraph;
use std::f64::consts::PI;

#[cfg(feature = "cuda")]
use cast::simulator::Cuda;

const TOL: f64 = 1e-10;

// ── Circuit builders ────────────────────────────────────────────────────────

/// QFT on `n` qubits: H, controlled-phase rotations, and SWAP reversal.
fn qft_circuit(n: u32) -> QuantumCircuit {
    let mut c = QuantumCircuit::new(n);
    for i in 0..n {
        c.add(QuantumGate::h(i));
        for j in (i + 1)..n {
            let theta = PI / (1u64 << (j - i)) as f64;
            c.add(QuantumGate::cp(theta, j, i));
        }
    }
    for i in 0..n / 2 {
        c.add(QuantumGate::swap(i, n - 1 - i));
    }
    c
}

/// Hardware-efficient ansatz: alternating Ry/Rz rotation layers and CX
/// entangling layers.  `depth` controls the number of entangling layers.
fn hea_circuit(n: u32, depth: usize) -> QuantumCircuit {
    let mut c = QuantumCircuit::new(n);
    let mut angle = 0.3;
    for d in 0..depth {
        for q in 0..n {
            c.add(QuantumGate::ry(angle, q));
            angle += 0.17;
            c.add(QuantumGate::rz(angle, q));
            angle += 0.13;
        }
        let offset = d % 2;
        let mut q = offset as u32;
        while q + 1 < n {
            c.add(QuantumGate::cx(q, q + 1));
            q += 2;
        }
    }
    c
}

/// Brick-wall circuit with Haar-random 2-qubit gates.
fn brick_wall_circuit(n: u32, layers: usize, seed: u64) -> QuantumCircuit {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = QuantumCircuit::new(n);
    for layer in 0..layers {
        let offset = layer % 2;
        let mut q = offset as u32;
        while q + 1 < n {
            c.add(QuantumGate::random_unitary_with_rng(&[q, q + 1], &mut rng));
            q += 2;
        }
    }
    c
}

/// QFT on a computational-basis input state (apply X to selected qubits first).
fn qft_on_input(n: u32, input_bits: u64) -> QuantumCircuit {
    let mut c = QuantumCircuit::new(n);
    for q in 0..n {
        if (input_bits >> q) & 1 == 1 {
            c.add(QuantumGate::x(q));
        }
    }
    for i in 0..n {
        c.add(QuantumGate::h(i));
        for j in (i + 1)..n {
            let theta = PI / (1u64 << (j - i)) as f64;
            c.add(QuantumGate::cp(theta, j, i));
        }
    }
    for i in 0..n / 2 {
        c.add(QuantumGate::swap(i, n - 1 - i));
    }
    c
}

// ── Fusion configs ──────────────────────────────────────────────────────────

const FUSION_LABELS: &[&str] = &["no_fusion", "size2", "size3", "size4"];

fn make_fusion(label: &str) -> Option<FusionConfig> {
    match label {
        "no_fusion" => None,
        "size2" => Some(FusionConfig::size_only(2)),
        "size3" => Some(FusionConfig::size_only(3)),
        "size4" => Some(FusionConfig::size_only(4)),
        _ => unreachable!(),
    }
}

// ── CPU spec builders ───────────────────────────────────────────────────────

fn cpu_spec(simd: SimdWidth) -> CPUKernelGenSpec {
    CPUKernelGenSpec {
        precision: Precision::F64,
        simd_width: simd,
        mode: MatrixLoadMode::ImmValue,
        ztol: 1e-12,
        otol: 1e-12,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn build_graph(circuit: &QuantumCircuit, fusion_cfg: Option<FusionConfig>) -> CircuitGraph {
    let mut graph = CircuitGraph::from_circuit(circuit);
    if let Some(cfg) = fusion_cfg {
        fusion::optimize(&mut graph, &cfg);
    }
    graph
}

fn run_cpu_with(
    circuit: &QuantumCircuit,
    spec: CPUKernelGenSpec,
    n_threads: u32,
    fusion_cfg: Option<FusionConfig>,
) -> Vec<f64> {
    let sim = Simulator::<Cpu>::new(spec).with_threads(n_threads);
    let graph = build_graph(circuit, fusion_cfg);
    sim.simulate(&graph, Representation::StateVector)
        .unwrap()
        .populations()
}

fn run_cpu_populations(circuit: &QuantumCircuit, fusion_cfg: Option<FusionConfig>) -> Vec<f64> {
    run_cpu_with(circuit, CPUKernelGenSpec::f64(), 0, fusion_cfg)
}

#[cfg(feature = "cuda")]
fn run_cuda_populations(circuit: &QuantumCircuit, fusion_cfg: Option<FusionConfig>) -> Vec<f64> {
    let sim = Simulator::<Cuda>::f64();
    let graph = build_graph(circuit, fusion_cfg);
    sim.simulate(&graph, Representation::StateVector)
        .unwrap()
        .populations()
        .unwrap()
}

fn assert_populations_close(a: &[f64], b: &[f64], label_a: &str, label_b: &str, tol: f64) {
    assert_eq!(a.len(), b.len(), "population length mismatch");
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_diff < tol,
        "populations differ: max_diff={max_diff:.2e} (tol={tol:.0e}) between {label_a} and {label_b}"
    );
}

fn assert_sum_one(pops: &[f64], name: &str) {
    let sum: f64 = pops.iter().sum();
    assert!(
        (sum - 1.0).abs() < TOL,
        "{name}: population sum = {sum}, expected 1.0"
    );
}

// ── Fusion invariance ───────────────────────────────────────────────────────

fn check_fusion_invariance_cpu(name: &str, circuit: &QuantumCircuit) {
    let reference = run_cpu_populations(circuit, make_fusion(FUSION_LABELS[0]));
    for &label in &FUSION_LABELS[1..] {
        let pops = run_cpu_populations(circuit, make_fusion(label));
        assert_populations_close(&reference, &pops, FUSION_LABELS[0], label, TOL);
    }
    assert_sum_one(&reference, name);
}

#[cfg(feature = "cuda")]
fn check_fusion_invariance_cuda(name: &str, circuit: &QuantumCircuit) {
    let reference = run_cuda_populations(circuit, make_fusion(FUSION_LABELS[0]));
    for &label in &FUSION_LABELS[1..] {
        let pops = run_cuda_populations(circuit, make_fusion(label));
        assert_populations_close(&reference, &pops, FUSION_LABELS[0], label, TOL);
    }
    assert_sum_one(&reference, name);
}

// ── SIMD + thread invariance ────────────────────────────────────────────────

/// Assert that all combinations of SIMD width and thread count produce
/// the same populations for the given circuit.
fn check_simd_thread_invariance(name: &str, circuit: &QuantumCircuit) {
    let simds = [SimdWidth::W128, SimdWidth::W256, SimdWidth::W512];
    let threads = [1u32, 2, 4];

    let reference = run_cpu_with(circuit, cpu_spec(simds[0]), threads[0], None);
    assert_sum_one(&reference, name);

    for &simd in &simds {
        for &nt in &threads {
            if simd == simds[0] && nt == threads[0] {
                continue; // skip reference
            }
            let label = format!("{simd:?}_t{nt}");
            let pops = run_cpu_with(circuit, cpu_spec(simd), nt, None);
            assert_populations_close(
                &reference,
                &pops,
                &format!("{:?}_t{}", simds[0], threads[0]),
                &label,
                TOL,
            );
        }
    }
}

// ── CPU tests: fusion invariance ────────────────────────────────────────────

#[test]
fn cpu_qft_12q() {
    check_fusion_invariance_cpu("qft_12q", &qft_circuit(12));
}

#[test]
fn cpu_qft_14q() {
    check_fusion_invariance_cpu("qft_14q", &qft_circuit(14));
}

#[test]
fn cpu_qft_on_input_14q() {
    check_fusion_invariance_cpu("qft_on_input_14q", &qft_on_input(14, 0b10_1011_0011_0101));
}

#[test]
fn cpu_hea_12q_depth8() {
    check_fusion_invariance_cpu("hea_12q_d8", &hea_circuit(12, 8));
}

#[test]
fn cpu_hea_16q_depth6() {
    check_fusion_invariance_cpu("hea_16q_d6", &hea_circuit(16, 6));
}

#[test]
fn cpu_brick_wall_12q() {
    check_fusion_invariance_cpu("brick_wall_12q", &brick_wall_circuit(12, 10, 42));
}

#[test]
fn cpu_brick_wall_14q() {
    check_fusion_invariance_cpu("brick_wall_14q", &brick_wall_circuit(14, 8, 99));
}

// ── CPU tests: SIMD width + thread count invariance ─────────────────────────

#[test]
fn cpu_simd_threads_qft_12q() {
    check_simd_thread_invariance("qft_12q", &qft_circuit(12));
}

#[test]
fn cpu_simd_threads_hea_12q() {
    check_simd_thread_invariance("hea_12q", &hea_circuit(12, 8));
}

#[test]
fn cpu_simd_threads_brick_wall_12q() {
    check_simd_thread_invariance("brick_wall_12q", &brick_wall_circuit(12, 10, 42));
}

// ── CUDA tests: fusion invariance ───────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn cuda_qft_12q() {
    check_fusion_invariance_cuda("qft_12q", &qft_circuit(12));
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_hea_12q_depth8() {
    check_fusion_invariance_cuda("hea_12q_d8", &hea_circuit(12, 8));
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_brick_wall_12q() {
    check_fusion_invariance_cuda("brick_wall_12q", &brick_wall_circuit(12, 10, 42));
}

// ── CPU vs CUDA cross-check ─────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn cross_qft_14q() {
    let cpu = run_cpu_populations(&qft_circuit(14), None);
    let cuda = run_cuda_populations(&qft_circuit(14), None);
    assert_populations_close(&cpu, &cuda, "cpu", "cuda", TOL);
}

#[test]
#[cfg(feature = "cuda")]
fn cross_hea_14q_depth6() {
    let cpu = run_cpu_populations(&hea_circuit(14, 6), None);
    let cuda = run_cuda_populations(&hea_circuit(14, 6), None);
    assert_populations_close(&cpu, &cuda, "cpu", "cuda", TOL);
}

#[test]
#[cfg(feature = "cuda")]
fn cross_brick_wall_14q() {
    let c = brick_wall_circuit(14, 8, 99);
    let cpu = run_cpu_populations(&c, None);
    let cuda = run_cuda_populations(&c, None);
    assert_populations_close(&cpu, &cuda, "cpu", "cuda", TOL);
}

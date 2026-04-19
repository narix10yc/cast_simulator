//! Execution-order correctness tests for the CUDA LRU module cache.
//!
//! Each test runs a random intermediate-scale circuit through the CPU backend
//! (reference) and the CUDA backend (with and without gate fusion), then asserts
//! that all results agree to within [`TOLERANCE`].
//!
//! Because `LRU_SIZE = 2` and every Haar-random gate produces a unique kernel,
//! a 60- or 100-gate unfused circuit forces ~58 / ~98 module evictions.  The
//! repeating-3-kernel test deliberately cycles through three distinct kernels so
//! that every apply call hits a different LRU case (hit, miss-with-empty-slot,
//! miss-with-eviction).
//!
//! Run with:
//! ```sh
//! LLVM_CONFIG=.../llvm-config cargo test --features cuda --test circuit_execution_order //! ```

#![cfg(feature = "cuda")]

use std::sync::Arc;

use rand::{rngs::StdRng, Rng, SeedableRng};

use cast::{
    cost_model::FusionConfig,
    cpu::{CPUKernelGenSpec, CPUStatevector, CpuKernelManager, MatrixLoadMode, SimdWidth},
    cuda::{device_sm, CudaKernelGenSpec, CudaKernelManager, CudaPrecision, CudaStatevector},
    fusion,
    types::{Complex, ComplexSquareMatrix, Precision, QuantumGate},
    CircuitGraph,
};

// ── Configuration ─────────────────────────────────────────────────────────────

const TOLERANCE: f64 = 1e-9;

fn cpu_spec() -> CPUKernelGenSpec {
    CPUKernelGenSpec {
        precision: Precision::F64,
        simd_width: SimdWidth::W128,
        mode: MatrixLoadMode::ImmValue,
        ztol: 1e-12,
        otol: 1e-12,
    }
}

fn cuda_spec() -> CudaKernelGenSpec {
    let (sm_major, sm_minor) = device_sm().expect("query device SM");
    CudaKernelGenSpec {
        precision: CudaPrecision::F64,
        ztol: 1e-12,
        otol: 1e-12,
        sm_major,
        sm_minor,
        maxnreg: 128,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Deterministic, normalised initial statevector.
fn seeded_state(n_qubits: usize) -> Vec<(f64, f64)> {
    let len = 1usize << n_qubits;
    let mut amps: Vec<(f64, f64)> = (0..len)
        .map(|i| ((i as f64) + 1.0, (i as f64) * 0.5 - 0.25))
        .collect();
    let norm = amps
        .iter()
        .map(|(re, im)| re * re + im * im)
        .sum::<f64>()
        .sqrt();
    for (re, im) in &mut amps {
        *re /= norm;
        *im /= norm;
    }
    amps
}

/// Random circuit with 50/50 mix of 1- and 2-qubit Haar-random gates.
fn random_circuit(n_qubits: u32, n_gates: usize, seed: u64) -> CircuitGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cg = CircuitGraph::new();
    for _ in 0..n_gates {
        let k = if rng.gen_bool(0.5) { 1_u32 } else { 2 }.min(n_qubits);
        let mut pool: Vec<u32> = (0..n_qubits).collect();
        let mut targets: Vec<u32> = (0..k)
            .map(|_| {
                let i = rng.gen_range(0..pool.len());
                pool.remove(i)
            })
            .collect();
        targets.sort();
        let matrix = ComplexSquareMatrix::random_unitary_with_rng(1 << k, &mut rng);
        cg.insert_gate(QuantumGate::new(matrix, targets));
    }
    cg
}

/// Extract gates in row-major order, deduplicating multi-qubit gates.
fn ordered_gates(cg: &CircuitGraph) -> Vec<Arc<QuantumGate>> {
    cg.gates_in_row_order()
}

/// Apply `gates` sequentially on the CPU JIT backend.
fn run_cpu(gates: &[Arc<QuantumGate>], n_qubits: u32, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let spec = cpu_spec();
    let mgr = CpuKernelManager::new();
    let kids: Vec<_> = gates
        .iter()
        .map(|g| mgr.generate_gate(spec, g).expect("cpu: generate"))
        .collect();

    let mut sv = CPUStatevector::new(n_qubits, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        sv.set_amp(i, Complex::new(re, im));
    }
    for kid in kids {
        mgr.apply(kid, &mut sv, 1).expect("cpu: apply");
    }
    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

/// Apply `gates` via the CUDA manager (all queued then synced in one shot).
fn run_cuda(gates: &[Arc<QuantumGate>], n_qubits: u32, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let spec = cuda_spec();
    let mgr = CudaKernelManager::new(spec);
    let kids: Vec<_> = gates
        .iter()
        .map(|g| mgr.generate(g).expect("cuda: generate"))
        .collect();

    let mut sv = CudaStatevector::new(n_qubits, CudaPrecision::F64).expect("cuda: alloc");
    sv.upload(init).expect("cuda: upload");
    for kid in &kids {
        mgr.apply(*kid, &mut sv).expect("cuda: apply");
    }
    mgr.sync().expect("cuda: sync");
    sv.download().expect("cuda: download")
}

fn assert_amps_close(a: &[(f64, f64)], b: &[(f64, f64)], label: &str, tol: f64) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&(ar, ai), &(br, bi))) in a.iter().zip(b).enumerate() {
        let diff = ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt();
        assert!(
            diff < tol,
            "{label}: amp[{i}] a=({ar:.4e},{ai:.4e}) b=({br:.4e},{bi:.4e}) |Δ|={diff:.2e} > {tol:.2e}",
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 12-qubit, 60-gate circuit. All gates are unique Haar-random → ~58 LRU evictions.
/// CPU vs unfused CUDA.
#[test]
fn lru_stress_12q_60g_unfused() {
    let n = 12_u32;
    let cg = random_circuit(n, 60, 42);
    let gates = ordered_gates(&cg);
    let init = seeded_state(n as usize);

    let cpu = run_cpu(&gates, n, &init);
    let cuda = run_cuda(&gates, n, &init);
    assert_amps_close(&cpu, &cuda, "12q/60g unfused", TOLERANCE);
}

/// Same circuit as above but with gate fusion (SizeOnlyCostModel, max_size=3).
/// Fused result must match the unfused CPU reference.
#[test]
fn lru_stress_12q_60g_fused_size3() {
    let n = 12_u32;
    let cg = random_circuit(n, 60, 42);
    let init = seeded_state(n as usize);

    let cpu = run_cpu(&ordered_gates(&cg), n, &init);

    let mut fused = cg.clone();
    fusion::optimize(&mut fused, &FusionConfig::size_only(3));
    let cuda_fused = run_cuda(&ordered_gates(&fused), n, &init);

    assert_amps_close(&cpu, &cuda_fused, "12q/60g fused(3) vs CPU", TOLERANCE);
}

/// 14-qubit, 80-gate circuit: unfused CUDA vs fused CUDA (max_size=3).
/// Tests that fusion preserves the circuit unitary end-to-end on the GPU.
#[test]
fn lru_stress_14q_80g_unfused_vs_fused() {
    let n = 14_u32;
    let cg = random_circuit(n, 80, 7);
    let init = seeded_state(n as usize);

    let cuda_unfused = run_cuda(&ordered_gates(&cg), n, &init);

    let mut fused = cg.clone();
    fusion::optimize(&mut fused, &FusionConfig::size_only(3));
    let cuda_fused = run_cuda(&ordered_gates(&fused), n, &init);

    assert_amps_close(
        &cuda_unfused,
        &cuda_fused,
        "14q/80g unfused vs fused",
        TOLERANCE,
    );
}

/// 16-qubit, 100-gate circuit: three-way comparison CPU / unfused CUDA / fused CUDA.
/// Maximum LRU pressure (~98 evictions for unfused).
#[test]
fn lru_stress_16q_100g_three_way() {
    let n = 16_u32;
    let cg = random_circuit(n, 100, 99);
    let init = seeded_state(n as usize);
    let unfused = ordered_gates(&cg);

    let cpu = run_cpu(&unfused, n, &init);
    let cuda_unfused = run_cuda(&unfused, n, &init);

    let mut fused_cg = cg.clone();
    fusion::optimize(&mut fused_cg, &FusionConfig::size_only(3));
    let cuda_fused = run_cuda(&ordered_gates(&fused_cg), n, &init);

    assert_amps_close(&cpu, &cuda_unfused, "16q/100g CPU vs unfused", TOLERANCE);
    assert_amps_close(&cpu, &cuda_fused, "16q/100g CPU vs fused(3)", TOLERANCE);
}

/// Repeating 3-kernel pattern on a 12-qubit statevector.
///
/// Three distinct Haar-random 1-qubit gates are applied in round-robin order
/// for 30 rounds (90 total apply calls).  With LRU_SIZE=2:
///  - Round 1: gate A → load into slot 0 (empty)
///  - Round 2: gate B → load into slot 1 (empty)
///  - Round 3: gate C → evict slot 0 (A, LRU), load C
///  - Round 4: gate A → evict slot 1 (B, LRU), reload A
///  - … alternating evictions every step thereafter.
///
/// Verifies that repeated eviction and reload does not perturb execution order.
#[test]
fn lru_repeating_3kernel_pattern_12q() {
    let n = 12_u32;
    let mut rng = StdRng::seed_from_u64(1337);
    // Three 1-qubit gates, each on a distinct qubit so they commute spatially.
    let gates: Vec<Arc<QuantumGate>> = (0..3_u32)
        .map(|q| {
            Arc::new(QuantumGate::new(
                ComplexSquareMatrix::random_unitary_with_rng(2, &mut rng),
                vec![q],
            ))
        })
        .collect();

    // Sequence: [0, 1, 2] × 30 = 90 gate applications.
    let seq: Vec<usize> = (0..30).flat_map(|_| 0..3_usize).collect();

    let init = seeded_state(n as usize);

    // ── CPU reference ─────────────────────────────────────────────────────────
    let spec = cpu_spec();
    let cpu_mgr = CpuKernelManager::new();
    let cpu_kids: Vec<_> = gates
        .iter()
        .map(|g| cpu_mgr.generate_gate(spec, g).unwrap())
        .collect();
    let mut cpu_sv = CPUStatevector::new(n, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    for &idx in &seq {
        cpu_mgr.apply(cpu_kids[idx], &mut cpu_sv, 1).unwrap();
    }
    let cpu: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // ── CUDA ──────────────────────────────────────────────────────────────────
    let spec = cuda_spec();
    let mgr = CudaKernelManager::new(spec);
    let cuda_kids: Vec<_> = gates.iter().map(|g| mgr.generate(g).unwrap()).collect();
    let mut sv = CudaStatevector::new(n, CudaPrecision::F64).unwrap();
    sv.upload(&init).unwrap();
    for &idx in &seq {
        mgr.apply(cuda_kids[idx], &mut sv).unwrap();
    }
    mgr.sync().unwrap();
    let cuda = sv.download().unwrap();

    assert_amps_close(&cpu, &cuda, "repeating-3-kernel/12q", TOLERANCE);
}

/// Verifies that splitting apply calls across two syncs gives the same result
/// as a single sync — i.e., the LRU cache is preserved across sync boundaries.
#[test]
fn lru_split_sync_12q() {
    let n = 12_u32;
    let cg = random_circuit(n, 40, 555);
    let gates = ordered_gates(&cg);
    let mid = gates.len() / 2;
    let init = seeded_state(n as usize);

    let cpu = run_cpu(&gates, n, &init);

    // CUDA: first half, sync, second half, sync.
    let spec = cuda_spec();
    let mgr = CudaKernelManager::new(spec);
    let kids: Vec<_> = gates
        .iter()
        .map(|g| mgr.generate(g).expect("cuda: generate"))
        .collect();
    let mut sv = CudaStatevector::new(n, CudaPrecision::F64).unwrap();
    sv.upload(&init).unwrap();

    for kid in &kids[..mid] {
        mgr.apply(*kid, &mut sv).unwrap();
    }
    mgr.sync().unwrap();

    for kid in &kids[mid..] {
        mgr.apply(*kid, &mut sv).unwrap();
    }
    mgr.sync().unwrap();

    let cuda = sv.download().unwrap();
    assert_amps_close(&cpu, &cuda, "split-sync/12q", TOLERANCE);
}

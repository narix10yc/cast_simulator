//! End-to-end CPU vs CUDA simulation comparison tests.
//!
//! Each test initialises both backends from the same statevector, applies the
//! same gate, and asserts that every amplitude agrees to within [`TOLERANCE`].
//!
//! All tests are `#[ignore]` because they require a CUDA-capable GPU (≥ sm_80).
//! Override the target SM by setting `CUDA_SM=<xy>` in the environment before
//! running (e.g. `CUDA_SM=75` for sm_75).
//!
//! Run with:
//! ```sh
//! LLVM_CONFIG=.../llvm-config cargo test --features cuda --test cpu_cuda_compare -- --ignored
//! ```

#![cfg(feature = "cuda")]

use cast::{
    cpu::{CPUKernelGenSpec, CPUKernelGenerator, CPUStatevector, MatrixLoadMode, Precision,
          SimdWidth},
    cuda::{CudaExecSession, CudaKernelGenSpec, CudaKernelGenerator, CudaPrecision,
           CudaStatevector},
    types::{Complex, ComplexSquareMatrix, QuantumGate},
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Maximum amplitude-wise L2 distance tolerated between CPU and CUDA results.
const TOLERANCE: f64 = 1e-10;

/// Returns the CUDA SM target to use, read from `CUDA_SM` env var (e.g. "80")
/// or falling back to sm_80.
fn sm() -> (u32, u32) {
    if let Ok(v) = std::env::var("CUDA_SM") {
        let s = v.trim();
        if s.len() >= 2 {
            let major: u32 = s[..s.len() - 1].parse().unwrap_or(8);
            let minor: u32 = s[s.len() - 1..].parse().unwrap_or(0);
            return (major, minor);
        }
    }
    (8, 0)
}

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
    let (sm_major, sm_minor) = sm();
    CudaKernelGenSpec {
        precision: CudaPrecision::F64,
        ztol: 1e-12,
        otol: 1e-12,
        sm_major,
        sm_minor,
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Builds a deterministic, normalised statevector of 2^n_qubits amplitudes.
///
/// Uses a simple formula so the state is non-trivial (non-zero in every
/// component) and reproducible: `amp[i] = (i+1) + i*0.5i`, then normalised.
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

/// Applies `gate` to an n-qubit statevector via the CPU JIT backend and returns
/// all amplitudes as `(re, im)` pairs.
fn apply_cpu(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros() as usize;
    let spec = cpu_spec();

    let mut gen = CPUKernelGenerator::new().expect("cpu: create generator");
    let kid = gen
        .generate(&spec, gate.matrix().data(), gate.qubits())
        .expect("cpu: generate kernel");
    let mut jit = gen.init_jit().expect("cpu: init_jit");

    let mut sv = CPUStatevector::new(n_sv_qubits, spec.precision, spec.simd_width);
    for (idx, &(re, im)) in init.iter().enumerate() {
        sv.set_amp(idx, Complex::new(re, im));
    }

    jit.apply(kid, &mut sv, Some(1)).expect("cpu: apply");

    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

/// Applies `gate` to a CUDA device statevector initialised from `init` and
/// returns all amplitudes downloaded to the host.
fn apply_cuda(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros() as usize;
    let spec = cuda_spec();

    let ffi_matrix: Vec<(f64, f64)> = gate
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();

    let mut gen = CudaKernelGenerator::new().expect("cuda: create generator");
    let kid = gen
        .generate(&spec, &ffi_matrix, gate.qubits())
        .expect("cuda: generate kernel");
    let session = gen.compile().expect("cuda: compile");
    let exec = CudaExecSession::new(&session).expect("cuda: create exec session");

    let mut sv =
        CudaStatevector::new(n_sv_qubits as u32, CudaPrecision::F64)
            .expect("cuda: alloc statevector");
    sv.upload(init).expect("cuda: upload");
    exec.apply(kid, &mut sv).expect("cuda: apply");
    sv.download().expect("cuda: download")
}

/// Panics if any corresponding amplitude pair differs by more than `tol`.
fn assert_amps_close(cpu: &[(f64, f64)], cuda: &[(f64, f64)], label: &str, tol: f64) {
    assert_eq!(
        cpu.len(),
        cuda.len(),
        "{label}: CPU returned {} amps, CUDA returned {}",
        cpu.len(),
        cuda.len()
    );
    let mut max_diff = 0.0_f64;
    for (i, (&(cr, ci), &(gr, gi))) in cpu.iter().zip(cuda.iter()).enumerate() {
        let diff = ((cr - gr).powi(2) + (ci - gi).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < tol,
            "{label}: amp[{i}] mismatch — cpu=({cr:.6e},{ci:.6e}) cuda=({gr:.6e},{gi:.6e}) \
             |Δ|={diff:.2e} > tol={tol:.2e}",
        );
    }
    // Informational: uncomment to see max error per test:
    // eprintln!("{label}: max |Δ| = {max_diff:.2e}");
    let _ = max_diff;
}

/// Convenience wrapper: builds the gate, runs both backends, checks closeness.
fn compare(gate: QuantumGate, n_sv_qubits: usize, label: &str) {
    let init = seeded_state(n_sv_qubits);
    let cpu_result = apply_cpu(&gate, &init);
    let cuda_result = apply_cuda(&gate, &init);
    assert_amps_close(&cpu_result, &cuda_result, label, TOLERANCE);
}

// ── Single-qubit gates ────────────────────────────────────────────────────────

#[test]
#[ignore = "requires CUDA device"]
fn compare_h_gate_1qubit_sv() {
    // The CPU F64/W128 layout requires n_sv ≥ n_gate + simd_s (= 1), so the
    // minimum statevector for a 1-qubit gate is 2 qubits.
    compare(QuantumGate::h(0), 2, "H on 2-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_x_gate_1qubit_sv() {
    compare(QuantumGate::x(0), 2, "X on 2-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_h_gate_on_larger_sv() {
    // H on qubit 0 of an 8-qubit statevector.
    compare(QuantumGate::h(0), 8, "H on 8-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_h_gate_non_zero_qubit() {
    // H targeting qubit 3 of a 6-qubit statevector.
    compare(QuantumGate::h(3), 6, "H[q3] on 6-qubit SV");
}

/// T gate: [[1,0],[0,e^{iπ/4}]] — exercises complex (non-real) matrix entries.
#[test]
#[ignore = "requires CUDA device"]
fn compare_t_gate() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let t_matrix = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(s, s), // e^{iπ/4}
    ];
    let gate = QuantumGate::new(
        ComplexSquareMatrix::from_vec(2, t_matrix),
        vec![0],
    );
    compare(gate, 5, "T gate");
}

/// Rx(θ) gate — arbitrary rotation, exercises non-trivial ImmValue constants.
#[test]
#[ignore = "requires CUDA device"]
fn compare_rx_gate() {
    compare(QuantumGate::rx(std::f64::consts::PI / 3.0, 0), 5, "Rx(π/3)");
}

/// Rz(θ) gate — diagonal matrix.
#[test]
#[ignore = "requires CUDA device"]
fn compare_rz_gate() {
    compare(QuantumGate::rz(std::f64::consts::PI / 5.0, 0), 5, "Rz(π/5)");
}

// ── Two-qubit gates ───────────────────────────────────────────────────────────

#[test]
#[ignore = "requires CUDA device"]
fn compare_cx_adjacent_qubits() {
    compare(QuantumGate::cx(0, 1), 5, "CX(0,1) on 5-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_cx_non_adjacent_qubits() {
    // Control on qubit 0, target on qubit 4 — exercises non-contiguous bit scatter.
    compare(QuantumGate::cx(0, 4), 6, "CX(0,4) on 6-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_cx_reversed_qubit_order() {
    compare(QuantumGate::cx(3, 1), 5, "CX(3,1) on 5-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_haar_random_2qubit() {
    let gate = QuantumGate::new(
        ComplexSquareMatrix::random_unitary(4),
        vec![0, 1],
    );
    compare(gate, 5, "Haar-2q on 5-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_haar_random_2qubit_non_adjacent() {
    let gate = QuantumGate::new(
        ComplexSquareMatrix::random_unitary(4),
        vec![1, 4],
    );
    compare(gate, 6, "Haar-2q[1,4] on 6-qubit SV");
}

// ── Three-qubit gates ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires CUDA device"]
fn compare_ccx_gate() {
    compare(QuantumGate::ccx(0, 1, 2), 5, "CCX on 5-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_ccx_non_adjacent() {
    compare(QuantumGate::ccx(0, 2, 5), 6, "CCX(0,2,5) on 6-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_haar_random_3qubit() {
    let gate = QuantumGate::new(
        ComplexSquareMatrix::random_unitary(8),
        vec![0, 1, 2],
    );
    compare(gate, 5, "Haar-3q on 5-qubit SV");
}

// ── Larger statevectors ───────────────────────────────────────────────────────

#[test]
#[ignore = "requires CUDA device"]
fn compare_h_gate_20qubit_sv() {
    // 2^20 = 1 M amplitudes.  The persistent-grid loop ensures the CUDA kernel
    // processes all of them even with a capped grid dimension.
    compare(QuantumGate::h(0), 20, "H on 20-qubit SV");
}

#[test]
#[ignore = "requires CUDA device"]
fn compare_cx_gate_20qubit_sv() {
    compare(QuantumGate::cx(0, 19), 20, "CX(0,19) on 20-qubit SV");
}

// ── Sequential gates (statefulness check) ────────────────────────────────────

/// Apply two gates in sequence on both backends and compare the final state.
/// This verifies that state is correctly updated in-place between applies.
#[test]
#[ignore = "requires CUDA device"]
fn compare_sequential_h_then_x() {
    let n_sv = 4;
    let init = seeded_state(n_sv);

    // CPU: H then X on qubit 0.
    let h_spec = cpu_spec();
    let mut cpu_gen = CPUKernelGenerator::new().unwrap();
    let h_cpu = cpu_gen
        .generate(&h_spec, QuantumGate::h(0).matrix().data(), &[0])
        .unwrap();
    let x_cpu = cpu_gen
        .generate(&h_spec, QuantumGate::x(0).matrix().data(), &[0])
        .unwrap();
    let mut jit = cpu_gen.init_jit().unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, h_spec.precision, h_spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    jit.apply(h_cpu, &mut cpu_sv, Some(1)).unwrap();
    jit.apply(x_cpu, &mut cpu_sv, Some(1)).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect();

    // CUDA: H then X on qubit 0.
    let c_spec = cuda_spec();
    let h_mat: Vec<(f64, f64)> = QuantumGate::h(0).matrix().data().iter().map(|c| (c.re, c.im)).collect();
    let x_mat: Vec<(f64, f64)> = QuantumGate::x(0).matrix().data().iter().map(|c| (c.re, c.im)).collect();
    let mut cuda_gen = CudaKernelGenerator::new().unwrap();
    let h_kid = cuda_gen.generate(&c_spec, &h_mat, &[0]).unwrap();
    let x_kid = cuda_gen.generate(&c_spec, &x_mat, &[0]).unwrap();
    let session = cuda_gen.compile().unwrap();
    let exec = CudaExecSession::new(&session).unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv as u32, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    exec.apply(h_kid, &mut cuda_sv).unwrap();
    exec.apply(x_kid, &mut cuda_sv).unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    assert_amps_close(&cpu_result, &cuda_result, "H then X sequential", TOLERANCE);
}

/// Verify that H applied twice is the identity: CPU and CUDA should both return
/// a state within TOLERANCE of the original.
#[test]
#[ignore = "requires CUDA device"]
fn compare_h_squared_is_identity() {
    let n_sv = 3;
    let init = seeded_state(n_sv);

    // CPU: H^2
    let spec = cpu_spec();
    let mut cpu_gen = CPUKernelGenerator::new().unwrap();
    let kid = cpu_gen
        .generate(&spec, QuantumGate::h(0).matrix().data(), &[0])
        .unwrap();
    let mut jit = cpu_gen.init_jit().unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    jit.apply(kid, &mut cpu_sv, Some(1)).unwrap();
    jit.apply(kid, &mut cpu_sv, Some(1)).unwrap();
    let cpu_result: Vec<(f64, f64)> =
        cpu_sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect();

    // CUDA: H^2
    let c_spec = cuda_spec();
    let h_mat: Vec<(f64, f64)> = QuantumGate::h(0)
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let mut cuda_gen = CudaKernelGenerator::new().unwrap();
    let kid = cuda_gen.generate(&c_spec, &h_mat, &[0]).unwrap();
    let session = cuda_gen.compile().unwrap();
    let exec = CudaExecSession::new(&session).unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv as u32, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    exec.apply(kid, &mut cuda_sv).unwrap();
    exec.apply(kid, &mut cuda_sv).unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    // Both should agree with each other.
    assert_amps_close(&cpu_result, &cuda_result, "H^2 cpu vs cuda", TOLERANCE);

    // And both should agree with the original state (H is its own inverse).
    assert_amps_close(&init, &cpu_result, "H^2 ≈ identity (CPU)", TOLERANCE);
    assert_amps_close(&init, &cuda_result, "H^2 ≈ identity (CUDA)", TOLERANCE);
}

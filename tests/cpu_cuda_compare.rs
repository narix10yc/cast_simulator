//! End-to-end CPU vs CUDA simulation comparison tests.
//!
//! Each test initialises both backends from the same statevector, applies the
//! same gate, and asserts that every amplitude agrees to within [`TOLERANCE`].
//!
//! All tests require a CUDA-capable GPU and run when `--features cuda` is set.
//! The target SM is detected automatically via [`cast::cuda::query_device_sm`].
//!
//! Run with:
//! ```sh
//! LLVM_CONFIG=.../llvm-config cargo test --features cuda --test cpu_cuda_compare -- --ignored
//! ```

#![cfg(feature = "cuda")]

use cast::{
    cpu::{CPUKernelGenSpec, CPUKernelGenerator, CPUStatevector, MatrixLoadMode, SimdWidth},
    cuda::{
        query_device_sm, CudaJitSession, CudaKernelGenSpec, CudaKernelGenerator, CudaPrecision,
        CudaStatevector,
    },
    types::{Complex, ComplexSquareMatrix, Precision, QuantumGate},
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Maximum amplitude-wise L2 distance tolerated between CPU and CUDA results.
const TOLERANCE: f64 = 1e-10;

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
    let (sm_major, sm_minor) = query_device_sm().expect("query device SM");
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
    let n_sv_qubits = init.len().trailing_zeros();
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

    jit.apply(kid, &mut sv, 1).expect("cpu: apply");

    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

/// Applies `gate` to a CUDA device statevector initialised from `init` and
/// returns all amplitudes downloaded to the host.
fn apply_cuda(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let spec = cuda_spec();

    let ffi_matrix: Vec<(f64, f64)> = gate.matrix().data().iter().map(|c| (c.re, c.im)).collect();

    let mut gen = CudaKernelGenerator::new().expect("cuda: create generator");
    let kid = gen
        .generate(&spec, &ffi_matrix, gate.qubits())
        .expect("cuda: generate kernel");
    let session = gen.compile().expect("cuda: compile");
    let exec = CudaJitSession::new(&session).expect("cuda: create jit session");

    let mut sv = CudaStatevector::new(n_sv_qubits as u32, CudaPrecision::F64)
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
fn compare(gate: QuantumGate, n_sv_qubits: u32, label: &str) {
    let init = seeded_state(n_sv_qubits as usize);
    let cpu_result = apply_cpu(&gate, &init);
    let cuda_result = apply_cuda(&gate, &init);
    assert_amps_close(&cpu_result, &cuda_result, label, TOLERANCE);
}

// ── Single-qubit gates ────────────────────────────────────────────────────────

#[test]
fn compare_h_gate_1qubit_sv() {
    // The CPU F64/W128 layout requires n_sv ≥ n_gate + simd_s (= 1), so the
    // minimum statevector for a 1-qubit gate is 2 qubits.
    compare(QuantumGate::h(0), 2, "H on 2-qubit SV");
}

#[test]
fn compare_x_gate_1qubit_sv() {
    compare(QuantumGate::x(0), 2, "X on 2-qubit SV");
}

#[test]
fn compare_h_gate_on_larger_sv() {
    // H on qubit 0 of an 8-qubit statevector.
    compare(QuantumGate::h(0), 8, "H on 8-qubit SV");
}

#[test]
fn compare_h_gate_non_zero_qubit() {
    // H targeting qubit 3 of a 6-qubit statevector.
    compare(QuantumGate::h(3), 6, "H[q3] on 6-qubit SV");
}

/// T gate: [[1,0],[0,e^{iπ/4}]] — exercises complex (non-real) matrix entries.
#[test]
fn compare_t_gate() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let t_matrix = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(s, s), // e^{iπ/4}
    ];
    let gate = QuantumGate::new(ComplexSquareMatrix::from_vec(2, t_matrix), vec![0]);
    compare(gate, 5, "T gate");
}

/// Rx(θ) gate — arbitrary rotation, exercises non-trivial ImmValue constants.
#[test]
fn compare_rx_gate() {
    compare(QuantumGate::rx(std::f64::consts::PI / 3.0, 0), 5, "Rx(π/3)");
}

/// Rz(θ) gate — diagonal matrix.
#[test]
fn compare_rz_gate() {
    compare(QuantumGate::rz(std::f64::consts::PI / 5.0, 0), 5, "Rz(π/5)");
}

// ── Two-qubit gates ───────────────────────────────────────────────────────────

#[test]
fn compare_cx_adjacent_qubits() {
    compare(QuantumGate::cx(0, 1), 5, "CX(0,1) on 5-qubit SV");
}

#[test]
fn compare_cx_non_adjacent_qubits() {
    // Control on qubit 0, target on qubit 4 — exercises non-contiguous bit scatter.
    compare(QuantumGate::cx(0, 4), 6, "CX(0,4) on 6-qubit SV");
}

#[test]
fn compare_cx_reversed_qubit_order() {
    compare(QuantumGate::cx(3, 1), 5, "CX(3,1) on 5-qubit SV");
}

#[test]
fn compare_haar_random_2qubit() {
    let gate = QuantumGate::new(ComplexSquareMatrix::random_unitary(4), vec![0, 1]);
    compare(gate, 5, "Haar-2q on 5-qubit SV");
}

#[test]
fn compare_haar_random_2qubit_non_adjacent() {
    let gate = QuantumGate::new(ComplexSquareMatrix::random_unitary(4), vec![1, 4]);
    compare(gate, 6, "Haar-2q[1,4] on 6-qubit SV");
}

// ── Three-qubit gates ─────────────────────────────────────────────────────────

#[test]
fn compare_ccx_gate() {
    compare(QuantumGate::ccx(0, 1, 2), 5, "CCX on 5-qubit SV");
}

#[test]
fn compare_ccx_non_adjacent() {
    compare(QuantumGate::ccx(0, 2, 5), 6, "CCX(0,2,5) on 6-qubit SV");
}

#[test]
fn compare_haar_random_3qubit() {
    let gate = QuantumGate::new(ComplexSquareMatrix::random_unitary(8), vec![0, 1, 2]);
    compare(gate, 5, "Haar-3q on 5-qubit SV");
}

// ── Larger statevectors ───────────────────────────────────────────────────────

#[test]
fn compare_h_gate_20qubit_sv() {
    // 2^20 = 1 M amplitudes.  The persistent-grid loop ensures the CUDA kernel
    // processes all of them even with a capped grid dimension.
    compare(QuantumGate::h(0), 20, "H on 20-qubit SV");
}

#[test]
fn compare_cx_gate_20qubit_sv() {
    compare(QuantumGate::cx(0, 19), 20, "CX(0,19) on 20-qubit SV");
}

// ── Sequential gates (statefulness check) ────────────────────────────────────

/// Apply two gates in sequence on both backends and compare the final state.
/// This verifies that state is correctly updated in-place between applies.
#[test]
fn compare_sequential_h_then_x() {
    let n_sv = 4_u32;
    let init = seeded_state(n_sv as usize);

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
    jit.apply(h_cpu, &mut cpu_sv, 1).unwrap();
    jit.apply(x_cpu, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // CUDA: H then X on qubit 0.
    let c_spec = cuda_spec();
    let h_mat: Vec<(f64, f64)> = QuantumGate::h(0)
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let x_mat: Vec<(f64, f64)> = QuantumGate::x(0)
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let mut cuda_gen = CudaKernelGenerator::new().unwrap();
    let h_kid = cuda_gen.generate(&c_spec, &h_mat, &[0]).unwrap();
    let x_kid = cuda_gen.generate(&c_spec, &x_mat, &[0]).unwrap();
    let session = cuda_gen.compile().unwrap();
    let exec = CudaJitSession::new(&session).unwrap();
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
fn compare_h_squared_is_identity() {
    let n_sv = 3_u32;
    let init = seeded_state(n_sv as usize);

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
    jit.apply(kid, &mut cpu_sv, 1).unwrap();
    jit.apply(kid, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

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
    let exec = CudaJitSession::new(&session).unwrap();
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

// ── More gate types ───────────────────────────────────────────────────────────

#[test]
fn compare_y_gate() {
    // Y has imaginary off-diagonal entries (±i) — exercises complex constant folding.
    compare(QuantumGate::y(0), 4, "Y gate");
}

#[test]
fn compare_ry_gate() {
    compare(QuantumGate::ry(std::f64::consts::PI / 4.0, 0), 5, "Ry(π/4)");
}

#[test]
fn compare_swap_gate() {
    compare(QuantumGate::swap(0, 1), 5, "SWAP(0,1) on 5-qubit SV");
}

#[test]
fn compare_swap_non_adjacent() {
    compare(QuantumGate::swap(1, 4), 6, "SWAP(1,4) on 6-qubit SV");
}

#[test]
fn compare_cz_gate() {
    compare(QuantumGate::cz(0, 1), 5, "CZ(0,1) on 5-qubit SV");
}

#[test]
fn compare_haar_random_4qubit() {
    let gate = QuantumGate::new(ComplexSquareMatrix::random_unitary(16), vec![0, 1, 2, 3]);
    compare(gate, 6, "Haar-4q on 6-qubit SV");
}

// ── Multi-kernel same session ─────────────────────────────────────────────────

/// Compile H and CX into one [`CudaKernelArtifacts`]/[`CudaJitSession`] and
/// apply them in sequence.  Verifies that a single session with multiple
/// kernels works correctly end-to-end.
#[test]
fn compare_multi_kernel_same_session() {
    let n_sv = 5_u32;
    let init = seeded_state(n_sv as usize);

    // CPU: H on q0, then CX(0,1).
    let spec = cpu_spec();
    let mut cpu_gen = CPUKernelGenerator::new().unwrap();
    let h_cpu = cpu_gen
        .generate(&spec, QuantumGate::h(0).matrix().data(), &[0])
        .unwrap();
    let cx_cpu = cpu_gen
        .generate(&spec, QuantumGate::cx(0, 1).matrix().data(), &[0, 1])
        .unwrap();
    let mut jit = cpu_gen.init_jit().unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    jit.apply(h_cpu, &mut cpu_sv, 1).unwrap();
    jit.apply(cx_cpu, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // CUDA: compile H and CX into a single artifacts/session.
    let c_spec = cuda_spec();
    let h_mat: Vec<(f64, f64)> = QuantumGate::h(0)
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let cx_mat: Vec<(f64, f64)> = QuantumGate::cx(0, 1)
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let mut cuda_gen = CudaKernelGenerator::new().unwrap();
    let h_kid = cuda_gen.generate(&c_spec, &h_mat, &[0]).unwrap();
    let cx_kid = cuda_gen.generate(&c_spec, &cx_mat, &[0, 1]).unwrap();
    let artifacts = cuda_gen.compile().unwrap();
    let jit_session = CudaJitSession::new(&artifacts).unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    jit_session.apply(h_kid, &mut cuda_sv).unwrap();
    jit_session.apply(cx_kid, &mut cuda_sv).unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    assert_amps_close(
        &cpu_result,
        &cuda_result,
        "multi-kernel same session",
        TOLERANCE,
    );
}

// ── F32 precision ─────────────────────────────────────────────────────────────

/// Tolerance for F32 comparisons: single precision accumulates ~1e-7 error.
const F32_TOLERANCE: f64 = 1e-5;

fn apply_cpu_f32(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let spec = CPUKernelGenSpec {
        precision: Precision::F32,
        simd_width: SimdWidth::W128,
        mode: MatrixLoadMode::ImmValue,
        ztol: 1e-6,
        otol: 1e-6,
    };
    let mut gen = CPUKernelGenerator::new().expect("cpu f32: create generator");
    let kid = gen
        .generate(&spec, gate.matrix().data(), gate.qubits())
        .expect("cpu f32: generate kernel");
    let mut jit = gen.init_jit().expect("cpu f32: init_jit");
    let mut sv = CPUStatevector::new(n_sv_qubits, spec.precision, spec.simd_width);
    for (idx, &(re, im)) in init.iter().enumerate() {
        sv.set_amp(idx, Complex::new(re, im));
    }
    jit.apply(kid, &mut sv, 1).expect("cpu f32: apply");
    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

fn apply_cuda_f32(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let (sm_major, sm_minor) = query_device_sm().expect("query device SM");
    let spec = CudaKernelGenSpec {
        precision: CudaPrecision::F32,
        ztol: 1e-6,
        otol: 1e-6,
        sm_major,
        sm_minor,
    };
    let ffi_matrix: Vec<(f64, f64)> = gate.matrix().data().iter().map(|c| (c.re, c.im)).collect();
    let mut gen = CudaKernelGenerator::new().expect("cuda f32: create generator");
    let kid = gen
        .generate(&spec, &ffi_matrix, gate.qubits())
        .expect("cuda f32: generate");
    let artifacts = gen.compile().expect("cuda f32: compile");
    let jit = CudaJitSession::new(&artifacts).expect("cuda f32: create jit session");
    let mut sv =
        CudaStatevector::new(n_sv_qubits as u32, CudaPrecision::F32).expect("cuda f32: alloc");
    sv.upload(init).expect("cuda f32: upload");
    jit.apply(kid, &mut sv).expect("cuda f32: apply");
    sv.download().expect("cuda f32: download")
}

fn compare_f32(gate: QuantumGate, n_sv_qubits: u32, label: &str) {
    let init = seeded_state(n_sv_qubits as usize);
    let cpu_result = apply_cpu_f32(&gate, &init);
    let cuda_result = apply_cuda_f32(&gate, &init);
    assert_amps_close(&cpu_result, &cuda_result, label, F32_TOLERANCE);
}

#[test]
fn compare_f32_h_gate() {
    compare_f32(QuantumGate::h(0), 4, "H (F32) on 4-qubit SV");
}

#[test]
fn compare_f32_cx_gate() {
    compare_f32(QuantumGate::cx(0, 1), 5, "CX (F32) on 5-qubit SV");
}

#[test]
fn compare_f32_rx_gate() {
    compare_f32(
        QuantumGate::rx(std::f64::consts::PI / 7.0, 0),
        5,
        "Rx(π/7) (F32)",
    );
}

#[test]
fn compare_f32_haar_2qubit() {
    let gate = QuantumGate::new(ComplexSquareMatrix::random_unitary(4), vec![0, 1]);
    compare_f32(gate, 5, "Haar-2q (F32) on 5-qubit SV");
}

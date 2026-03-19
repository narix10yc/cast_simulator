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

use std::sync::Arc;

use cast::{
    cpu::{CPUKernelGenSpec, CPUStatevector, CpuKernelManager, MatrixLoadMode, SimdWidth},
    cuda::{device_sm, CudaKernelGenSpec, CudaKernelManager, CudaPrecision, CudaStatevector},
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
    let (sm_major, sm_minor) = device_sm().expect("query device SM");
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

/// Applies `gate` to an n-qubit statevector via the CPU JIT backend.
fn apply_cpu(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let spec = cpu_spec();
    let gate = Arc::new(gate.clone());

    let mgr = CpuKernelManager::new();
    let kid = mgr.generate(&spec, &gate).expect("cpu: generate kernel");

    let mut sv = CPUStatevector::new(n_sv_qubits, spec.precision, spec.simd_width);
    for (idx, &(re, im)) in init.iter().enumerate() {
        sv.set_amp(idx, Complex::new(re, im));
    }

    mgr.apply(kid, &mut sv, 1).expect("cpu: apply");
    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

/// Applies `gate` to a CUDA device statevector initialised from `init`.
fn apply_cuda(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let gate = Arc::new(gate.clone());
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&gate, cuda_spec())
        .expect("cuda: generate kernel");

    let mut sv = CudaStatevector::new(n_sv_qubits as u32, CudaPrecision::F64)
        .expect("cuda: alloc statevector");
    sv.upload(init).expect("cuda: upload");
    mgr.apply(kid, &mut sv).expect("cuda: apply");
    mgr.sync().expect("cuda: sync");
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
    compare(QuantumGate::h(0), 2, "H on 2-qubit SV");
}

#[test]
fn compare_x_gate_1qubit_sv() {
    compare(QuantumGate::x(0), 2, "X on 2-qubit SV");
}

#[test]
fn compare_h_gate_on_larger_sv() {
    compare(QuantumGate::h(0), 8, "H on 8-qubit SV");
}

#[test]
fn compare_h_gate_non_zero_qubit() {
    compare(QuantumGate::h(3), 6, "H[q3] on 6-qubit SV");
}

#[test]
fn compare_t_gate() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let t_matrix = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(s, s),
    ];
    let gate = QuantumGate::new(ComplexSquareMatrix::from_vec(2, t_matrix), vec![0]);
    compare(gate, 5, "T gate");
}

#[test]
fn compare_rx_gate() {
    compare(QuantumGate::rx(std::f64::consts::PI / 3.0, 0), 5, "Rx(π/3)");
}

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
    compare(QuantumGate::h(0), 20, "H on 20-qubit SV");
}

#[test]
fn compare_cx_gate_20qubit_sv() {
    compare(QuantumGate::cx(0, 19), 20, "CX(0,19) on 20-qubit SV");
}

// ── Sequential gates ──────────────────────────────────────────────────────────

#[test]
fn compare_sequential_h_then_x() {
    let n_sv = 4_u32;
    let init = seeded_state(n_sv as usize);

    // CPU: H then X on qubit 0.
    let h_spec = cpu_spec();
    let cpu_mgr = CpuKernelManager::new();
    let h_gate = Arc::new(QuantumGate::h(0));
    let x_gate = Arc::new(QuantumGate::x(0));
    let h_cpu = cpu_mgr.generate(&h_spec, &h_gate).unwrap();
    let x_cpu = cpu_mgr.generate(&h_spec, &x_gate).unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, h_spec.precision, h_spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    cpu_mgr.apply(h_cpu, &mut cpu_sv, 1).unwrap();
    cpu_mgr.apply(x_cpu, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // CUDA: H then X enqueued on the same stream.
    let mgr = CudaKernelManager::new();
    let h_kid = mgr
        .generate(&Arc::new(QuantumGate::h(0)), cuda_spec())
        .unwrap();
    let x_kid = mgr
        .generate(&Arc::new(QuantumGate::x(0)), cuda_spec())
        .unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv as u32, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    mgr.apply(h_kid, &mut cuda_sv).unwrap();
    mgr.apply(x_kid, &mut cuda_sv).unwrap();
    mgr.sync().unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    assert_amps_close(&cpu_result, &cuda_result, "H then X sequential", TOLERANCE);
}

#[test]
fn compare_h_squared_is_identity() {
    let n_sv = 3_u32;
    let init = seeded_state(n_sv as usize);

    // CPU: H^2
    let spec = cpu_spec();
    let cpu_mgr = CpuKernelManager::new();
    let h_gate = Arc::new(QuantumGate::h(0));
    let kid = cpu_mgr.generate(&spec, &h_gate).unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    cpu_mgr.apply(kid, &mut cpu_sv, 1).unwrap();
    cpu_mgr.apply(kid, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // CUDA: H^2 — apply the same kernel twice.
    let mgr = CudaKernelManager::new();
    let h_kid = mgr
        .generate(&Arc::new(QuantumGate::h(0)), cuda_spec())
        .unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv as u32, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    mgr.apply(h_kid, &mut cuda_sv).unwrap();
    mgr.apply(h_kid, &mut cuda_sv).unwrap();
    mgr.sync().unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    assert_amps_close(&cpu_result, &cuda_result, "H^2 cpu vs cuda", TOLERANCE);
    assert_amps_close(&init, &cpu_result, "H^2 ≈ identity (CPU)", TOLERANCE);
    assert_amps_close(&init, &cuda_result, "H^2 ≈ identity (CUDA)", TOLERANCE);
}

// ── More gate types ───────────────────────────────────────────────────────────

#[test]
fn compare_y_gate() {
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

// ── Multi-kernel same manager ─────────────────────────────────────────────────

#[test]
fn compare_multi_kernel_same_manager() {
    let n_sv = 5_u32;
    let init = seeded_state(n_sv as usize);

    // CPU: H on q0, then CX(0,1).
    let spec = cpu_spec();
    let cpu_mgr = CpuKernelManager::new();
    let h_gate = Arc::new(QuantumGate::h(0));
    let cx_gate = Arc::new(QuantumGate::cx(0, 1));
    let h_cpu = cpu_mgr.generate(&spec, &h_gate).unwrap();
    let cx_cpu = cpu_mgr.generate(&spec, &cx_gate).unwrap();
    let mut cpu_sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    for (i, &(re, im)) in init.iter().enumerate() {
        cpu_sv.set_amp(i, Complex::new(re, im));
    }
    cpu_mgr.apply(h_cpu, &mut cpu_sv, 1).unwrap();
    cpu_mgr.apply(cx_cpu, &mut cpu_sv, 1).unwrap();
    let cpu_result: Vec<(f64, f64)> = cpu_sv
        .amplitudes()
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect();

    // CUDA: H and CX in one manager, enqueued in order.
    let mgr = CudaKernelManager::new();
    let h_kid = mgr
        .generate(&Arc::new(QuantumGate::h(0)), cuda_spec())
        .unwrap();
    let cx_kid = mgr
        .generate(&Arc::new(QuantumGate::cx(0, 1)), cuda_spec())
        .unwrap();
    let mut cuda_sv = CudaStatevector::new(n_sv, CudaPrecision::F64).unwrap();
    cuda_sv.upload(&init).unwrap();
    mgr.apply(h_kid, &mut cuda_sv).unwrap();
    mgr.apply(cx_kid, &mut cuda_sv).unwrap();
    mgr.sync().unwrap();
    let cuda_result = cuda_sv.download().unwrap();

    assert_amps_close(
        &cpu_result,
        &cuda_result,
        "multi-kernel same manager",
        TOLERANCE,
    );
}

// ── F32 precision ─────────────────────────────────────────────────────────────

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
    let gate = Arc::new(gate.clone());
    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(&spec, &gate)
        .expect("cpu f32: generate kernel");
    let mut sv = CPUStatevector::new(n_sv_qubits, spec.precision, spec.simd_width);
    for (idx, &(re, im)) in init.iter().enumerate() {
        sv.set_amp(idx, Complex::new(re, im));
    }
    mgr.apply(kid, &mut sv, 1).expect("cpu f32: apply");
    sv.amplitudes().into_iter().map(|c| (c.re, c.im)).collect()
}

fn apply_cuda_f32(gate: &QuantumGate, init: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n_sv_qubits = init.len().trailing_zeros();
    let (sm_major, sm_minor) = device_sm().expect("query device SM");
    let spec = CudaKernelGenSpec {
        precision: CudaPrecision::F32,
        ztol: 1e-6,
        otol: 1e-6,
        sm_major,
        sm_minor,
    };
    let gate = Arc::new(gate.clone());
    let mgr = CudaKernelManager::new();
    let kid = mgr.generate(&gate, spec).expect("cuda f32: generate");
    let mut sv =
        CudaStatevector::new(n_sv_qubits as u32, CudaPrecision::F32).expect("cuda f32: alloc");
    sv.upload(init).expect("cuda f32: upload");
    mgr.apply(kid, &mut sv).expect("cuda f32: apply");
    mgr.sync().expect("cuda f32: sync");
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

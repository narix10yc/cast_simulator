use crate::types::QuantumGate;

use super::{CudaKernelGenSpec, CudaKernelManager, CudaPrecision, CudaStatevector};

fn hadamard() -> QuantumGate {
    QuantumGate::h(0)
}

fn cnot() -> QuantumGate {
    QuantumGate::cx(0, 1)
}

fn default_spec() -> CudaKernelGenSpec {
    CudaKernelGenSpec::f64_sm80()
}

// ── PTX generation (no CUDA device required) ──────────────────────────────────

#[test]
fn test_h_gate_emit_ptx() {
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H kernel");
    let ptx = mgr.emit_ptx(kid).expect("emit PTX");
    assert!(
        ptx.contains(".visible .entry"),
        "PTX should contain .visible .entry; got:\n{ptx}"
    );
}

#[test]
fn test_cnot_ptx_differs_from_h() {
    let mgr = CudaKernelManager::new();
    let h_kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H kernel");
    let cnot_kid = mgr
        .generate(&cnot(), default_spec())
        .expect("generate CNOT kernel");
    let ptx_h = mgr.emit_ptx(h_kid).expect("emit H PTX");
    let ptx_cnot = mgr.emit_ptx(cnot_kid).expect("emit CNOT PTX");
    assert_ne!(ptx_h, ptx_cnot, "H and CNOT should produce different PTX");
}

#[test]
fn test_emit_ptx_idempotent() {
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate kernel");
    let ptx1 = mgr.emit_ptx(kid).expect("first emit_ptx");
    let ptx2 = mgr.emit_ptx(kid).expect("second emit_ptx");
    assert_eq!(ptx1, ptx2, "emit_ptx should be idempotent");
}

#[test]
fn test_multi_kernel_manager() {
    let mgr = CudaKernelManager::new();
    let h_kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H");
    let cnot_kid = mgr
        .generate(&cnot(), default_spec())
        .expect("generate CNOT");
    assert_ne!(h_kid, cnot_kid, "kernel IDs must be distinct");
    let ptx_h = mgr.emit_ptx(h_kid).expect("emit H PTX");
    let ptx_cnot = mgr.emit_ptx(cnot_kid).expect("emit CNOT PTX");
    assert!(ptx_h.contains(".visible .entry"));
    assert!(ptx_cnot.contains(".visible .entry"));
    assert_ne!(ptx_h, ptx_cnot);
}

#[test]
fn test_unknown_kernel_id_returns_error() {
    let mgr = CudaKernelManager::new();
    assert!(
        mgr.emit_ptx(9999).is_none(),
        "emit_ptx with unknown id should fail"
    );
}

// ── GPU execution tests (require a CUDA device) ───────────────────────────────

#[test]
fn test_statevector_zero_state() {
    let mut sv = CudaStatevector::new(3, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    let amps = sv.download().expect("download");
    assert_eq!(amps.len(), 8);
    assert!((amps[0].0 - 1.0).abs() < 1e-14, "|0> re should be 1");
    for i in 1..8 {
        assert!(
            amps[i].0.abs() < 1e-14 && amps[i].1.abs() < 1e-14,
            "amp[{i}] should be 0"
        );
    }
}

#[test]
fn test_statevector_upload_download_roundtrip() {
    let data: Vec<(f64, f64)> = (0..4u32)
        .map(|i| (i as f64 * 0.1, i as f64 * -0.05))
        .collect();
    let mut sv = CudaStatevector::new(2, CudaPrecision::F64).expect("alloc statevector");
    sv.upload(&data).expect("upload");
    let got = sv.download().expect("download");
    for (i, (&want, got)) in data.iter().zip(got.iter()).enumerate() {
        assert!((want.0 - got.0).abs() < 1e-14, "re[{i}] mismatch");
        assert!((want.1 - got.1).abs() < 1e-14, "im[{i}] mismatch");
    }
}

#[test]
fn test_h_gate_apply_to_zero_state() {
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H kernel");

    let mut sv = CudaStatevector::new(1, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    mgr.apply(kid, &mut sv).expect("apply H");
    mgr.sync().expect("sync");

    let amps = sv.download().expect("download");
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amps[0].0 - s).abs() < 1e-10, "amp[0].re ≈ 1/√2");
    assert!(amps[0].1.abs() < 1e-10, "amp[0].im ≈ 0");
    assert!((amps[1].0 - s).abs() < 1e-10, "amp[1].re ≈ 1/√2");
    assert!(amps[1].1.abs() < 1e-10, "amp[1].im ≈ 0");
}

#[test]
fn test_x_gate_apply_to_zero_state() {
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&QuantumGate::x(0), default_spec())
        .expect("generate X kernel");

    let mut sv = CudaStatevector::new(1, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    mgr.apply(kid, &mut sv).expect("apply X");
    mgr.sync().expect("sync");

    let amps = sv.download().expect("download");
    assert!(amps[0].0.abs() < 1e-10, "amp[0] ≈ 0 after X");
    assert!((amps[1].0 - 1.0).abs() < 1e-10, "amp[1] ≈ 1 after X");
}

#[test]
fn test_apply_on_larger_statevector() {
    let mgr = CudaKernelManager::new();
    let kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H kernel");

    let mut sv = CudaStatevector::new(4, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    mgr.apply(kid, &mut sv).expect("apply H on 4-qubit SV");
    mgr.sync().expect("sync");

    let amps = sv.download().expect("download");
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amps[0].0 - s).abs() < 1e-10, "amp[0] ≈ 1/√2");
    assert!((amps[1].0 - s).abs() < 1e-10, "amp[1] ≈ 1/√2");
    for i in 2..16 {
        assert!(
            amps[i].0.abs() < 1e-10 && amps[i].1.abs() < 1e-10,
            "amp[{i}] should be 0"
        );
    }
}

#[test]
fn test_sequential_apply() {
    // Apply X then H on the same statevector via two enqueued launches.
    // X|0⟩ = |1⟩, then H|1⟩ = |−⟩ = (1/√2)|0⟩ − (1/√2)|1⟩.
    let mgr = CudaKernelManager::new();
    let x_kid = mgr
        .generate(&QuantumGate::x(0), default_spec())
        .expect("generate X");
    let h_kid = mgr
        .generate(&hadamard(), default_spec())
        .expect("generate H");

    let mut sv = CudaStatevector::new(1, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");

    // Enqueue both launches; GPU executes them in order on the single stream.
    mgr.apply(x_kid, &mut sv).expect("apply X");
    mgr.apply(h_kid, &mut sv).expect("apply H");
    mgr.sync().expect("sync");

    let amps = sv.download().expect("download");
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amps[0].0 - s).abs() < 1e-10, "amp[0] ≈ 1/√2");
    assert!((amps[1].0 + s).abs() < 1e-10, "amp[1] ≈ −1/√2");
}

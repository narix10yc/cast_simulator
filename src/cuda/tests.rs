use super::{CudaKernelGenSpec, CudaKernelGenerator, CudaPrecision, CudaStatevector};

fn hadamard_matrix() -> Vec<(f64, f64)> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![(s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0)]
}

fn cnot_matrix() -> Vec<(f64, f64)> {
    vec![
        (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0),
        (0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0),
    ]
}

fn default_spec() -> CudaKernelGenSpec {
    CudaKernelGenSpec::f64_sm80()
}

#[test]
fn test_h_gate_emit_ir() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let ir = gen.emit_ir(kid).expect("emit IR");
    assert!(
        ir.contains("nvptx64"),
        "IR should mention nvptx64 target; got:\n{ir}"
    );
    assert!(
        ir.contains("define"),
        "IR should contain function definitions; got:\n{ir}"
    );
}

#[test]
fn test_h_gate_emit_ptx() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let session = gen.compile().expect("compile");
    let ptx = session.emit_ptx(kid).expect("emit PTX");
    assert!(
        ptx.contains(".visible .entry"),
        "PTX should contain .visible .entry; got:\n{ptx}"
    );
}

#[test]
fn test_compilation_session_kernels_accessible() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let session = gen.compile().expect("compile");
    assert_eq!(session.kernels.len(), 1);
    assert_eq!(session.kernels[0].kernel_id, kid);
    assert_eq!(session.kernels[0].n_gate_qubits, 1);
    assert!(!session.kernels[0].ptx.is_empty());
    assert!(!session.kernels[0].func_name.is_empty());
}

#[test]
fn test_cnot_ptx_differs_from_h() {
    let spec = default_spec();

    let mut gen_h = CudaKernelGenerator::new().expect("create H generator");
    let kid_h = gen_h
        .generate(&spec, &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let session_h = gen_h.compile().expect("compile H");
    let ptx_h = session_h.emit_ptx(kid_h).expect("emit H PTX");

    let mut gen_cnot = CudaKernelGenerator::new().expect("create CNOT generator");
    let kid_cnot = gen_cnot
        .generate(&spec, &cnot_matrix(), &[0, 1])
        .expect("generate CNOT kernel");
    let session_cnot = gen_cnot.compile().expect("compile CNOT");
    let ptx_cnot = session_cnot.emit_ptx(kid_cnot).expect("emit CNOT PTX");

    assert_ne!(ptx_h, ptx_cnot, "H and CNOT should produce different PTX");
}

#[test]
fn test_emit_ir_idempotent() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate kernel");
    let ir1 = gen.emit_ir(kid).expect("first emit_ir");
    let ir2 = gen.emit_ir(kid).expect("second emit_ir");
    assert_eq!(ir1, ir2, "emit_ir should be idempotent");
}

#[test]
fn test_emit_ptx_idempotent() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate kernel");
    let session = gen.compile().expect("compile");
    let ptx1 = session.emit_ptx(kid).expect("first emit_ptx");
    let ptx2 = session.emit_ptx(kid).expect("second emit_ptx");
    assert_eq!(ptx1, ptx2, "emit_ptx should be idempotent");
}

#[test]
fn test_multi_kernel_session() {
    let spec = default_spec();
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid_h = gen
        .generate(&spec, &hadamard_matrix(), &[0])
        .expect("generate H");
    let kid_cnot = gen
        .generate(&spec, &cnot_matrix(), &[0, 1])
        .expect("generate CNOT");
    let session = gen.compile().expect("compile");
    assert_eq!(session.kernels.len(), 2);
    let ptx_h = session.emit_ptx(kid_h).expect("emit H PTX");
    let ptx_cnot = session.emit_ptx(kid_cnot).expect("emit CNOT PTX");
    assert!(ptx_h.contains(".visible .entry"), "H PTX should have entry");
    assert!(ptx_cnot.contains(".visible .entry"), "CNOT PTX should have entry");
    assert_ne!(ptx_h, ptx_cnot);
}

#[test]
#[ignore = "requires CUDA device with nvJitLink support"]
fn test_h_gate_emit_cubin() {
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&default_spec(), &hadamard_matrix(), &[0])
        .expect("generate kernel");
    let session = gen.compile().expect("compile");
    let cubin = session.emit_cubin(kid).expect("emit cubin");
    assert!(cubin.len() >= 4, "cubin should be non-empty");
    assert_eq!(&cubin[..4], b"\x7fELF", "cubin should start with ELF magic");
}

// ── GPU execution tests (require a CUDA device) ───────────────────────────────

#[test]
#[ignore = "requires CUDA device"]
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
#[ignore = "requires CUDA device"]
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
#[ignore = "requires CUDA device"]
fn test_h_gate_apply_to_zero_state() {
    let spec = default_spec();
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&spec, &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let session = gen.compile().expect("compile");
    let exec = super::CudaExecSession::new(&session).expect("create exec session");

    let mut sv = CudaStatevector::new(1, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    exec.apply(kid, &mut sv).expect("apply H");

    let amps = sv.download().expect("download");
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amps[0].0 - s).abs() < 1e-10, "amp[0].re ≈ 1/√2");
    assert!(amps[0].1.abs() < 1e-10, "amp[0].im ≈ 0");
    assert!((amps[1].0 - s).abs() < 1e-10, "amp[1].re ≈ 1/√2");
    assert!(amps[1].1.abs() < 1e-10, "amp[1].im ≈ 0");
}

#[test]
#[ignore = "requires CUDA device"]
fn test_x_gate_apply_to_zero_state() {
    let x_matrix = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)];
    let spec = default_spec();
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&spec, &x_matrix, &[0])
        .expect("generate X kernel");
    let session = gen.compile().expect("compile");
    let exec = super::CudaExecSession::new(&session).expect("create exec session");

    let mut sv = CudaStatevector::new(1, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    exec.apply(kid, &mut sv).expect("apply X");

    let amps = sv.download().expect("download");
    assert!(amps[0].0.abs() < 1e-10, "amp[0] ≈ 0 after X");
    assert!((amps[1].0 - 1.0).abs() < 1e-10, "amp[1] ≈ 1 after X");
}

#[test]
#[ignore = "requires CUDA device"]
fn test_apply_on_larger_statevector() {
    let spec = default_spec();
    let mut gen = CudaKernelGenerator::new().expect("create generator");
    let kid = gen
        .generate(&spec, &hadamard_matrix(), &[0])
        .expect("generate H kernel");
    let session = gen.compile().expect("compile");
    let exec = super::CudaExecSession::new(&session).expect("create exec session");

    let mut sv = CudaStatevector::new(4, CudaPrecision::F64).expect("alloc statevector");
    sv.zero().expect("zero");
    exec.apply(kid, &mut sv).expect("apply H on 4-qubit SV");

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

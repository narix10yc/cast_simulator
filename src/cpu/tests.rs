use std::sync::Arc;

use crate::cost_model::FusionConfig;
use crate::cpu::*;
use crate::fusion;
use crate::types::{Complex, Precision, QuantumGate};
use crate::CircuitGraph;

/// Creates a normalized statevector with deterministic amplitudes.
fn seeded_statevector(
    n_qubits: u32,
    precision: Precision,
    simd_width: SimdWidth,
) -> CPUStatevector {
    let mut sv = CPUStatevector::new(n_qubits, precision, simd_width);
    for idx in 0..sv.len() {
        let re = (idx as f64) + 1.0;
        let im = (idx as f64) * 0.5 - 0.25;
        sv.set_amp(idx, Complex::new(re, im));
    }
    sv.normalize();
    sv
}

fn assert_statevectors_close(lhs: &CPUStatevector, rhs: &CPUStatevector, tol: f64) {
    assert_eq!(lhs.n_qubits(), rhs.n_qubits(), "statevector size mismatch");
    for idx in 0..lhs.len() {
        let diff = lhs.amp(idx) - rhs.amp(idx);
        assert!(
            diff.norm() < tol,
            "statevectors differ at index {}: lhs={:?}, rhs={:?}, diff={:?}",
            idx,
            lhs.amp(idx),
            rhs.amp(idx),
            diff
        );
    }
}

/// Generates a JIT kernel for `gate` and checks the result against the scalar path.
/// `n_threads`: number of worker threads passed to `apply`.
fn run_jit_and_compare_full(
    gate: &QuantumGate,
    n_qubits_sv: u32,
    spec: CPUKernelGenSpec,
    n_threads: u32,
    tol: f64,
) {
    let gate = Arc::new(gate.clone());
    let mgr = CpuKernelManager::new();
    let kernel_id = mgr.generate_gate(spec, &gate).expect("generate kernel");

    let mut sv_jit = seeded_statevector(n_qubits_sv, spec.precision, spec.simd_width);
    let mut sv_ref = sv_jit.clone();

    sv_ref.apply_gate(&gate);
    mgr.apply(kernel_id, &mut sv_jit, n_threads)
        .expect("apply kernel");

    assert_statevectors_close(&sv_jit, &sv_ref, tol);
    assert!((sv_jit.norm() - 1.0).abs() < tol);
}

fn default_spec(precision: Precision, simd_width: SimdWidth) -> CPUKernelGenSpec {
    CPUKernelGenSpec {
        precision,
        simd_width,
        mode: MatrixLoadMode::ImmValue,
        ztol: match precision {
            Precision::F32 => 1e-6,
            Precision::F64 => 1e-12,
        },
        otol: match precision {
            Precision::F32 => 1e-6,
            Precision::F64 => 1e-12,
        },
    }
}

fn run_jit_and_compare(
    gate: QuantumGate,
    n_qubits_sv: u32,
    precision: Precision,
    simd_width: SimdWidth,
    tol: f64,
) {
    run_jit_and_compare_full(
        &gate,
        n_qubits_sv,
        default_spec(precision, simd_width),
        1,
        tol,
    );
}

fn circuit_gates_in_row_order(graph: &CircuitGraph) -> Vec<Arc<QuantumGate>> {
    graph.gates_in_row_order()
}

fn run_circuit_scalar(
    graph: &CircuitGraph,
    n_qubits_sv: u32,
    precision: Precision,
    simd_width: SimdWidth,
) -> CPUStatevector {
    let mut sv = seeded_statevector(n_qubits_sv, precision, simd_width);
    for gate in circuit_gates_in_row_order(graph) {
        sv.apply_gate(&gate);
    }
    sv
}

fn run_circuit_jit(
    graph: &CircuitGraph,
    n_qubits_sv: u32,
    spec: CPUKernelGenSpec,
    n_threads: u32,
) -> CPUStatevector {
    let gates = circuit_gates_in_row_order(graph);
    let mgr = CpuKernelManager::new();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in &gates {
        let kernel_id = mgr.generate_gate(spec, gate).expect("generate kernel");
        kernel_ids.push(kernel_id);
    }
    let mut sv = seeded_statevector(n_qubits_sv, spec.precision, spec.simd_width);
    for kernel_id in kernel_ids {
        mgr.apply(kernel_id, &mut sv, n_threads)
            .expect("apply circuit kernel");
    }
    sv
}

fn assert_fused_and_unfused_circuits_agree(
    original: &CircuitGraph,
    n_qubits_sv: u32,
    spec: CPUKernelGenSpec,
    tol: f64,
) {
    let scalar_ref = run_circuit_scalar(original, n_qubits_sv, spec.precision, spec.simd_width);
    let unfused_jit = run_circuit_jit(original, n_qubits_sv, spec, 1);

    let mut fused = original.clone();
    fusion::optimize(&mut fused, &FusionConfig::default());
    let fused_jit = run_circuit_jit(&fused, n_qubits_sv, spec, 1);

    assert_statevectors_close(&unfused_jit, &scalar_ref, tol);
    assert_statevectors_close(&fused_jit, &scalar_ref, tol);
    assert_statevectors_close(&fused_jit, &unfused_jit, tol);
    assert!((fused_jit.norm() - 1.0).abs() < tol);
}

#[test]
fn initializes_to_zero_state() {
    let mut sv = CPUStatevector::new(3, Precision::F64, SimdWidth::W128);
    sv.initialize();

    assert_eq!(sv.amp(0), Complex::new(1.0, 0.0));
    for idx in 1..sv.len() {
        assert_eq!(sv.amp(idx), Complex::new(0.0, 0.0));
    }
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn applies_single_qubit_gate() {
    let mut sv = CPUStatevector::new(1, Precision::F64, SimdWidth::W128);
    sv.initialize();
    sv.apply_gate(&QuantumGate::h(0));

    let expected = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sv.amp(0) - Complex::new(expected, 0.0)).norm() < 1e-12);
    assert!((sv.amp(1) - Complex::new(expected, 0.0)).norm() < 1e-12);
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn jit_applies_single_qubit_gate() {
    let spec = CPUKernelGenSpec::f64();
    let gate = Arc::new(QuantumGate::h(0));
    let mgr = CpuKernelManager::new();
    let kernel_id = mgr.generate_gate(spec, &gate).expect("generate kernel");

    let mut sv = CPUStatevector::new(6, spec.precision, spec.simd_width);
    sv.initialize();
    mgr.apply(kernel_id, &mut sv, 1).expect("apply kernel");

    let expected = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sv.amp(0) - Complex::new(expected, 0.0)).norm() < 1e-10);
    assert!((sv.amp(1) - Complex::new(expected, 0.0)).norm() < 1e-10);
    for idx in 2..sv.len() {
        assert!((sv.amp(idx) - Complex::new(0.0, 0.0)).norm() < 1e-10);
    }
    assert!((sv.norm() - 1.0).abs() < 1e-10);
}

#[test]
fn jit_matches_scalar_for_x_gate_imm_mode() {
    run_jit_and_compare(QuantumGate::x(1), 3, Precision::F64, SimdWidth::W128, 1e-10);
}

#[test]
fn jit_matches_scalar_for_cx_gate_imm_mode() {
    run_jit_and_compare(
        QuantumGate::cx(0, 1),
        3,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

#[test]
fn jit_matches_scalar_for_nonadjacent_gate_imm_mode() {
    run_jit_and_compare(
        QuantumGate::cx(0, 2),
        4,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

#[test]
fn jit_matches_scalar_for_fp32_imm_mode() {
    run_jit_and_compare(QuantumGate::h(2), 4, Precision::F32, SimdWidth::W128, 5e-5);
}

// ── SIMD width coverage ────────────────────────────────────────────────────

// F64/W256: simd_s=2, needs ≥3 qubits for a 1-qubit gate.
#[test]
fn jit_f64_w256() {
    run_jit_and_compare(QuantumGate::h(1), 4, Precision::F64, SimdWidth::W256, 1e-10);
}

// F64/W512: simd_s=3, needs ≥4 qubits for a 1-qubit gate.
#[test]
fn jit_f64_w512() {
    run_jit_and_compare(QuantumGate::h(1), 5, Precision::F64, SimdWidth::W512, 1e-10);
}

// F32/W256: simd_s=3, needs ≥4 qubits for a 1-qubit gate.
#[test]
fn jit_f32_w256() {
    run_jit_and_compare(QuantumGate::h(2), 5, Precision::F32, SimdWidth::W256, 5e-5);
}

// F32/W512: simd_s=4, needs ≥5 qubits for a 1-qubit gate.
#[test]
fn jit_f32_w512() {
    run_jit_and_compare(QuantumGate::h(2), 6, Precision::F32, SimdWidth::W512, 5e-5);
}

// ── StackLoad mode ─────────────────────────────────────────────────────────

// StackLoad embeds a runtime matrix pointer rather than immediate constants;
// the numerical result must be identical to ImmValue for the same gate.
#[test]
fn jit_stack_load_matches_imm_value() {
    let gate = Arc::new(QuantumGate::cx(0, 2));
    let n_qubits_sv = 4;
    let precision = Precision::F64;
    let simd_width = SimdWidth::W128;
    let tol = 1e-10;

    let mut sv_imm = seeded_statevector(n_qubits_sv, precision, simd_width);
    let mut sv_stack = sv_imm.clone();

    let spec_imm = default_spec(precision, simd_width);
    let spec_stack = CPUKernelGenSpec {
        mode: MatrixLoadMode::StackLoad,
        ..spec_imm
    };

    let mgr_imm = CpuKernelManager::new();
    let kid_imm = mgr_imm
        .generate_gate(spec_imm, &gate)
        .expect("generate imm kernel");
    mgr_imm.apply(kid_imm, &mut sv_imm, 1).expect("apply imm");

    let mgr_stack = CpuKernelManager::new();
    let kid_stack = mgr_stack
        .generate_gate(spec_stack, &gate)
        .expect("generate stack kernel");
    mgr_stack
        .apply(kid_stack, &mut sv_stack, 1)
        .expect("apply stack");

    assert_statevectors_close(&sv_imm, &sv_stack, tol);
}

// ── Gate variety ───────────────────────────────────────────────────────────

#[test]
fn jit_swap_nonadjacent() {
    // SWAP(0,2): non-adjacent, exercises hi_bits path same as CX(0,2).
    run_jit_and_compare(
        QuantumGate::swap(0, 2),
        4,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

#[test]
fn jit_cz_nonadjacent() {
    run_jit_and_compare(
        QuantumGate::cz(1, 3),
        5,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

#[test]
fn jit_ccx_gate() {
    // 3-qubit Toffoli: exercises the multi-qubit hi_bits partitioning.
    run_jit_and_compare(
        QuantumGate::ccx(0, 1, 2),
        5,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

// Rx has a dense, fully complex matrix — no zero or ±1 entries — so the
// kernel exercises FMA paths that sparse gates like CX skip.
#[test]
fn jit_rx_gate() {
    run_jit_and_compare(
        QuantumGate::rx(std::f64::consts::PI / 3.0, 1),
        3,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

#[test]
fn jit_rz_gate() {
    run_jit_and_compare(
        QuantumGate::rz(std::f64::consts::PI / 5.0, 0),
        3,
        Precision::F64,
        SimdWidth::W128,
        1e-10,
    );
}

// ── Multi-kernel session ────────────────────────────────────────────────────

// Generates H and CX kernels in one session and applies them sequentially;
// verifies that the session correctly dispatches by kernel id.
#[test]
fn jit_multiple_kernels_in_one_manager() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let h_gate = Arc::new(QuantumGate::h(0));
    let cx_gate = Arc::new(QuantumGate::cx(0, 1));

    let mgr = CpuKernelManager::new();
    let kid_h = mgr.generate_gate(spec, &h_gate).expect("generate H kernel");
    let kid_cx = mgr
        .generate_gate(spec, &cx_gate)
        .expect("generate CX kernel");

    let mut sv_jit = seeded_statevector(3, Precision::F64, SimdWidth::W128);
    let mut sv_ref = sv_jit.clone();

    sv_ref.apply_gate(&h_gate);
    sv_ref.apply_gate(&cx_gate);

    mgr.apply(kid_h, &mut sv_jit, 1).expect("apply H");
    mgr.apply(kid_cx, &mut sv_jit, 1).expect("apply CX");

    assert_statevectors_close(&sv_jit, &sv_ref, 1e-10);
    assert!((sv_jit.norm() - 1.0).abs() < 1e-10);
}

// ── emit_ir ───────────────────────────────────────────────────────────────

#[test]
fn emit_ir_returns_valid_llvm_ir() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(KernelGenRequest::from_gate(spec, &QuantumGate::h(0)).with_ir())
        .expect("generate kernel");

    let ir = mgr.emit_ir(kid).expect("emit_ir should be Some");

    // The IR must be a non-empty LLVM text module.
    assert!(!ir.is_empty(), "IR should not be empty");
    assert!(
        ir.contains("define"),
        "IR should contain a function definition"
    );
}

#[test]
fn emit_ir_is_idempotent() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(KernelGenRequest::from_gate(spec, &QuantumGate::x(0)).with_ir())
        .expect("generate kernel");

    let ir_first = mgr.emit_ir(kid).expect("first emit_ir");
    let ir_second = mgr.emit_ir(kid).expect("second emit_ir");

    assert_eq!(ir_first, ir_second, "emit_ir should be idempotent");
}

#[test]
fn emit_ir_with_diagnostics_still_produces_correct_kernel() {
    // Diagnostics capture must not corrupt the kernel.
    let gate = Arc::new(QuantumGate::h(1));
    let spec = default_spec(Precision::F64, SimdWidth::W128);

    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(KernelGenRequest::from_gate(spec, &gate).with_ir())
        .expect("generate kernel");

    let ir = mgr.emit_ir(kid).expect("emit_ir should be Some");
    assert!(!ir.is_empty());

    let mut sv_jit = seeded_statevector(3, Precision::F64, SimdWidth::W128);
    let mut sv_ref = sv_jit.clone();
    sv_ref.apply_gate(&gate);
    mgr.apply(kid, &mut sv_jit, 1).expect("apply kernel");

    assert_statevectors_close(&sv_jit, &sv_ref, 1e-10);
}

#[test]
fn emit_ir_per_kernel_independent() {
    // Each kernel_id returns its own IR; they must differ (H ≠ CX).
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let h_gate = Arc::new(QuantumGate::h(0));
    let cx_gate = Arc::new(QuantumGate::cx(0, 1));

    let mgr = CpuKernelManager::new();
    let kid_h = mgr
        .generate(KernelGenRequest::from_gate(spec, &h_gate).with_ir())
        .expect("H kernel");
    let kid_cx = mgr
        .generate(KernelGenRequest::from_gate(spec, &cx_gate).with_ir())
        .expect("CX kernel");

    let ir_h = mgr.emit_ir(kid_h).expect("H IR");
    let ir_cx = mgr.emit_ir(kid_cx).expect("CX IR");

    assert_ne!(ir_h, ir_cx, "different gates must produce different IR");
}

#[test]
fn emit_ir_returns_none_without_diagnostics() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate_gate(spec, &QuantumGate::h(0))
        .expect("generate kernel");
    assert!(
        mgr.emit_ir(kid).is_none(),
        "IR should be None without diagnostics"
    );
}

#[test]
fn emit_ir_returns_none_for_unknown_kernel_id() {
    let mgr = CpuKernelManager::new();
    assert!(
        mgr.emit_ir(9999).is_none(),
        "unknown kernel_id should return None"
    );
}

// ── emit_asm ──────────────────────────────────────────────────────────────

fn compile_h_manager_with_asm() -> (CpuKernelManager, KernelId) {
    let gate = Arc::new(QuantumGate::h(0));
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(KernelGenRequest::from_gate(spec, &gate).with_asm())
        .expect("generate kernel");
    (mgr, kid)
}

#[test]
fn emit_asm_returns_nonempty_text() {
    let (mgr, kid) = compile_h_manager_with_asm();
    let asm = mgr.emit_asm(kid).expect("emit_asm should be Some");
    assert!(!asm.is_empty(), "assembly should not be empty");
}

#[test]
fn emit_asm_is_ascii() {
    let (mgr, kid) = compile_h_manager_with_asm();
    let asm = mgr.emit_asm(kid).expect("emit_asm should be Some");
    assert!(asm.is_ascii(), "assembly should be ASCII text");
}

#[test]
fn emit_asm_is_idempotent() {
    let (mgr, kid) = compile_h_manager_with_asm();
    let first = mgr.emit_asm(kid).expect("first emit_asm");
    let second = mgr.emit_asm(kid).expect("second emit_asm");
    assert_eq!(first, second);
}

#[test]
fn emit_asm_per_kernel_independent() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let h_gate = Arc::new(QuantumGate::h(0));
    let cx_gate = Arc::new(QuantumGate::cx(0, 1));

    let mgr = CpuKernelManager::new();
    let kid_h = mgr
        .generate(KernelGenRequest::from_gate(spec, &h_gate).with_asm())
        .expect("H kernel");
    let kid_cx = mgr
        .generate(KernelGenRequest::from_gate(spec, &cx_gate).with_asm())
        .expect("CX kernel");

    let asm_h = mgr.emit_asm(kid_h).expect("H asm");
    let asm_cx = mgr.emit_asm(kid_cx).expect("CX asm");
    assert_ne!(
        asm_h, asm_cx,
        "different gates must produce different assembly"
    );
}

#[test]
fn emit_asm_consistent_with_emit_ir() {
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let gate = Arc::new(QuantumGate::h(0));

    let mgr = CpuKernelManager::new();
    let kid = mgr
        .generate(
            KernelGenRequest::from_gate(spec, &gate)
                .with_ir()
                .with_asm(),
        )
        .expect("generate");
    let ir = mgr.emit_ir(kid).expect("emit_ir should be Some");
    let asm = mgr.emit_asm(kid).expect("emit_asm should be Some");

    assert!(!ir.is_empty());
    assert!(!asm.is_empty());
    assert!(
        !asm.contains("define"),
        "assembly should not contain LLVM IR syntax"
    );
}

#[test]
fn emit_asm_returns_none_without_diagnostics() {
    let gate = Arc::new(QuantumGate::h(0));
    let spec = default_spec(Precision::F64, SimdWidth::W128);
    let mgr = CpuKernelManager::new();
    let kid = mgr.generate_gate(spec, &gate).expect("generate kernel");
    assert!(
        mgr.emit_asm(kid).is_none(),
        "emit_asm should be None without diagnostics"
    );
}

#[test]
fn emit_asm_returns_none_for_unknown_kernel_id() {
    let mgr = CpuKernelManager::new();
    assert!(
        mgr.emit_asm(9999).is_none(),
        "unknown kernel_id should return None"
    );
}

// ── Multithreaded apply ────────────────────────────────────────────────────

// Runs `graph` with `n_threads` and asserts the result equals 1-thread output.
fn assert_multithreaded_matches_single(
    graph: &CircuitGraph,
    n_qubits_sv: u32,
    spec: CPUKernelGenSpec,
    n_threads: u32,
    tol: f64,
) {
    let sv_single = run_circuit_jit(graph, n_qubits_sv, spec, 1);
    let sv_multi = run_circuit_jit(graph, n_qubits_sv, spec, n_threads);
    assert_statevectors_close(&sv_multi, &sv_single, tol);
    assert!((sv_multi.norm() - 1.0).abs() < tol);
}

// Result must be identical to single-threaded regardless of how work is split.
#[test]
fn jit_multithreaded_matches_single_thread() {
    let gate = QuantumGate::rx(std::f64::consts::PI / 7.0, 2);
    let spec = default_spec(Precision::F64, SimdWidth::W128);

    run_jit_and_compare_full(&gate, 6, spec, 4, 1e-10);
}

// 2-thread split on a CX gate over 8 qubits (128 tasks → 64 per thread).
#[test]
fn jit_multithreaded_2_threads_cx_gate() {
    run_jit_and_compare_full(
        &QuantumGate::cx(0, 3),
        8,
        default_spec(Precision::F64, SimdWidth::W128),
        2,
        1e-10,
    );
}

// 8 threads on a 10-qubit statevector (1024 tasks → 128 per thread).
#[test]
fn jit_multithreaded_8_threads_large_sv() {
    run_jit_and_compare_full(
        &QuantumGate::rx(std::f64::consts::PI / 5.0, 4),
        10,
        default_spec(Precision::F64, SimdWidth::W128),
        8,
        1e-10,
    );
}

// Multi-kernel session: 8 gates compiled in one session, applied with 4 threads.
// Verifies that thread-local counter ranges are correct across kernel boundaries.
#[test]
fn jit_multithreaded_circuit_4_threads() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::rx(0.3, 2));
    graph.insert_gate(QuantumGate::cx(1, 3));
    graph.insert_gate(QuantumGate::rz(0.7, 0));
    graph.insert_gate(QuantumGate::h(4));
    graph.insert_gate(QuantumGate::cx(4, 5));
    graph.insert_gate(QuantumGate::ry(1.1, 3));

    assert_multithreaded_matches_single(
        &graph,
        8,
        default_spec(Precision::F64, SimdWidth::W128),
        4,
        1e-10,
    );
}

// StackLoad mode: each thread receives the same p_mat pointer; verifies that
// the Rust-side matrix buffer allocation is correct and all threads read it
// consistently without data races.
#[test]
fn jit_multithreaded_stack_load_4_threads() {
    let spec = CPUKernelGenSpec {
        mode: MatrixLoadMode::StackLoad,
        ..default_spec(Precision::F64, SimdWidth::W128)
    };
    run_jit_and_compare_full(&QuantumGate::cx(0, 3), 8, spec, 4, 1e-10);
}

// StackLoad + multi-kernel circuit + threads.
#[test]
fn jit_multithreaded_stack_load_circuit_4_threads() {
    let spec = CPUKernelGenSpec {
        mode: MatrixLoadMode::StackLoad,
        ..default_spec(Precision::F64, SimdWidth::W128)
    };
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::cx(0, 2));
    graph.insert_gate(QuantumGate::rz(0.5, 1));
    graph.insert_gate(QuantumGate::cx(1, 3));
    graph.insert_gate(QuantumGate::ry(0.9, 0));

    assert_multithreaded_matches_single(&graph, 7, spec, 4, 1e-10);
}

// F32 precision with 4 threads.
#[test]
fn jit_multithreaded_f32_4_threads() {
    run_jit_and_compare_full(
        &QuantumGate::rx(std::f64::consts::PI / 4.0, 2),
        8,
        default_spec(Precision::F32, SimdWidth::W128),
        4,
        5e-5,
    );
}

// W256 SIMD (simd_s = 2 for F64) with 4 threads.
#[test]
fn jit_multithreaded_w256_4_threads() {
    run_jit_and_compare_full(
        &QuantumGate::rx(0.9, 1),
        8,
        default_spec(Precision::F64, SimdWidth::W256),
        4,
        1e-10,
    );
}

// W512 SIMD (simd_s = 3 for F64) with 4 threads.
#[test]
fn jit_multithreaded_w512_4_threads() {
    run_jit_and_compare_full(
        &QuantumGate::rx(0.5, 2),
        9,
        default_spec(Precision::F64, SimdWidth::W512),
        4,
        1e-10,
    );
}

// Large brick-wall circuit over 12 qubits, 8 threads.
#[test]
fn jit_multithreaded_large_circuit_8_threads() {
    let mut graph = CircuitGraph::new();
    for i in (0u32..10).step_by(2) {
        graph.insert_gate(QuantumGate::cx(i, i + 1));
    }
    for i in (1u32..9).step_by(2) {
        graph.insert_gate(QuantumGate::cx(i, i + 1));
    }
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::rz(0.4, 5));
    graph.insert_gate(QuantumGate::rx(0.8, 9));
    for i in (0u32..10).step_by(2) {
        graph.insert_gate(QuantumGate::cx(i, i + 1));
    }
    graph.insert_gate(QuantumGate::swap(2, 7));
    graph.insert_gate(QuantumGate::ccx(0, 3, 6));

    assert_multithreaded_matches_single(
        &graph,
        12,
        default_spec(Precision::F64, SimdWidth::W128),
        8,
        1e-10,
    );
}

// Thread count exactly equals n_tasks: each thread handles exactly one task.
// For a 1-qubit gate (simd_s=1) on a 3-qubit sv: n_tasks = 2^(3-1-1) = 2.
#[test]
fn jit_multithreaded_thread_count_equals_task_count() {
    run_jit_and_compare_full(
        &QuantumGate::h(0),
        3,
        default_spec(Precision::F64, SimdWidth::W128),
        2,
        1e-10,
    );
}

// Thread count exceeds n_tasks: clamping logic must prevent empty threads from
// corrupting the statevector. Same setup as above (n_tasks=2), 8 threads → clamped to 2.
#[test]
fn jit_multithreaded_thread_count_exceeds_task_count() {
    run_jit_and_compare_full(
        &QuantumGate::h(0),
        3,
        default_spec(Precision::F64, SimdWidth::W128),
        8,
        1e-10,
    );
}

// ── End-to-end circuits with and without fusion ─────────────────────────

#[test]
fn e2e_bell_style_circuit_matches_with_and_without_fusion() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::rz(0.3, 1));
    graph.insert_gate(QuantumGate::cx(1, 2));
    graph.insert_gate(QuantumGate::h(2));

    assert_fused_and_unfused_circuits_agree(
        &graph,
        5,
        default_spec(Precision::F64, SimdWidth::W128),
        1e-10,
    );
}

#[test]
fn e2e_mixed_parametric_circuit_matches_with_and_without_fusion() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::u3(0.7, 0.2, -0.4, 0));
    graph.insert_gate(QuantumGate::rx(0.5, 2));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::swap(1, 2));
    graph.insert_gate(QuantumGate::rz(-0.9, 0));
    graph.insert_gate(QuantumGate::ccx(0, 2, 3));
    graph.insert_gate(QuantumGate::ry(0.25, 1));

    assert_fused_and_unfused_circuits_agree(
        &graph,
        6,
        default_spec(Precision::F64, SimdWidth::W128),
        1e-10,
    );
}

#[test]
fn e2e_brick_wall_circuit_matches_with_and_without_fusion() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::cx(2, 3));
    graph.insert_gate(QuantumGate::cx(4, 5));
    graph.insert_gate(QuantumGate::cx(1, 2));
    graph.insert_gate(QuantumGate::cx(3, 4));
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::rz(0.2, 5));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::cx(2, 3));
    graph.insert_gate(QuantumGate::cx(4, 5));

    assert_fused_and_unfused_circuits_agree(
        &graph,
        7,
        default_spec(Precision::F64, SimdWidth::W128),
        1e-10,
    );
}

#[test]
fn e2e_mixed_parametric_circuit_w512_matches_with_and_without_fusion() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::u3(0.7, 0.2, -0.4, 0));
    graph.insert_gate(QuantumGate::rx(0.5, 2));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::swap(1, 2));
    graph.insert_gate(QuantumGate::rz(-0.9, 0));
    graph.insert_gate(QuantumGate::ccx(0, 2, 3));
    graph.insert_gate(QuantumGate::ry(0.25, 1));
    graph.insert_gate(QuantumGate::cx(3, 4));
    graph.insert_gate(QuantumGate::h(5));

    // W512/F64 uses simd_s=3, so give the fused path extra headroom.
    assert_fused_and_unfused_circuits_agree(
        &graph,
        9,
        default_spec(Precision::F64, SimdWidth::W512),
        1e-10,
    );
}

#[test]
fn e2e_brick_wall_circuit_w512_matches_with_and_without_fusion() {
    let mut graph = CircuitGraph::new();
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::cx(2, 3));
    graph.insert_gate(QuantumGate::cx(4, 5));
    graph.insert_gate(QuantumGate::cx(6, 7));
    graph.insert_gate(QuantumGate::cx(1, 2));
    graph.insert_gate(QuantumGate::cx(3, 4));
    graph.insert_gate(QuantumGate::cx(5, 6));
    graph.insert_gate(QuantumGate::h(0));
    graph.insert_gate(QuantumGate::rz(0.2, 7));
    graph.insert_gate(QuantumGate::cx(0, 1));
    graph.insert_gate(QuantumGate::cx(2, 3));
    graph.insert_gate(QuantumGate::cx(4, 5));
    graph.insert_gate(QuantumGate::cx(6, 7));

    // Use a larger circuit and statevector so W512 exercises fused kernels
    // that need more than the minimal W128-style headroom.
    assert_fused_and_unfused_circuits_agree(
        &graph,
        10,
        default_spec(Precision::F64, SimdWidth::W512),
        1e-10,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel-generator bit-layout coverage
//
// Pins the numerical output of the CPU JIT across every bit-layout
// shape the generator can produce and every supported {precision,
// simd_width} combination.  Serves as a safety net for refactors of
// emit_load_amplitudes / emit_merge_and_store (src/cpp/cpu/cpu_gen.cpp)
// — e.g. tiling the lo-heavy `<vec_size x double>` mega-load into
// native-width aligned loads so LLVM type legalisation stops inflating
// compile time.  Reference is `CPUStatevector::apply_gate` (scalar).
// ═══════════════════════════════════════════════════════════════════════

mod layout_coverage {
    use super::*;
    use crate::types::ComplexSquareMatrix;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // ── Seed-stable gate constructors ────────────────────────────────

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn random_unitary(qubits: &[u32], seed: u64) -> QuantumGate {
        let mut rng = seeded_rng(seed);
        QuantumGate::random_unitary_with_rng(qubits, &mut rng)
    }

    fn random_sparse(qubits: &[u32], sparsity: f64, seed: u64) -> QuantumGate {
        let mut rng = seeded_rng(seed);
        QuantumGate::random_sparse_with_rng(qubits, sparsity, &mut rng)
    }

    fn identity_gate(qubits: &[u32]) -> QuantumGate {
        let n = 1usize << qubits.len();
        let mut data = vec![Complex::default(); n * n];
        for i in 0..n {
            data[i * n + i] = Complex::new(1.0, 0.0);
        }
        QuantumGate::new(ComplexSquareMatrix::from_vec(n, data), qubits.to_vec())
    }

    fn zero_gate(qubits: &[u32]) -> QuantumGate {
        let n = 1usize << qubits.len();
        let data = vec![Complex::default(); n * n];
        QuantumGate::new(ComplexSquareMatrix::from_vec(n, data), qubits.to_vec())
    }

    // Random diagonal unitary: |d_ii| = 1, off-diagonals exactly 0.
    // Stresses structural-zero detection (ztol) uniformly.
    fn diagonal_unitary(qubits: &[u32], seed: u64) -> QuantumGate {
        let mut rng = seeded_rng(seed);
        let n = 1usize << qubits.len();
        let mut data = vec![Complex::default(); n * n];
        for i in 0..n {
            let theta: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            data[i * n + i] = Complex::new(theta.cos(), theta.sin());
        }
        QuantumGate::new(ComplexSquareMatrix::from_vec(n, data), qubits.to_vec())
    }

    // Random signed-permutation unitary: exactly one ±1 per row/column.
    // Every non-zero entry must fold to the ±1 branch.
    fn permutation_gate(qubits: &[u32], seed: u64) -> QuantumGate {
        let mut rng = seeded_rng(seed);
        let n = 1usize << qubits.len();
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let mut data = vec![Complex::default(); n * n];
        for row in 0..n {
            let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            data[row * n + perm[row]] = Complex::new(sign, 0.0);
        }
        QuantumGate::new(ComplexSquareMatrix::from_vec(n, data), qubits.to_vec())
    }

    // Conjugate transpose (unitary inverse when the input is unitary).
    fn dagger(g: &QuantumGate) -> QuantumGate {
        let edge = g.matrix().edge_size();
        let src = g.matrix();
        let mut data = vec![Complex::default(); edge * edge];
        for r in 0..edge {
            for c in 0..edge {
                let v = src.get(r, c);
                data[c * edge + r] = Complex::new(v.re, -v.im);
            }
        }
        QuantumGate::new(
            ComplexSquareMatrix::from_vec(edge, data),
            g.qubits().to_vec(),
        )
    }

    // ── Spec matrix ─────────────────────────────────────────────────

    // All six {precision, simd_width} combinations the CPU path supports.
    fn all_specs() -> [(Precision, SimdWidth, f64); 6] {
        [
            (Precision::F64, SimdWidth::W128, 1e-10),
            (Precision::F64, SimdWidth::W256, 1e-10),
            (Precision::F64, SimdWidth::W512, 1e-10),
            (Precision::F32, SimdWidth::W128, 5e-5),
            (Precision::F32, SimdWidth::W256, 5e-5),
            (Precision::F32, SimdWidth::W512, 5e-5),
        ]
    }

    // Statevector size generous enough that every spec variant's
    // sep_bit fits (simd_s ∈ [1, 4]).
    fn sv_size_for(qubits: &[u32]) -> u32 {
        let max_q = qubits.iter().copied().max().unwrap_or(0);
        let k = qubits.len() as u32;
        (max_q + 4).max(k + 6)
    }

    fn check_one(
        gate: &QuantumGate,
        n_qubits_sv: u32,
        precision: Precision,
        simd_width: SimdWidth,
        tol: f64,
        n_threads: u32,
        check_norm: bool,
    ) {
        let spec = default_spec(precision, simd_width);
        let gate_arc = Arc::new(gate.clone());
        let mgr = CpuKernelManager::new();
        let id = mgr.generate_gate(spec, &gate_arc).expect("generate kernel");

        let mut sv_jit = seeded_statevector(n_qubits_sv, precision, simd_width);
        let mut sv_ref = sv_jit.clone();

        sv_ref.apply_gate(&gate_arc);
        mgr.apply(id, &mut sv_jit, n_threads).expect("apply kernel");

        assert_statevectors_close(&sv_jit, &sv_ref, tol);
        if check_norm {
            assert!(
                (sv_jit.norm() - 1.0).abs() < tol,
                "unit-norm violated ({:?}/{:?}): norm={}",
                precision,
                simd_width,
                sv_jit.norm()
            );
        }
    }

    // Run the gate against every spec variant at n_threads=1.
    fn check_all_specs(gate: &QuantumGate, check_norm: bool) {
        let n_qubits_sv = sv_size_for(gate.qubits());
        for (precision, simd_width, tol) in all_specs() {
            check_one(gate, n_qubits_sv, precision, simd_width, tol, 1, check_norm);
        }
    }

    // Single-spec convenience: production default (F64, W512).
    // Used when we want broad scenario coverage without paying the
    // 6× compile cost of sweeping all specs.
    fn check_default_spec(gate: &QuantumGate, check_norm: bool) {
        check_one(
            gate,
            sv_size_for(gate.qubits()),
            Precision::F64,
            SimdWidth::W512,
            1e-10,
            1,
            check_norm,
        );
    }

    // ── Shape A: pure lo-heavy, contiguous targets q=[0..k) ─────────
    //
    // This is the bit-layout shape that produces the `<vec_size x f64>`
    // mega-load in `emit_load_amplitudes`.  lk=k, hk=0.

    #[test]
    fn shape_all_lo_contiguous_k_sweep_f64_w512() {
        // k-sweep at a single spec to keep runtime bounded; F64 W512
        // is the production default and has simd_s=3.
        for k in 2..=5u32 {
            let qs: Vec<u32> = (0..k).collect();
            let gate = random_unitary(&qs, 0x1_0000 + k as u64);
            check_one(
                &gate,
                sv_size_for(&qs),
                Precision::F64,
                SimdWidth::W512,
                1e-10,
                1,
                true,
            );
        }
    }

    #[test]
    fn shape_all_lo_contiguous_k5_every_spec() {
        // Largest lo-heavy size under the k≤5 coverage cap; full spec matrix.
        let qs: Vec<u32> = (0..5).collect();
        let gate = random_unitary(&qs, 0x1_A005);
        check_all_specs(&gate, true);
    }

    #[test]
    fn shape_all_lo_contiguous_k4_every_spec() {
        // Mid-size lo-heavy — runs fast enough to cover all six specs.
        let qs: Vec<u32> = (0..4).collect();
        let gate = random_unitary(&qs, 0x1_A004);
        check_all_specs(&gate, true);
    }

    // ── Shape B: all-lo with gaps (lk=k still, sep_bit varies) ──────

    #[test]
    fn shape_all_lo_with_gaps() {
        // Single spec to keep runtime bounded — each scenario is a
        // 5-qubit gate whose compile dominates.
        let scenarios: &[&[u32]] = &[
            &[0, 1, 2, 3, 5], // one gap at position 4
            &[0, 1, 2, 4, 5], // gap inside the target span
            &[0, 1, 3, 4, 6], // two interior gaps
            &[0, 1, 2, 3, 6], // one wide gap
        ];
        for (i, qs) in scenarios.iter().enumerate() {
            let gate = random_unitary(qs, 0x2_0000 + i as u64);
            check_default_spec(&gate, true);
        }
    }

    // ── Shape C: pure all-hi (existing fast path — regression guard) ─

    #[test]
    fn shape_all_hi_contiguous() {
        // First target ≥ simd_s on every variant (simd_s ≤ 4).
        // These compile fast (~300 ms), so every spec is affordable.
        let scenarios: &[&[u32]] = &[&[5, 6, 7, 8, 9], &[6, 7, 8, 9, 10], &[5], &[5, 6]];
        for (i, qs) in scenarios.iter().enumerate() {
            let gate = random_unitary(qs, 0x3_0000 + i as u64);
            check_all_specs(&gate, true);
        }
    }

    // ── Shape D: mixed layouts, 0 < lk < k ───────────────────────────

    #[test]
    fn shape_mixed_lk_varies() {
        let scenarios: &[&[u32]] = &[
            &[0, 5, 6, 7, 8], // lk=1 on simd_s ≥ 1
            &[0, 1, 5, 6, 7], // lk=2 on simd_s ≥ 2
            &[0, 1, 2, 5, 6], // lk=3 on simd_s ≥ 3
            &[0, 2, 4, 6, 8], // spread, lk varies per simd_s
        ];
        for (i, qs) in scenarios.iter().enumerate() {
            let gate = random_unitary(qs, 0x4_0000 + i as u64);
            check_default_spec(&gate, true);
        }
    }

    // ── Shape E: targets straddling the simd_s boundary ──────────────
    //
    // Different simd_s values place the same qubit on different sides
    // of the boundary.  Single- and two-qubit cases exercise every
    // placement relative to simd_s ∈ {1, 2, 3, 4}.

    #[test]
    fn shape_straddles_simd_boundary_1q() {
        for pos in [0u32, 1, 2, 3, 4, 5] {
            let gate = random_unitary(&[pos], 0x5_0000 + pos as u64);
            check_all_specs(&gate, true);
        }
    }

    #[test]
    fn shape_straddles_simd_boundary_2q() {
        let scenarios: &[&[u32]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 4], &[0, 4], &[1, 5]];
        for (i, qs) in scenarios.iter().enumerate() {
            let gate = random_unitary(qs, 0x6_0000 + i as u64);
            check_all_specs(&gate, true);
        }
    }

    // ── Matrix content classes on the worst-case lo-heavy layout ────

    #[test]
    fn content_identity_on_all_lo_k5() {
        // Identity → every off-diagonal = 0 folded by InstCombine, so
        // this runs fast enough to sweep all specs.
        let qs: Vec<u32> = (0..5).collect();
        check_all_specs(&identity_gate(&qs), true);
    }

    #[test]
    fn content_diagonal_on_all_lo_k5() {
        // Diagonal unitary — off-diagonals are structural zeros; fast.
        let qs: Vec<u32> = (0..5).collect();
        check_all_specs(&diagonal_unitary(&qs, 0x7_D001), true);
    }

    #[test]
    fn content_permutation_on_all_lo_k5() {
        // Permutation — one ±1 per row, exercises the otol branch.
        let qs: Vec<u32> = (0..5).collect();
        check_all_specs(&permutation_gate(&qs, 0x7_D002), true);
    }

    #[test]
    fn content_sparse_densities_on_all_lo_k5() {
        // Sparsity triggers the zero-fold path in the kernel generator.
        // random_sparse is NOT unitary, so we skip the norm check.
        let qs: Vec<u32> = (0..5).collect();
        for (i, &d) in [0.5f64, 0.25, 0.1].iter().enumerate() {
            let gate = random_sparse(&qs, d, 0x7_5000 + i as u64);
            check_default_spec(&gate, false);
        }
    }

    #[test]
    fn content_zero_matrix_on_all_lo_k5() {
        // Degenerate: every output amp must be 0.  Guards against
        // NaN/inf leakage in the accumulator-init path.
        let qs: Vec<u32> = (0..5).collect();
        let gate = zero_gate(&qs);
        let n_qubits_sv = sv_size_for(&qs);
        for (precision, simd_width, tol) in all_specs() {
            check_one(&gate, n_qubits_sv, precision, simd_width, tol, 1, false);
        }
    }

    // ── StackLoad mode over the worst-case lo-heavy layout ──────────

    #[test]
    fn stack_load_mode_on_all_lo_k5() {
        let qs: Vec<u32> = (0..5).collect();
        let gate_arc = Arc::new(random_unitary(&qs, 0x8_F001));
        let n_qubits_sv = sv_size_for(&qs);
        for (precision, simd_width, tol) in all_specs() {
            let spec = CPUKernelGenSpec {
                precision,
                simd_width,
                mode: MatrixLoadMode::StackLoad,
                ztol: match precision {
                    Precision::F32 => 1e-6,
                    Precision::F64 => 1e-12,
                },
                otol: match precision {
                    Precision::F32 => 1e-6,
                    Precision::F64 => 1e-12,
                },
            };
            let mgr = CpuKernelManager::new();
            let id = mgr.generate_gate(spec, &gate_arc).expect("generate");
            let mut sv_jit = seeded_statevector(n_qubits_sv, precision, simd_width);
            let mut sv_ref = sv_jit.clone();
            sv_ref.apply_gate(&gate_arc);
            mgr.apply(id, &mut sv_jit, 1).expect("apply");
            assert_statevectors_close(&sv_jit, &sv_ref, tol);
            assert!((sv_jit.norm() - 1.0).abs() < tol);
        }
    }

    // ── Multi-threaded execution on the worst-case layout ────────────

    #[test]
    fn multi_thread_on_all_lo_k5() {
        let qs: Vec<u32> = (0..5).collect();
        let gate = random_unitary(&qs, 0x9_0001);
        let n_qubits_sv = sv_size_for(&qs);
        for &n_threads in &[1u32, 2, 4, 8] {
            check_one(
                &gate,
                n_qubits_sv,
                Precision::F64,
                SimdWidth::W512,
                1e-10,
                n_threads,
                true,
            );
        }
    }

    // ── U · U† = I round-trip on the worst-case layout ───────────────

    #[test]
    fn unitary_dagger_roundtrip_on_all_lo_k5() {
        let qs: Vec<u32> = (0..5).collect();
        let u = random_unitary(&qs, 0xA_0001);
        let ud = dagger(&u);

        let spec = default_spec(Precision::F64, SimdWidth::W512);
        let u_arc = Arc::new(u);
        let ud_arc = Arc::new(ud);
        let mgr = CpuKernelManager::new();
        let id_u = mgr.generate_gate(spec, &u_arc).expect("gen u");
        let id_ud = mgr.generate_gate(spec, &ud_arc).expect("gen ud");

        let n_qubits_sv = sv_size_for(u_arc.qubits());
        let sv_start = seeded_statevector(n_qubits_sv, spec.precision, spec.simd_width);
        let mut sv = sv_start.clone();
        mgr.apply(id_u, &mut sv, 1).expect("apply u");
        mgr.apply(id_ud, &mut sv, 1).expect("apply ud");
        assert_statevectors_close(&sv, &sv_start, 1e-10);
    }

    // ── Minimum SV size (edge of valid input) ────────────────────────
    //
    // Exercises the smallest SV that still produces a non-zero task
    // count and the next few sizes up — boundary bugs in the
    // sv-base-pointer math tend to appear at the minimum.

    #[test]
    fn minimum_sv_size_edges() {
        let qs: Vec<u32> = (0..5).collect();
        let gate = random_unitary(&qs, 0xB_0001);
        for n_qubits_sv in [8u32, 9, 10, 11] {
            check_one(
                &gate,
                n_qubits_sv,
                Precision::F64,
                SimdWidth::W512,
                1e-10,
                1,
                true,
            );
        }
    }

    // ── Multiple random seeds (property-style robustness) ────────────

    #[test]
    fn random_seed_robustness_all_lo_k5() {
        // Eight independent random unitaries — catches bugs that depend
        // on specific matrix patterns (e.g. accidental ±1 coincidence
        // within ImmValue ztol/otol tolerances).
        let qs: Vec<u32> = (0..5).collect();
        for seed in 0..8u64 {
            let gate = random_unitary(&qs, 0xC_0000 + seed);
            check_one(
                &gate,
                sv_size_for(&qs),
                Precision::F64,
                SimdWidth::W512,
                1e-10,
                1,
                true,
            );
        }
    }

    // ── Kernel-manager dedup must return the same numerical result ──

    #[test]
    fn dedup_returns_identical_results() {
        let qs: Vec<u32> = (0..5).collect();
        let gate = Arc::new(random_unitary(&qs, 0xD_0001));
        let spec = default_spec(Precision::F64, SimdWidth::W512);
        let mgr = CpuKernelManager::new();
        let id_a = mgr.generate_gate(spec, &gate).expect("generate a");
        let id_b = mgr.generate_gate(spec, &gate).expect("generate b");
        // Dedup should have returned the same kernel id.
        assert_eq!(id_a, id_b);

        let mut sv_a =
            seeded_statevector(sv_size_for(gate.qubits()), spec.precision, spec.simd_width);
        let mut sv_b = sv_a.clone();
        mgr.apply(id_a, &mut sv_a, 1).expect("apply a");
        mgr.apply(id_b, &mut sv_b, 1).expect("apply b");
        // Bit-identical: dedup maps both ids to the same compiled kernel.
        for idx in 0..sv_a.len() {
            assert_eq!(sv_a.amp(idx), sv_b.amp(idx));
        }
    }
}

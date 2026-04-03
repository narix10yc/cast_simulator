#[cfg(test)]
mod tests {

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
        let mgr = CpuKernelManager::new(spec);
        let kernel_id = mgr.generate(&gate).expect("generate kernel");

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
        let mgr = CpuKernelManager::new(spec);
        let mut kernel_ids = Vec::with_capacity(gates.len());
        for gate in &gates {
            let kernel_id = mgr.generate(gate).expect("generate kernel");
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
        let mgr = CpuKernelManager::new(spec);
        let kernel_id = mgr.generate(&gate).expect("generate kernel");

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

        let mgr_imm = CpuKernelManager::new(spec_imm);
        let kid_imm = mgr_imm.generate(&gate).expect("generate imm kernel");
        mgr_imm.apply(kid_imm, &mut sv_imm, 1).expect("apply imm");

        let mgr_stack = CpuKernelManager::new(spec_stack);
        let kid_stack = mgr_stack.generate(&gate).expect("generate stack kernel");
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

        let mgr = CpuKernelManager::new(spec);
        let kid_h = mgr.generate(&h_gate).expect("generate H kernel");
        let kid_cx = mgr.generate(&cx_gate).expect("generate CX kernel");

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
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        let kid = mgr
            .generate_with_diagnostics(&Arc::new(QuantumGate::h(0)), true, false)
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
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        let kid = mgr
            .generate_with_diagnostics(&Arc::new(QuantumGate::x(0)), true, false)
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

        let mgr = CpuKernelManager::new(spec);
        let kid = mgr
            .generate_with_diagnostics(&gate, true, false)
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

        let mgr = CpuKernelManager::new(spec);
        let kid_h = mgr
            .generate_with_diagnostics(&h_gate, true, false)
            .expect("H kernel");
        let kid_cx = mgr
            .generate_with_diagnostics(&cx_gate, true, false)
            .expect("CX kernel");

        let ir_h = mgr.emit_ir(kid_h).expect("H IR");
        let ir_cx = mgr.emit_ir(kid_cx).expect("CX IR");

        assert_ne!(ir_h, ir_cx, "different gates must produce different IR");
    }

    #[test]
    fn emit_ir_returns_none_without_diagnostics() {
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        let kid = mgr
            .generate(&Arc::new(QuantumGate::h(0)))
            .expect("generate kernel");
        assert!(
            mgr.emit_ir(kid).is_none(),
            "IR should be None without diagnostics"
        );
    }

    #[test]
    fn emit_ir_returns_none_for_unknown_kernel_id() {
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        assert!(
            mgr.emit_ir(9999).is_none(),
            "unknown kernel_id should return None"
        );
    }

    // ── emit_asm ──────────────────────────────────────────────────────────────

    fn compile_h_manager_with_asm() -> (CpuKernelManager, KernelId) {
        let gate = Arc::new(QuantumGate::h(0));
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        let kid = mgr
            .generate_with_diagnostics(&gate, false, true)
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

        let mgr = CpuKernelManager::new(spec);
        let kid_h = mgr
            .generate_with_diagnostics(&h_gate, false, true)
            .expect("H kernel");
        let kid_cx = mgr
            .generate_with_diagnostics(&cx_gate, false, true)
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

        let mgr = CpuKernelManager::new(spec);
        let kid = mgr
            .generate_with_diagnostics(&gate, true, true)
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
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
        let kid = mgr.generate(&gate).expect("generate kernel");
        assert!(
            mgr.emit_asm(kid).is_none(),
            "emit_asm should be None without diagnostics"
        );
    }

    #[test]
    fn emit_asm_returns_none_for_unknown_kernel_id() {
        let mgr = CpuKernelManager::new(default_spec(Precision::F64, SimdWidth::W128));
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
}

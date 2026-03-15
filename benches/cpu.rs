//! CPU kernel generation and JIT compilation benchmarks.
//!
//! All groups measure one-time latency costs paid before the first gate apply,
//! not steady-state simulation throughput.
//!
//! ## Groups
//!
//! | Group | What it measures |
//! |-------|-----------------|
//! | `ir_gen` | LLVM IR generation only (`generate`), no compilation |
//! | `jit_compile` | End-to-end: `generate` + `init_jit` for one kernel |
//! | `jit_compile_batch` | `generate` × N + `init_jit` for N kernels in one session |
//!
//! The LLVM compilation cost for a single kernel is approximately:
//! `jit_compile` − `ir_gen`.
//!
//! The batch group shows how compilation cost scales with session size and
//! whether LLVM parallelises work across kernels.
//!
//! ## Usage
//!
//! ```sh
//! cargo bench                          # all groups
//! cargo bench --bench cpu ir_gen       # IR generation only
//! cargo bench --bench cpu jit_compile  # end-to-end compile latency
//! cargo bench --bench cpu batch        # batch compile scaling
//! ```

use cast::cpu::{CPUKernelGenSpec, CPUKernelGenerator, CPUStatevector, Precision, SimdWidth};
use cast::types::QuantumGate;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// ── Shared fixtures ───────────────────────────────────────────────────────────

/// Gates that represent qualitatively different code-generation paths:
///
/// - `H`       : 1-qubit, dense (all entries non-zero)
/// - `Rx(π/3)` : 1-qubit, dense, no ±1 entries → exercises FMA paths
/// - `CX`      : 2-qubit, sparse (many zeros)
/// - `CCX`     : 3-qubit, sparse (Toffoli) — largest standard gate
fn gates() -> Vec<(&'static str, QuantumGate)> {
    vec![
        ("H", QuantumGate::h(0)),
        ("Rx(pi/3)", QuantumGate::rx(std::f64::consts::PI / 3.0, 0)),
        ("CX", QuantumGate::cx(0, 1)),
        ("CCX", QuantumGate::ccx(0, 1, 2)),
    ]
}

/// Specs to benchmark. F64/W128 is the primary production configuration;
/// F32/W128 is included to show precision impact on IR size and compile time.
fn specs() -> Vec<(&'static str, CPUKernelGenSpec)> {
    vec![
        ("f64/w128", CPUKernelGenSpec::f64()),
        ("f32/w128", CPUKernelGenSpec::f32()),
    ]
}

// ── IR generation ─────────────────────────────────────────────────────────────

/// Cost of LLVM IR construction only (`CPUKernelGenerator::new` + `generate`).
///
/// `init_jit` is deliberately omitted so this measures only the Rust-side
/// graph walk and LLVM IR builder calls, without any LLVM optimisation or
/// machine-code emission.
fn bench_ir_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("ir_gen");

    for (gate_name, gate) in gates() {
        let matrix = gate.matrix().data().to_vec();
        let qubits = gate.qubits().to_vec();

        for (spec_name, spec) in specs() {
            group.bench_function(format!("{gate_name}/{spec_name}"), |b| {
                b.iter(|| {
                    let mut gen = CPUKernelGenerator::new().expect("create generator");
                    gen.generate(&spec, &matrix, &qubits).expect("generate");
                    // Drop without compiling — measures pure IR construction.
                });
            });
        }
    }

    group.finish();
}

// ── Single-kernel compile ─────────────────────────────────────────────────────

/// End-to-end latency for a single kernel: `new + generate + init_jit`.
///
/// This is the cold-start cost a simulator pays the first time it encounters
/// a new gate shape. Subtract `ir_gen` to isolate the LLVM backend time.
fn bench_jit_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");
    group.sample_size(10);

    for (gate_name, gate) in gates() {
        let matrix = gate.matrix().data().to_vec();
        let qubits = gate.qubits().to_vec();

        for (spec_name, spec) in specs() {
            group.bench_function(format!("{gate_name}/{spec_name}"), |b| {
                b.iter(|| {
                    let mut gen = CPUKernelGenerator::new().expect("create generator");
                    gen.generate(&spec, &matrix, &qubits).expect("generate");
                    gen.init_jit().expect("init jit")
                    // JitSession dropped here; drop latency is negligible.
                });
            });
        }
    }

    group.finish();
}

// ── Batch compile ─────────────────────────────────────────────────────────────

/// Compile `n_kernels` distinct kernels in a single `init_jit` call.
///
/// All kernels in a session are compiled in parallel by LLVM's concurrent
/// compile thread pool. This group shows how per-kernel compile time changes
/// as the session grows, and how well LLVM parallelises across kernels.
///
/// Each slot `i` gets a unique gate to prevent LLVM from deduplicating IR.
fn make_batch_gates(n: usize) -> Vec<QuantumGate> {
    // Cycle through a variety of single- and two-qubit gates so every kernel
    // has distinct IR (different matrix immediates or different qubit counts).
    let angles: Vec<f64> = (0..n)
        .map(|i| std::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64)
        .collect();
    angles
        .iter()
        .enumerate()
        .map(|(i, &theta)| {
            if i % 3 == 0 {
                QuantumGate::rx(theta, 0)
            } else if i % 3 == 1 {
                QuantumGate::ry(theta, 0)
            } else {
                QuantumGate::rz(theta, 0)
            }
        })
        .collect()
}

fn bench_jit_compile_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile_batch");
    group.sample_size(10);

    let spec = CPUKernelGenSpec::f64();
    // A statevector is needed only to validate the minimum qubit requirement;
    // use a small one just to check the spec is valid for these gates.
    let _ = CPUStatevector::new(3, Precision::F64, SimdWidth::W128);

    for &n_kernels in &[1usize, 4, 16] {
        let gates = make_batch_gates(n_kernels);
        let prepared: Vec<_> = gates
            .iter()
            .map(|g| (g.matrix().data().to_vec(), g.qubits().to_vec()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("f64/w128", n_kernels),
            &n_kernels,
            |b, _| {
                b.iter(|| {
                    let mut gen = CPUKernelGenerator::new().expect("create generator");
                    for (matrix, qubits) in &prepared {
                        gen.generate(&spec, matrix, qubits).expect("generate");
                    }
                    gen.init_jit().expect("init jit")
                });
            },
        );
    }

    group.finish();
}

// ── Entry point ───────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_ir_gen,
    bench_jit_compile,
    bench_jit_compile_batch
);
criterion_main!(benches);

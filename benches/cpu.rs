//! CPU gate simulation benchmarks.
//!
//! Reports two metrics:
//!
//! - **Memory update speed** (GB/s via Criterion throughput): bytes read + written per
//!   second while applying a gate, divided by wall time. Since every amplitude is read
//!   once and written once, `bytes = 2 × statevector_byte_size`. This value is
//!   independent of statevector size: a bandwidth-limited kernel should plateau at the
//!   same GB/s regardless of `n_qubits`, while a compute-limited one will drop as the
//!   working set grows past each cache level.
//!
//! - **JIT compile overhead**: one-time latency of IR generation + LLVM compilation
//!   (`CPUKernelGenerator::new` + `generate` + `init_jit`) per gate kernel.
//!
//! ## Qubit counts
//!
//! | n_qubits | sv size (F64) | sv size (F32) | Likely location |
//! |----------|--------------|--------------|-----------------|
//! | 14       | 256 KB       | 128 KB       | L2/L3 cache     |
//! | 18       |   4 MB       |   2 MB       | L3 cache        |
//! | 20       |  16 MB       |   8 MB       | DRAM            |
//!
//! ## Usage
//!
//! ```sh
//! cargo bench                        # all groups
//! cargo bench --bench cpu scalar     # scalar group only
//! cargo bench --bench cpu jit_1t     # JIT single-thread only
//! ```

use cast::cpu::{CPUKernelGenSpec, CPUKernelGenerator, CPUStatevector};
use cast::types::QuantumGate;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ── Shared fixtures ───────────────────────────────────────────────────────────

/// Qubit counts sweeping L2 cache → L3 cache → DRAM.
const N_QUBITS: &[usize] = &[14, 18, 20];

/// Gates to benchmark.
///   - H  : dense 1-qubit (all entries non-zero → no sparsity optimisation)
///   - CX : sparse 2-qubit (many zeros → exercises the ImmValue zero-tolerance path)
fn gates() -> Vec<(&'static str, QuantumGate)> {
    vec![
        ("H(q0)", QuantumGate::h(0)),
        ("CX(q0,q1)", QuantumGate::cx(0, 1)),
    ]
}

/// Precision × SIMD-width specs to benchmark.
fn specs() -> Vec<(&'static str, CPUKernelGenSpec)> {
    vec![
        ("f64", CPUKernelGenSpec::f64()),
        ("f32", CPUKernelGenSpec::f32()),
    ]
}

/// Total bytes read + written when applying any k-qubit gate to a statevector.
///
/// Every amplitude is read once and written once, so the transfer is 2 × the full
/// statevector buffer (real + imaginary scalars for each amplitude).
fn bytes_rw(sv: &CPUStatevector) -> u64 {
    (sv.byte_len() * 2) as u64
}

// ── Scalar reference path ─────────────────────────────────────────────────────

/// `CPUStatevector::apply_gate` — pure Rust, no JIT.
///
/// Useful as a baseline to quantify the JIT speedup.
fn bench_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");

    for (gate_name, gate) in gates() {
        for (spec_name, spec) in specs() {
            for &n in N_QUBITS {
                let mut sv = CPUStatevector::new(n, spec.precision, spec.simd_width);
                sv.initialize();

                group.throughput(Throughput::Bytes(bytes_rw(&sv)));
                group.bench_with_input(
                    BenchmarkId::new(format!("{gate_name}/{spec_name}"), n),
                    &n,
                    |b, _| b.iter(|| sv.apply_gate(&gate)),
                );
            }
        }
    }

    group.finish();
}

// ── JIT kernel path ───────────────────────────────────────────────────────────

/// `JitSession::apply` with a single worker thread.
///
/// Isolates per-core memory throughput; compare with `jit_mt` to see parallel scaling.
fn bench_jit_1t(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_1t");

    for (gate_name, gate) in gates() {
        for (spec_name, spec) in specs() {
            // Compile once; reuse across all n_qubits for this (gate, spec) pair.
            let mut gen = CPUKernelGenerator::new().expect("create generator");
            let kid = gen
                .generate(&spec, gate.matrix().data(), gate.qubits())
                .expect("generate kernel");
            let mut jit = gen.init_jit().expect("init jit");

            for &n in N_QUBITS {
                let mut sv = CPUStatevector::new(n, spec.precision, spec.simd_width);
                sv.initialize();

                group.throughput(Throughput::Bytes(bytes_rw(&sv)));
                group.bench_with_input(
                    BenchmarkId::new(format!("{gate_name}/{spec_name}"), n),
                    &n,
                    |b, _| b.iter(|| jit.apply(kid, &mut sv, Some(1)).expect("apply")),
                );
            }
        }
    }

    group.finish();
}

/// `JitSession::apply` letting the C++ side pick thread count (`hardware_concurrency`).
///
/// Compare GB/s with `jit_1t` to evaluate parallel scaling efficiency.
fn bench_jit_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_mt");

    for (gate_name, gate) in gates() {
        for (spec_name, spec) in specs() {
            let mut gen = CPUKernelGenerator::new().expect("create generator");
            let kid = gen
                .generate(&spec, gate.matrix().data(), gate.qubits())
                .expect("generate kernel");
            let mut jit = gen.init_jit().expect("init jit");

            for &n in N_QUBITS {
                let mut sv = CPUStatevector::new(n, spec.precision, spec.simd_width);
                sv.initialize();

                group.throughput(Throughput::Bytes(bytes_rw(&sv)));
                group.bench_with_input(
                    BenchmarkId::new(format!("{gate_name}/{spec_name}"), n),
                    &n,
                    // None → C++ calls hardware_concurrency()
                    |b, _| b.iter(|| jit.apply(kid, &mut sv, None).expect("apply")),
                );
            }
        }
    }

    group.finish();
}

// ── JIT compile overhead ──────────────────────────────────────────────────────

/// One-time cost of LLVM IR generation + compilation per gate kernel.
///
/// Measures `CPUKernelGenerator::new` + `generate` + `init_jit` end-to-end.
/// This is the latency you pay once before the first `apply` call on a new gate shape.
fn bench_jit_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");
    // Compilation takes ~10–200 ms; reduce sample count so the suite finishes quickly.
    group.sample_size(10);

    for (gate_name, gate) in gates() {
        for (spec_name, spec) in specs() {
            // Extract matrix/qubits upfront so heap allocation isn't included.
            let matrix = gate.matrix().data().to_vec();
            let qubits = gate.qubits().to_vec();

            group.bench_function(format!("{gate_name}/{spec_name}"), |b| {
                b.iter(|| {
                    let mut gen = CPUKernelGenerator::new().expect("create generator");
                    gen.generate(&spec, &matrix, &qubits).expect("generate");
                    gen.init_jit().expect("init jit")
                    // JitSession is dropped here; its drop time (~free) is included but negligible.
                });
            });
        }
    }

    group.finish();
}

// ── Entry point ───────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_scalar,
    bench_jit_1t,
    bench_jit_mt,
    bench_jit_compile,
);
criterion_main!(benches);

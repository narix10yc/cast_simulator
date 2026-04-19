//! Microbenchmark: single-gate CPU kernel JIT compile time.
//!
//! Decomposes the per-kernel compile cost into three phases:
//!
//!   * **ir**        — LLVM IRBuilder emission in `cast_cpu_generate_kernel_ir`
//!     (runs during `CpuKernelManager::generate`).
//!   * **opt**       — O1 module pipeline (`cast_cpu_optimize_kernel_ir`).
//!   * **codegen**   — native code generation + LLJIT symbol resolution
//!     (`cast_cpu_jit_compile_kernel` → `jit.addIRModule`
//!     → `jit.lookup`).
//!
//! Measurement recipe (three fresh managers per rep, same matrix):
//!
//!   1. Plain `generate()`                         → `ir`
//!   2. `generate_with_diagnostics(ir=true)`       → `ir + opt`
//!   3. `generate() + emit_asm()`                  → `ir + opt + codegen`
//!
//! From which `opt = (2) - (1)` and `codegen = (3) - (2)`.
//!
//! The microbench supports two independent axes: gate size (`k`, auto-swept
//! by default) and either explicit target qubit placement (`--qubits`) or
//! matrix density (`--density`). Non-sweep modes run one row.

use cast::cpu::{CPUKernelGenSpec, CPUStatevector, CpuKernelManager, KernelGenRequest, SimdWidth};
use cast::timing::{fmt_duration, TimingStats};
use cast::types::{ComplexSquareMatrix, QuantumGate};
use clap::{Parser, ValueEnum};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliPrecision {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliSimd {
    Native,
    #[value(name = "128")]
    W128,
    #[value(name = "256")]
    W256,
    #[value(name = "512")]
    W512,
}

impl CliSimd {
    fn resolve(self) -> SimdWidth {
        match self {
            CliSimd::Native => cast::cpu::native_simd_width(),
            CliSimd::W128 => SimdWidth::W128,
            CliSimd::W256 => SimdWidth::W256,
            CliSimd::W512 => SimdWidth::W512,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Per-kernel CPU compile-time microbenchmark (ir / opt / codegen phases)")]
struct Args {
    /// Largest gate size to sweep (1..=max). Ignored when --qubits is given.
    #[arg(long, default_value_t = 6)]
    max_qubits: u32,

    /// Reps per row (fresh manager each rep; one extra untimed warmup).
    #[arg(long, default_value_t = 3)]
    reps: u32,

    #[arg(long, value_enum, default_value_t = CliPrecision::F64)]
    precision: CliPrecision,

    #[arg(long, value_enum, default_value_t = CliSimd::Native)]
    simd: CliSimd,

    /// Explicit target qubit positions, e.g. "0,1,2,3,4,5". Sorted ascending.
    /// When given, k-sweep is disabled and only one row is produced.
    #[arg(long, value_delimiter = ',')]
    qubits: Option<Vec<u32>>,

    /// Fraction of structurally non-zero entries (0, 1]. 1.0 = dense Haar
    /// unitary (default). <1.0 = sparse matrix from `random_sparse`.
    #[arg(long, default_value_t = 1.0)]
    density: f64,

    /// Force dense codegen (ztol = otol = 0).
    #[arg(long)]
    force_dense: bool,

    /// Write the optimized LLVM IR for each row to `<prefix>_<label>.ll`.
    #[arg(long)]
    dump_ir: Option<std::path::PathBuf>,

    /// Write the native assembly for each row to `<prefix>_<label>.s`.
    #[arg(long)]
    dump_asm: Option<std::path::PathBuf>,

    /// Budget (seconds) for the apply-time exec benchmark.  0 = skip.
    /// When > 0, every row additionally compiles the kernel, applies it
    /// to a statevector, and reports per-thread-count adaptive timings.
    #[arg(long, default_value_t = 0.0)]
    apply_budget: f64,

    /// Statevector size for exec measurement, in qubits.  Default 26 gives
    /// a 16 MB F64 statevector that exceeds L3 on typical workstation
    /// CPUs, so DRAM behavior is exercised.
    #[arg(long, default_value_t = 26)]
    sv_qubits: u32,

    /// Thread count(s) to sweep for the exec benchmark.  Accepts a
    /// comma-separated list, e.g. "1,4,8,32".  Fewer threads shift the
    /// workload toward the compute-bound regime.
    #[arg(long, value_delimiter = ',', default_value = "1,4,32")]
    threads: Vec<u32>,
}

struct Row {
    label: String,
    ir_ms: f64,
    opt_ms: f64,
    cg_ms: f64,
    total_ms: f64,
    ir_lines: usize,
    ir_text: Option<String>,
    asm_text: Option<String>,
    exec: Vec<(u32, TimingStats)>, // (thread_count, stats) when apply_budget > 0
}

fn make_gate(qubits: &[u32], density: f64) -> Arc<QuantumGate> {
    let gate = if density >= 1.0 {
        QuantumGate::random_unitary(qubits)
    } else {
        QuantumGate::random_sparse(qubits, density)
    };
    Arc::new(gate)
}

fn clone_gate(g: &QuantumGate) -> Arc<QuantumGate> {
    let m = ComplexSquareMatrix::from_vec(g.matrix().edge_size(), g.matrix().data().to_vec());
    Arc::new(QuantumGate::new(m, g.qubits().to_vec()))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure_runtime(
    spec: CPUKernelGenSpec,
    gate: &Arc<QuantumGate>,
    sv_qubits: u32,
    threads: &[u32],
    apply_budget_s: f64,
) -> anyhow::Result<Vec<(u32, TimingStats)>> {
    let mgr = CpuKernelManager::new();
    let id = mgr.generate_gate(spec, gate)?;
    // Force compile now so the first exec doesn't absorb finalize.
    let _ = mgr.emit_asm(id);

    let mut sv = CPUStatevector::new(sv_qubits, spec.precision, spec.simd_width);
    sv.randomize();

    let mut out = Vec::with_capacity(threads.len());
    for &n_threads in threads {
        // Untimed warmup to populate caches for this thread count.
        mgr.apply(id, &mut sv, n_threads)?;
        let stats = mgr.time_adaptive(id, &mut sv, n_threads, apply_budget_s)?;
        out.push((n_threads, stats));
    }
    Ok(out)
}

fn measure(
    spec: CPUKernelGenSpec,
    qubits: &[u32],
    density: f64,
    reps: u32,
    capture: bool,
    runtime_opts: Option<(u32, &[u32], f64)>, // (sv_qubits, threads, budget)
) -> anyhow::Result<Row> {
    // Warm-up — a cold LLVM InitializeNativeTarget + module ctx allocation
    // adds ~5 ms on the first call in the process.
    {
        let mgr = CpuKernelManager::new();
        let gate = make_gate(qubits, density);
        let id = mgr.generate_gate(spec, &gate)?;
        let _ = mgr.emit_asm(id);
    }

    let mut ir_samples = Vec::with_capacity(reps as usize);
    let mut ir_opt_samples = Vec::with_capacity(reps as usize);
    let mut total_samples = Vec::with_capacity(reps as usize);
    let mut ir_lines_samples = Vec::with_capacity(reps as usize);

    for _ in 0..reps {
        // Regenerate a fresh random matrix each rep so each phase sees a new
        // instance — filtering out any accidental cache effects from matrix
        // layout in memory.  The geometry (qubits, edge size) is fixed.
        let gate_proto = make_gate(qubits, density);

        // Phase A: IR emission only.
        {
            let mgr = CpuKernelManager::new();
            let gate = clone_gate(&gate_proto);
            let t0 = Instant::now();
            let _ = mgr.generate_gate(spec, &gate)?;
            ir_samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }

        // Phase B: IR emission + O1 optimization (+ IR capture to measure size).
        {
            let mgr = CpuKernelManager::new();
            let gate = clone_gate(&gate_proto);
            let t0 = Instant::now();
            let id = mgr.generate(KernelGenRequest::from_gate(spec, &gate).with_ir())?;
            ir_opt_samples.push(t0.elapsed().as_secs_f64() * 1e3);
            if let Some(ir) = mgr.emit_ir(id) {
                ir_lines_samples.push(ir.lines().count());
            }
        }

        // Phase C: IR emission + O1 + native codegen + JIT.
        {
            let mgr = CpuKernelManager::new();
            let gate = clone_gate(&gate_proto);
            let t0 = Instant::now();
            let id = mgr.generate_gate(spec, &gate)?;
            let _ = mgr.emit_asm(id);
            total_samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }

    let ir = median(&mut ir_samples);
    let ir_opt = median(&mut ir_opt_samples);
    let total = median(&mut total_samples);
    let ir_lines = ir_lines_samples.into_iter().max().unwrap_or(0);

    // Capture IR and asm from one extra run with both diagnostics enabled.
    let (ir_text, asm_text) = if capture {
        let mgr = CpuKernelManager::new();
        let gate = make_gate(qubits, density);
        let id = mgr.generate(
            KernelGenRequest::from_gate(spec, &gate)
                .with_ir()
                .with_asm(),
        )?;
        let asm = mgr.emit_asm(id);
        let ir_text = mgr.emit_ir(id);
        (ir_text, asm)
    } else {
        (None, None)
    };

    // Runtime exec sweep (optional).
    let exec = if let Some((sv_qubits, threads, budget)) = runtime_opts {
        let gate = make_gate(qubits, density);
        measure_runtime(spec, &gate, sv_qubits, threads, budget)?
    } else {
        Vec::new()
    };

    Ok(Row {
        label: String::new(),
        ir_ms: ir,
        opt_ms: (ir_opt - ir).max(0.0),
        cg_ms: (total - ir_opt).max(0.0),
        total_ms: total,
        ir_lines,
        ir_text,
        asm_text,
        exec,
    })
}

fn print_header() {
    println!(
        "{:<28} {:>8} {:>9} {:>10} {:>10} {:>10}",
        "row", "ir (ms)", "opt (ms)", "cg (ms)", "total (ms)", "ir lines"
    );
}

fn print_row(r: &Row) {
    println!(
        "{:<28} {:>8.1} {:>9.1} {:>10.1} {:>10.1} {:>10}",
        r.label, r.ir_ms, r.opt_ms, r.cg_ms, r.total_ms, r.ir_lines
    );
    for (t, stats) in &r.exec {
        println!(
            "  exec @ {:>2} thr  {:>10}  ±{:>9}  ({:>3} iters, cv={:.3})",
            t,
            fmt_duration(stats.mean_s),
            fmt_duration(stats.stddev_s),
            stats.n_iters,
            stats.cv,
        );
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut spec = match args.precision {
        CliPrecision::F32 => CPUKernelGenSpec::f32(),
        CliPrecision::F64 => CPUKernelGenSpec::f64(),
    };
    spec.simd_width = args.simd.resolve();
    if args.force_dense {
        spec.ztol = 0.0;
        spec.otol = 0.0;
    }

    println!(
        "Precision {:?}  SIMD {:?}  ztol={:.0e}  density={}  force_dense={}",
        spec.precision, spec.simd_width, spec.ztol, args.density, args.force_dense
    );
    print_header();

    let capture = args.dump_ir.is_some() || args.dump_asm.is_some();
    let write_dumps = |row: &Row, slug: &str| -> anyhow::Result<()> {
        if let (Some(prefix), Some(ir)) = (&args.dump_ir, row.ir_text.as_ref()) {
            let path = prefix.with_file_name(format!(
                "{}_{}.ll",
                prefix.file_name().and_then(|s| s.to_str()).unwrap_or("ir"),
                slug,
            ));
            std::fs::write(&path, ir)?;
            eprintln!("  wrote {}", path.display());
        }
        if let (Some(prefix), Some(asm)) = (&args.dump_asm, row.asm_text.as_ref()) {
            let path = prefix.with_file_name(format!(
                "{}_{}.s",
                prefix.file_name().and_then(|s| s.to_str()).unwrap_or("asm"),
                slug,
            ));
            std::fs::write(&path, asm)?;
            eprintln!("  wrote {}", path.display());
        }
        Ok(())
    };

    let runtime_opts: Option<(u32, &[u32], f64)> = if args.apply_budget > 0.0 {
        Some((args.sv_qubits, &args.threads, args.apply_budget))
    } else {
        None
    };

    if let Some(qubits) = args.qubits.as_ref() {
        let mut qs = qubits.clone();
        qs.sort_unstable();
        qs.dedup();
        let label = format!("q={:?}", qs);
        let slug = qs
            .iter()
            .map(|q| q.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let mut row = measure(spec, &qs, args.density, args.reps, capture, runtime_opts)?;
        row.label = label;
        print_row(&row);
        write_dumps(&row, &slug)?;
    } else {
        for k in 1..=args.max_qubits {
            let qubits: Vec<u32> = (0..k).collect();
            let label = format!("k={}  q=[0..{})", k, k);
            let slug = format!("k{}", k);
            let mut row = measure(
                spec,
                &qubits,
                args.density,
                args.reps,
                capture,
                runtime_opts,
            )?;
            row.label = label;
            print_row(&row);
            write_dumps(&row, &slug)?;
        }
    }

    Ok(())
}

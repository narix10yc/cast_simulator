//! Computation-memory balance analysis for JIT gate simulation kernels.
//!
//! Sweeps gate size and density (arithmetic intensity) from purely memory-bound
//! (random sparse gates with one nonzero scalar per row) to compute-bound
//! (dense random real gates up to 5 qubits, arithmetic intensity up to 32), reporting
//! GiB/s, GFLOPs/s, and timing variance side-by-side so the roofline
//! crossover is visible.
//!
//! ## Adaptive timing
//!
//! Each gate is first probed with a small number of iterations to estimate its
//! per-iteration cost. The remaining per-gate time budget (total budget divided
//! evenly across gates) is then filled with as many iterations as fit. This
//! means fast kernels accumulate hundreds of samples while slow kernels at large
//! qubit counts get fewer — but each gate spends roughly the same wall time.
//!
//! ## Metric
//!
//! ```text
//! ai       = scalar_nnz(M) / edge_size(M)   (re and im parts counted separately)
//! FLOPs    = ai × |ψ| × 2                   (2 real FLOPs per nonzero scalar component)
//! GFLOPs/s = FLOPs / mean_time
//! GiB/s    = 2 × sv_bytes / mean_time   (read + write every amplitude once)
//! CV       = stddev / mean              (coefficient of variation; scale-free noise)
//! ```
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin cpu_crossover --release
//! cargo run --bin cpu_crossover --release -- --n-qubits 30 --threads 10 --budget-secs 120
//! cargo run --bin cpu_crossover --release -- --help
//! ```

use anyhow::Result;
use cast::cpu;
use cast::types::{Complex, ComplexSquareMatrix, Precision, QuantumGate};
use clap::{Parser, ValueEnum};
use rand::Rng;
use std::time::Instant;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
enum CliPrecision {
    F32,
    F64,
}

impl From<CliPrecision> for Precision {
    fn from(value: CliPrecision) -> Self {
        match value {
            CliPrecision::F32 => Precision::F32,
            CliPrecision::F64 => Precision::F64,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "cpu_crossover")]
struct Args {
    /// log2 of statevector length
    #[arg(long, default_value_t = 20)]
    n_qubits: usize,

    /// worker threads; 0 = hardware_concurrency
    #[arg(long = "threads", default_value_t = 1)]
    threads: usize,

    #[arg(long, value_enum, default_value_t = CliPrecision::F64)]
    precision: CliPrecision,

    #[arg(skip = cpu::SimdWidth::W128)]
    simd_width: cpu::SimdWidth,

    /// Total wall-time budget for the entire sweep (seconds).
    #[arg(long, default_value_t = 120.0)]
    budget_secs: f64,

    /// Minimum timed iterations per gate regardless of budget.
    #[arg(long, default_value_t = 5)]
    min_iters: usize,

    /// Warmup iterations (not timed, not counted toward budget).
    #[arg(long, default_value_t = 3)]
    n_warmup: usize,
}

impl Args {
    fn n_threads(&self) -> Option<usize> {
        if self.threads == 0 {
            None
        } else {
            Some(self.threads)
        }
    }

    fn spec(&self) -> cpu::CPUKernelGenSpec {
        let precision: Precision = self.precision.into();
        cpu::CPUKernelGenSpec {
            precision,
            simd_width: self.simd_width,
            mode: cpu::MatrixLoadMode::ImmValue,
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

    fn scalar_size(&self) -> usize {
        match self.precision {
            CliPrecision::F32 => 4,
            CliPrecision::F64 => 8,
        }
    }
}

// ── Gate sweep ────────────────────────────────────────────────────────────────

struct Case {
    label: String,
    gate: QuantumGate,
}

/// Sweeps arithmetic intensity 1 → 32 using only generated dense and sparse matrices.
///
/// Dense cases use full real matrices so `ai = edge_size = 2^k`
/// exactly. Sparse cases use `s` real nonzeros per row so `ai = s`
/// exactly. This keeps the plotted x-axis aligned with the actual metric
/// reported by `QuantumGate::arithmatic_intensity`.
fn gate_sweep() -> Vec<Case> {
    let mut cases = Vec::new();

    for &s in &[1usize, 2, 4, 8] {
        cases.push(Case {
            label: format!("Sparse-4q   [rand sparse, s={s}]"),
            gate: sparse_real(4, s),
        });
        cases.push(Case {
            label: format!("Sparse-5q   [rand sparse, s={s}]"),
            gate: sparse_real(5, s),
        });
    }

    cases.push(Case {
        label: "Sparse-5q   [rand sparse, s=16]".to_string(),
        gate: sparse_real(5, 16),
    });

    for k in 1..=5 {
        cases.push(Case {
            label: format!("Dense-{k}q    [rand dense]"),
            gate: dense_real(k),
        });
    }

    cases
}

fn dense_real(k: usize) -> QuantumGate {
    sparse_real(k, 1usize << k)
}

/// Builds a k-qubit gate matrix with exactly `s` nonzero real entries per row.
///
/// Using real-only coefficients makes `ai = s` exactly:
///
/// `scalar_nnz(M) = n_rows × s`, `edge_size(M) = n_rows`, so
/// `ai = scalar_nnz / edge_size = s`.
///
/// The matrix is intentionally non-unitary; this tool measures kernel
/// throughput, not physical validity.
fn sparse_real(k: usize, s: usize) -> QuantumGate {
    let n = 1usize << k;
    assert!(s <= n, "s must be ≤ edge_size");
    let stride = n / s; // exact because sweep uses powers of two
    let mut rng = rand::thread_rng();
    let mut m = ComplexSquareMatrix::zeros(n);
    for row in 0..n {
        for t in 0..s {
            // Offset by n/2 so that s=1 is still a non-trivial permutation.
            let col = (row + n / 2 + t * stride) % n;
            m.set(row, col, Complex::new(sample_nonzero_real(&mut rng), 0.0));
        }
    }
    let qubits: Vec<u32> = (0..k as u32).collect();
    QuantumGate::new(m, qubits)
}

fn sample_nonzero_real<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let mag = rng.gen_range(0.25_f64..1.0);
    if rng.gen_bool(0.5) {
        mag
    } else {
        -mag
    }
}

// ── Adaptive timing ───────────────────────────────────────────────────────────

struct Timing {
    /// Number of timed iterations actually collected.
    n_iters: usize,
    mean_s: f64,
    stddev_s: f64,
    /// Coefficient of variation: stddev / mean (scale-free noise metric).
    cv: f64,
    min_s: f64,
    max_s: f64,
}

/// Runs `n_warmup` un-timed iterations, then collects timed samples until
/// `per_gate_budget_s` is exhausted, with at least `min_iters` samples.
///
/// Strategy:
///   1. Probe `N_PROBE` iterations to estimate per-iteration cost.
///   2. Compute how many more iterations fit in the remaining budget.
///   3. Collect all probe + fill samples; compute statistics together.
fn time_adaptive(
    jit: &mut cpu::JitSession,
    kid: cpu::KernelId,
    sv: &mut cpu::CPUStatevector,
    n_threads: Option<usize>,
    n_warmup: usize,
    min_iters: usize,
    per_gate_budget_s: f64,
) -> Timing {
    const N_PROBE: usize = 3;

    for _ in 0..n_warmup {
        jit.apply(kid, sv, n_threads).unwrap();
    }

    // Phase 1: probe to estimate per-iteration time.
    let mut samples: Vec<f64> = Vec::new();
    let probe_wall = Instant::now();
    for _ in 0..N_PROBE {
        let t = Instant::now();
        jit.apply(kid, sv, n_threads).unwrap();
        samples.push(t.elapsed().as_secs_f64());
    }
    let probe_elapsed = probe_wall.elapsed().as_secs_f64();

    // Phase 2: fill remaining budget.
    let est_per_iter = probe_elapsed / N_PROBE as f64;
    let remaining = (per_gate_budget_s - probe_elapsed).max(0.0);
    let n_fill = ((remaining / est_per_iter) as usize).max(min_iters.saturating_sub(N_PROBE));
    for _ in 0..n_fill {
        let t = Instant::now();
        jit.apply(kid, sv, n_threads).unwrap();
        samples.push(t.elapsed().as_secs_f64());
    }

    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Timing {
        n_iters: samples.len(),
        mean_s: mean,
        stddev_s: stddev,
        cv: stddev / mean,
        min_s: min,
        max_s: max,
    }
}

// ── Row ───────────────────────────────────────────────────────────────────────

struct Row {
    label: String,
    k: usize,
    nnz: usize,
    ai: f64,
    timing: Timing,
    gib_s: f64,
    gflops_s: f64,
}

struct AiBucket {
    ai: f64,
    mean_gflops_s: f64,
    max_k: usize,
}

fn ai_buckets(rows: &[Row]) -> Vec<AiBucket> {
    const AI_EPS: f64 = 1e-9;

    let mut buckets = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let ai = rows[i].ai;
        let mut sum_gflops = 0.0;
        let mut count = 0usize;
        let mut max_k = rows[i].k;

        while i < rows.len() && (rows[i].ai - ai).abs() < AI_EPS {
            sum_gflops += rows[i].gflops_s;
            count += 1;
            max_k = max_k.max(rows[i].k);
            i += 1;
        }

        buckets.push(AiBucket {
            ai,
            mean_gflops_s: sum_gflops / count as f64,
            max_k,
        });
    }

    buckets
}

fn measure(case: &Case, args: &Args, per_gate_budget_s: f64) -> Result<Row> {
    let spec = args.spec();
    let k = case.gate.n_qubits();
    let ai = case.gate.arithmatic_intensity(spec.ztol);
    let n = case.gate.matrix().edge_size();
    // scalar_nnz = ai × edge_size; recover as integer for display
    let scalar_nnz = (ai * n as f64).round() as usize;

    let mut gen = cpu::CPUKernelGenerator::new()?;
    let kid = gen.generate(&spec, case.gate.matrix().data(), case.gate.qubits())?;
    let mut jit = gen.init_jit()?;

    let mut sv = cpu::CPUStatevector::new(n, spec.precision, spec.simd_width);
    sv.initialize();

    let timing = time_adaptive(
        &mut jit,
        kid,
        &mut sv,
        args.n_threads(),
        args.n_warmup,
        args.min_iters,
        per_gate_budget_s,
    );

    let gib_s = 2.0 * sv.byte_len() as f64 / timing.mean_s / (1u64 << 30) as f64;
    let gflops_s = ai * sv.len() as f64 * 2.0 / timing.mean_s / 1e9;

    Ok(Row {
        label: case.label.clone(),
        k,
        nnz: scalar_nnz,
        ai,
        timing,
        gib_s,
        gflops_s,
    })
}

// ── Report ────────────────────────────────────────────────────────────────────

fn print_report(rows: &[Row], args: &Args) {
    let sv_mib = (1usize << args.n_qubits) * args.scalar_size() * 2 / (1 << 20);
    let threads = match args.n_threads() {
        Some(n) => n.to_string(),
        None => "auto (hardware_concurrency)".to_string(),
    };

    let bar = "═".repeat(100);
    let line = "─".repeat(100);

    println!();
    println!("  {bar}");
    println!("  Computation-Memory Balance Report");
    println!("  {bar}");
    println!(
        "  statevector  : {} qubits = 2^{} amplitudes = {} MiB  ({:?})",
        args.n_qubits, args.n_qubits, sv_mib, args.precision,
    );
    println!("  threads      : {threads}");
    println!(
        "  time budget  : {:.0} s total / {:.1} s per gate",
        args.budget_secs,
        args.budget_secs / rows.len() as f64,
    );
    println!("  FLOPs/scalar : 2 real  (1×mul + 1×add per nonzero scalar component)");
    println!("  ai           : scalar_nnz(M) / edge_size(M) — re and im parts counted separately");
    println!("  CV           : stddev / mean — scale-free timing noise");
    println!();

    // ── Performance table ─────────────────────────────────────────────────────
    println!(
        "  {:<32}  {:>2}  {:>5}  {:>7}  {:>10}  {:>9}  {:>10}  {:>8}",
        "gate", "k", "snnz", "ai", "mean (ms)", "GiB/s", "GFLOPs/s", "regime",
    );
    println!("  {line}");

    let buckets = ai_buckets(rows);
    let first = buckets.first().unwrap();
    let gflops_per_ai = first.mean_gflops_s / first.ai;

    for row in rows {
        let expected_linear = gflops_per_ai * row.ai;
        let regime = if row.gflops_s >= expected_linear * 0.85 {
            "mem-bound"
        } else {
            "compute↑"
        };

        println!(
            "  {:<32}  {:>2}  {:>5}  {:>7.1}  {:>10.2}  {:>9.1}  {:>10.1}  {:>8}",
            row.label,
            row.k,
            row.nnz,
            row.ai,
            row.timing.mean_s * 1e3,
            row.gib_s,
            row.gflops_s,
            regime,
        );
    }

    println!("  {line}");

    // ── Variance table ────────────────────────────────────────────────────────
    println!();
    println!(
        "  {:<32}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}  {:>6}",
        "gate", "n_iters", "mean (ms)", "stddev", "min (ms)", "max (ms)", "CV %",
    );
    println!("  {line}");

    for row in rows {
        let t = &row.timing;
        println!(
            "  {:<32}  {:>7}  {:>10.3}  {:>10.3}  {:>10.3}  {:>8.3}  {:>5.2}%",
            row.label,
            t.n_iters,
            t.mean_s * 1e3,
            t.stddev_s * 1e3,
            t.min_s * 1e3,
            t.max_s * 1e3,
            t.cv * 100.0,
        );
    }

    println!("  {line}");

    // ── Summary ───────────────────────────────────────────────────────────────
    let peak_gib = rows.iter().map(|r| r.gib_s).fold(0.0_f64, f64::max);
    let peak_gflops = rows.iter().map(|r| r.gflops_s).fold(0.0_f64, f64::max);
    let knee_ai = peak_gflops / gflops_per_ai;

    let mut prev_bucket_regime = "";
    let mut prev_bucket_ai = 0.0_f64;
    let mut crossover: Option<(f64, f64, usize)> = None;
    for bucket in &buckets {
        let expected_linear = gflops_per_ai * bucket.ai;
        let regime = if bucket.mean_gflops_s >= expected_linear * 0.85 {
            "mem-bound"
        } else {
            "compute↑"
        };
        if prev_bucket_regime == "mem-bound" && regime == "compute↑" && crossover.is_none() {
            crossover = Some((prev_bucket_ai, bucket.ai, bucket.max_k));
        }
        prev_bucket_regime = regime;
        prev_bucket_ai = bucket.ai;
    }

    println!();
    println!("  Peak memory bandwidth  : {peak_gib:.1} GiB/s");
    println!("  Peak compute observed  : {peak_gflops:.1} GFLOPs/s");
    println!("  Roofline knee (est.)   : ai ≈ {knee_ai:.1}");

    if let Some((ai_before, ai_after, k)) = crossover {
        println!();
        println!(
            "  ⚑ Crossover: mem-bound → compute-bound between ai {ai_before:.0}–{ai_after:.0} ({k}-qubit gate)",
        );
    } else {
        println!();
        println!("  ⚑ All measured gates are memory-bound; increase gate size to see crossover.");
    }
    println!();
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let cases = gate_sweep();
    let per_gate_budget = args.budget_secs / cases.len() as f64;
    let sv_mib = (1usize << args.n_qubits) * args.scalar_size() * 2 / (1 << 20);

    eprintln!(
        "Balance sweep: n_qubits={} ({} MiB, {:?}), threads={:?}",
        args.n_qubits,
        sv_mib,
        args.precision,
        args.n_threads(),
    );
    eprintln!(
        "Budget: {:.0}s total → {:.1}s per gate  (≥{} iters, {} warmup)",
        args.budget_secs, per_gate_budget, args.min_iters, args.n_warmup,
    );
    eprintln!();

    let mut rows: Vec<Row> = Vec::new();
    for case in &cases {
        eprint!("  {:<40} ", case.label);
        let row = measure(case, &args, per_gate_budget)?;
        eprintln!(
            "{:>6} iters  {:.2} ms ± {:.3} ms  CV={:.2}%  {:>7.1} GiB/s  {:>7.1} GFLOPs/s",
            row.timing.n_iters,
            row.timing.mean_s * 1e3,
            row.timing.stddev_s * 1e3,
            row.timing.cv * 100.0,
            row.gib_s,
            row.gflops_s,
        );
        rows.push(row);
    }

    rows.sort_by(|a, b| {
        a.ai.total_cmp(&b.ai)
            .then_with(|| a.k.cmp(&b.k))
            .then_with(|| a.label.cmp(&b.label))
    });
    print_report(&rows, &args);
    Ok(())
}

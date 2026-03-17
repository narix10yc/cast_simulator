//! Computation-memory balance analysis for JIT gate simulation kernels.
//!
//! Sweeps gate size and density (arithmetic intensity) using
//! [`QuantumGate::random_sparse_with_rng`], reporting GiB/s, GFLOPs/s, and
//! timing variance side-by-side so the roofline crossover is visible.
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
use cast::cpu::{self, CPUStatevector};
use cast::types::{Precision, QuantumGate};
use clap::{Parser, ValueEnum};
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
    /// number of qubits in the statevector
    #[arg(long, default_value_t = 28)]
    n_qubits: u32,

    /// worker threads; 0 = default
    #[arg(long = "threads", default_value_t = 0)]
    threads: u32,

    #[arg(long, value_enum, default_value_t = CliPrecision::F64)]
    precision: CliPrecision,

    /// Total wall-time budget for the entire sweep (seconds).
    #[arg(long, default_value_t = 120.0)]
    budget_secs: f64,

    /// Minimum timed iterations per gate regardless of budget.
    #[arg(long, default_value_t = 3)]
    min_iters: u32,

    /// Warmup iterations (not timed, not counted toward budget).
    #[arg(long, default_value_t = 1)]
    n_warmup: u32,
}

impl Args {
    fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    fn n_threads(&self) -> u32 {
        if self.threads == 0 {
            num_cpus::get() as u32
        } else {
            self.threads
        }
    }

    fn precision(&self) -> Precision {
        self.precision.into()
    }

    fn spec(&self) -> cpu::CPUKernelGenSpec {
        match self.precision {
            CliPrecision::F32 => cpu::CPUKernelGenSpec::f32(),
            CliPrecision::F64 => cpu::CPUKernelGenSpec::f64(),
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

/// The zero tolerance for arithmetic intensity calculation
const AI_EPS: f64 = 1e-9;
const NUM_INITIAL_CASES: usize = 5;

fn initial_sweep(n_qubits_sv: u32) -> Vec<Case> {
    let mut cases = Vec::new();
    cases.reserve(NUM_INITIAL_CASES);
    let rng = &mut rand::thread_rng();

    for i in 0..NUM_INITIAL_CASES {
        let qubits = rand::seq::index::sample(rng, n_qubits_sv as usize, 4)
            .iter()
            .map(|i| i as u32)
            .collect::<Vec<u32>>();

        let f = 1.0 / (NUM_INITIAL_CASES + 1) as f64;
        let sparsity = (i + 1) as f64 * f;
        let gate = QuantumGate::random_sparse(qubits.as_slice(), sparsity);
        let ai = gate.arithmatic_intensity(AI_EPS);
        cases.push(Case {
            label: format!("Rand-4q [ai={:>4.1}]", ai),
            gate,
        });
    }

    cases
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
/// `budget_s` is exhausted, with at least `min_iters` samples. Returns timing
/// statistics.
fn time_adaptive(
    jit: &mut cpu::JitSession,
    kid: cpu::KernelId,
    sv: &mut cpu::CPUStatevector,
    n_threads: u32,
    n_warmup: u32,
    min_iters: u32,
    budget_s: f64,
) -> Timing {
    const N_PROBE: u32 = 3;

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
    let remaining = (budget_s - probe_elapsed).max(0.0);
    let n_fill = ((remaining / est_per_iter) as u32).max(min_iters.saturating_sub(N_PROBE));
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
    k: u32,
    nnz: usize,
    ai: f64,
    timing: Timing,
    gib_s: f64,
    gflops_s: f64,
}

struct AiBucket {
    ai: f64,
    mean_gflops_s: f64,
    max_k: u32,
}

fn ai_buckets(rows: &[Row]) -> Vec<AiBucket> {
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

fn measure(
    sv: &mut CPUStatevector,
    case: &Case,
    args: &Args,
    per_gate_budget_s: f64,
) -> Result<Row> {
    let spec = args.spec();
    let k = case.gate.n_qubits() as u32;
    let ai = case.gate.arithmatic_intensity(spec.ztol);
    let n = case.gate.matrix().edge_size();
    // scalar_nnz = ai × edge_size; recover as integer for display
    let scalar_nnz = (ai * n as f64).round() as usize;

    let mut gen = cpu::CPUKernelGenerator::new()?;
    let kid = gen.generate(&spec, case.gate.matrix().data(), case.gate.qubits())?;
    let mut jit = gen.init_jit()?;

    let timing = time_adaptive(
        &mut jit,
        kid,
        sv,
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
    let threads = args.n_threads().to_string();

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
    println!("  FLOPs/scalar : 2 real  (1xmul + 1×add per nonzero scalar component)");
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
    let crossover_ai = peak_gflops / gflops_per_ai;

    let mut prev_bucket_regime = "";
    let mut prev_bucket_ai = 0.0_f64;
    let mut crossover: Option<(f64, f64, u32)> = None;
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
    println!("  Roofline crossover (est.) : ai ≈ {crossover_ai:.1}");

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

    let cases = initial_sweep(args.n_qubits());
    let per_gate_budget = args.budget_secs / cases.len() as f64;
    let sv_mib = (1usize << args.n_qubits()) * args.scalar_size() * 2 / (1 << 20);

    eprintln!(
        "Balance sweep: n_qubits={} ({} MiB, {:?}), {}-thread",
        args.n_qubits(),
        sv_mib,
        args.precision(),
        args.n_threads(),
    );
    eprintln!(
        "Budget: {:.0}s total → {:.1}s per gate  (≥{} iters, {} warmup run)",
        args.budget_secs, per_gate_budget, args.min_iters, args.n_warmup,
    );
    eprintln!();

    // --- Actual run ---
    let mut sv =
        unsafe { CPUStatevector::uninit(args.n_qubits(), args.precision(), cpu::SimdWidth::W128) };

    let mut rows: Vec<Row> = Vec::new();
    for case in &cases {
        eprint!("  {:<20} ", case.label);
        let row = measure(&mut sv, case, &args, per_gate_budget)?;
        eprintln!(
            "{:.2} ms (cv={:.2}%)  {:>7.1} GiB/s  {:>7.1} GFLOPs/s",
            row.timing.mean_s * 1e3,
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

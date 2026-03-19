//! CPU kernel roofline profiler: adaptive crossover detection.
//!
//! Locates the arithmetic-intensity crossover between the memory-bound and
//! compute-bound regimes of the JIT gate simulation kernel using three phases:
//!
//! 1. **BW calibration** — probes an AI=1 gate (near-pure memory traffic) to
//!    estimate peak DRAM bandwidth.  No separate memcpy test on CPU.
//! 2. **Heuristic escalation** — probes AI = 2, 4, 8, … (doubling) with the
//!    smallest gate that achieves each AI, stopping at the first non-mem-bound
//!    measurement.  Never pre-commits to a large dense gate.
//! 3. **Coarse sweep + bisection** — probes the remaining power-of-2 AI
//!    points to build the full curve, then bisects between the last mem-bound
//!    and first non-mem-bound measurement to pinpoint the crossover.
//!
//! ## Statevector size
//!
//! The benchmark automatically clamps the profiling statevector to
//! [`PROFILE_N_QUBITS`] qubits regardless of `--n-qubits`, because:
//!   - Too small → LLC dominates, masking the true DRAM bandwidth.
//!   - Too large → allocation + first-touch time dominates the budget.
//!
//! The `--n-qubits` argument is recorded in the report header only.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin cpu_crossover --release                                        # defaults
//! cargo run --bin cpu_crossover --release -- --n-qubits 30 --threads 32          # override
//! CAST_NUM_THREADS=32 cargo run --bin cpu_crossover --release                    # via env
//! cargo run --bin cpu_crossover --release -- --help
//! ```

use anyhow::Result;
use cast::cpu::{self, CPUKernelGenSpec, CpuKernelManager, CPUStatevector};
use cast::types::{Complex, ComplexSquareMatrix, Precision, QuantumGate};
use clap::{Parser, ValueEnum};
use rand::Rng;

// ── Profiling constants ───────────────────────────────────────────────────────

/// Statevector size used for all gate benchmarks.
/// 28 qubits ≈ 4 GiB (F64) / 2 GiB (F32) — large enough to spill all CPU
/// caches while remaining quickly allocatable.
const PROFILE_N_QUBITS: u32 = 28;

/// Smallest gate size (qubits) used in any probe, regardless of AI.
const MIN_GATE_QUBITS: usize = 4;

/// Largest gate size (qubits); caps the probed AI range at 2^MAX_GATE_QUBITS.
const MAX_GATE_QUBITS: usize = 6;

/// GFLOPs/s ≥ this fraction of the mem-bound linear prediction → mem-bound.
const MEM_BOUND_RATIO: f64 = 0.85;

/// GFLOPs/s < this fraction → compute-bound (below = transition).
const COMP_BOUND_RATIO: f64 = 0.65;

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
    /// Target qubit count for large simulations (recorded in report only).
    /// The profiling statevector is clamped to PROFILE_N_QUBITS internally.
    #[arg(long, default_value_t = 30)]
    n_qubits: u32,

    /// Worker threads; 0 = CAST_NUM_THREADS or logical CPU count.
    #[arg(long = "threads", default_value_t = 0)]
    threads: u32,

    #[arg(long, value_enum, default_value_t = CliPrecision::F64)]
    precision: CliPrecision,

    /// Total wall-time budget for the gate sweep (seconds).
    #[arg(long, default_value_t = 120.0)]
    budget_secs: f64,
}

impl Args {
    fn n_threads(&self) -> u32 {
        if self.threads == 0 {
            cpu::get_num_threads()
        } else {
            self.threads
        }
    }

    fn precision(&self) -> Precision {
        self.precision.into()
    }

    fn spec(&self) -> CPUKernelGenSpec {
        match self.precision {
            CliPrecision::F32 => CPUKernelGenSpec::f32(),
            CliPrecision::F64 => CPUKernelGenSpec::f64(),
        }
    }

    fn scalar_size(&self) -> usize {
        match self.precision {
            CliPrecision::F32 => 4,
            CliPrecision::F64 => 8,
        }
    }

    fn profile_n_qubits(&self) -> u32 {
        PROFILE_N_QUBITS.min(self.n_qubits)
    }

    fn profile_sv_bytes(&self) -> usize {
        (1usize << self.profile_n_qubits()) * 2 * self.scalar_size()
    }
}

// ── Gate construction ─────────────────────────────────────────────────────────

/// Returns the smallest gate size k ≥ MIN_GATE_QUBITS such that 2^k ≥ target_ai.
fn gate_qubits_for_ai(target_ai: usize) -> usize {
    let k = (target_ai as f64).log2().ceil() as usize;
    k.max(MIN_GATE_QUBITS)
}

/// Build a k-qubit sparse-real gate with exactly `s` nonzero real entries per row.
/// Arithmetic intensity = s (re parts; im parts are all zero).
/// Requires 1 ≤ s ≤ 2^k.
fn gate_for_ai(k: usize, s: usize) -> QuantumGate {
    let n = 1usize << k;
    debug_assert!(s >= 1 && s <= n);
    let stride = if s < n { n / s } else { 1 };
    let mut rng = rand::thread_rng();
    let mut m = ComplexSquareMatrix::zeros(n);
    for row in 0..n {
        for t in 0..s {
            let col = (row + n / 2 + t * stride) % n;
            let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            let v = rng.gen_range(0.25_f64..1.0) * sign;
            m.set(row, col, Complex::new(v, 0.0));
        }
    }
    QuantumGate::new(m, (0..k as u32).collect())
}

// ── Measurement ───────────────────────────────────────────────────────────────

struct Measurement {
    ai_target: usize,
    k: usize,
    ai: f64,
    nnz: usize,
    timing: cpu::TimingStats,
    gib_s: f64,
    gflops_s: f64,
}

fn measure_at(
    target_ai: usize,
    sv: &mut CPUStatevector,
    args: &Args,
    budget_s: f64,
) -> Result<Measurement> {
    let spec = args.spec();
    let k = gate_qubits_for_ai(target_ai);
    let gate = gate_for_ai(k, target_ai);
    let ai = gate.arithmatic_intensity(spec.ztol);
    let nnz = gate.scalar_nnz(spec.ztol);

    let mgr = CpuKernelManager::new();
    let kid = mgr.generate(&spec, gate.matrix().data(), gate.qubits())?;

    let timing = mgr.time_adaptive(kid, sv, args.n_threads(), budget_s)?;

    let sv_bytes = args.profile_sv_bytes();
    let n_sv = args.profile_n_qubits();
    let gib_s = 2.0 * sv_bytes as f64 / timing.mean_s / (1u64 << 30) as f64;
    let gflops_s = ai * (1usize << n_sv) as f64 * 2.0 / timing.mean_s / 1e9;

    Ok(Measurement {
        ai_target: target_ai,
        k,
        ai,
        nnz,
        timing,
        gib_s,
        gflops_s,
    })
}

// ── Regime classification ─────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Regime {
    MemBound,
    Transition,
    CompBound,
}

fn classify(m: &Measurement, gflops_per_ai: f64) -> Regime {
    let ratio = m.gflops_s / (gflops_per_ai * m.ai);
    if ratio >= MEM_BOUND_RATIO {
        Regime::MemBound
    } else if ratio < COMP_BOUND_RATIO {
        Regime::CompBound
    } else {
        Regime::Transition
    }
}

// ── Hardware profile ──────────────────────────────────────────────────────────

struct HardwareProfile {
    /// Peak memory bandwidth inferred from the AI=1 gate probe.
    peak_bw_gib_s: f64,
    /// GFLOPs/s per unit AI predicted by the roofline model for a mem-bound gate.
    gflops_per_ai: f64,
    /// Best GFLOPs/s observed across all probes.
    peak_gflops_s: f64,
    /// Theoretical crossover AI = peak_gflops_s / gflops_per_ai.
    theoretical_crossover_ai: f64,
}

// ── Adaptive exploration ──────────────────────────────────────────────────────

fn explore(args: &Args) -> Result<(HardwareProfile, Vec<Measurement>)> {
    let spec = args.spec();
    let n_qubits = args.profile_n_qubits();

    // PROFILE_N_QUBITS must exceed MAX_GATE_QUBITS + max_simd_s + 1.
    // max_simd_s is 4 (F32/W512), so the minimum is MAX_GATE_QUBITS + 5 = 11.
    debug_assert!(n_qubits as usize > MAX_GATE_QUBITS + 5);

    let mut sv = CPUStatevector::new(n_qubits, args.precision(), spec.simd_width);
    sv.initialize();

    let sv_mib = args.profile_sv_bytes() / (1 << 20);
    eprintln!(
        "Profile SV: {} qubits = {} MiB ({:?}),  {}-thread,  budget {:.0}s",
        n_qubits,
        sv_mib,
        args.precision,
        args.n_threads(),
        args.budget_secs,
    );
    eprintln!();

    // ── Phase 1a: BW calibration (AI=1 gate probe) ──────────────────────────

    eprintln!("Phase 1a: BW calibration (AI=1 gate probe)");
    let bw_budget = args.budget_secs * 0.10;
    eprint!("  ai=   1 k={} ... ", MIN_GATE_QUBITS);
    let bw_m = measure_at(1, &mut sv, args, bw_budget)?;
    let peak_bw_gib_s = bw_m.gib_s;
    // gflops_per_ai: from the roofline model.
    // T_mem_bound = 2 × sv_bytes / BW_bytes_s
    // GFLOPs/s    = ai × 2^n_sv × 2 / T / 1e9
    //             = ai × BW_bytes_s / (2 × scalar_size) / 1e9
    let gflops_per_ai =
        peak_bw_gib_s * (1u64 << 30) as f64 / (2.0 * args.scalar_size() as f64) / 1e9;
    eprintln!(
        "{:.3} ms  {:.1} GiB/s  →  {:.2} GFLOPs/s per unit AI",
        bw_m.timing.mean_s * 1e3,
        peak_bw_gib_s,
        gflops_per_ai,
    );
    eprintln!();

    let mut measurements: Vec<Measurement> = vec![bw_m];

    // ── Phase 1b: Heuristic escalation to find the compute-bound regime ─────

    eprintln!("Phase 1b: Heuristic escalation (AI = 2, 4, 8, … until non-mem-bound)");
    let max_ai = 1usize << MAX_GATE_QUBITS;

    let n_escalation_steps = MAX_GATE_QUBITS as f64;
    let escalation_budget_per = args.budget_secs * 0.20 / n_escalation_steps;

    let mut ai_probe = 2usize;
    while ai_probe <= max_ai {
        let k = gate_qubits_for_ai(ai_probe);
        eprint!("  ai={ai_probe:>4} k={k} ... ");
        let m = measure_at(ai_probe, &mut sv, args, escalation_budget_per)?;
        let r = classify(&m, gflops_per_ai);
        eprintln!(
            "{:.3} ms  {:.1} GiB/s  {:.1}/{:.1} GFLOPs/s  [{r:?}]",
            m.timing.mean_s * 1e3,
            m.gib_s,
            m.gflops_s,
            gflops_per_ai * m.ai,
        );
        let found_non_mem = r != Regime::MemBound;
        measurements.push(m);
        if found_non_mem {
            break;
        }
        if ai_probe >= max_ai {
            break;
        }
        ai_probe = (ai_probe * 2).min(max_ai);
    }

    if classify(measurements.last().unwrap(), gflops_per_ai) == Regime::MemBound {
        eprintln!(
            "  Warning: still mem-bound at ai={}. Crossover may be above ai={}.",
            ai_probe, max_ai,
        );
        eprintln!("  Consider increasing MAX_GATE_QUBITS or using a CPU with higher peak FLOPS.");
    }
    eprintln!();

    // ── Phase 2: Coarse sweep — fill in remaining power-of-2 AI points ──────

    eprintln!("Phase 2: Coarse sweep (remaining power-of-2 AI values)");
    let already_measured: std::collections::HashSet<usize> =
        measurements.iter().map(|m| m.ai_target).collect();
    let coarse_ais: Vec<usize> = (1..=MAX_GATE_QUBITS)
        .map(|e| 1usize << e)
        .filter(|&ai| !already_measured.contains(&ai))
        .collect();

    let coarse_budget_per = if coarse_ais.is_empty() {
        0.0
    } else {
        args.budget_secs * 0.45 / coarse_ais.len() as f64
    };

    // Stop early once a crossover bracket is established: a mem-bound or
    // transition point at some AI, followed by a compute-bound point at a
    // higher AI.  Beyond the bracket, larger gates are slow and redundant.
    let mut seen_non_comp = measurements
        .iter()
        .any(|m| classify(m, gflops_per_ai) != Regime::CompBound);

    for &ai in &coarse_ais {
        let k = gate_qubits_for_ai(ai);
        eprint!("  ai={ai:>4} k={k} ... ");
        let m = measure_at(ai, &mut sv, args, coarse_budget_per)?;
        let r = classify(&m, gflops_per_ai);
        eprintln!(
            "{:.3} ms  {:.1} GiB/s  {:.1}/{:.1} GFLOPs/s  [{r:?}]",
            m.timing.mean_s * 1e3,
            m.gib_s,
            m.gflops_s,
            gflops_per_ai * m.ai,
        );
        if r != Regime::CompBound {
            seen_non_comp = true;
        }
        measurements.push(m);
        if seen_non_comp && r == Regime::CompBound {
            eprintln!("  (crossover bracket found — skipping higher AI values)");
            break;
        }
    }
    eprintln!();

    // ── Phase 3: Integer bisection to refine the crossover ──────────────────

    eprintln!("Phase 3: Bisection refinement");

    measurements.sort_by(|a, b| a.ai_target.cmp(&b.ai_target));
    let mut bracket_lo = measurements.first().unwrap().ai_target;
    let mut bracket_hi = measurements.last().unwrap().ai_target;

    for pair in measurements.windows(2) {
        if classify(&pair[0], gflops_per_ai) == Regime::MemBound
            && classify(&pair[1], gflops_per_ai) != Regime::MemBound
        {
            bracket_lo = pair[0].ai_target;
            bracket_hi = pair[1].ai_target;
            break;
        }
    }

    let bisect_budget_per = args.budget_secs * 0.25 / 5.0;
    for _ in 0..5 {
        if bracket_hi - bracket_lo <= 1 {
            break;
        }
        let ai_mid = (bracket_lo + bracket_hi) / 2;
        let k = gate_qubits_for_ai(ai_mid);
        eprint!("  ai={ai_mid:>4} k={k} [{bracket_lo}..{bracket_hi}] ... ");
        let m = measure_at(ai_mid, &mut sv, args, bisect_budget_per)?;
        let r = classify(&m, gflops_per_ai);
        eprintln!(
            "{:.3} ms  {:.1} GFLOPs/s  [{r:?}]",
            m.timing.mean_s * 1e3,
            m.gflops_s,
        );
        if r == Regime::MemBound {
            bracket_lo = ai_mid;
        } else {
            bracket_hi = ai_mid;
        }
        measurements.push(m);
    }
    eprintln!();

    // ── Build hardware profile from all measurements ────────────────────────

    let peak_gflops_s = measurements
        .iter()
        .map(|m| m.gflops_s)
        .fold(0.0_f64, f64::max);
    let theoretical_crossover_ai = if peak_gflops_s > 0.0 {
        peak_gflops_s / gflops_per_ai
    } else {
        f64::INFINITY
    };
    let profile = HardwareProfile {
        peak_bw_gib_s,
        gflops_per_ai,
        peak_gflops_s,
        theoretical_crossover_ai,
    };

    eprintln!(
        "  Theoretical crossover: ai ≈ {:.1}  (peak observed: {:.1} GFLOPs/s)",
        theoretical_crossover_ai, peak_gflops_s,
    );
    eprintln!();

    Ok((profile, measurements))
}

// ── Report ────────────────────────────────────────────────────────────────────

fn print_report(profile: &HardwareProfile, rows: &[Measurement], args: &Args) {
    let sv_mib = args.profile_sv_bytes() / (1 << 20);
    let bar = "═".repeat(105);
    let line = "─".repeat(105);

    println!();
    println!("  {bar}");
    println!("  CPU Roofline Profile Report");
    println!("  {bar}");
    println!(
        "  target simulation  : {} qubits ({:?})",
        args.n_qubits, args.precision,
    );
    println!(
        "  profile SV         : {} qubits = {} MiB  (clamped to avoid cache/alloc artifacts)",
        args.profile_n_qubits(),
        sv_mib,
    );
    println!("  threads            : {}", args.n_threads());
    println!(
        "  budget             : {:.0} s gate sweep",
        args.budget_secs
    );
    println!();
    println!(
        "  ── Hardware calibration (AI=1 gate probe) ────────────────────────────────────────────"
    );
    println!(
        "  Peak memory BW          : {:.1} GiB/s",
        profile.peak_bw_gib_s
    );
    println!(
        "  GFLOPs/s per unit AI    : {:.2}  (= BW_bytes_s / (2 × scalar_size) / 1e9)",
        profile.gflops_per_ai
    );
    println!(
        "  Peak compute (observed) : {:.1} GFLOPs/s",
        profile.peak_gflops_s
    );
    println!(
        "  Theoretical crossover   : ai ≈ {:.1}",
        profile.theoretical_crossover_ai
    );
    println!();

    // ── Performance table ─────────────────────────────────────────────────────

    let mut sorted: Vec<&Measurement> = rows.iter().collect();
    sorted.sort_by(|a, b| a.ai_target.cmp(&b.ai_target).then(a.k.cmp(&b.k)));

    println!(
        "  {:<7}  {:>2}  {:>5}  {:>7}  {:>11}  {:>9}  {:>11}  {:>11}  {:>7}  {:>10}",
        "ai",
        "k",
        "snnz",
        "ai_meas",
        "mean (ms)",
        "GiB/s",
        "GFLOPs/s",
        "expected",
        "ratio",
        "regime",
    );
    println!("  {line}");

    for m in &sorted {
        let expected = profile.gflops_per_ai * m.ai;
        let ratio = m.gflops_s / expected;
        let regime_str = match classify(m, profile.gflops_per_ai) {
            Regime::MemBound => "mem-bound",
            Regime::Transition => "transit.",
            Regime::CompBound => "comp-bnd",
        };
        println!(
            "  {:<7}  {:>2}  {:>5}  {:>7.1}  {:>11.4}  {:>9.1}  {:>11.1}  {:>11.1}  {:>7.3}  {:>10}",
            m.ai_target, m.k, m.nnz, m.ai,
            m.timing.mean_s * 1e3,
            m.gib_s,
            m.gflops_s,
            expected,
            ratio,
            regime_str,
        );
    }

    println!("  {line}");

    // ── Variance table ────────────────────────────────────────────────────────

    println!();
    println!(
        "  {:<7}  {:>2}  {:>7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>6}",
        "ai", "k", "n_iters", "mean (ms)", "stddev ms", "min ms", "max ms", "CV %",
    );
    println!("  {line}");
    for m in &sorted {
        let t = &m.timing;
        println!(
            "  {:<7}  {:>2}  {:>7}  {:>11.4}  {:>11.4}  {:>11.4}  {:>11.4}  {:>5.2}%",
            m.ai_target,
            m.k,
            t.n_iters,
            t.mean_s * 1e3,
            t.stddev_s * 1e3,
            t.min_s * 1e3,
            t.max_s * 1e3,
            t.cv * 100.0,
        );
    }
    println!("  {line}");

    // ── Crossover summary ─────────────────────────────────────────────────────

    let mut crossover_lo: Option<usize> = None;
    let mut crossover_hi: Option<usize> = None;
    for pair in sorted.windows(2) {
        if classify(pair[0], profile.gflops_per_ai) == Regime::MemBound
            && classify(pair[1], profile.gflops_per_ai) != Regime::MemBound
            && crossover_lo.is_none()
        {
            crossover_lo = Some(pair[0].ai_target);
            crossover_hi = Some(pair[1].ai_target);
        }
    }

    println!();
    println!(
        "  ── Crossover summary ─────────────────────────────────────────────────────────────────"
    );
    match (crossover_lo, crossover_hi) {
        (Some(lo), Some(hi)) => {
            println!("  Observed crossover    : ai between {lo} and {hi}");
            println!(
                "  Theoretical crossover : ai ≈ {:.1}",
                profile.theoretical_crossover_ai
            );
            let mid = (lo + hi) as f64 / 2.0;
            let deviation =
                (mid - profile.theoretical_crossover_ai).abs() / profile.theoretical_crossover_ai;
            if deviation < 0.30 {
                println!(
                    "  Model accuracy        : good (theory and observation agree within 30%)"
                );
            } else {
                println!(
                    "  Model accuracy        : fair (theory: {:.1}, observed midpoint: {mid:.1})",
                    profile.theoretical_crossover_ai,
                );
            }
        }
        _ => {
            let last = sorted.last().unwrap();
            if classify(last, profile.gflops_per_ai) == Regime::MemBound {
                println!("  No crossover observed — all gates are memory-bound.");
                println!("  Increase MAX_GATE_QUBITS or use a CPU with higher peak FLOPS.");
            } else {
                println!("  No mem-bound baseline found — all probes appear compute-bound.");
                println!("  This is unexpected; check that PROFILE_N_QUBITS is large enough.");
            }
        }
    }
    println!();
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let (profile, mut measurements) = explore(&args)?;
    measurements.sort_by(|a, b| a.ai_target.cmp(&b.ai_target).then(a.k.cmp(&b.k)));
    print_report(&profile, &measurements, &args);
    Ok(())
}

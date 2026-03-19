//! CUDA kernel roofline profiler: adaptive crossover detection.
//!
//! Locates the arithmetic-intensity crossover between the memory-bound and
//! compute-bound regimes of the gate simulation kernel using three phases:
//!
//! 1. **BW calibration** — measures device peak memory bandwidth via D2D
//!    async memcpy (no gate involved).
//! 2. **Heuristic escalation** — probes AI = 2, 4, 8, … (doubling) with the
//!    smallest gate that achieves each AI, stopping at the first non-mem-bound
//!    measurement.  Never pre-commits to a large dense gate.
//! 3. **Coarse sweep + bisection** — probes the remaining power-of-2 AI
//!    points to build the full curve, then bisects between the last mem-bound
//!    and first non-mem-bound measurement to pinpoint the crossover.
//!
//! GPU time is measured via CUDA events (`SyncStats::kernels[*].gpu_time`),
//! not CPU wall-clock, for sub-microsecond accuracy.
//!
//! ## Statevector size
//!
//! The benchmark automatically clamps the profiling statevector to
//! [`PROFILE_N_QUBITS`] qubits regardless of `--n-qubits`, because:
//!   - Too small → L2 cache dominates, masking the true HBM/GDDR bandwidth.
//!   - Too large → allocation + first-touch time dominates the budget.
//! The `--n-qubits` argument is recorded in the report header only.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin cuda_crossover --features cuda --release             # auto-detect SM
//! cargo run --bin cuda_crossover --features cuda --release -- --sm 89  # override SM
//! cargo run --bin cuda_crossover --features cuda --release -- --help
//! ```

use anyhow::Result;
use cast::cuda::{
    device_sm, measure_peak_bw_gib_s, CudaKernelGenSpec, CudaKernelId, CudaKernelManager,
    CudaPrecision, CudaStatevector,
};
use cast::types::{Complex, ComplexSquareMatrix, QuantumGate};
use clap::{Parser, ValueEnum};
use rand::Rng;
use std::str::FromStr;
use std::time::Instant;

// ── Profiling constants ───────────────────────────────────────────────────────

/// Statevector size used for all gate benchmarks and the BW test.
/// 28 qubits ≈ 4 GiB (F64) / 2 GiB (F32) — large enough to spill all GPU
/// caches on any current device while remaining quickly allocatable.
const PROFILE_N_QUBITS: usize = 28;

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
enum CliCudaPrecision {
    F32,
    F64,
}

impl From<CliCudaPrecision> for CudaPrecision {
    fn from(v: CliCudaPrecision) -> Self {
        match v {
            CliCudaPrecision::F32 => CudaPrecision::F32,
            CliCudaPrecision::F64 => CudaPrecision::F64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SmTarget {
    major: u32,
    minor: u32,
}

impl FromStr for SmTarget {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() < 2 {
            return Err("expected e.g. 86 or 120".into());
        }
        let (major, minor) = s.split_at(s.len() - 1);
        Ok(Self {
            major: major
                .parse()
                .map_err(|_| format!("bad SM major in {s:?}"))?,
            minor: minor
                .parse()
                .map_err(|_| format!("bad SM minor in {s:?}"))?,
        })
    }
}

#[derive(Parser, Debug)]
#[command(name = "cuda_crossover")]
struct Args {
    /// Target qubit count for large simulations (recorded in report only).
    /// The profiling statevector is clamped to PROFILE_N_QUBITS internally.
    #[arg(long, default_value_t = 30)]
    n_qubits: usize,

    #[arg(long, value_enum, default_value_t = CliCudaPrecision::F64)]
    precision: CliCudaPrecision,

    /// CUDA SM target, e.g. 86 or 120. Auto-detected from device 0 if omitted.
    #[arg(long = "sm")]
    sm: Option<SmTarget>,

    /// Total wall-time budget for the gate sweep (seconds; BW test is extra).
    #[arg(long, default_value_t = 120.0)]
    budget_secs: f64,

    /// Minimum timed GPU iterations per gate probe.
    #[arg(long, default_value_t = 10)]
    min_iters: usize,

    /// Warmup GPU iterations per gate probe (not timed).
    #[arg(long, default_value_t = 5)]
    n_warmup: usize,

    /// Warmup iterations for the D2D memcpy BW test.
    #[arg(long, default_value_t = 10)]
    bw_warmup: usize,

    /// Timed iterations for the D2D memcpy BW test.
    #[arg(long, default_value_t = 50)]
    bw_iters: usize,
}

impl Args {
    /// Resolve `--sm`: use the explicit value if given, otherwise auto-detect from device 0.
    fn resolve_sm(&mut self) -> Result<()> {
        if self.sm.is_none() {
            let (major, minor) = device_sm()?;
            self.sm = Some(SmTarget { major, minor });
        }
        Ok(())
    }

    fn sm(&self) -> SmTarget {
        self.sm.expect("SM must be resolved before use")
    }

    fn precision(&self) -> CudaPrecision {
        self.precision.into()
    }

    fn spec(&self) -> CudaKernelGenSpec {
        let sm = self.sm();
        let p = self.precision();
        let tol = if matches!(self.precision, CliCudaPrecision::F32) {
            1e-6
        } else {
            1e-12
        };
        CudaKernelGenSpec {
            precision: p,
            ztol: tol,
            otol: tol,
            sm_major: sm.major,
            sm_minor: sm.minor,
        }
    }

    fn scalar_size(&self) -> usize {
        match self.precision {
            CliCudaPrecision::F32 => 4,
            CliCudaPrecision::F64 => 8,
        }
    }

    /// Number of qubits used for the actual profiling statevector.
    fn profile_n_qubits(&self) -> usize {
        PROFILE_N_QUBITS.min(self.n_qubits)
    }

    /// Total bytes in the profiling statevector (2 scalars × 2^n amplitudes).
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

// ── GPU event timing ──────────────────────────────────────────────────────────

struct GpuTiming {
    n_iters: usize,
    mean_s: f64,
    stddev_s: f64,
    cv: f64,
    min_s: f64,
    max_s: f64,
}

/// Times `apply + sync` using CUDA event GPU time.
///
/// Runs `n_warmup` un-recorded warm-up iterations, then probes [`N_PROBE`]
/// iterations to estimate the per-iteration cost and budget remaining time.
fn time_gpu(
    mgr: &CudaKernelManager,
    kid: CudaKernelId,
    sv: &mut CudaStatevector,
    n_warmup: usize,
    min_iters: usize,
    budget_s: f64,
) -> Result<GpuTiming> {
    const N_PROBE: usize = 3;

    for _ in 0..n_warmup {
        mgr.apply(kid, sv)?;
        mgr.sync()?;
    }

    // Phase 1: probe iterations — wall-clock tracks budget, GPU events track time.
    let mut samples: Vec<f64> = Vec::new();
    let probe_wall = Instant::now();
    for _ in 0..N_PROBE {
        mgr.apply(kid, sv)?;
        let stats = mgr.sync()?;
        samples.push(stats.kernels[0].gpu_time.as_secs_f64());
    }
    let probe_elapsed = probe_wall.elapsed().as_secs_f64();

    // Phase 2: fill remaining budget.
    let est_per_iter = probe_elapsed / N_PROBE as f64;
    let remaining = (budget_s - probe_elapsed).max(0.0);
    let n_fill = ((remaining / est_per_iter) as usize).max(min_iters.saturating_sub(N_PROBE));
    for _ in 0..n_fill {
        mgr.apply(kid, sv)?;
        let stats = mgr.sync()?;
        samples.push(stats.kernels[0].gpu_time.as_secs_f64());
    }

    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(GpuTiming {
        n_iters: samples.len(),
        mean_s: mean,
        stddev_s: stddev,
        cv: stddev / mean,
        min_s: min,
        max_s: max,
    })
}

// ── Measurement ───────────────────────────────────────────────────────────────

struct Measurement {
    ai_target: usize,
    k: usize,
    ai: f64,
    nnz: usize,
    timing: GpuTiming,
    gib_s: f64,
    gflops_s: f64,
}

fn measure_at(
    target_ai: usize,
    mgr: &CudaKernelManager,
    sv: &mut CudaStatevector,
    args: &Args,
    budget_s: f64,
) -> Result<Measurement> {
    let spec = args.spec();
    let k = gate_qubits_for_ai(target_ai);
    let gate = gate_for_ai(k, target_ai);
    let ai = gate.arithmatic_intensity(spec.ztol);
    let nnz = gate.scalar_nnz(spec.ztol);
    let kid = mgr.generate(&gate, spec)?;
    let timing = time_gpu(mgr, kid, sv, args.n_warmup, args.min_iters, budget_s)?;

    let n_sv = args.profile_n_qubits();
    let sv_bytes = args.profile_sv_bytes();
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
    /// Bidirectional peak device memory bandwidth from the D2D memcpy test.
    peak_bw_gib_s: f64,
    /// GFLOPs/s per unit AI predicted by the roofline model for a mem-bound gate.
    /// Derived from peak_bw_gib_s: BW_bytes_s / (2 × scalar_size) / 1e9.
    gflops_per_ai: f64,
    /// Best GFLOPs/s observed across all compute-bound (or transition) probes.
    peak_gflops_s: f64,
    /// Theoretical crossover AI = peak_gflops_s / gflops_per_ai.
    theoretical_crossover_ai: f64,
}

// ── Adaptive exploration ──────────────────────────────────────────────────────

fn explore(args: &Args) -> Result<(HardwareProfile, Vec<Measurement>)> {
    let spec = args.spec();
    let mgr = CudaKernelManager::new();
    let mut sv = CudaStatevector::new(args.profile_n_qubits() as u32, spec.precision)?;
    sv.zero()?;

    let sv_mib = args.profile_sv_bytes() / (1 << 20);
    eprintln!(
        "Profile SV: {} qubits = {} MiB ({:?}),  sm_{}{},  budget {:.0}s",
        args.profile_n_qubits(),
        sv_mib,
        args.precision,
        args.sm().major,
        args.sm().minor,
        args.budget_secs,
    );
    eprintln!();

    // ── Phase 1a: BW calibration (D2D memcpy, no gate) ───────────────────────

    eprintln!(
        "Phase 1a: BW calibration (D2D memcpy, {} warmup + {} timed)",
        args.bw_warmup, args.bw_iters
    );
    eprint!("  {} × {} MiB ... ", args.bw_iters, sv_mib);
    let peak_bw_gib_s =
        measure_peak_bw_gib_s(args.profile_sv_bytes(), args.bw_warmup, args.bw_iters)?;
    // gflops_per_ai: from the roofline model (see cuda_roofline_model memory entry).
    // T_mem_bound = 2 × sv_bytes / BW_bytes_s
    // GFLOPs/s    = ai × 2^n_sv × 2 / T / 1e9
    //             = ai × BW_bytes_s / (2 × scalar_size) / 1e9
    let gflops_per_ai =
        peak_bw_gib_s * (1u64 << 30) as f64 / (2.0 * args.scalar_size() as f64) / 1e9;
    eprintln!(
        "{:.1} GiB/s  →  {:.2} GFLOPs/s per unit AI",
        peak_bw_gib_s, gflops_per_ai
    );
    eprintln!();

    // ── Phase 1b: Heuristic escalation to find the compute-bound regime ───────

    eprintln!("Phase 1b: Heuristic escalation (AI = 2, 4, 8, … until non-mem-bound)");
    let mut measurements: Vec<Measurement> = Vec::new();
    let max_ai = 1usize << MAX_GATE_QUBITS;

    // Budget for Phase 1b: ~20% of total, split evenly over log2(max_ai) steps.
    let n_escalation_steps = MAX_GATE_QUBITS as f64;
    let escalation_budget_per = args.budget_secs * 0.20 / n_escalation_steps;

    let mut ai_probe = 2usize;
    let mut escalation_anchor_gflops: f64 = 0.0;
    while ai_probe <= max_ai {
        let k = gate_qubits_for_ai(ai_probe);
        eprint!("  ai={ai_probe:>4} k={k} ... ");
        let m = measure_at(ai_probe, &mgr, &mut sv, args, escalation_budget_per)?;
        let r = classify(&m, gflops_per_ai);
        eprintln!(
            "{:.3} ms  {:.1} GiB/s  {:.1}/{:.1} GFLOPs/s  [{r:?}]",
            m.timing.mean_s * 1e3,
            m.gib_s,
            m.gflops_s,
            gflops_per_ai * m.ai,
        );
        let found_non_mem = r != Regime::MemBound;
        escalation_anchor_gflops = escalation_anchor_gflops.max(m.gflops_s);
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
        eprintln!("  Consider increasing MAX_GATE_QUBITS or using a GPU with higher peak FLOPS.");
    }
    eprintln!();

    eprintln!();

    // ── Phase 2: Coarse sweep — fill in remaining power-of-2 AI points ───────

    eprintln!("Phase 2: Coarse sweep (remaining power-of-2 AI values)");
    let already_measured: std::collections::HashSet<usize> =
        measurements.iter().map(|m| m.ai_target).collect();
    let coarse_ais: Vec<usize> = (1..=MAX_GATE_QUBITS)
        .map(|e| 1usize << e)
        .filter(|&ai| !already_measured.contains(&ai))
        .collect();

    // Budget: remaining after Phase 1b, divided over coarse + bisection probes.
    // Allocate ~55% of total to coarse and ~25% to bisection.
    let coarse_budget_per = if coarse_ais.is_empty() {
        0.0
    } else {
        args.budget_secs * 0.55 / coarse_ais.len() as f64
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
        let m = measure_at(ai, &mgr, &mut sv, args, coarse_budget_per)?;
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

    // ── Phase 3: Integer bisection to refine the crossover ───────────────────

    eprintln!("Phase 3: Bisection refinement");

    // Sort to find the tightest mem-bound / non-mem-bound bracket.
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
        let m = measure_at(ai_mid, &mgr, &mut sv, args, bisect_budget_per)?;
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

    // ── Build hardware profile from all measurements ─────────────────────────

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
    println!("  CUDA Roofline Profile Report");
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
    println!(
        "  CUDA SM target     : sm_{}{}",
        args.sm().major,
        args.sm().minor
    );
    println!(
        "  budget             : {:.0} s gate sweep  +  BW test",
        args.budget_secs
    );
    println!();
    println!(
        "  ── Hardware calibration (D2D memcpy) ─────────────────────────────────────────────────"
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
        "mean GPU ms",
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
        "ai", "k", "n_iters", "mean GPU ms", "stddev ms", "min ms", "max ms", "CV %",
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
                println!("  Increase MAX_GATE_QUBITS or use a GPU with higher peak FLOPS.");
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
    let mut args = Args::parse();
    args.resolve_sm()?;
    let (profile, mut measurements) = explore(&args)?;
    measurements.sort_by(|a, b| a.ai_target.cmp(&b.ai_target).then(a.k.cmp(&b.k)));
    print_report(&profile, &measurements, &args);
    Ok(())
}

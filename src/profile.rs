//! Adaptive roofline profiler for CPU and CUDA backends.
//!
//! Measures the memory/compute crossover point by sweeping gate kernels across
//! a range of arithmetic intensities and fitting a **two-segment piecewise
//! roofline model**.
//!
//! # The roofline model
//!
//! A gate kernel's throughput is limited by whichever hardware resource
//! bottlenecks first — memory bandwidth or compute:
//!
//! ```text
//!   GFLOPs/s = min(bw_slope × AI,  peak_gflops)
//!
//!   where  AI  = arithmetic intensity (FLOPs per element)
//! ```
//!
//! This gives two regimes:
//!
//! - **Memory-bound** (AI < crossover): throughput grows linearly with AI.
//!   The slope (`bw_slope`, in GFLOPs/s per unit AI) is set by memory
//!   bandwidth — more FLOPs per byte transferred means more useful work per
//!   memory pass, but the memory wall limits how fast data arrives.
//!
//! - **Compute-bound** (AI ≥ crossover): throughput plateaus at `peak_gflops`.
//!   The ALUs are fully saturated; adding more FLOPs per element only makes
//!   the kernel slower without saving memory traffic.
//!
//! The **crossover AI** is where the two segments meet:
//!
//! ```text
//!   crossover_ai = peak_gflops / bw_slope
//! ```
//!
//! For gate fusion this means: fusing gates whose combined AI stays below the
//! crossover is free (same memory-bound cost, fewer kernel launches).  Fusing
//! past the crossover pushes into compute-bound territory with diminishing
//! returns.
//!
//! # Fitting procedure
//!
//! 1. **Seed phase**: probe at AI = 1, 2, 4, 8, …, 2^MAX_GATE_QUBITS.
//! 2. **Refinement**: fit the two-segment model, add probes near the estimated
//!    crossover, repeat until R² stabilises or the time budget runs out.
//! 3. **Output**: a [`HardwareProfile`] with the fitted parameters and
//!    (optionally) the raw sweep data for plotting.

use std::sync::Arc;
use std::time::Instant;

use crate::cost_model::{Device, HardwareProfile, ProfileConfig, SweepEntry};
use crate::types::QuantumGate;

// ── Constants ────────────────────────────────────────────────────────────────

/// Largest gate size (qubits); caps the probed AI range at `2^MAX_GATE_QUBITS`.
const MAX_GATE_QUBITS: u32 = 6;

/// R² threshold for declaring the roofline fit stable.
const R2_STABLE: f64 = 0.95;

/// Maximum change in R² between rounds before the fit is declared stable.
const R2_DELTA: f64 = 0.01;

/// R² below which we emit a warning.
const R2_WARN: f64 = 0.90;

/// Maximum refinement rounds in the adaptive sweep.
const MAX_REFINE_ROUNDS: u32 = 5;

/// Maximum new probe candidates per refinement round.
const MAX_CANDIDATES_PER_ROUND: usize = 3;

// ── Probe result ─────────────────────────────────────────────────────────────

/// A single measured data point from the roofline sweep.
///
/// Each probe runs a gate kernel at a target arithmetic intensity and records
/// the observed throughput in two units: compute (GFLOPs/s) and memory (GiB/s).
struct ProbeResult {
    /// Actual arithmetic intensity of the probed gate (FLOPs per element).
    ai: f64,
    /// Observed compute throughput (GFLOPs/s).
    gflops_s: f64,
    /// Observed memory bandwidth (GiB/s, bidirectional read + write).
    gib_s: f64,
}

// ── Roofline parameters ──────────────────────────────────────────────────────

/// Parameters of a fitted two-segment roofline model.
///
/// ```text
///   predicted GFLOPs/s = min(bw_slope × AI,  peak_gflops)
///                      = min(bw_slope × AI,  bw_slope × crossover_ai)
/// ```
struct RooflineParams {
    /// The AI at which memory-bound meets compute-bound.
    crossover_ai: f64,
    /// GFLOPs/s per unit AI in the memory-bound regime (the slope of the
    /// rising segment).  Derived from memory bandwidth and scalar size.
    bw_slope: f64,
}

impl RooflineParams {
    /// Peak compute throughput implied by the fit: `bw_slope × crossover_ai`.
    fn peak_gflops(&self) -> f64 {
        self.bw_slope * self.crossover_ai
    }

    /// Predicted GFLOPs/s for a given AI under this roofline.
    fn predicted_gflops(&self, ai: f64) -> f64 {
        (self.bw_slope * ai).min(self.peak_gflops())
    }
}

// ── Roofline fitting ─────────────────────────────────────────────────────────

/// Fits a two-segment piecewise roofline to measured sweep data.
///
/// **Slope estimation**: takes the lower half of the sweep (sorted by AI),
/// computes `gflops_s / ai` for each point, and uses the median as `bw_slope`.
/// The median is robust against outliers near the crossover where throughput
/// starts to plateau.
///
/// **Crossover**: `peak_gflops / bw_slope`, where `peak_gflops` is the maximum
/// observed throughput across all sweep points.
fn fit_roofline(sweep: &[ProbeResult]) -> RooflineParams {
    let n = sweep.len();
    let lower = &sweep[..n.div_ceil(2)];
    let mut ratios: Vec<f64> = lower.iter().map(|p| p.gflops_s / p.ai).collect();
    ratios.sort_by(|a, b| a.total_cmp(b));
    let bw_slope = ratios[ratios.len() / 2]; // median

    let peak_gflops = sweep.iter().map(|p| p.gflops_s).fold(0.0_f64, f64::max);

    let crossover_ai = if bw_slope > 0.0 {
        peak_gflops / bw_slope
    } else {
        f64::INFINITY
    };

    RooflineParams {
        crossover_ai,
        bw_slope,
    }
}

/// Coefficient of determination (R²) for the roofline fit.
///
/// Measures how well the two-segment model explains the observed GFLOPs/s
/// values.  R² = 1.0 means a perfect fit; values below [`R2_WARN`] suggest
/// the hardware doesn't follow a clean roofline (e.g. thermal throttling,
/// cache effects).
fn roofline_r2(sweep: &[ProbeResult], params: &RooflineParams) -> f64 {
    let gflops: Vec<f64> = sweep.iter().map(|p| p.gflops_s).collect();
    let mean = gflops.iter().sum::<f64>() / gflops.len() as f64;
    let ss_tot: f64 = gflops.iter().map(|g| (g - mean).powi(2)).sum();
    if ss_tot < 1e-30 {
        return 1.0;
    }
    let ss_res: f64 = sweep
        .iter()
        .map(|p| (p.gflops_s - params.predicted_gflops(p.ai)).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

// ── Progress bar ─────────────────────────────────────────────────────────────

/// Renders a single-line progress bar on stderr, overwriting the previous line.
///
/// Format: `label  [████░░░░░░]  phase  pts  R²  elapsed/budget`
fn progress_bar(
    label: &str,
    phase: &str,
    n_points: usize,
    max_points: usize,
    r2: f64,
    elapsed_s: f64,
    budget_s: f64,
) {
    use std::io::Write;

    const BAR_WIDTH: usize = 20;
    let frac = (elapsed_s / budget_s).clamp(0.0, 1.0);
    let filled = (frac * BAR_WIDTH as f64).round() as usize;
    let empty = BAR_WIDTH - filled;
    let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(empty);

    let r2_str = if r2 > 0.0 {
        format!("R\u{00b2}={r2:.3}")
    } else {
        "R\u{00b2}=---".to_string()
    };

    // \x1b[2K erases the entire line, \r returns to column 0.
    eprint!(
        "\x1b[2K\r  {label}  [{bar}]  {phase:<8} {n_points:>2}/{max_points} pts  \
         {r2_str}  {elapsed_s:.0}/{budget_s:.0}s",
    );
    let _ = std::io::stderr().flush();
}

// ── Adaptive sweep ───────────────────────────────────────────────────────────

/// Drives the adaptive roofline sweep shared between CPU and CUDA profiling.
///
/// `probe_fn` is called with `(target_ai, per_point_budget_s)` and must return
/// a [`ProbeResult`] with the measured throughput at (approximately) that AI.
///
/// The sweep proceeds in two phases:
/// 1. **Seed**: probe at powers of two (AI = 1, 2, 4, …, 2^MAX_GATE_QUBITS).
/// 2. **Refine**: fit the roofline, add probes near the estimated crossover,
///    repeat until R² stabilises or the wall-clock budget is exhausted.
fn adaptive_sweep(
    mut probe_fn: impl FnMut(f64, f64) -> anyhow::Result<ProbeResult>,
    config: ProfileConfig,
    budget_s: f64,
    label: &str,
) -> anyhow::Result<HardwareProfile> {
    let seed_ais: Vec<f64> = (0..=MAX_GATE_QUBITS).map(|e| (1u64 << e) as f64).collect();

    let max_points = seed_ais.len() + MAX_REFINE_ROUNDS as usize * MAX_CANDIDATES_PER_ROUND;
    let per_point_budget = budget_s / max_points as f64;
    let wall_start = Instant::now();

    // ── Seed phase ──────────────────────────────────────────────────────────
    let mut sweep: Vec<ProbeResult> = Vec::new();
    for &target_ai in &seed_ais {
        progress_bar(
            label,
            "seed",
            sweep.len(),
            max_points,
            0.0,
            wall_start.elapsed().as_secs_f64(),
            budget_s,
        );
        sweep.push(probe_fn(target_ai, per_point_budget)?);
    }
    sweep.sort_by(|a, b| a.ai.total_cmp(&b.ai));

    // ── Refinement loop ─────────────────────────────────────────────────────
    let mut prev_r2 = 0.0_f64;
    for round in 0..MAX_REFINE_ROUNDS {
        let params = fit_roofline(&sweep);
        let r2 = roofline_r2(&sweep, &params);

        progress_bar(
            label,
            &format!("refine {}", round + 1),
            sweep.len(),
            max_points,
            r2,
            wall_start.elapsed().as_secs_f64(),
            budget_s,
        );

        if r2 >= R2_STABLE && (r2 - prev_r2).abs() < R2_DELTA {
            break;
        }
        prev_r2 = r2;

        if wall_start.elapsed().as_secs_f64() >= budget_s {
            break;
        }

        // Probe near the estimated crossover to sharpen the fit.
        let cx = params.crossover_ai;
        let candidates: Vec<f64> = [cx * 0.5, cx, cx * 1.5]
            .iter()
            .map(|&v| v.round().max(1.0))
            .filter(|&ai| !sweep.iter().any(|p| (p.ai - ai).abs() < 0.5))
            .take(MAX_CANDIDATES_PER_ROUND)
            .collect();
        if candidates.is_empty() {
            break;
        }
        for target_ai in candidates {
            progress_bar(
                label,
                &format!("refine {}", round + 1),
                sweep.len(),
                max_points,
                prev_r2,
                wall_start.elapsed().as_secs_f64(),
                budget_s,
            );
            sweep.push(probe_fn(target_ai, per_point_budget)?);
        }
        sweep.sort_by(|a, b| a.ai.total_cmp(&b.ai));
    }

    // ── Finalize ────────────────────────────────────────────────────────────
    let params = fit_roofline(&sweep);
    let r2 = roofline_r2(&sweep, &params);

    progress_bar(
        label,
        "done",
        sweep.len(),
        max_points,
        r2,
        wall_start.elapsed().as_secs_f64(),
        budget_s,
    );
    eprint!("\x1b[2K\r");

    if r2 < R2_WARN {
        eprintln!(
            "  {label}: Warning: roofline fit R\u{00b2} = {r2:.3} (< {R2_WARN}); \
             crossover estimate may be unreliable."
        );
    }

    let peak_bw_gib_s = sweep.iter().map(|p| p.gib_s).fold(0.0_f64, f64::max);
    let peak_gflops_s = sweep.iter().map(|p| p.gflops_s).fold(0.0_f64, f64::max);
    let raw: Vec<SweepEntry> = sweep
        .iter()
        .map(|p| SweepEntry {
            ai: p.ai,
            gflops_s: p.gflops_s,
            gib_s: p.gib_s,
        })
        .collect();
    let mut profile =
        HardwareProfile::from_measurements(config, peak_bw_gib_s, peak_gflops_s, params.bw_slope);
    profile.raw = raw;
    Ok(profile)
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Profiles hardware by sweeping gate kernels across arithmetic intensities.
///
/// This is the unified profiling entry point for all backends.  The caller
/// provides a `time_kernel` closure that, given a [`QuantumGate`] and a
/// wall-clock budget in seconds, **compiles and executes** the kernel and
/// returns the **mean execution time in seconds** (excluding JIT compilation
/// time).
///
/// `config` describes the profiling environment (backend, precision, etc.).
/// `n_qubits` and `scalar_bytes` determine the statevector geometry used to
/// convert execution time into GiB/s and GFLOPs/s.
pub fn measure(
    config: ProfileConfig,
    n_qubits: u32,
    scalar_bytes: usize,
    budget_s: f64,
    mut time_kernel: impl FnMut(&QuantumGate, f64) -> anyhow::Result<f64>,
) -> anyhow::Result<HardwareProfile> {
    let ztol = 1e-12;
    let label = format!(
        "{} {}",
        config.device,
        config.precision.to_ascii_uppercase(),
    );

    let sv_n_elements = 1usize << n_qubits; // number of complex amplitudes
    let sv_bytes = sv_n_elements * 2 * scalar_bytes; // total bytes (re + im)

    adaptive_sweep(
        |target_ai, per_point_budget| {
            let gate = QuantumGate::random_arithmatic_intensity(n_qubits, target_ai, ztol);
            let actual_ai = gate.arithmatic_intensity(ztol);
            let mean_s = time_kernel(&gate, per_point_budget)?;

            let gib_s = 2.0 * sv_bytes as f64 / mean_s / (1u64 << 30) as f64;
            let gflops_s = actual_ai * sv_n_elements as f64 * 2.0 / mean_s / 1e9;
            Ok(ProbeResult {
                ai: actual_ai,
                gflops_s,
                gib_s,
            })
        },
        config,
        budget_s,
        &label,
    )
}

// ── Backend convenience functions ────────────────────────────────────────────

/// Profiles CPU hardware.
///
/// `n_qubits` sets the statevector size for profiling.  Thread count comes
/// from `CAST_NUM_THREADS` (see [`crate::cpu::get_num_threads`]).
/// **Profile at the thread count you will simulate with.**
pub fn measure_cpu(
    spec: &crate::cpu::CPUKernelGenSpec,
    n_qubits: u32,
    budget_s: f64,
) -> anyhow::Result<HardwareProfile> {
    use crate::cpu::{get_num_threads, get_simd_s, CPUStatevector, CpuKernelManager};

    let n_threads = get_num_threads();
    let min_qubits = MAX_GATE_QUBITS + get_simd_s(spec.simd_width, spec.precision) + 1;
    anyhow::ensure!(
        n_qubits >= min_qubits,
        "n_qubits ({n_qubits}) too small for CPU profiling; need at least {min_qubits} \
         (MAX_GATE_QUBITS={MAX_GATE_QUBITS} + simd_s={} + 1)",
        get_simd_s(spec.simd_width, spec.precision),
    );
    let scalar_bytes = spec.precision.scalar_bytes();

    let mut sv = CPUStatevector::new(n_qubits, spec.precision, spec.simd_width);
    sv.initialize();
    let mgr = CpuKernelManager::new();

    let config = ProfileConfig {
        device: Device::Cpu {
            name: String::new(),
            n_threads,
        },
        precision: format!("f{}", scalar_bytes * 8),
        n_qubits,
    };

    measure(config, n_qubits, scalar_bytes, budget_s, |gate, budget| {
        let gate = Arc::new(gate.clone());
        let kid = mgr.generate(spec, &gate)?;
        let timing = mgr.time_adaptive(kid, &mut sv, n_threads, budget)?;
        Ok(timing.mean_s)
    })
}

/// Profiles CUDA hardware.
///
/// `n_qubits` sets the statevector size for profiling.  GPU time is measured
/// via CUDA events for sub-microsecond accuracy (reported by
/// [`CudaKernelManager::time_adaptive`]).
#[cfg(feature = "cuda")]
pub fn measure_cuda(
    spec: &crate::cuda::CudaKernelGenSpec,
    n_qubits: u32,
    budget_s: f64,
) -> anyhow::Result<HardwareProfile> {
    use crate::cuda::{CudaKernelManager, CudaStatevector};

    let scalar_bytes = spec.precision.scalar_bytes();
    let mgr = CudaKernelManager::new();
    let mut sv = CudaStatevector::new(n_qubits, spec.precision)?;
    sv.zero()?;

    let config = ProfileConfig {
        device: Device::Cuda {
            name: String::new(),
            sm_major: spec.sm_major,
            sm_minor: spec.sm_minor,
        },
        precision: format!("f{}", scalar_bytes * 8),
        n_qubits,
    };

    measure(config, n_qubits, scalar_bytes, budget_s, |gate, budget| {
        let gate = Arc::new(gate.clone());
        let kid = mgr.generate(&gate, *spec)?;
        let timing = mgr.time_adaptive(kid, &mut sv, budget)?;
        Ok(timing.mean_s)
    })
}

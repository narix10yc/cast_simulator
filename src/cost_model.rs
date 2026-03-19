use crate::types::QuantumGate;
use std::time::Instant;

/// Estimates the cost of applying a gate, in units of GiB-time (seconds per GiB
/// of statevector memory updated).
///
/// The fusion optimizer accepts a fusion when the ratio of pre-fusion to
/// post-fusion cost exceeds `1 + benefit_margin`.
pub trait CostModel: Send + Sync {
    fn cost_of(&self, gate: &QuantumGate) -> f64;
}

// ── Size-only cost model ─────────────────────────────────────────────────────

/// Binary accept/reject model: gates that fit within the qubit and
/// arithmetic-intensity budgets are "free" (`1e-10`); everything else is
/// "expensive" (`1.0`).
pub struct SizeOnlyCostModel {
    pub max_size: usize,
    pub max_ai: usize,
    pub zero_tol: f64,
}

impl CostModel for SizeOnlyCostModel {
    fn cost_of(&self, gate: &QuantumGate) -> f64 {
        if gate.n_qubits() > self.max_size {
            return 1.0;
        }
        if gate.arithmatic_intensity(self.zero_tol) > self.max_ai as f64 {
            return 1.0;
        }
        1e-10
    }
}

// ── Hardware profile ─────────────────────────────────────────────────────────

/// Roofline hardware profile used by [`HardwareAdaptiveCostModel`] and the
/// crossover profiling binaries.
///
/// Captures the three independent hardware roofline parameters — peak memory
/// bandwidth, peak compute throughput, and scalar element size — and caches
/// the derived quantities (memory-bound GFLOPs/s slope and crossover AI).
///
/// The crossover AI is the arithmetic intensity where gate simulation
/// transitions from memory-bound to compute-bound.  Below the crossover
/// every gate costs the same (one statevector pass); above it compute
/// dominates and fusion yields diminishing benefit.
///
/// # Construction
///
/// ```ignore
/// // Automatic CPU profiling (adaptive roofline sweep):
/// let p = HardwareProfile::measure_cpu(&spec, 30.0)?;
///
/// // Automatic CUDA profiling (requires `cuda` feature):
/// let p = HardwareProfile::measure_cuda(&spec, 30.0)?;
///
/// // From raw roofline parameters:
/// let p = HardwareProfile::from_roofline(peak_bw_gib_s, peak_gflops_s, 8);
///
/// // From directly measured quantities (crossover binaries):
/// let p = HardwareProfile::from_measurements(peak_bw_gib_s, peak_gflops_s, 8, gflops_per_ai);
/// ```
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Peak effective bidirectional bandwidth (GiB/s, read + write).
    peak_bw_gib_s: f64,
    /// Peak observed compute throughput (GFLOPs/s).
    peak_gflops_s: f64,
    /// Scalar element size in bytes (8 for f64, 4 for f32).
    scalar_bytes: usize,
    /// Memory-bound slope: GFLOPs/s per unit arithmetic intensity.
    /// Derived: `peak_bw_gib_s × 2^30 / (2 × scalar_bytes) / 1e9`.
    gflops_per_ai: f64,
    /// Roofline crossover point: the AI where memory-bound meets compute-bound.
    /// Derived: `peak_gflops_s / gflops_per_ai`.
    crossover_ai: f64,
}

/// Computes the memory-bound slope (GFLOPs/s per unit AI) from bandwidth and
/// scalar size.
///
/// In the memory-bound regime the statevector is read and written once per gate
/// application:  `T = 2 × sv_bytes / BW`.  The GFLOPs/s at arithmetic
/// intensity `ai` is therefore `ai × BW_bytes / (2 × scalar_bytes) / 1e9`.
fn compute_gflops_per_ai(peak_bw_gib_s: f64, scalar_bytes: usize) -> f64 {
    peak_bw_gib_s * (1u64 << 30) as f64 / (2.0 * scalar_bytes as f64) / 1e9
}

impl HardwareProfile {
    /// Derives the full profile from peak bandwidth and peak compute.
    ///
    /// `scalar_bytes`: 8 for f64, 4 for f32.  **Measure `peak_gflops_s` at the
    /// thread count you will simulate with** — compute scales with threads while
    /// DRAM bandwidth does not.
    pub fn from_roofline(peak_bw_gib_s: f64, peak_gflops_s: f64, scalar_bytes: usize) -> Self {
        let gflops_per_ai = compute_gflops_per_ai(peak_bw_gib_s, scalar_bytes);
        let crossover_ai = if gflops_per_ai > 0.0 && peak_gflops_s > 0.0 {
            peak_gflops_s / gflops_per_ai
        } else {
            f64::INFINITY
        };
        Self {
            peak_bw_gib_s,
            peak_gflops_s,
            scalar_bytes,
            gflops_per_ai,
            crossover_ai,
        }
    }

    /// Constructs from directly measured quantities.
    ///
    /// Use this when `gflops_per_ai` has been measured via BW calibration
    /// (e.g. D2D memcpy or AI=1 gate probe) rather than derived from
    /// `peak_bw_gib_s` and `scalar_bytes`.  The measured slope is preferred
    /// because it reflects actual memory-system behaviour more accurately.
    pub fn from_measurements(
        peak_bw_gib_s: f64,
        peak_gflops_s: f64,
        scalar_bytes: usize,
        gflops_per_ai: f64,
    ) -> Self {
        let crossover_ai = if gflops_per_ai > 0.0 && peak_gflops_s > 0.0 {
            peak_gflops_s / gflops_per_ai
        } else {
            f64::INFINITY
        };
        Self {
            peak_bw_gib_s,
            peak_gflops_s,
            scalar_bytes,
            gflops_per_ai,
            crossover_ai,
        }
    }

    pub fn peak_bw_gib_s(&self) -> f64 {
        self.peak_bw_gib_s
    }
    pub fn peak_gflops_s(&self) -> f64 {
        self.peak_gflops_s
    }
    pub fn scalar_bytes(&self) -> usize {
        self.scalar_bytes
    }
    pub fn gflops_per_ai(&self) -> f64 {
        self.gflops_per_ai
    }
    pub fn crossover_ai(&self) -> f64 {
        self.crossover_ai
    }
}

// ── Hardware-adaptive cost model ─────────────────────────────────────────────

/// Roofline-based cost model: `cost = max(1.0, ai / crossover_ai)`.
///
/// Memory-bound gates (below the crossover) all cost 1.0 — fusing them always
/// helps. Compute-bound gates cost proportionally more, so the optimizer
/// rejects fusions that push past the crossover. Gates exceeding `max_size`
/// qubits return infinity to prevent them from ever being proposed.
pub struct HardwareAdaptiveCostModel {
    pub crossover_ai: f64,
    pub max_size: usize,
    pub zero_tol: f64,
}

impl HardwareAdaptiveCostModel {
    pub fn new(profile: &HardwareProfile, max_size: usize) -> Self {
        Self {
            crossover_ai: profile.crossover_ai(),
            max_size,
            zero_tol: 1e-12,
        }
    }
}

impl CostModel for HardwareAdaptiveCostModel {
    fn cost_of(&self, gate: &QuantumGate) -> f64 {
        if gate.n_qubits() > self.max_size {
            return f64::INFINITY;
        }
        (gate.arithmatic_intensity(self.zero_tol) / self.crossover_ai).max(1.0)
    }
}

// ── Fusion config ────────────────────────────────────────────────────────────

/// Configuration for the adaptive gate-fusion optimizer.
///
/// `size_max` caps the qubit count of any fused gate. `benefit_margin` is the
/// minimum `old_cost / (new_cost + ε) - 1` for a fusion to be accepted. Phase 1
/// (size-2 canonicalization) always runs before the agglomerative phase.
pub struct FusionConfig {
    pub size_max: usize,
    pub benefit_margin: f64,
    pub cost_model: Box<dyn CostModel>,
}

impl FusionConfig {
    /// Pure size-gated fusion up to `max_size` qubits. No AI cap.
    pub fn size_only(max_size: usize) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size,
                max_ai: usize::MAX,
                zero_tol: 0.0,
            }),
        }
    }

    pub fn mild() -> Self {
        Self::size_only(5)
    }
    pub fn balanced() -> Self {
        Self::size_only(6)
    }
    pub fn aggressive() -> Self {
        Self::size_only(7)
    }

    /// Roofline-adaptive fusion. The crossover in `profile` must match the
    /// precision and thread count of your simulation.
    pub fn hardware_adaptive(profile: &HardwareProfile, max_size: usize) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(HardwareAdaptiveCostModel::new(profile, max_size)),
        }
    }
}

// ── Profiling ────────────────────────────────────────────────────────────────

/// Statevector size used for all profiling benchmarks.
/// 28 qubits ~ 4 GiB (F64) / 2 GiB (F32) — large enough to spill all caches
/// while remaining quickly allocatable.
const PROFILE_N_QUBITS: u32 = 28;

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

/// Fits a two-segment piecewise roofline to `(ai, gflops_s, gib_s)` sweep data.
///
/// Returns `(crossover_ai, c_slope)` where `c_slope` is GFLOPs/s per AI unit
/// in the memory-bound regime. The crossover is `peak_gflops / c_slope`.
fn fit_crossover(sweep: &[(f64, f64, f64)]) -> (f64, f64) {
    let n = sweep.len();
    let lower = &sweep[..n.div_ceil(2)];
    let mut ratios: Vec<f64> = lower.iter().map(|(ai, g, _)| g / ai).collect();
    ratios.sort_by(|a, b| a.total_cmp(b));
    let c_slope = ratios[ratios.len() / 2]; // median
    let peak_gflops = sweep.iter().map(|(_, g, _)| *g).fold(0.0_f64, f64::max);
    let crossover = if c_slope > 0.0 {
        peak_gflops / c_slope
    } else {
        f64::INFINITY
    };
    (crossover, c_slope)
}

/// R² of the piecewise roofline fit.
fn roofline_r2(sweep: &[(f64, f64, f64)], crossover: f64, c_slope: f64) -> f64 {
    let peak = c_slope * crossover;
    let gflops: Vec<f64> = sweep.iter().map(|(_, g, _)| *g).collect();
    let mean = gflops.iter().sum::<f64>() / gflops.len() as f64;
    let ss_tot: f64 = gflops.iter().map(|g| (g - mean).powi(2)).sum();
    if ss_tot < 1e-30 {
        return 1.0;
    }
    let ss_res: f64 = sweep
        .iter()
        .map(|(ai, g, _)| (g - (c_slope * ai).min(peak)).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

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
    let bar: String =
        "\u{2588}".repeat(filled) + &"\u{2591}".repeat(empty);

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

/// Drives the adaptive roofline sweep shared between CPU and CUDA profiling.
///
/// `probe` is called with `(target_ai, per_point_budget)` and must return
/// `(actual_ai, gflops_s, gib_s)`.
fn adaptive_sweep(
    mut probe: impl FnMut(f64, f64) -> anyhow::Result<(f64, f64, f64)>,
    scalar_bytes: usize,
    budget_s: f64,
    label: &str,
) -> anyhow::Result<HardwareProfile> {
    let seed_ais: Vec<f64> = (0..=MAX_GATE_QUBITS)
        .map(|e| (1u64 << e) as f64)
        .filter(|&ai| ai <= (1u64 << MAX_GATE_QUBITS) as f64)
        .collect();

    let max_points =
        seed_ais.len() + MAX_REFINE_ROUNDS as usize * MAX_CANDIDATES_PER_ROUND;
    let per_point_budget = budget_s / max_points as f64;
    let wall_start = Instant::now();

    // ── Seed phase ──────────────────────────────────────────────────────────
    let mut sweep: Vec<(f64, f64, f64)> = Vec::new();
    for &target_ai in &seed_ais {
        progress_bar(label, "seed", sweep.len(), max_points, 0.0,
                     wall_start.elapsed().as_secs_f64(), budget_s);
        sweep.push(probe(target_ai, per_point_budget)?);
    }
    sweep.sort_by(|a, b| a.0.total_cmp(&b.0));

    // ── Refinement loop ─────────────────────────────────────────────────────
    let mut prev_r2 = 0.0_f64;
    for round in 0..MAX_REFINE_ROUNDS {
        let (crossover, c_slope) = fit_crossover(&sweep);
        let r2 = roofline_r2(&sweep, crossover, c_slope);

        progress_bar(label, &format!("refine {}", round + 1), sweep.len(),
                     max_points, r2, wall_start.elapsed().as_secs_f64(), budget_s);

        if r2 >= R2_STABLE && (r2 - prev_r2).abs() < R2_DELTA {
            break;
        }
        prev_r2 = r2;

        if wall_start.elapsed().as_secs_f64() >= budget_s {
            break;
        }

        let candidates: Vec<f64> = [crossover * 0.5, crossover, crossover * 1.5]
            .iter()
            .map(|&v| v.round().max(1.0))
            .filter(|&ai| !sweep.iter().any(|(m, _, _)| (*m - ai).abs() < 0.5))
            .take(MAX_CANDIDATES_PER_ROUND)
            .collect();
        if candidates.is_empty() {
            break;
        }
        for target_ai in candidates {
            progress_bar(label, &format!("refine {}", round + 1), sweep.len(),
                         max_points, prev_r2, wall_start.elapsed().as_secs_f64(),
                         budget_s);
            sweep.push(probe(target_ai, per_point_budget)?);
        }
        sweep.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    // ── Finalize ────────────────────────────────────────────────────────────
    let (crossover, c_slope) = fit_crossover(&sweep);
    let r2 = roofline_r2(&sweep, crossover, c_slope);

    progress_bar(label, "done", sweep.len(), max_points, r2,
                 wall_start.elapsed().as_secs_f64(), budget_s);
    // Clear the progress line and move to a fresh line.
    eprint!("\x1b[2K\r");

    if r2 < R2_WARN {
        eprintln!(
            "  {label}: Warning: roofline fit R\u{00b2} = {r2:.3} (< {R2_WARN}); \
             crossover estimate may be unreliable."
        );
    }

    let peak_bw_gib_s = sweep.iter().map(|(_, _, g)| *g).fold(0.0_f64, f64::max);
    let peak_gflops_s = sweep.iter().map(|(_, g, _)| *g).fold(0.0_f64, f64::max);
    Ok(HardwareProfile::from_measurements(
        peak_bw_gib_s,
        peak_gflops_s,
        scalar_bytes,
        c_slope,
    ))
}

impl HardwareProfile {
    /// Profiles CPU hardware by timing JIT gate kernels across a range of
    /// arithmetic intensities and fitting a roofline model.
    ///
    /// Thread count comes from `CAST_NUM_THREADS` (see [`crate::cpu::get_num_threads`]).
    /// **Profile at the thread count you will simulate with**: compute scales with
    /// threads while DRAM bandwidth does not, so the crossover shifts.
    pub fn measure_cpu(
        spec: &crate::cpu::CPUKernelGenSpec,
        budget_s: f64,
    ) -> anyhow::Result<Self> {
        use crate::cpu::{get_num_threads, get_simd_s, CPUStatevector, CpuKernelManager};

        let n_threads = get_num_threads();
        let ztol = spec.ztol;

        // Create a reusable statevector and kernel manager.
        let n_qubits = PROFILE_N_QUBITS.max(
            MAX_GATE_QUBITS + get_simd_s(spec.simd_width, spec.precision) + 1,
        );
        let mut sv = CPUStatevector::new(n_qubits, spec.precision, spec.simd_width);
        sv.initialize();
        let mgr = CpuKernelManager::new();

        let label = format!("CPU F{} {}-thread", spec.precision.scalar_bytes() * 8, n_threads);
        adaptive_sweep(
            |target_ai, per_point_budget| {
                let gate = QuantumGate::random_arithmatic_intensity(n_qubits, target_ai, ztol);
                let actual_ai = gate.arithmatic_intensity(ztol);
                let kid = mgr.generate(spec, &gate)?;
                let timing = mgr.time_adaptive(kid, &mut sv, n_threads, per_point_budget)?;

                let gib_s =
                    2.0 * sv.byte_len() as f64 / timing.mean_s / (1u64 << 30) as f64;
                let gflops_s =
                    actual_ai * sv.len() as f64 * 2.0 / timing.mean_s / 1e9;
                Ok((actual_ai, gflops_s, gib_s))
            },
            spec.precision.scalar_bytes(),
            budget_s,
            &label,
        )
    }

    /// Profiles CUDA hardware by timing generated GPU kernels across a range
    /// of arithmetic intensities and fitting a roofline model.
    ///
    /// GPU time is measured via CUDA events for sub-microsecond accuracy.
    /// BW calibration uses an AI=1 gate probe (not a separate D2D memcpy).
    #[cfg(feature = "cuda")]
    pub fn measure_cuda(
        spec: &crate::cuda::CudaKernelGenSpec,
        budget_s: f64,
    ) -> anyhow::Result<Self> {
        use crate::cuda::{CudaKernelManager, CudaStatevector};

        let ztol = spec.ztol;
        let n_qubits = PROFILE_N_QUBITS;
        let mgr = CudaKernelManager::new();
        let mut sv = CudaStatevector::new(n_qubits, spec.precision)?;
        sv.zero()?;

        const N_WARMUP: usize = 3;
        const MIN_ITERS: usize = 10;
        const N_PROBE: usize = 3;

        let label = format!(
            "CUDA F{} sm_{}{}",
            spec.precision.scalar_bytes() * 8, spec.sm_major, spec.sm_minor,
        );
        adaptive_sweep(
            |target_ai, per_point_budget| {
                let gate = QuantumGate::random_arithmatic_intensity(n_qubits, target_ai, ztol);
                let actual_ai = gate.arithmatic_intensity(ztol);
                let kid = mgr.generate(&gate, *spec)?;

                // Warmup iterations (not timed).
                for _ in 0..N_WARMUP {
                    mgr.apply(kid, &mut sv)?;
                    mgr.sync()?;
                }

                // Probe iterations — wall-clock tracks budget, GPU events track time.
                let mut samples: Vec<f64> = Vec::new();
                let probe_wall = Instant::now();
                for _ in 0..N_PROBE {
                    mgr.apply(kid, &mut sv)?;
                    let stats = mgr.sync()?;
                    samples.push(stats.kernels[0].gpu_time.as_secs_f64());
                }
                let probe_elapsed = probe_wall.elapsed().as_secs_f64();

                // Fill remaining budget.
                let est_per_iter = probe_elapsed / N_PROBE as f64;
                let remaining = (per_point_budget - probe_elapsed).max(0.0);
                let n_fill = ((remaining / est_per_iter) as usize)
                    .max(MIN_ITERS.saturating_sub(N_PROBE));
                for _ in 0..n_fill {
                    mgr.apply(kid, &mut sv)?;
                    let stats = mgr.sync()?;
                    samples.push(stats.kernels[0].gpu_time.as_secs_f64());
                }

                let mean_s = samples.iter().copied().sum::<f64>() / samples.len() as f64;

                let sv_bytes = (1usize << n_qubits) * 2 * spec.precision.scalar_bytes();
                let gib_s = 2.0 * sv_bytes as f64 / mean_s / (1u64 << 30) as f64;
                let gflops_s =
                    actual_ai * (1usize << n_qubits) as f64 * 2.0 / mean_s / 1e9;
                Ok((actual_ai, gflops_s, gib_s))
            },
            spec.precision.scalar_bytes(),
            budget_s,
            &label,
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantumGate;
    use std::mem::size_of;

    fn size_model(max_size: usize) -> SizeOnlyCostModel {
        SizeOnlyCostModel {
            max_size,
            max_ai: usize::MAX,
            zero_tol: 0.0,
        }
    }

    fn adaptive_model(bw: f64, gflops: f64, max_size: usize) -> HardwareAdaptiveCostModel {
        let profile = HardwareProfile::from_roofline(bw, gflops, size_of::<f64>());
        HardwareAdaptiveCostModel::new(&profile, max_size)
    }

    // ── SizeOnlyCostModel ────────────────────────────────────────────────────

    #[test]
    fn small_gate_is_free() {
        let m = size_model(3);
        assert!(m.cost_of(&QuantumGate::x(0)) < 1e-9);
        assert!(m.cost_of(&QuantumGate::cx(0, 1)) < 1e-9);
        assert!(m.cost_of(&QuantumGate::ccx(0, 1, 2)) < 1e-9);
    }

    #[test]
    fn oversized_gate_is_expensive() {
        let m = size_model(3);
        let g4 = QuantumGate::cx(0, 1)
            .matmul(&QuantumGate::cx(2, 3))
            .matmul(&QuantumGate::cx(0, 2));
        assert_eq!(g4.n_qubits(), 4);
        assert_eq!(m.cost_of(&g4), 1.0);
    }

    #[test]
    fn gate_at_size_limit_is_free() {
        let m = size_model(2);
        assert!(m.cost_of(&QuantumGate::cx(0, 1)) < 1e-9);
    }

    // ── HardwareAdaptiveCostModel ────────────────────────────────────────────

    #[test]
    fn crossover_ai_formula() {
        // 50 GiB/s bw, 200 GFLOPs/s peak, f64 (8 bytes).
        // C_slope = 50·2^30 / (2·8·1e9) ≈ 3.355 → crossover ≈ 200/3.355 ≈ 59.6
        let profile = HardwareProfile::from_roofline(50.0, 200.0, 8);
        assert!(
            (profile.crossover_ai() - 59.6).abs() < 1.0,
            "crossover = {}",
            profile.crossover_ai()
        );
    }

    #[test]
    fn memory_bound_gate_costs_one() {
        let m = adaptive_model(50.0, 200.0, 4);
        let cost = m.cost_of(&QuantumGate::x(0));
        assert!((cost - 1.0).abs() < 1e-9, "cost = {cost}");
    }

    #[test]
    fn compute_bound_gate_costs_more_than_one() {
        // Tiny crossover (≈ 0.015) — everything is compute-bound.
        let m = adaptive_model(1000.0, 1.0, 5);
        let g4 = QuantumGate::cx(0, 1)
            .matmul(&QuantumGate::cx(2, 3))
            .matmul(&QuantumGate::cx(0, 2));
        assert!(m.cost_of(&g4) > 1.0);
    }

    #[test]
    fn oversized_gate_returns_infinity() {
        let m = adaptive_model(50.0, 200.0, 2);
        assert_eq!(m.cost_of(&QuantumGate::ccx(0, 1, 2)), f64::INFINITY);
    }

    #[test]
    fn fusing_two_memory_bound_gates_yields_positive_benefit() {
        let m = adaptive_model(50.0, 200.0, 4);
        let cx01 = QuantumGate::cx(0, 1);
        let cx23 = QuantumGate::cx(2, 3);
        let fused = cx01.matmul(&cx23);
        let benefit = (m.cost_of(&cx01) + m.cost_of(&cx23)) / (m.cost_of(&fused) + 1e-10) - 1.0;
        assert!(benefit > 0.0, "benefit = {benefit}");
    }
}

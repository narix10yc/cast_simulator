use crate::types::QuantumGate;
use serde::{Deserialize, Serialize};

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
    /// When true, use `n_qubits()` (trajectory) instead of
    /// `effective_n_qubits()` (density-matrix) for the size check.
    pub trajectory_mode: bool,
}

impl CostModel for SizeOnlyCostModel {
    fn cost_of(&self, gate: &QuantumGate) -> f64 {
        let size = if self.trajectory_mode {
            gate.n_qubits()
        } else {
            gate.effective_n_qubits()
        };
        if size > self.max_size {
            return 1.0;
        }
        if gate.arithmetic_intensity(self.zero_tol) > self.max_ai as f64 {
            return 1.0;
        }
        1e-10
    }
}

// ── Hardware profile ─────────────────────────────────────────────────────────

/// The device a [`HardwareProfile`] was measured on.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend")]
pub enum Device {
    #[serde(rename = "cpu")]
    Cpu {
        /// Human-readable name (e.g. `"AMD EPYC 9654"`).
        name: String,
        /// Number of CPU worker threads used during profiling.
        n_threads: u32,
    },
    #[serde(rename = "cuda")]
    Cuda {
        /// Human-readable name (e.g. `"RTX 5090"`).
        name: String,
        /// CUDA compute capability major version (e.g. 12 for sm_120).
        sm_major: u32,
        /// CUDA compute capability minor version (e.g. 0 for sm_120).
        sm_minor: u32,
    },
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu { name, n_threads } if !name.is_empty() => {
                write!(f, "CPU {} ({n_threads}-thread)", name)
            }
            Device::Cpu { n_threads, .. } => write!(f, "CPU {n_threads}-thread"),
            Device::Cuda {
                name,
                sm_major,
                sm_minor,
            } if !name.is_empty() => {
                write!(f, "CUDA {} (sm_{sm_major}{sm_minor})", name)
            }
            Device::Cuda {
                sm_major, sm_minor, ..
            } => {
                write!(f, "CUDA sm_{sm_major}{sm_minor}")
            }
        }
    }
}

/// Metadata describing the conditions under which a [`HardwareProfile`] was
/// measured.  Stored inside the profile so that cached JSON files are
/// self-documenting and consumers can verify the profile matches their
/// intended simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Device and backend that was profiled.
    pub device: Device,
    /// Floating-point precision (`"f32"` or `"f64"`).
    pub precision: String,
    /// Number of statevector qubits used during profiling.
    pub n_qubits: u32,
}

/// A single data point from the adaptive roofline sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepEntry {
    /// Arithmetic intensity of the probed gate.
    pub ai: f64,
    /// Observed compute throughput (GFLOPs/s).
    pub gflops_s: f64,
    /// Observed memory bandwidth (GiB/s, bidirectional).
    pub gib_s: f64,
}

/// Roofline hardware profile: the measured parameters that determine when
/// gate fusion helps and when it hurts.
///
/// Three independent measured values fully define the roofline:
///
/// - `peak_bw_gib_s` — peak memory bandwidth (GiB/s, bidirectional)
/// - `peak_gflops_s` — peak compute throughput (GFLOPs/s)
/// - `bw_slope` — memory-bound GFLOPs/s per unit AI (slope of rising segment)
///
/// The crossover AI (`peak_gflops_s / bw_slope`) is cached for fast access
/// by the cost model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Profiling configuration: backend, device, precision, qubit count, etc.
    pub config: ProfileConfig,
    /// Peak effective bidirectional bandwidth (GiB/s, read + write).
    pub peak_bw_gib_s: f64,
    /// Peak observed compute throughput (GFLOPs/s).
    pub peak_gflops_s: f64,
    /// Memory-bound slope: GFLOPs/s per unit arithmetic intensity.
    /// This is the slope of the rising (memory-bound) segment of the roofline.
    pub bw_slope: f64,
    /// Roofline crossover point: the AI where memory-bound meets compute-bound.
    /// Cached from `peak_gflops_s / bw_slope`.
    pub crossover_ai: f64,
    /// Raw sweep data from profiling.  Empty when constructed manually;
    /// populated by the profiling functions in [`crate::profile`].
    #[serde(default)]
    pub raw: Vec<SweepEntry>,
}

/// Computes the memory-bound slope (GFLOPs/s per unit AI) from bandwidth and
/// scalar size.
///
/// In the memory-bound regime the statevector is read and written once per gate
/// application:  `T = 2 × sv_bytes / BW`.  The GFLOPs/s at arithmetic
/// intensity `ai` is therefore `ai × BW_bytes / (2 × scalar_bytes) / 1e9`.
fn compute_bw_slope(peak_bw_gib_s: f64, scalar_bytes: usize) -> f64 {
    peak_bw_gib_s * (1u64 << 30) as f64 / (2.0 * scalar_bytes as f64) / 1e9
}

fn compute_crossover(peak_gflops_s: f64, bw_slope: f64) -> f64 {
    if bw_slope > 0.0 && peak_gflops_s > 0.0 {
        peak_gflops_s / bw_slope
    } else {
        f64::INFINITY
    }
}

impl HardwareProfile {
    /// Constructs a profile from theoretical roofline parameters.
    ///
    /// `scalar_bytes`: 8 for f64, 4 for f32.  The BW slope is derived from
    /// `peak_bw_gib_s` and `scalar_bytes`.
    pub fn from_roofline(
        config: ProfileConfig,
        peak_bw_gib_s: f64,
        peak_gflops_s: f64,
        scalar_bytes: usize,
    ) -> Self {
        let bw_slope = compute_bw_slope(peak_bw_gib_s, scalar_bytes);
        Self {
            config,
            peak_bw_gib_s,
            peak_gflops_s,
            bw_slope,
            crossover_ai: compute_crossover(peak_gflops_s, bw_slope),
            raw: Vec::new(),
        }
    }

    /// Constructs from directly measured quantities.
    ///
    /// Use this when `bw_slope` has been measured via the adaptive sweep
    /// rather than derived from `peak_bw_gib_s` and `scalar_bytes`.  The
    /// measured slope is preferred because it reflects actual memory-system
    /// behaviour more accurately.
    pub fn from_measurements(
        config: ProfileConfig,
        peak_bw_gib_s: f64,
        peak_gflops_s: f64,
        bw_slope: f64,
    ) -> Self {
        Self {
            config,
            peak_bw_gib_s,
            peak_gflops_s,
            bw_slope,
            crossover_ai: compute_crossover(peak_gflops_s, bw_slope),
            raw: Vec::new(),
        }
    }

    /// Serializes this profile to a JSON file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserializes a profile from a JSON file.
    pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let profile: Self = serde_json::from_str(&json)?;
        Ok(profile)
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
    /// When true, use `n_qubits()` (trajectory) instead of
    /// `effective_n_qubits()` (density-matrix) for the size check.
    pub trajectory_mode: bool,
}

impl HardwareAdaptiveCostModel {
    pub fn new(profile: &HardwareProfile, max_size: usize, trajectory_mode: bool) -> Self {
        Self {
            crossover_ai: profile.crossover_ai,
            max_size,
            zero_tol: 1e-12,
            trajectory_mode,
        }
    }
}

impl CostModel for HardwareAdaptiveCostModel {
    fn cost_of(&self, gate: &QuantumGate) -> f64 {
        let size = if self.trajectory_mode {
            gate.n_qubits()
        } else {
            gate.effective_n_qubits()
        };
        if size > self.max_size {
            return f64::INFINITY;
        }
        (gate.arithmetic_intensity(self.zero_tol) / self.crossover_ai).max(1.0)
    }
}

// ── Fusion config ────────────────────────────────────────────────────────────

/// Configuration for the adaptive gate-fusion optimizer.
///
/// `size_max` caps the qubit count of any fused gate. `benefit_margin` is the
/// minimum `old_cost / (new_cost + ε) - 1` for a fusion to be accepted. Phase 1
/// (size-2 canonicalization) always runs before the agglomerative phase.
///
/// Noisy gates always participate in fusion. When `trajectory_mode` is `true`,
/// the cost model evaluates the dominant (noiseless) branch using `n_qubits()`.
/// When `false` (default, density-matrix), the cost model uses
/// `effective_n_qubits()` which doubles the qubit count for noisy gates to
/// reflect the superoperator size.
pub struct FusionConfig {
    pub size_max: usize,
    pub benefit_margin: f64,
    pub cost_model: Box<dyn CostModel>,
    pub trajectory_mode: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self::size_only(3)
    }
}

impl FusionConfig {
    /// Pure size-gated fusion up to `max_size` qubits. No AI cap.
    pub fn size_only(max_size: usize) -> Self {
        Self::size_only_inner(max_size, false)
    }

    /// Size-gated fusion for trajectory mode — noisy gates participate.
    pub fn size_only_trajectory(max_size: usize) -> Self {
        Self::size_only_inner(max_size, true)
    }

    fn size_only_inner(max_size: usize, trajectory_mode: bool) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size,
                max_ai: usize::MAX,
                zero_tol: 0.0,
                trajectory_mode,
            }),
            trajectory_mode,
        }
    }

    pub fn aggressive() -> Self {
        Self::size_only(4)
    }

    /// Roofline-adaptive fusion. The crossover in `profile` must match the
    /// precision and thread count of your simulation.
    pub fn hardware_adaptive(profile: &HardwareProfile, max_size: usize) -> Self {
        Self::hardware_adaptive_inner(profile, max_size, false)
    }

    /// Roofline-adaptive fusion for trajectory mode — noisy gates participate.
    pub fn hardware_adaptive_trajectory(profile: &HardwareProfile, max_size: usize) -> Self {
        Self::hardware_adaptive_inner(profile, max_size, true)
    }

    fn hardware_adaptive_inner(
        profile: &HardwareProfile,
        max_size: usize,
        trajectory_mode: bool,
    ) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(HardwareAdaptiveCostModel::new(
                profile,
                max_size,
                trajectory_mode,
            )),
            trajectory_mode,
        }
    }
}

// ── Fusion log ───────────────────────────────────────────────────────────

/// A single recorded fusion decision from the agglomerative optimizer.
#[derive(Debug, Clone)]
pub struct FusionDecision {
    /// Size level at which this decision was made (cdd_size).
    pub cdd_size: usize,
    /// Number of candidate gates in the cluster (including the seed).
    pub n_candidates: usize,
    /// Per-candidate: (n_qubits, arithmetic_intensity).
    pub candidates: Vec<(usize, f64)>,
    /// Number of qubits in the product gate.
    pub product_n_qubits: usize,
    /// Arithmetic intensity of the product gate.
    pub product_ai: f64,
    /// Sum of individual candidate costs.
    pub old_cost: f64,
    /// Cost of the product gate.
    pub new_cost: f64,
    /// Benefit ratio: old_cost / (new_cost + eps) - 1.
    pub benefit: f64,
    /// Whether the fusion was accepted.
    pub accepted: bool,
}

/// Accumulates fusion decisions during optimization for analysis and
/// reproducibility.  Pass to [`crate::fusion::optimize_with_log`] to collect.
#[derive(Debug, Clone, Default)]
pub struct FusionLog {
    pub decisions: Vec<FusionDecision>,
}

impl FusionLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of accepted fusions.
    pub fn n_accepted(&self) -> usize {
        self.decisions.iter().filter(|d| d.accepted).count()
    }

    /// Number of rejected fusions.
    pub fn n_rejected(&self) -> usize {
        self.decisions.iter().filter(|d| !d.accepted).count()
    }

    /// Print a human-readable summary to stderr.
    pub fn print_summary(&self) {
        let accepted = self.n_accepted();
        let rejected = self.n_rejected();
        eprintln!(
            "  Fusion log: {} decisions ({} accepted, {} rejected)",
            self.decisions.len(),
            accepted,
            rejected,
        );
        if self.decisions.is_empty() {
            return;
        }
        eprintln!(
            "  {:>8} {:>6} {:>8} {:>10} {:>10} {:>10} {:>8}",
            "Size", "NCand", "ProdQ", "OldCost", "NewCost", "Benefit", "Accept"
        );
        eprintln!("  {}", "-".repeat(68));
        for d in &self.decisions {
            eprintln!(
                "  {:>8} {:>6} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>8}",
                d.cdd_size,
                d.n_candidates,
                d.product_n_qubits,
                d.old_cost,
                d.new_cost,
                d.benefit,
                if d.accepted { "yes" } else { "no" },
            );
        }
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
            trajectory_mode: false,
        }
    }

    fn test_config() -> ProfileConfig {
        ProfileConfig {
            device: Device::Cpu {
                name: String::new(),
                n_threads: 1,
            },
            precision: "f64".into(),
            n_qubits: 10,
        }
    }

    fn adaptive_model(bw: f64, gflops: f64, max_size: usize) -> HardwareAdaptiveCostModel {
        let profile = HardwareProfile::from_roofline(test_config(), bw, gflops, size_of::<f64>());
        HardwareAdaptiveCostModel::new(&profile, max_size, false)
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

    // ── Noisy gate size accounting ─────────────────────────────────────────

    #[test]
    fn noisy_gate_counts_as_double_qubits() {
        // A 2-qubit channel has a 4^2=16×16 superoperator — equivalent to a
        // 4-qubit unitary. A max_size=3 model must reject it.
        let m = size_model(3);
        // A 2-qubit gate with noise has effective_n_qubits = 4.
        let ch2 =
            crate::types::QuantumGate::new(crate::types::ComplexSquareMatrix::eye(4), vec![0, 1])
                .with_noise(vec![(1.0, crate::types::ComplexSquareMatrix::eye(4))]);
        assert_eq!(ch2.n_qubits(), 2);
        assert_eq!(ch2.effective_n_qubits(), 4);
        assert_eq!(
            m.cost_of(&ch2),
            1.0,
            "2-qubit noisy gate must be rejected by max_size=3"
        );
    }

    #[test]
    fn single_qubit_noisy_gate_fits_within_size2_limit() {
        let m = size_model(2);
        let ch = crate::types::QuantumGate::depolarizing(0, 0.1);
        assert_eq!(ch.effective_n_qubits(), 2);
        assert!(
            m.cost_of(&ch) < 1e-9,
            "1-qubit noisy gate fits within max_size=2"
        );
    }

    // ── HardwareAdaptiveCostModel ────────────────────────────────────────────

    #[test]
    fn crossover_ai_formula() {
        // 50 GiB/s bw, 200 GFLOPs/s peak, f64 (8 bytes).
        // C_slope = 50·2^30 / (2·8·1e9) ≈ 3.355 → crossover ≈ 200/3.355 ≈ 59.6
        let profile = HardwareProfile::from_roofline(test_config(), 50.0, 200.0, 8);
        assert!(
            (profile.crossover_ai - 59.6).abs() < 1.0,
            "crossover = {}",
            profile.crossover_ai
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
    fn adaptive_channel_gate_uses_effective_size() {
        // 1-qubit channel → effective size 2: fits in max_size=2, blocked at max_size=1.
        let ch = crate::types::QuantumGate::depolarizing(0, 0.1);
        let m_ok = adaptive_model(50.0, 200.0, 2);
        let m_blocked = adaptive_model(50.0, 200.0, 1);
        assert!(m_ok.cost_of(&ch).is_finite());
        assert_eq!(m_blocked.cost_of(&ch), f64::INFINITY);
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

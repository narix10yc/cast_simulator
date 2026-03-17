use crate::types::QuantumGate;

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

/// Roofline hardware profile used by [`HardwareAdaptiveCostModel`].
///
/// The key quantity is `crossover_ai` — the arithmetic intensity where gate
/// simulation transitions from memory-bound to compute-bound. Below the
/// crossover every gate costs the same (one statevector pass); above it compute
/// dominates and fusion yields diminishing benefit.
///
/// Build from explicit roofline numbers:
/// ```ignore
/// let profile = HardwareProfile::from_roofline(peak_bw_gib_s, peak_gflops_s, 8);
/// ```
/// Or from an automated JIT sweep:
/// ```ignore
/// let profile = cast::cpu::measure_cpu_profile(&CPUKernelGenSpec::f64())?;
/// ```
pub struct HardwareProfile {
    /// Roofline crossover point in arithmetic-intensity units.
    pub crossover_ai: f64,
    /// Peak effective bandwidth (GiB/s, read + write).
    pub peak_bw_gib_s: f64,
}

impl HardwareProfile {
    /// Derives the crossover point from peak bandwidth and peak compute.
    ///
    /// `scalar_bytes`: 8 for f64, 4 for f32. **Measure `peak_gflops_s` at the
    /// thread count you will simulate with** — compute scales with threads while
    /// DRAM bandwidth does not.
    ///
    /// Derivation: in the memory-bound regime GFLOPs/s =
    /// `ai · bw_bytes / (2 · scalar_bytes · 1e9)`. Setting that equal to
    /// `peak_gflops` gives `crossover = peak_gflops · 2 · scalar_bytes · 1e9 / bw_bytes`.
    pub fn from_roofline(peak_bw_gib_s: f64, peak_gflops_s: f64, scalar_bytes: usize) -> Self {
        let crossover = if peak_bw_gib_s > 0.0 && peak_gflops_s > 0.0 {
            let bw_bytes = peak_bw_gib_s * (1u64 << 30) as f64;
            peak_gflops_s * 2.0 * scalar_bytes as f64 * 1e9 / bw_bytes
        } else {
            f64::INFINITY
        };
        Self {
            crossover_ai: crossover,
            peak_bw_gib_s,
        }
    }

    /// Constructs directly from a fitted crossover point and bandwidth.
    pub(crate) fn from_fit(crossover_ai: f64, peak_bw_gib_s: f64) -> Self {
        Self {
            crossover_ai,
            peak_bw_gib_s,
        }
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
            crossover_ai: profile.crossover_ai,
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
    fn fusing_two_memory_bound_gates_yields_positive_benefit() {
        let m = adaptive_model(50.0, 200.0, 4);
        let cx01 = QuantumGate::cx(0, 1);
        let cx23 = QuantumGate::cx(2, 3);
        let fused = cx01.matmul(&cx23);
        let benefit = (m.cost_of(&cx01) + m.cost_of(&cx23)) / (m.cost_of(&fused) + 1e-10) - 1.0;
        assert!(benefit > 0.0, "benefit = {benefit}");
    }
}

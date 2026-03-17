use crate::types::QuantumGate;

/// Abstract cost model: estimates the GiB-time (seconds per GiB of statevector
/// memory updated) for applying a gate. Lower cost = cheaper to execute.
///
/// The optimizer accepts a fusion when the ratio of pre-fusion total cost to
/// post-fusion cost exceeds 1 + `benefit_margin`.
pub trait CostModel: Send + Sync {
    fn cost_of(&self, gate: &QuantumGate) -> f64;
}

/// Size-only cost model: a gate is "free" (1e-10) if it fits within the qubit
/// and op-count budget, and "expensive" (1.0) otherwise. This yields a pure
/// size-gated accept/reject decision with no empirical profiling required.
///
/// Corresponds to C++ `SizeOnlyCostModel(maxSize, maxOp, zeroTol)`.
pub struct SizeOnlyCostModel {
    pub max_size: usize,
    pub max_op: usize,
    pub zero_tol: f64,
}

impl CostModel for SizeOnlyCostModel {
    fn cost_of(&self, gate: &QuantumGate) -> f64 {
        if gate.n_qubits() > self.max_size {
            return 1.0;
        }
        if gate.opcount(self.zero_tol) > self.max_op as f64 {
            return 1.0;
        }
        1e-10
    }
}

// ── Hardware profile ──────────────────────────────────────────────────────────

/// Observed hardware performance profile for the roofline cost model.
///
/// The key quantity is `knee_opcount`: the `QuantumGate::opcount` value at
/// which gate simulation transitions from memory-bound to compute-bound.
/// Below the knee every gate costs the same (one statevector pass), so fusing
/// is always beneficial. Above it compute dominates and fusion yields less gain.
///
/// # Building a profile
///
/// From `cpu_crossover` output:
/// ```ignore
/// let profile = HardwareProfile::from_roofline(peak_bw_gib_s, peak_gflops_s, 8);
/// ```
///
/// Programmatically via the JIT sweep:
/// ```ignore
/// let profile = cast::cpu::measure_cpu_profile(&CPUKernelGenSpec::f64())?;
/// ```
pub struct HardwareProfile {
    /// Roofline knee in `QuantumGate::opcount` units.
    pub knee_opcount: f64,
    /// Peak effective memory bandwidth (GiB/s, read + write), for reference.
    pub peak_bw_gib_s: f64,
}

impl HardwareProfile {
    /// Constructs a profile from two directly measured roofline quantities.
    ///
    /// `peak_bw_gib_s`: effective bandwidth in GiB/s (read + write combined).
    /// `peak_gflops_s`: peak compute in GFLOPs/s. **Must be measured at the
    /// thread count you intend to simulate with.**
    /// `scalar_bytes`: byte width of one real scalar (8 for f64, 4 for f32).
    ///
    /// # Derivation
    ///
    /// In the memory-bound regime: GFLOPs/s = `opcount · bw_bytes_s / (2·scalar_bytes·1e9)`.
    /// At the knee this equals `peak_gflops`, giving:
    /// `knee = peak_gflops · 2 · scalar_bytes · 1e9 / (peak_bw · GIB)`
    pub fn from_roofline(peak_bw_gib_s: f64, peak_gflops_s: f64, scalar_bytes: usize) -> Self {
        let knee = if peak_bw_gib_s > 0.0 && peak_gflops_s > 0.0 {
            let bw_bytes_s = peak_bw_gib_s * (1u64 << 30) as f64;
            peak_gflops_s * 2.0 * scalar_bytes as f64 * 1e9 / bw_bytes_s
        } else {
            f64::INFINITY
        };
        Self {
            knee_opcount: knee,
            peak_bw_gib_s,
        }
    }

    /// Constructs a profile directly from a pre-fitted knee and bandwidth.
    /// Intended for use by [`crate::cpu::measure_cpu_profile`].
    pub(crate) fn from_fit(knee_opcount: f64, peak_bw_gib_s: f64) -> Self {
        Self {
            knee_opcount,
            peak_bw_gib_s,
        }
    }
}

// ── Hardware-adaptive cost model ──────────────────────────────────────────────

/// Roofline-based cost model.
///
/// A gate costs `max(1.0, opcount / knee_opcount)`. Memory-bound gates (below
/// the knee) all cost 1.0; compute-bound gates cost proportionally more.
///
/// - Fusing two memory-bound gates always yields benefit ≈ 1.0 → accepted.
/// - Fusing into a compute-bound result reduces benefit, and the optimizer
///   rejects once the fused gate becomes more expensive than the originals.
///
/// Gates exceeding `max_size` qubits return `f64::INFINITY`.
pub struct HardwareAdaptiveCostModel {
    pub knee_opcount: f64,
    pub max_size: usize,
    pub zero_tol: f64,
}

impl HardwareAdaptiveCostModel {
    pub fn new(profile: &HardwareProfile, max_size: usize) -> Self {
        Self {
            knee_opcount: profile.knee_opcount,
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
        (gate.opcount(self.zero_tol) / self.knee_opcount).max(1.0)
    }
}

/// Configuration for the adaptive gate-fusion optimizer.
///
/// `size_max` is the maximum number of qubits a fused gate may touch.
/// `benefit_margin` is the minimum required benefit ratio for a fusion to be
/// accepted: `benefit = old_cost / (new_cost + 1e-10) - 1.0 >= benefit_margin`.
///
/// Phase 1 (size-2 canonicalization) always runs before the agglomerative
/// phase; `size_min` is therefore fixed at 3 and not stored here.
pub struct FusionConfig {
    pub size_max: usize,
    pub benefit_margin: f64,
    pub cost_model: Box<dyn CostModel>,
}

impl FusionConfig {
    /// Pure size-gated fusion up to `max_size` qubits. No op-count cap.
    pub fn size_only(max_size: usize) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size,
                max_op: usize::MAX,
                zero_tol: 0.0,
            }),
        }
    }

    /// Mild fusion: fuse gates up to 5 qubits.
    pub fn mild() -> Self {
        Self::size_only(5)
    }

    /// Balanced fusion: fuse gates up to 6 qubits.
    pub fn balanced() -> Self {
        Self::size_only(6)
    }

    /// Aggressive fusion: fuse gates up to 7 qubits.
    pub fn aggressive() -> Self {
        Self::size_only(7)
    }

    /// Hardware-adaptive fusion using the roofline cost model.
    ///
    /// `max_size` caps the qubit count of any fused gate. The knee embedded in
    /// `profile` was measured (or derived) for the precision and thread count
    /// you intend to simulate with — pass a profile built from
    /// [`HardwareProfile::from_roofline`] or [`crate::cpu::measure_cpu_profile`].
    pub fn hardware_adaptive(profile: &HardwareProfile, max_size: usize) -> Self {
        Self {
            size_max: max_size,
            benefit_margin: 0.0,
            cost_model: Box::new(HardwareAdaptiveCostModel::new(profile, max_size)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantumGate;
    use std::mem::size_of;

    fn model(max_size: usize) -> SizeOnlyCostModel {
        SizeOnlyCostModel {
            max_size,
            max_op: usize::MAX,
            zero_tol: 0.0,
        }
    }

    #[test]
    fn small_gate_is_free() {
        let m = model(3);
        assert!(m.cost_of(&QuantumGate::x(0)) < 1e-9);
        assert!(m.cost_of(&QuantumGate::cx(0, 1)) < 1e-9);
        assert!(m.cost_of(&QuantumGate::ccx(0, 1, 2)) < 1e-9);
    }

    #[test]
    fn oversized_gate_is_expensive() {
        // Build a 4-qubit gate by fusing; then check against a max_size=3 model.
        let m = model(3);
        let g4 = QuantumGate::cx(0, 1)
            .matmul(&QuantumGate::cx(2, 3))
            .matmul(&QuantumGate::cx(0, 2));
        assert_eq!(g4.n_qubits(), 4);
        assert_eq!(m.cost_of(&g4), 1.0);
    }

    #[test]
    fn gate_at_size_limit_is_free() {
        let m = model(2);
        assert!(m.cost_of(&QuantumGate::cx(0, 1)) < 1e-9);
    }

    // ── HardwareAdaptiveCostModel tests ───────────────────────────────────────

    fn adaptive_model(
        peak_bw_gib_s: f64,
        peak_gflops_s: f64,
        max_size: usize,
    ) -> HardwareAdaptiveCostModel {
        let profile =
            HardwareProfile::from_roofline(peak_bw_gib_s, peak_gflops_s, size_of::<f64>());
        HardwareAdaptiveCostModel::new(&profile, max_size)
    }

    #[test]
    fn knee_opcount_formula() {
        // 50 GiB/s bandwidth, 200 GFLOPs/s peak, f64 (8 bytes).
        // C_slope = bw_bytes_s / (2 * 8 * 1e9) = 50*2^30 / 16e9 ≈ 3.355
        // knee = 200 / 3.355 ≈ 59.6
        let profile = HardwareProfile::from_roofline(50.0, 200.0, 8);
        assert!(
            (profile.knee_opcount - 59.6).abs() < 1.0,
            "knee = {}",
            profile.knee_opcount
        );
    }

    #[test]
    fn memory_bound_gate_costs_one() {
        // With knee ≈ 60, a 1-qubit X gate (opcount = 2) is well below knee.
        let m = adaptive_model(50.0, 200.0, 4);
        let cost = m.cost_of(&QuantumGate::x(0));
        assert!((cost - 1.0).abs() < 1e-9, "cost = {cost}");
    }

    #[test]
    fn compute_bound_gate_costs_more_than_one() {
        // Use a profile with a tiny knee so even small opcounts exceed it.
        // knee = 1e9 * 2 * 8 / (1000 * 2^30) ≈ 0.015
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
        // CX(0,1) and CX(2,3): opcount = 1.0 each, knee ≈ 60 → memory-bound.
        // Fused (4-qubit permutation): opcount = 1.0 → also memory-bound.
        // old_cost = 2.0, new_cost = 1.0, benefit > 0.
        let m = adaptive_model(50.0, 200.0, 4);
        let cx01 = QuantumGate::cx(0, 1);
        let cx23 = QuantumGate::cx(2, 3);
        let fused = cx01.matmul(&cx23);
        let benefit = (m.cost_of(&cx01) + m.cost_of(&cx23)) / (m.cost_of(&fused) + 1e-10) - 1.0;
        assert!(benefit > 0.0, "benefit = {benefit}");
    }
}

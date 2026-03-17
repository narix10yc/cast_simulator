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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantumGate;

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
}

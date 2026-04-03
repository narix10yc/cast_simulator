use super::ComplexSquareMatrix;

/// A probability-weighted set of full Kraus operators.
///
/// Each entry `(p_i, K_i)` pairs a probability with a full Kraus operator
/// `K_i = V_i · U` (the noise-only operator `V_i` pre-composed with the
/// gate's base unitary `U`). Probabilities must sum to 1.0.
///
/// Storing full Kraus operators (rather than noise-only parts) makes gate
/// fusion trivial: the fused Kraus ops are just `K_i^self · K_j^other`.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseModel {
    pub(crate) branches: Vec<(f64, ComplexSquareMatrix)>,
}

impl NoiseModel {
    /// Create a noise model from a list of `(probability, kraus_operator)` branches.
    ///
    /// # Panics
    /// Panics if probabilities do not sum to 1.0 (within 1e-10) or any
    /// probability is negative.
    pub fn new(branches: Vec<(f64, ComplexSquareMatrix)>) -> Self {
        let mut prob_sum = 0.0f64;
        for (p, _) in &branches {
            assert!(*p >= 0.0, "noise probability must be non-negative, got {p}");
            prob_sum += p;
        }
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "noise probabilities must sum to 1.0, got {prob_sum}"
        );
        Self { branches }
    }

    /// The Kraus branches `[(probability, full_kraus_operator)]`.
    pub fn branches(&self) -> &[(f64, ComplexSquareMatrix)] {
        &self.branches
    }

    /// Number of Kraus branches.
    pub fn len(&self) -> usize {
        self.branches.len()
    }

    /// Whether the noise model has no branches.
    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }
}

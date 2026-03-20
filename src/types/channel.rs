use super::{Complex, ComplexSquareMatrix, QuantumGate};

/// Returns the n-qubit Pauli operator encoded by `pauli_index`.
///
/// `pauli_index` is an integer in `0..4^n`. Bits `(2q, 2q+1)` of `pauli_index` encode
/// qubit `q`'s Pauli type (0=I, 1=X, 2=Y, 3=Z), consistent with the LSB-first qubit
/// ordering used by [`QuantumGate`].
fn n_qubit_pauli(pauli_index: usize, n_qubits: usize) -> ComplexSquareMatrix {
    // Qubit 0 → LSB of the matrix index.  We grow the product as:
    //   result = kron(P_q, result_so_far)
    // so each new qubit q ends up at bit q of the combined row/col index.
    let mut result = ComplexSquareMatrix::pauli(pauli_index & 3);
    for q in 1..n_qubits {
        let p_q = ComplexSquareMatrix::pauli((pauli_index >> (2 * q)) & 3);
        result = p_q.kron(&result);
    }
    result
}

/// A quantum channel on `k` qubits, given by its Kraus operators `{K_i}`.
///
/// A physical quantum channel (completely positive, trace-preserving map) acts on a
/// density matrix ρ as:
///
/// ```text
/// ε(ρ) = Σ_i  K_i  ρ  K_i†
/// ```
///
/// The Kraus operators must satisfy the completeness relation Σ_i K_i†K_i = I for the
/// channel to be trace-preserving (CPTP). Use [`Self::check_cptp`] to verify numerically.
///
/// # Density-matrix simulation
///
/// To apply a channel using the existing statevector engine, convert it to a superoperator
/// with [`Self::to_superoperator_gate`]. The density matrix ρ on `n` qubits is stored as a
/// statevector on `2n` virtual qubits, with basis index layout:
///
/// ```text
/// virtual_index = ket_index  |  (bra_index << n)
/// ```
///
/// The resulting gate acts on `2k` virtual qubits: the ket copies of `self.qubits` and
/// the bra copies `self.qubits[i] + n_total_qubits`.
#[derive(Clone, Debug, PartialEq)]
pub struct KrausChannel {
    operators: Vec<ComplexSquareMatrix>,
    qubits: Vec<u32>,
}

impl KrausChannel {
    /// Creates a Kraus channel from a list of operators and sorted target qubits.
    ///
    /// Each operator must be square with dimension `2^n`, where `n = qubits.len()`.
    /// Qubits must be provided in **strictly ascending order**; unlike [`QuantumGate::new`],
    /// this constructor does not permute the operators for a different qubit ordering.
    ///
    /// This constructor does **not** verify the CPTP condition. Call [`Self::check_cptp`]
    /// to validate.
    ///
    /// # Panics
    /// - `operators` is empty.
    /// - Any qubit appears more than once, or qubits are not sorted ascending.
    /// - Any operator's dimension is not `2^n`.
    pub fn new(operators: Vec<ComplexSquareMatrix>, qubits: Vec<u32>) -> Self {
        assert!(!operators.is_empty(), "KrausChannel requires at least one Kraus operator");
        assert!(
            qubits.windows(2).all(|w| w[0] < w[1]),
            "channel qubits must be distinct and in ascending order"
        );
        let expected_dim = 1usize << qubits.len();
        for (i, op) in operators.iter().enumerate() {
            assert_eq!(
                op.edge_size(),
                expected_dim,
                "Kraus operator {i}: dimension {} does not match 2^{} = {expected_dim}",
                op.edge_size(),
                qubits.len(),
            );
        }
        Self { operators, qubits }
    }

    /// Wraps a unitary gate as a single-Kraus-operator channel (no noise).
    pub fn from_gate(gate: &QuantumGate) -> Self {
        Self {
            operators: vec![gate.matrix().clone()],
            qubits: gate.qubits().to_vec(),
        }
    }

    /// Single-qubit depolarizing channel with error probability `p ∈ [0, 1]`.
    ///
    /// ```text
    /// ε(ρ) = (1 − p) ρ  +  (p/3)(X ρ X  +  Y ρ Y  +  Z ρ Z)
    /// ```
    ///
    /// Kraus operators: `√(1−p)·I`, `√(p/3)·X`, `√(p/3)·Y`, `√(p/3)·Z`.
    ///
    /// # Panics
    /// Panics if `p` is outside `[0, 1]`.
    pub fn depolarizing(qubit: u32, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "depolarizing probability p must be in [0, 1], got {p}"
        );
        let q = p / 3.0;
        Self::pauli_channel(qubit, q, q, q)
    }

    /// Single-qubit amplitude-damping channel with decay parameter `gamma ∈ [0, 1]`.
    ///
    /// Models spontaneous emission: `|1⟩` decays to `|0⟩` with probability `γ`.
    ///
    /// Kraus operators:
    /// ```text
    /// K₀ = [[1,      0       ],   K₁ = [[0, √γ],
    ///        [0, √(1−γ)]]                [0,  0 ]]
    /// ```
    ///
    /// # Panics
    /// Panics if `gamma` is outside `[0, 1]`.
    pub fn amplitude_damping(qubit: u32, gamma: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "amplitude damping gamma must be in [0, 1], got {gamma}"
        );
        let k0 = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new((1.0 - gamma).sqrt(), 0.0),
            ],
        );
        let k1 = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(gamma.sqrt(), 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        );
        Self {
            operators: vec![k0, k1],
            qubits: vec![qubit],
        }
    }

    /// Single-qubit phase-damping channel with dephasing parameter `lambda ∈ [0, 1]`.
    ///
    /// Models pure dephasing: off-diagonal elements (coherences) decay while populations
    /// `ρ₀₀` and `ρ₁₁` are preserved.
    ///
    /// Kraus operators:
    /// ```text
    /// K₀ = [[1,       0      ],   K₁ = [[0,   0 ],
    ///        [0, √(1−λ)]]                [0, √λ]]
    /// ```
    ///
    /// # Panics
    /// Panics if `lambda` is outside `[0, 1]`.
    pub fn phase_damping(qubit: u32, lambda: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "phase damping lambda must be in [0, 1], got {lambda}"
        );
        let k0 = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new((1.0 - lambda).sqrt(), 0.0),
            ],
        );
        let k1 = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(lambda.sqrt(), 0.0),
            ],
        );
        Self {
            operators: vec![k0, k1],
            qubits: vec![qubit],
        }
    }

    /// Returns the target qubit indices in ascending order.
    pub fn qubits(&self) -> &[u32] {
        &self.qubits
    }

    /// Returns the number of target qubits.
    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns the Kraus operators.
    pub fn operators(&self) -> &[ComplexSquareMatrix] {
        &self.operators
    }

    /// Returns `true` if the CPTP completeness condition `Σ_i K_i†K_i ≈ I` holds within
    /// the given element-wise tolerance.
    pub fn check_cptp(&self, tol: f64) -> bool {
        let n = self.operators[0].edge_size();
        let mut sum = ComplexSquareMatrix::zeros(n);
        for k in &self.operators {
            // Accumulate K†K: (K†K)[r,c] = Σ_j conj(K[j,r]) · K[j,c]
            for r in 0..n {
                for c in 0..n {
                    let mut acc = Complex::default();
                    for j in 0..n {
                        acc += k.get(j, r).conj() * k.get(j, c);
                    }
                    sum.set(r, c, sum.get(r, c) + acc);
                }
            }
        }
        sum.maximum_norm_diff(&ComplexSquareMatrix::eye(n)) < tol
    }

    /// Computes the superoperator (transfer matrix) for this channel.
    ///
    /// Returns a `4^k × 4^k` matrix `S` such that for any density matrix ρ vectorized as
    ///
    /// ```text
    /// vec(ρ)[ ket_idx | (bra_idx << k) ]  =  ρ[ket_idx, bra_idx]
    /// ```
    ///
    /// the channel action satisfies `vec(ε(ρ)) = S · vec(ρ)`.
    ///
    /// Matrix entries:
    /// ```text
    /// S[r, c]  =  Σ_i  K_i[i_ket, j_ket]  ·  conj(K_i[i_bra, j_bra])
    /// ```
    /// where `i_ket = r & (2^k − 1)`, `i_bra = r >> k`, and similarly for `j_*`.
    pub fn superoperator_matrix(&self) -> ComplexSquareMatrix {
        let k = self.n_qubits();
        let dim = 1usize << k; // 2^k — gate Hilbert-space dimension
        let super_dim = dim * dim; // 4^k — superoperator dimension

        let mut s = ComplexSquareMatrix::zeros(super_dim);
        for r in 0..super_dim {
            let i_ket = r & (dim - 1);
            let i_bra = r >> k;
            for c in 0..super_dim {
                let j_ket = c & (dim - 1);
                let j_bra = c >> k;
                let mut acc = Complex::default();
                for op in &self.operators {
                    acc += op.get(i_ket, j_ket) * op.get(i_bra, j_bra).conj();
                }
                s.set(r, c, acc);
            }
        }
        s
    }

    /// Symmetric depolarizing channel on `n` qubits with error probability `p ∈ [0, 1]`.
    ///
    /// All `4ⁿ − 1` non-identity n-qubit Pauli operators are applied with equal weight:
    ///
    /// ```text
    /// ε(ρ) = (1 − p) ρ  +  p/(4ⁿ−1) · Σ_{P ≠ Iⁿ} P ρ P†
    /// ```
    ///
    /// Kraus operators: `√(1−p)·Iⁿ` plus `√(p/(4ⁿ−1))·P` for each of the `4ⁿ−1`
    /// non-identity n-qubit Paulis. For `n = 1` this is identical to [`Self::depolarizing`].
    ///
    /// Qubits must be provided in **strictly ascending order**.
    ///
    /// # Panics
    /// Panics if `p` is outside `[0, 1]`, qubits are not sorted/distinct, or `n > 10`.
    pub fn symmetric_depolarizing(qubits: &[u32], p: f64) -> Self {
        assert!(!qubits.is_empty(), "symmetric_depolarizing requires at least one qubit");
        assert!(
            (0.0..=1.0).contains(&p),
            "symmetric_depolarizing probability p must be in [0, 1], got {p}"
        );
        assert!(
            qubits.windows(2).all(|w| w[0] < w[1]),
            "qubits must be distinct and in ascending order"
        );
        let n = qubits.len();
        assert!(n <= 10, "symmetric_depolarizing: n > 10 would produce {} Kraus operators", 4usize.pow(n as u32));

        let n_paulis = 4usize.pow(n as u32); // 4^n
        let non_id = (n_paulis - 1) as f64;
        let s_id = (1.0 - p).sqrt();
        let s_noise = (p / non_id).sqrt();

        let operators = (0..n_paulis)
            .map(|i| {
                let scale = if i == 0 { s_id } else { s_noise };
                &n_qubit_pauli(i, n) * scale
            })
            .collect();

        Self { operators, qubits: qubits.to_vec() }
    }

    /// General single-qubit Pauli channel.
    ///
    /// Applies X, Y, Z errors with independent probabilities `p_x`, `p_y`, `p_z`:
    ///
    /// ```text
    /// ε(ρ) = p_I·ρ  +  p_x·XρX  +  p_y·YρY  +  p_z·ZρZ
    /// ```
    ///
    /// where `p_I = 1 − p_x − p_y − p_z`. Special cases:
    /// - `pauli_channel(q, p/3, p/3, p/3)` = `depolarizing(q, p)`
    /// - `pauli_channel(q, p, 0, 0)` = `bit_flip(q, p)`
    /// - `pauli_channel(q, 0, 0, p)` = `phase_flip(q, p)`
    ///
    /// # Panics
    /// Panics if any probability is negative or `p_x + p_y + p_z > 1`.
    pub fn pauli_channel(qubit: u32, p_x: f64, p_y: f64, p_z: f64) -> Self {
        let p_i = 1.0 - p_x - p_y - p_z;
        assert!(p_x >= 0.0 && p_y >= 0.0 && p_z >= 0.0 && p_i >= -1e-12,
            "pauli_channel: probabilities must be non-negative and sum to at most 1 \
             (got p_x={p_x}, p_y={p_y}, p_z={p_z}, implied p_I={p_i})");
        let p_i = p_i.max(0.0);
        Self {
            operators: vec![
                &ComplexSquareMatrix::eye(2) * p_i.sqrt(),
                &ComplexSquareMatrix::x() * p_x.sqrt(),
                &ComplexSquareMatrix::y() * p_y.sqrt(),
                &ComplexSquareMatrix::z() * p_z.sqrt(),
            ],
            qubits: vec![qubit],
        }
    }

    /// Single-qubit bit-flip channel: applies X with probability `p`.
    ///
    /// Kraus operators: `√(1−p)·I`, `√p·X`.
    ///
    /// # Panics
    /// Panics if `p` is outside `[0, 1]`.
    pub fn bit_flip(qubit: u32, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "bit_flip probability p must be in [0, 1], got {p}");
        Self::pauli_channel(qubit, p, 0.0, 0.0)
    }

    /// Single-qubit phase-flip channel: applies Z with probability `p`.
    ///
    /// Kraus operators: `√(1−p)·I`, `√p·Z`.
    ///
    /// # Panics
    /// Panics if `p` is outside `[0, 1]`.
    pub fn phase_flip(qubit: u32, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "phase_flip probability p must be in [0, 1], got {p}");
        Self::pauli_channel(qubit, 0.0, 0.0, p)
    }

    /// Single-qubit generalized amplitude-damping channel.
    ///
    /// Models thermalization toward a thermal state at finite temperature. The parameter
    /// `p ∈ [0, 1]` is the excited-state population at thermal equilibrium (p = 0 gives
    /// zero temperature, reducing to [`Self::amplitude_damping`]). `gamma ∈ [0, 1]` is
    /// the relaxation probability per application.
    ///
    /// Kraus operators:
    /// ```text
    /// K₀ = √(1−p) [[1,       0      ],   K₁ = √(1−p) [[0, √γ],
    ///               [0, √(1−γ)]]                        [0,  0 ]]
    ///
    /// K₂ = √p [[√(1−γ), 0],              K₃ = √p [[0,   0 ],
    ///           [0,      1]]                        [√γ,  0 ]]
    /// ```
    ///
    /// At `p = 0` only K₀ and K₁ survive, recovering the standard amplitude-damping channel.
    ///
    /// # Panics
    /// Panics if `p` or `gamma` is outside `[0, 1]`.
    pub fn generalized_amplitude_damping(qubit: u32, p: f64, gamma: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1], got {p}");
        assert!((0.0..=1.0).contains(&gamma), "gamma must be in [0, 1], got {gamma}");
        let sg = gamma.sqrt();
        let s1g = (1.0 - gamma).sqrt();
        let sp = p.sqrt();
        let s1p = (1.0 - p).sqrt();

        let k0 = ComplexSquareMatrix::from_vec(2, vec![
            Complex::new(s1p, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(s1p * s1g, 0.0),
        ]);
        let k1 = ComplexSquareMatrix::from_vec(2, vec![
            Complex::new(0.0, 0.0), Complex::new(s1p * sg, 0.0),
            Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
        ]);
        let k2 = ComplexSquareMatrix::from_vec(2, vec![
            Complex::new(sp * s1g, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),      Complex::new(sp, 0.0),
        ]);
        let k3 = ComplexSquareMatrix::from_vec(2, vec![
            Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
            Complex::new(sp * sg, 0.0), Complex::new(0.0, 0.0),
        ]);
        Self { operators: vec![k0, k1, k2, k3], qubits: vec![qubit] }
    }

    /// Wraps this channel as a [`QuantumGate`].
    ///
    /// The superoperator is computed eagerly and stored inside the gate. The gate's
    /// `qubits()` returns the `k` **physical** qubit indices; the simulation dispatch layer
    /// is responsible for mapping those to the correct ket/bra positions in the
    /// `2n`-qubit density-matrix statevector.
    pub fn to_gate(self) -> QuantumGate {
        QuantumGate::from_channel(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ComplexSquareMatrix;

    const TOL: f64 = 1e-12;

    // ── CPTP validation ───────────────────────────────────────────────────────

    #[test]
    fn depolarizing_is_cptp() {
        for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                KrausChannel::depolarizing(0, p).check_cptp(TOL),
                "depolarizing(p={p}) failed CPTP check"
            );
        }
    }

    #[test]
    fn amplitude_damping_is_cptp() {
        for &gamma in &[0.0, 0.1, 0.5, 1.0] {
            assert!(
                KrausChannel::amplitude_damping(0, gamma).check_cptp(TOL),
                "amplitude_damping(gamma={gamma}) failed CPTP check"
            );
        }
    }

    #[test]
    fn phase_damping_is_cptp() {
        for &lambda in &[0.0, 0.3, 0.7, 1.0] {
            assert!(
                KrausChannel::phase_damping(0, lambda).check_cptp(TOL),
                "phase_damping(lambda={lambda}) failed CPTP check"
            );
        }
    }

    #[test]
    fn from_gate_is_cptp() {
        // Any unitary gate wrapped as a channel is CPTP.
        let gates = [
            QuantumGate::h(0),
            QuantumGate::cx(0, 1),
            QuantumGate::random_unitary(&[0, 1]),
        ];
        for gate in &gates {
            assert!(KrausChannel::from_gate(gate).check_cptp(TOL));
        }
    }

    // ── superoperator correctness ─────────────────────────────────────────────

    /// Apply the superoperator S to a vectorized density matrix and return the result
    /// as a 2×2 ComplexSquareMatrix.
    fn apply_super(s: &ComplexSquareMatrix, rho: &ComplexSquareMatrix) -> ComplexSquareMatrix {
        let n = rho.edge_size(); // 2 for a 1-qubit state
        let super_dim = n * n;
        assert_eq!(s.edge_size(), super_dim);

        // Vectorize ρ: vec[ket | (bra << log2(n))]
        let k = n.ilog2() as usize; // = 1 for n=2
        let mut vec_rho = vec![Complex::default(); super_dim];
        for ket in 0..n {
            for bra in 0..n {
                vec_rho[ket | (bra << k)] = rho.get(ket, bra);
            }
        }

        // Multiply S * vec_rho
        let mut out_vec = vec![Complex::default(); super_dim];
        for r in 0..super_dim {
            for c in 0..super_dim {
                out_vec[r] += s.get(r, c) * vec_rho[c];
            }
        }

        // Unvectorize back to a matrix
        let mut result = ComplexSquareMatrix::zeros(n);
        for ket in 0..n {
            for bra in 0..n {
                result.set(ket, bra, out_vec[ket | (bra << k)]);
            }
        }
        result
    }

    /// Compute K ρ K† directly.
    fn apply_kraus_direct(k: &ComplexSquareMatrix, rho: &ComplexSquareMatrix) -> ComplexSquareMatrix {
        let n = rho.edge_size();
        // (K ρ K†)[i,j] = Σ_{a,b} K[i,a] ρ[a,b] conj(K[j,b])
        let mut out = ComplexSquareMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex::default();
                for a in 0..n {
                    for b in 0..n {
                        acc += k.get(i, a) * rho.get(a, b) * k.get(j, b).conj();
                    }
                }
                out.set(i, j, acc);
            }
        }
        out
    }

    #[test]
    fn x_gate_superoperator_matches_direct() {
        let channel = KrausChannel::from_gate(&QuantumGate::x(0));
        let s = channel.superoperator_matrix();

        // Test with a non-trivial density matrix
        let rho = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(0.7, 0.0),
                Complex::new(0.1, 0.2),
                Complex::new(0.1, -0.2),
                Complex::new(0.3, 0.0),
            ],
        );

        let via_super = apply_super(&s, &rho);
        let direct = apply_kraus_direct(&ComplexSquareMatrix::x(), &rho);

        assert!(
            via_super.maximum_norm_diff(&direct) < TOL,
            "X superoperator mismatch: max diff = {}",
            via_super.maximum_norm_diff(&direct)
        );
    }

    #[test]
    fn identity_channel_superoperator_is_identity_map() {
        let channel = KrausChannel::from_gate(&QuantumGate::h(0));
        let s_h = channel.superoperator_matrix();

        let rho = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(0.6, 0.0),
                Complex::new(0.0, 0.3),
                Complex::new(0.0, -0.3),
                Complex::new(0.4, 0.0),
            ],
        );

        let via_super = apply_super(&s_h, &rho);
        let direct = apply_kraus_direct(&ComplexSquareMatrix::h(), &rho);

        assert!(via_super.maximum_norm_diff(&direct) < TOL);
    }

    #[test]
    fn depolarizing_p0_is_identity_channel() {
        let s = KrausChannel::depolarizing(0, 0.0).superoperator_matrix();
        let eye4 = ComplexSquareMatrix::eye(4);
        assert!(
            s.maximum_norm_diff(&eye4) < TOL,
            "depolarizing(p=0) superoperator should be identity"
        );
    }

    #[test]
    fn amplitude_damping_gamma1_maps_excited_to_ground() {
        // γ=1: any state should decay to |0⟩⟨0|
        let s = KrausChannel::amplitude_damping(0, 1.0).superoperator_matrix();

        // ρ = |1⟩⟨1| = [[0,0],[0,1]]
        let excited = ComplexSquareMatrix::from_reals(2, &[0.0, 0.0, 0.0, 1.0]);
        let ground = ComplexSquareMatrix::from_reals(2, &[1.0, 0.0, 0.0, 0.0]);

        let result = apply_super(&s, &excited);
        assert!(
            result.maximum_norm_diff(&ground) < TOL,
            "amplitude damping γ=1 should map |1⟩⟨1| → |0⟩⟨0|"
        );
    }

    // ── new standard channels ─────────────────────────────────────────────────

    #[test]
    fn symmetric_depolarizing_is_cptp() {
        // 1-qubit
        for &p in &[0.0, 0.5, 1.0] {
            assert!(KrausChannel::symmetric_depolarizing(&[0], p).check_cptp(TOL));
        }
        // 2-qubit
        for &p in &[0.0, 0.3, 1.0] {
            assert!(KrausChannel::symmetric_depolarizing(&[0, 1], p).check_cptp(TOL));
        }
    }

    #[test]
    fn symmetric_depolarizing_n1_matches_depolarizing() {
        // For a single qubit these two must produce the same superoperator matrix.
        for &p in &[0.0, 0.25, 0.9] {
            let sym = KrausChannel::symmetric_depolarizing(&[0], p).superoperator_matrix();
            let dep = KrausChannel::depolarizing(0, p).superoperator_matrix();
            assert!(
                sym.maximum_norm_diff(&dep) < TOL,
                "mismatch at p={p}: diff={}",
                sym.maximum_norm_diff(&dep)
            );
        }
    }

    #[test]
    fn symmetric_depolarizing_p0_is_identity() {
        let s = KrausChannel::symmetric_depolarizing(&[0, 1], 0.0).superoperator_matrix();
        assert!(s.maximum_norm_diff(&ComplexSquareMatrix::eye(16)) < TOL);
    }

    #[test]
    fn pauli_channel_is_cptp() {
        assert!(KrausChannel::pauli_channel(0, 0.1, 0.2, 0.3).check_cptp(TOL));
        assert!(KrausChannel::pauli_channel(0, 0.0, 0.0, 0.0).check_cptp(TOL));
        assert!(KrausChannel::pauli_channel(0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0).check_cptp(TOL));
    }

    #[test]
    fn pauli_channel_recovers_depolarizing() {
        let p = 0.6_f64;
        let via_pauli = KrausChannel::pauli_channel(0, p / 3.0, p / 3.0, p / 3.0)
            .superoperator_matrix();
        let via_dep = KrausChannel::depolarizing(0, p).superoperator_matrix();
        assert!(via_pauli.maximum_norm_diff(&via_dep) < TOL);
    }

    #[test]
    fn bit_flip_is_cptp() {
        for &p in &[0.0, 0.5, 1.0] {
            assert!(KrausChannel::bit_flip(0, p).check_cptp(TOL));
        }
    }

    #[test]
    fn bit_flip_p1_flips_all_states() {
        let s = KrausChannel::bit_flip(0, 1.0).superoperator_matrix();
        let s_x = KrausChannel::from_gate(&QuantumGate::x(0)).superoperator_matrix();
        assert!(s.maximum_norm_diff(&s_x) < TOL);
    }

    #[test]
    fn phase_flip_is_cptp() {
        for &p in &[0.0, 0.5, 1.0] {
            assert!(KrausChannel::phase_flip(0, p).check_cptp(TOL));
        }
    }

    #[test]
    fn generalized_amplitude_damping_is_cptp() {
        for &p in &[0.0, 0.3, 0.5, 1.0] {
            for &gamma in &[0.0, 0.5, 1.0] {
                assert!(
                    KrausChannel::generalized_amplitude_damping(0, p, gamma).check_cptp(TOL),
                    "failed at p={p}, gamma={gamma}"
                );
            }
        }
    }

    #[test]
    fn generalized_amplitude_damping_p0_matches_amplitude_damping() {
        for &gamma in &[0.0, 0.3, 1.0] {
            let gen = KrausChannel::generalized_amplitude_damping(0, 0.0, gamma)
                .superoperator_matrix();
            let std = KrausChannel::amplitude_damping(0, gamma).superoperator_matrix();
            assert!(
                gen.maximum_norm_diff(&std) < TOL,
                "mismatch at gamma={gamma}: diff={}",
                gen.maximum_norm_diff(&std)
            );
        }
    }

    // ── to_gate ───────────────────────────────────────────────────────────────

    #[test]
    fn channel_gate_preserves_physical_qubits() {
        // Physical qubit indices survive round-trip through to_gate()
        let gate = KrausChannel::from_gate(&QuantumGate::x(0)).to_gate();
        assert_eq!(gate.qubits(), &[0]);
        assert!(!gate.is_unitary());

        let gate2 = KrausChannel::from_gate(&QuantumGate::cx(1, 2)).to_gate();
        assert_eq!(gate2.qubits(), &[1, 2]);
        assert!(!gate2.is_unitary());
    }

    #[test]
    fn channel_gate_matrix_is_superoperator() {
        // QuantumGate::from_channel stores the superoperator matrix
        let channel = KrausChannel::from_gate(&QuantumGate::x(0));
        let expected_super = channel.superoperator_matrix();
        let gate = channel.to_gate();
        assert_eq!(gate.matrix().edge_size(), 4); // 4^1 = 4
        assert!(gate.matrix().maximum_norm_diff(&expected_super) < TOL);
    }
}

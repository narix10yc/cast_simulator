use std::sync::Arc;

use rand::Rng;

use super::{Complex, ComplexSquareMatrix, NoiseModel};

// ── QuantumGate ─────────────────────────────────────────────────────────────

/// A quantum gate: a unitary matrix paired with target qubit indices, optionally
/// carrying a [`NoiseModel`] of full Kraus operators.
///
/// Qubits are stored in ascending order; [`QuantumGate::new`] permutes the
/// matrix if needed.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGate {
    /// Gate matrix.
    matrix: ComplexSquareMatrix,
    /// Physical qubit indices, sorted ascending.
    qubits: Vec<u32>,
    /// Optional noise model.
    noise: Option<Arc<NoiseModel>>,
}

impl QuantumGate {
    /// Creates a unitary gate from a matrix and a list of target qubits.
    ///
    /// The matrix dimension must equal $2^n$ where `n = qubits.len()`.
    /// If the qubits are not in ascending order, the matrix is permuted so that
    /// `qubits[k]` maps to bit position `k` in the canonical basis.
    ///
    /// # Panics
    /// Panics if the matrix dimension does not match the qubit count, or if any
    /// two qubit indices are equal.
    pub fn new(matrix: ComplexSquareMatrix, qubits: Vec<u32>) -> Self {
        assert_eq!(
            matrix.edge_size(),
            edge_size_for_n_qubits(qubits.len()),
            "matrix dimension does not match number of target qubits"
        );

        let (matrix, qubits) = canonicalize_qubits_and_matrix(matrix, qubits);
        Self {
            matrix,
            qubits,
            noise: None,
        }
    }

    /// Creates a gate from real-valued matrix entries (imaginary parts are zero).
    ///
    /// `data` must be a flat, row-major slice of length `(2^n)²`.
    pub fn from_reals(qubits: Vec<u32>, data: &[f64]) -> Self {
        let edge_size = edge_size_for_n_qubits(qubits.len());
        let matrix = ComplexSquareMatrix::from_reals(edge_size, data);
        Self::new(matrix, qubits)
    }

    pub fn x(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::x(), vec![qubit])
    }

    pub fn y(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::y(), vec![qubit])
    }

    pub fn z(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::z(), vec![qubit])
    }

    pub fn h(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::h(), vec![qubit])
    }

    pub fn s(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::s(), vec![qubit])
    }

    pub fn t(qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::t(), vec![qubit])
    }

    pub fn rx(theta: f64, qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::rx(theta), vec![qubit])
    }

    pub fn ry(theta: f64, qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::ry(theta), vec![qubit])
    }

    pub fn rz(theta: f64, qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::rz(theta), vec![qubit])
    }

    pub fn u3(theta: f64, phi: f64, lambda: f64, qubit: u32) -> Self {
        Self::new(ComplexSquareMatrix::u3(theta, phi, lambda), vec![qubit])
    }

    pub fn cx(ctrl: u32, targ: u32) -> Self {
        Self::new(ComplexSquareMatrix::cx(), vec![ctrl, targ])
    }

    pub fn cz(ctrl: u32, targ: u32) -> Self {
        Self::new(ComplexSquareMatrix::cz(), vec![ctrl, targ])
    }

    pub fn cp(theta: f64, ctrl: u32, targ: u32) -> Self {
        Self::new(ComplexSquareMatrix::cp(theta), vec![ctrl, targ])
    }

    pub fn swap(q0: u32, q1: u32) -> Self {
        Self::new(ComplexSquareMatrix::swap(), vec![q0, q1])
    }

    pub fn ccx(ctrl0: u32, ctrl1: u32, targ: u32) -> Self {
        Self::new(ComplexSquareMatrix::ccx(), vec![ctrl0, ctrl1, targ])
    }

    /// Returns the gate's unitary matrix.
    pub fn matrix(&self) -> &ComplexSquareMatrix {
        &self.matrix
    }

    /// Returns the gate's target qubit indices in ascending order.
    pub fn qubits(&self) -> &[u32] {
        &self.qubits
    }

    /// Number of target qubits.
    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Effective qubit count for cost estimation: `n_qubits` for pure gates,
    /// `2 * n_qubits` for noisy gates (superoperator size).
    pub fn effective_n_qubits(&self) -> usize {
        if self.is_unitary() {
            self.qubits.len()
        } else {
            2 * self.qubits.len()
        }
    }

    /// Returns `true` if this gate has no noise (pure unitary).
    pub fn is_unitary(&self) -> bool {
        self.noise.is_none()
    }

    /// Returns the noise model, or `None` for noiseless gates.
    pub fn noise_model(&self) -> Option<&Arc<NoiseModel>> {
        self.noise.as_ref()
    }

    /// Attach a noise model to this gate.
    ///
    /// Each entry `(p_i, V_i)` is a probability and a noise-only unitary operator
    /// of the same dimension as the gate matrix. Internally, each operator is
    /// pre-composed with the base unitary to form the full Kraus operator
    /// `K_i = V_i · U`. Probabilities must sum to 1.0.
    pub fn with_noise(mut self, noise_ops: Vec<(f64, ComplexSquareMatrix)>) -> Self {
        let edge = self.matrix.edge_size();
        let branches = noise_ops
            .into_iter()
            .map(|(p, v)| {
                assert_eq!(
                    v.edge_size(),
                    edge,
                    "noise operator dimension must match gate dimension"
                );
                (p, v.matmul(&self.matrix))
            })
            .collect();
        self.noise = Some(Arc::new(NoiseModel::new(branches)));
        self
    }

    /// Attach single-qubit depolarizing noise with error probability `p`.
    ///
    /// The noise model is `[(1-p, I), (p/3, X), (p/3, Y), (p/3, Z)]`.
    pub fn with_depolarizing(self, p: f64) -> Self {
        assert_eq!(self.n_qubits(), 1, "depolarizing noise is single-qubit");
        self.with_noise(vec![
            (1.0 - p, ComplexSquareMatrix::eye(2)),
            (p / 3.0, ComplexSquareMatrix::x()),
            (p / 3.0, ComplexSquareMatrix::y()),
            (p / 3.0, ComplexSquareMatrix::z()),
        ])
    }

    // ── Noise-only gate factories ────────────────────────────────────────

    /// Single-qubit depolarizing noise (identity gate + depolarizing channel).
    pub fn depolarizing(qubit: u32, p: f64) -> Self {
        Self::new(ComplexSquareMatrix::eye(2), vec![qubit]).with_noise(vec![
            (1.0 - p, ComplexSquareMatrix::eye(2)),
            (p / 3.0, ComplexSquareMatrix::x()),
            (p / 3.0, ComplexSquareMatrix::y()),
            (p / 3.0, ComplexSquareMatrix::z()),
        ])
    }

    /// Single-qubit bit-flip noise (X error with probability `p`).
    pub fn bit_flip(qubit: u32, p: f64) -> Self {
        Self::new(ComplexSquareMatrix::eye(2), vec![qubit]).with_noise(vec![
            (1.0 - p, ComplexSquareMatrix::eye(2)),
            (p, ComplexSquareMatrix::x()),
        ])
    }

    /// Single-qubit phase-flip noise (Z error with probability `p`).
    pub fn phase_flip(qubit: u32, p: f64) -> Self {
        Self::new(ComplexSquareMatrix::eye(2), vec![qubit]).with_noise(vec![
            (1.0 - p, ComplexSquareMatrix::eye(2)),
            (p, ComplexSquareMatrix::z()),
        ])
    }

    /// Single-qubit Pauli channel: X with probability `px`, Y with `py`, Z with `pz`.
    pub fn pauli_channel(qubit: u32, px: f64, py: f64, pz: f64) -> Self {
        let pi = 1.0 - px - py - pz;
        Self::new(ComplexSquareMatrix::eye(2), vec![qubit]).with_noise(vec![
            (pi, ComplexSquareMatrix::eye(2)),
            (px, ComplexSquareMatrix::x()),
            (py, ComplexSquareMatrix::y()),
            (pz, ComplexSquareMatrix::z()),
        ])
    }

    /// Returns a gate suitable for density-matrix simulation.
    ///
    /// The n-qubit density matrix ρ is stored as a vectorized `2n`-qubit statevector.
    ///
    /// - **Noiseless** gate `U`: superoperator `S = U ⊗ conj(U)`.
    /// - **Noisy** gate with Kraus ops `[(p_i, K_i)]`: superoperator
    ///   `S = Σ_i p_i · K_i ⊗ conj(K_i)`.
    ///
    /// # Panics
    /// Panics if any physical qubit index `≥ n_total`.
    pub fn to_density_matrix_gate(&self, n_total: usize) -> Self {
        for &q in &self.qubits {
            assert!(
                (q as usize) < n_total,
                "qubit {q} is out of range for n_total={n_total}"
            );
        }

        let super_mat = if self.is_unitary() {
            superoperator_of(&self.matrix)
        } else {
            let super_dim = self.matrix.edge_size().pow(2);
            let mut s = ComplexSquareMatrix::zeros(super_dim);
            for (p, kraus) in self.noise.as_ref().unwrap().branches() {
                s += &(&superoperator_of(kraus) * *p);
            }
            s
        };

        let mut virtual_qubits: Vec<u32> = self.qubits.clone();
        virtual_qubits.extend(self.qubits.iter().map(|&q| q + n_total as u32));
        Self::new(super_mat, virtual_qubits)
    }

    /// Number of non-zero real/imaginary scalars in the gate matrix (counted independently).
    pub fn scalar_nnz(&self, ztol: f64) -> usize {
        self.matrix
            .data()
            .iter()
            .map(|z| (z.re.abs() > ztol) as usize + (z.im.abs() > ztol) as usize)
            .sum::<usize>()
    }

    /// Arithmetic intensity of the gate: `scalar_nnz(ztol) / matrix.edge_size()`.
    pub fn arithmetic_intensity(&self, ztol: f64) -> f64 {
        self.scalar_nnz(ztol) as f64 / self.matrix.edge_size() as f64
    }

    /// Returns the product `self * other`, expanding both gates to act on the union
    /// of their qubit sets before multiplying.
    ///
    /// If either gate carries noise, the fused Kraus operators are the Cartesian
    /// product `K_i^self · K_j^other` (pure gates contribute their base unitary as
    /// the single branch). The fused base unitary is `self.matrix * other.matrix`.
    pub fn matmul(&self, other: &Self) -> Self {
        let qubits = union_qubits(&self.qubits, &other.qubits);
        let lhs = expand_matrix(&self.matrix, &self.qubits, &qubits);
        let rhs = expand_matrix(&other.matrix, &other.qubits, &qubits);
        let fused_matrix = lhs.matmul(&rhs);

        // Noise: expand each gate's Kraus ops (or base unitary if pure) to the
        // union qubit set, then take the Cartesian product K_i · K_j.
        let fused_noise = match (&self.noise, &other.noise) {
            (None, None) => None,
            (Some(ln), None) => {
                let branches = ln
                    .branches()
                    .iter()
                    .map(|(p, k)| (*p, expand_matrix(k, &self.qubits, &qubits).matmul(&rhs)))
                    .collect();
                Some(Arc::new(NoiseModel { branches }))
            }
            (None, Some(rn)) => {
                let branches = rn
                    .branches()
                    .iter()
                    .map(|(p, k)| (*p, lhs.matmul(&expand_matrix(k, &other.qubits, &qubits))))
                    .collect();
                Some(Arc::new(NoiseModel { branches }))
            }
            (Some(ln), Some(rn)) => {
                let mut branches = Vec::with_capacity(ln.len() * rn.len());
                for (p_i, k_i) in ln.branches() {
                    let exp_ki = expand_matrix(k_i, &self.qubits, &qubits);
                    for (q_j, k_j) in rn.branches() {
                        let exp_kj = expand_matrix(k_j, &other.qubits, &qubits);
                        branches.push((p_i * q_j, exp_ki.matmul(&exp_kj)));
                    }
                }
                Some(Arc::new(NoiseModel { branches }))
            }
        };

        let mut gate = Self::new(fused_matrix, qubits);
        gate.noise = fused_noise;
        gate
    }

    /// Generate a random unitary gate for the given qubits.
    ///
    /// The distribution is uniform according to the Haar measure on the unitary group.
    /// Delegates to [`ComplexSquareMatrix::random_unitary_with_rng`] for the actual matrix
    /// generation.
    ///
    /// For reproducible randomness, use [`Self::random_unitary_with_rng`].
    pub fn random_unitary(qubits: &[u32]) -> Self {
        let rng = &mut rand::thread_rng();
        Self::random_unitary_with_rng(qubits, rng)
    }

    /// Generate a random unitary gate for the given qubits.
    ///
    /// The distribution is uniform according to the Haar measure on the unitary group.
    /// Delegates to [`ComplexSquareMatrix::random_unitary_with_rng`] for the actual matrix
    /// generation.
    pub fn random_unitary_with_rng<R: Rng + ?Sized>(qubits: &[u32], rng: &mut R) -> Self {
        let mut sorted = qubits.to_vec();
        sorted.sort();
        let edge_size = edge_size_for_n_qubits(sorted.len());
        let matrix = ComplexSquareMatrix::random_unitary_with_rng(edge_size, rng);
        Self::new(matrix, sorted)
    }

    /// Generate a random sparse non-unitary gate for the given qubits.
    /// Note: the resulting matrix is not guaranteed to be unitary. This method is to be used for
    /// testing and benchmarking purposes, not for simulating actual quantum circuits.  
    ///
    /// For reproducible randomness, use [`Self::random_sparse_with_rng`].
    pub fn random_sparse(qubits: &[u32], sparsity: f64) -> Self {
        let rng = &mut rand::thread_rng();
        Self::random_sparse_with_rng(qubits, sparsity, rng)
    }

    /// Generate a random sparse non-unitary gate for the given qubits.
    /// Note: the resulting matrix is not guaranteed to be unitary. Only intended to be used
    /// for testing and benchmarking purposes.
    pub fn random_sparse_with_rng(qubits: &[u32], sparsity: f64, rng: &mut impl Rng) -> Self {
        let mut sorted = qubits.to_vec();
        sorted.sort();
        let edge_size = edge_size_for_n_qubits(sorted.len());
        let mut matrix = ComplexSquareMatrix::random_sparse_with_rng(edge_size, sparsity, rng);

        // Ensure each row contains at least one non-zero entry
        // This should happen with super-low prob. If it does, we just randomly sprinkle a non-zero
        // in that row.
        for row in 0..edge_size {
            if matrix.row(row).iter().all(|c| c.norm() <= 1e-8) {
                let col = rng.gen_range(0..edge_size);
                let v = rng.gen_range(0.25_f64..1.0);
                matrix.set(row, col, Complex::new(v, 0.0));
            }
        }

        Self::new(matrix, sorted)
    }

    /// Generate a random sparse gate targeting the given arithmetic intensity,
    /// with randomly chosen target qubits in `0..n_qubits_sv`.
    ///
    /// Auto-determines the gate size (number of qubits) needed to achieve `ai`
    /// with sparsity in `[0, 1]`, then picks that many random distinct qubits.
    /// The matrix is not guaranteed to be unitary — intended for benchmarking
    /// and profiling, not quantum circuit simulation.
    ///
    /// `ztol` controls the zero-tolerance used when computing the actual AI of
    /// the resulting gate.
    ///
    /// # Panics
    /// Panics if `n_qubits_sv` is too small for the required gate size.
    pub fn random_arithmetic_intensity(n_qubits_sv: u32, ai: f64, ztol: f64) -> Self {
        let rng = &mut rand::thread_rng();
        Self::random_arithmetic_intensity_with_rng(n_qubits_sv, ai, ztol, rng)
    }

    /// Seeded variant of [`Self::random_arithmetic_intensity`].
    pub fn random_arithmetic_intensity_with_rng(
        n_qubits_sv: u32,
        ai: f64,
        ztol: f64,
        rng: &mut impl Rng,
    ) -> Self {
        // Minimum gate qubits k such that edge_size = 2^k can achieve the
        // target AI with sparsity <= 1:  max_ai = 2 * edge_size, so we need
        // edge_size >= ceil(ai / 2), i.e. k >= ceil(log2(ceil(ai / 2))).
        const MIN_GATE_QUBITS: u32 = 4;
        let k = ((ai / 2.0).ceil().max(1.0).log2().ceil() as u32).max(MIN_GATE_QUBITS);
        assert!(
            n_qubits_sv >= k,
            "n_qubits_sv ({n_qubits_sv}) too small for gate requiring {k} qubits"
        );

        let edge_size = 1u64 << k;
        let max_ai = 2.0 * edge_size as f64;
        let sparsity = (ai / max_ai).clamp(0.0, 1.0);

        // Pick k distinct random qubits from 0..n_qubits_sv.
        let mut pool: Vec<u32> = (0..n_qubits_sv).collect();
        for i in 0..k as usize {
            let j = rng.gen_range(i..pool.len());
            pool.swap(i, j);
        }
        let qubits = &pool[..k as usize];

        let _ = ztol; // ztol is for the caller's use; gate construction doesn't need it
        Self::random_sparse_with_rng(qubits, sparsity, rng)
    }
}

/// Expands a matrix that acts on `gate_qubits` into a larger matrix that acts
/// on `union_qubits`, treating non-target qubits as identity wires.
fn expand_matrix(
    matrix: &ComplexSquareMatrix,
    gate_qubits: &[u32],
    union_qubits: &[u32],
) -> ComplexSquareMatrix {
    let union_edge_size = edge_size_for_n_qubits(union_qubits.len());
    let target_positions: Vec<usize> = gate_qubits
        .iter()
        .map(|qubit| {
            union_qubits
                .iter()
                .position(|candidate| candidate == qubit)
                .expect("gate qubit must exist in union qubits")
        })
        .collect();

    let target_mask = target_positions
        .iter()
        .fold(0usize, |mask, position| mask | (1usize << position));

    let mut expanded = ComplexSquareMatrix::zeros(union_edge_size);
    for row in 0..union_edge_size {
        for col in 0..union_edge_size {
            if ((row ^ col) & !target_mask) != 0 {
                continue;
            }
            let gate_row = compress_bits(row, &target_positions);
            let gate_col = compress_bits(col, &target_positions);
            expanded.set(row, col, matrix.get(gate_row, gate_col));
        }
    }
    expanded
}

/// Returns `2^n_qubits`, panicking on overflow.
fn edge_size_for_n_qubits(n_qubits: usize) -> usize {
    1usize
        .checked_shl(n_qubits as u32)
        .expect("too many qubits to represent matrix dimension")
}

/// Sorts `qubits` in ascending order and permutes `matrix` rows/columns to match.
///
/// `new_to_old_positions[k]` is the original position of the qubit that ends up at
/// sorted position `k`. The basis index mapping (`index_map`) is computed once in
/// O(n) and then applied in-place via row then column cycle permutations,
/// using O(n) temporary storage instead of a full O(n²) copy.
fn canonicalize_qubits_and_matrix(
    mut matrix: ComplexSquareMatrix,
    qubits: Vec<u32>,
) -> (ComplexSquareMatrix, Vec<u32>) {
    let mut permutation: Vec<(usize, u32)> = qubits.into_iter().enumerate().collect();
    permutation.sort_unstable_by_key(|(_, qubit)| *qubit);

    let sorted_qubits: Vec<u32> = permutation.iter().map(|(_, qubit)| *qubit).collect();
    assert!(
        sorted_qubits.windows(2).all(|window| window[0] < window[1]),
        "gate qubits must be distinct"
    );

    let new_to_old_positions: Vec<usize> = permutation
        .iter()
        .map(|(old_position, _)| *old_position)
        .collect();

    if new_to_old_positions
        .iter()
        .enumerate()
        .all(|(new_position, old_position)| new_position == *old_position)
    {
        return (matrix, sorted_qubits);
    }

    // Compute the basis index mapping once (O(n)) rather than O(n²) times inside the loop.
    let n = matrix.edge_size();
    let index_map: Vec<usize> = (0..n)
        .map(|i| permute_basis_index(i, &new_to_old_positions))
        .collect();

    // result[i][j] = input[index_map[i]][index_map[j]]:
    // apply the row permutation then the column permutation, both in-place.
    apply_row_permutation(&mut matrix, &index_map);
    apply_col_permutation(&mut matrix, &index_map);

    (matrix, sorted_qubits)
}

/// Permutes rows in-place: after the call, `matrix[i][*] == old_matrix[perm[i]][*]`.
///
/// Uses cycle decomposition so only one row-sized buffer is needed as scratch space.
/// Row copies are delegated to `copy_within` for contiguous, cache-friendly moves.
fn apply_row_permutation(matrix: &mut ComplexSquareMatrix, perm: &[usize]) {
    let n = matrix.edge_size();
    let mut visited = vec![false; n];
    let mut temp = vec![Complex::default(); n];

    for start in 0..n {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }

        // Save the starting row to close the cycle later.
        temp.copy_from_slice(&matrix.data()[start * n..start * n + n]);

        let mut current = start;
        loop {
            let next = perm[current];
            visited[current] = true;
            if next == start {
                matrix.data_mut()[current * n..current * n + n].copy_from_slice(&temp);
                break;
            }
            matrix
                .data_mut()
                .copy_within(next * n..next * n + n, current * n);
            current = next;
        }
    }
}

/// Permutes columns in-place: after the call, `matrix[*][j] == old_matrix[*][perm[j]]`.
///
/// Collects each permutation cycle once, then applies it across every row so that
/// the inner loop accesses a single contiguous row at a time.
fn apply_col_permutation(matrix: &mut ComplexSquareMatrix, perm: &[usize]) {
    let n = matrix.edge_size();
    let mut visited = vec![false; n];

    for start in 0..n {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }

        // Collect the full cycle before touching the matrix.
        let mut cycle = vec![start];
        let mut j = perm[start];
        while j != start {
            cycle.push(j);
            j = perm[j];
        }
        for &c in &cycle {
            visited[c] = true;
        }

        // Rotate cycle columns across every row.
        let k = cycle.len();
        let data = matrix.data_mut();
        for r in 0..n {
            let temp = data[r * n + cycle[0]];
            for i in 0..k - 1 {
                data[r * n + cycle[i]] = data[r * n + cycle[i + 1]];
            }
            data[r * n + cycle[k - 1]] = temp;
        }
    }
}

/// Returns the sorted union of two already-sorted qubit lists (merge step of merge sort).
fn union_qubits(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(lhs.len() + rhs.len());
    let mut i = 0;
    let mut j = 0;

    while i < lhs.len() || j < rhs.len() {
        if i == lhs.len() {
            out.extend_from_slice(&rhs[j..]);
            break;
        }
        if j == rhs.len() {
            out.extend_from_slice(&lhs[i..]);
            break;
        }

        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => {
                out.push(lhs[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(rhs[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(lhs[i]);
                i += 1;
                j += 1;
            }
        }
    }

    out
}

/// Packs selected bits of `index` into contiguous low-order positions.
///
/// For each `(target_bit, position)` in `positions.iter().enumerate()`, the bit at
/// `position` in `index` is placed at bit `target_bit` in the output. This effectively
/// projects an expanded basis index down to the gate's local basis.
pub(crate) fn compress_bits(index: usize, positions: &[usize]) -> usize {
    let mut out = 0usize;
    for (target_bit, position) in positions.iter().enumerate() {
        let bit = (index >> position) & 1usize;
        out |= bit << target_bit;
    }
    out
}

/// Re-scatters the bits of `index` from canonical positions to original qubit positions.
///
/// Bit `new_position` of `index` is placed at bit `new_to_old_positions[new_position]`
/// in the output, mapping a sorted basis index back to the pre-sort basis.
fn permute_basis_index(index: usize, new_to_old_positions: &[usize]) -> usize {
    let mut out = 0usize;
    for (new_position, old_position) in new_to_old_positions.iter().enumerate() {
        let bit = (index >> new_position) & 1usize;
        out |= bit << old_position;
    }
    out
}

/// Compute the superoperator `S = U ⊗ conj(U)` for a matrix `U`.
///
/// `S[r, c] = U[i_ket, j_ket] * conj(U[i_bra, j_bra])`
/// where `i_ket = r % dim, i_bra = r / dim, j_ket = c % dim, j_bra = c / dim`.
fn superoperator_of(u: &ComplexSquareMatrix) -> ComplexSquareMatrix {
    let dim = u.edge_size();
    let super_dim = dim * dim;
    let mut data = vec![Complex::new(0.0, 0.0); super_dim * super_dim];

    for r in 0..super_dim {
        let i_ket = r & (dim - 1);
        let i_bra = r >> dim.trailing_zeros();
        for c in 0..super_dim {
            let j_ket = c & (dim - 1);
            let j_bra = c >> dim.trailing_zeros();
            let val = u.get(i_ket, j_ket) * u.get(i_bra, j_bra).conj();
            data[r * super_dim + c] = val;
        }
    }

    ComplexSquareMatrix::from_vec(super_dim, data)
}

#[cfg(test)]
mod tests {
    use super::QuantumGate;
    use crate::types::ComplexSquareMatrix;

    fn x() -> QuantumGate {
        QuantumGate::from_reals(vec![0], &[0.0, 1.0, 1.0, 0.0])
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn canonicalizes_qubit_order() {
        // X⊗I supplied as qubits=[1,0] must equal I⊗X supplied as qubits=[0,1].
        let got = QuantumGate::from_reals(
            vec![1, 0],
            &[
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        );
        let expected = QuantumGate::from_reals(
            vec![0, 1],
            &[
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        );
        assert_eq!(got, expected);
    }

    #[test]
    #[should_panic(expected = "gate qubits must be distinct")]
    fn rejects_duplicate_qubits() {
        QuantumGate::new(ComplexSquareMatrix::eye(4), vec![0, 0]);
    }

    #[test]
    #[should_panic(expected = "matrix dimension does not match number of target qubits")]
    fn rejects_wrong_matrix_size() {
        QuantumGate::new(ComplexSquareMatrix::eye(2), vec![0, 1]);
    }

    // ── matmul ───────────────────────────────────────────────────────────────

    #[test]
    fn matmul_involution() {
        // X² = I
        let product = x().matmul(&x());
        assert_eq!(product.matrix(), &ComplexSquareMatrix::eye(2));
    }

    #[test]
    fn matmul_disjoint_qubits() {
        // X(q0) · X(q1) = X⊗X, flips both bits.
        let product = QuantumGate::from_reals(vec![1], &[0.0, 1.0, 1.0, 0.0]).matmul(&x());
        let expected = QuantumGate::from_reals(
            vec![0, 1],
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(product, expected);
    }

    #[test]
    fn matmul_non_adjacent_qubits() {
        // X(q0) · X(q2) expands to qubits [0,2], flips both bits.
        let x_q2 = QuantumGate::from_reals(vec![2], &[0.0, 1.0, 1.0, 0.0]);
        let product = x().matmul(&x_q2);
        let expected = QuantumGate::from_reals(
            vec![0, 2],
            &[
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
        assert_eq!(product, expected);
    }

    #[test]
    fn matmul_overlapping_qubits() {
        // X(q1) · I(q0,q1) = X on q1 expanded to {q0,q1}.
        let x_q1 = QuantumGate::from_reals(vec![1], &[0.0, 1.0, 1.0, 0.0]);
        let product = x_q1.matmul(&QuantumGate::new(ComplexSquareMatrix::eye(4), vec![0, 1]));
        let expected = QuantumGate::from_reals(
            vec![0, 1],
            &[
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        );
        assert_eq!(product, expected);
    }

    #[test]
    fn matmul_identity_neutral() {
        // G · I == G and I · G == G.
        let h = QuantumGate::h(0);
        let eye = QuantumGate::new(ComplexSquareMatrix::eye(2), vec![0]);
        assert_eq!(h.matmul(&eye), h);
        assert_eq!(eye.matmul(&h), h);
    }
}

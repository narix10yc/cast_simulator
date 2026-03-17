use super::{Complex, ComplexSquareMatrix};

/// A unitary quantum gate: a `2ⁿ × 2ⁿ` complex matrix paired with `n` target qubit indices.
///
/// Qubits are always stored in ascending order. If the caller supplies them in a different
/// order, [`QuantumGate::new`] permutes the matrix rows/columns to match the canonical ordering.
/// Bit `k` of a basis index corresponds to the `k`-th entry of `qubits`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGate {
    matrix: ComplexSquareMatrix,
    qubits: Vec<u32>,
}

impl QuantumGate {
    /// Creates a gate from a unitary matrix and a list of target qubits.
    ///
    /// The matrix dimension must equal `2^n` where `n = qubits.len()`.
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
        Self { matrix, qubits }
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

    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Arithmetic intensity of the gate: `scalar_nnz(M) / edge_size(M)`.
    ///
    /// Real and imaginary parts of each matrix entry are tested against `ztol`
    /// independently, because the kernel generator emits separate SIMD
    /// instructions for each nonzero scalar component.  Each such component
    /// contributes exactly 2 real FLOPs (1 multiply + 1 accumulate) to one
    /// output amplitude:
    ///
    /// ```text
    /// out_re += M_re * in_re  (if M_re > ztol)
    /// out_re -= M_im * in_im  (if M_im > ztol)
    /// out_im += M_re * in_im  (if M_re > ztol)
    /// out_im += M_im * in_re  (if M_im > ztol)
    ///
    /// FLOPs = arithmetic_intensity × |ψ| × 2
    /// ```
    ///
    /// Using complex-entry nnz with a fixed ×8 factor would overcount by 2×
    /// for purely real matrices (X, CX, H, …) where `M_im = 0` throughout.
    pub fn arithmatic_intensity(&self, ztol: f64) -> f64 {
        let scalar_nnz = self
            .matrix
            .data()
            .iter()
            .map(|z| (z.re.abs() > ztol) as usize + (z.im.abs() > ztol) as usize)
            .sum::<usize>();
        scalar_nnz as f64 / self.matrix.edge_size() as f64
    }

    /// Returns the product `self * other`, expanding both gates to act on the union
    /// of their qubit sets before multiplying.
    ///
    /// The gate acts as identity on qubits not originally in its target set.
    pub fn matmul(&self, other: &Self) -> Self {
        let qubits = union_qubits(&self.qubits, &other.qubits);
        let lhs = self.expand_to(&qubits);
        let rhs = other.expand_to(&qubits);
        Self::new(lhs.matmul(&rhs), qubits)
    }

    /// Expands `self` into a larger matrix that acts on `union_qubits`.
    ///
    /// Non-target qubits are treated as identity wires: an entry is non-zero only when
    /// the non-target bits of `row` and `col` are identical. Target-qubit bits are
    /// extracted with [`compress_bits`] to index into the original gate matrix.
    fn expand_to(&self, union_qubits: &[u32]) -> ComplexSquareMatrix {
        let union_edge_size = edge_size_for_n_qubits(union_qubits.len());
        let target_positions: Vec<usize> = self
            .qubits
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
                // Skip entries where non-target bits differ (identity on those wires).
                if ((row ^ col) & !target_mask) != 0 {
                    continue;
                }

                let gate_row = compress_bits(row, &target_positions);
                let gate_col = compress_bits(col, &target_positions);
                expanded.set(row, col, self.matrix.get(gate_row, gate_col));
            }
        }

        expanded
    }
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
fn compress_bits(index: usize, positions: &[usize]) -> usize {
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

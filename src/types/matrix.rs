use std::ops::{Add, AddAssign, Mul, MulAssign};

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, StandardNormal};

use super::Complex;

/// A dense, row-major `n×n` matrix of [`Complex`] (`f64`) values.
///
/// Elements are stored in a flat `Vec` of length `n²`, indexed as `data[row * n + col]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexSquareMatrix {
    data: Vec<Complex>,
    edge_size: usize,
}

impl ComplexSquareMatrix {
    fn len_for_edge_size(edge_size: usize) -> usize {
        edge_size
            .checked_mul(edge_size)
            .expect("matrix edge_size overflow")
    }

    /// Creates an `n×n` zero matrix.
    pub fn zeros(edge_size: usize) -> Self {
        Self {
            data: vec![Complex::default(); Self::len_for_edge_size(edge_size)],
            edge_size,
        }
    }

    /// Creates an `n×n` matrix without initializing its elements.
    ///
    /// This is intended for performance-sensitive code that will write every
    /// element via raw pointers before any safe access occurs.
    ///
    /// # Safety
    /// The returned matrix must have all `edge_size * edge_size` elements
    /// written with valid [`Complex`] values before calling any safe method
    /// that reads matrix contents (for example [`Self::data`], [`Self::get`],
    /// [`Self::matmul`], `Clone`, `Debug`, or `PartialEq`).
    #[allow(clippy::uninit_vec)]
    pub unsafe fn uninit(edge_size: usize) -> Self {
        let len = Self::len_for_edge_size(edge_size);
        let mut data = Vec::with_capacity(len);
        unsafe { data.set_len(len) };
        Self { data, edge_size }
    }

    /// Creates an `n×n` identity matrix.
    pub fn eye(edge_size: usize) -> Self {
        let mut matrix = Self::zeros(edge_size);
        for i in 0..edge_size {
            matrix.set(i, i, Complex::new(1.0, 0.0));
        }
        matrix
    }

    pub fn x() -> Self {
        Self::from_reals(2, &[0.0, 1.0, 1.0, 0.0])
    }

    pub fn y() -> Self {
        Self::from_vec(
            2,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -1.0),
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 0.0),
            ],
        )
    }

    pub fn z() -> Self {
        Self::from_reals(2, &[1.0, 0.0, 0.0, -1.0])
    }

    pub fn h() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self::from_reals(2, &[s, s, s, -s])
    }

    pub fn s() -> Self {
        Self::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 1.0),
            ],
        )
    }

    pub fn t() -> Self {
        let phase = std::f64::consts::FRAC_PI_4;
        Self::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(phase.cos(), phase.sin()),
            ],
        )
    }

    pub fn rx(theta: f64) -> Self {
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        Self::from_vec(
            2,
            vec![
                Complex::new(c, 0.0),
                Complex::new(0.0, -s),
                Complex::new(0.0, -s),
                Complex::new(c, 0.0),
            ],
        )
    }

    pub fn ry(theta: f64) -> Self {
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        Self::from_reals(2, &[c, -s, s, c])
    }

    pub fn rz(theta: f64) -> Self {
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        Self::from_vec(
            2,
            vec![
                Complex::new(c, -s),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(c, s),
            ],
        )
    }

    pub fn u3(theta: f64, phi: f64, lambda: f64) -> Self {
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        let eiphi = Complex::new(phi.cos(), phi.sin());
        let eilambda = Complex::new(lambda.cos(), lambda.sin());
        let eiphilambda = Complex::new((phi + lambda).cos(), (phi + lambda).sin());
        Self::from_vec(
            2,
            vec![
                Complex::new(c, 0.0),
                -eilambda * s,
                eiphi * s,
                eiphilambda * c,
            ],
        )
    }

    pub fn cx() -> Self {
        Self::from_reals(
            4,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
            ],
        )
    }

    pub fn cz() -> Self {
        Self::from_reals(
            4,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, -1.0, //
            ],
        )
    }

    /// Controlled-phase gate: `diag(1, 1, 1, e^{iθ})`.
    pub fn cp(theta: f64) -> Self {
        let mut m = Self::eye(4);
        m.set(3, 3, Complex::new(theta.cos(), theta.sin()));
        m
    }

    pub fn swap() -> Self {
        Self::from_reals(
            4,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        )
    }

    pub fn ccx() -> Self {
        let mut out = Self::eye(8);
        out.set(6, 6, Complex::new(0.0, 0.0));
        out.set(7, 7, Complex::new(0.0, 0.0));
        out.set(6, 7, Complex::new(1.0, 0.0));
        out.set(7, 6, Complex::new(1.0, 0.0));
        out
    }

    /// Returns the single-qubit Pauli matrix for `index`: 0 = I, 1 = X, 2 = Y, 3 = Z.
    ///
    /// # Panics
    /// Panics if `index` is not in `0..=3`.
    pub fn pauli(index: usize) -> Self {
        match index {
            0 => Self::eye(2),
            1 => Self::x(),
            2 => Self::y(),
            3 => Self::z(),
            _ => panic!("Pauli index must be 0–3, got {index}"),
        }
    }

    /// Returns the Kronecker (tensor) product `self ⊗ other`.
    ///
    /// If `self` is `m×m` and `other` is `n×n`, the result is `(mn)×(mn)` with entries:
    ///
    /// ```text
    /// (self ⊗ other)[i·n + k, j·n + l] = self[i,j] · other[k,l]
    /// ```
    pub fn kron(&self, other: &Self) -> Self {
        let m = self.edge_size;
        let n = other.edge_size;
        let mn = m * n;
        // SAFETY: every element is written exactly once by the four nested loops.
        let mut out = unsafe { Self::uninit(mn) };
        let ptr = out.data_ptr_mut();
        for i in 0..m {
            for k in 0..n {
                let out_row = i * n + k;
                for j in 0..m {
                    let a = self.get(i, j);
                    for l in 0..n {
                        let out_col = j * n + l;
                        unsafe {
                            ptr.add(out_row * mn + out_col).write(a * other.get(k, l));
                        }
                    }
                }
            }
        }
        out
    }

    /// Generates a Haar-random unitary matrix of size `n×n` using the system RNG.
    ///
    /// Delegates to [`Self::random_unitary_with_rng`].
    pub fn random_unitary(edge_size: usize) -> Self {
        let mut rng = thread_rng();
        Self::random_unitary_with_rng(edge_size, &mut rng)
    }

    /// Generates a Haar-random unitary matrix of size `n×n` using the provided RNG.
    ///
    /// Uses iterative Gram-Schmidt orthonormalization on rows seeded with i.i.d.
    /// standard-normal complex entries. Retries any row that collapses to near-zero
    /// after projection (probability zero in exact arithmetic).
    pub fn random_unitary_with_rng<R: Rng + ?Sized>(edge_size: usize, rng: &mut R) -> Self {
        let mut matrix = Self::zeros(edge_size);

        for row in 0..edge_size {
            let mut current_row = vec![Complex::default(); edge_size];

            loop {
                for value in &mut current_row {
                    let re: f64 = StandardNormal.sample(rng);
                    let im: f64 = StandardNormal.sample(rng);
                    *value = Complex::new(re, im);
                }

                // Subtract projections onto all previously completed rows.
                for prev_row in 0..row {
                    let prev = matrix.row(prev_row);
                    let projection = inner_product(prev, &current_row);
                    for (value, basis) in current_row.iter_mut().zip(prev.iter()) {
                        *value -= projection * *basis;
                    }
                }

                let norm = l2_norm(&current_row);
                // rejects small-norm rows to improve numerical stability
                if norm > 1e-8 {
                    for value in &mut current_row {
                        *value /= norm;
                    }
                    break;
                }
            }

            for (col, value) in current_row.into_iter().enumerate() {
                matrix.set(row, col, value);
            }
        }

        matrix
    }

    /// Generates a random sparse matrix of size `nxn` with the desired sparsity (fraction of
    /// non-zero entries).
    ///
    /// `sparsity` is clamped to `[0.0, 1.0]`
    ///
    /// Delegates to [`Self::random_sparse_with_rng`].
    pub fn random_sparse(edge_size: usize, sparsity: f64) -> Self {
        let mut rng = thread_rng();
        Self::random_sparse_with_rng(edge_size, sparsity, &mut rng)
    }

    /// Generates a random sparse matrix of size `nxn` with the desired sparsity (fraction of
    /// non-zero entries).
    ///
    /// Simple process: each entry is i.i.d Gaussian with probability `sparsity`, otherwise zero. No
    /// orthogonalization or normalization is performed. Useful for testing purposes.
    pub fn random_sparse_with_rng<R: Rng + ?Sized>(
        edge_size: usize,
        sparsity: f64,
        rng: &mut R,
    ) -> Self {
        if sparsity < 1e-8 {
            return Self::zeros(edge_size);
        }

        let mut matrix = unsafe { Self::uninit(edge_size) };
        let ptr = matrix.data_ptr_mut();
        for i in 0..matrix.len() {
            unsafe {
                ptr.add(i).write(if rng.gen::<f64>() < sparsity {
                    Complex::new(StandardNormal.sample(rng), StandardNormal.sample(rng))
                } else {
                    Complex::ZERO
                });
            }
        }

        matrix
    }

    /// Creates a matrix from a flat, row-major `Vec` of length `edge_size²`.
    ///
    /// # Panics
    /// Panics if `data.len() != edge_size * edge_size`.
    pub fn from_vec(edge_size: usize, data: Vec<Complex>) -> Self {
        assert_eq!(
            data.len(),
            Self::len_for_edge_size(edge_size),
            "matrix data size does not match edge_size"
        );
        Self { data, edge_size }
    }

    pub fn from_reals(edge_size: usize, data: &[f64]) -> Self {
        Self::from_vec(
            edge_size,
            data.iter().map(|value| Complex::new(*value, 0.0)).collect(),
        )
    }

    /// Returns `n`, the number of rows (and columns).
    pub fn edge_size(&self) -> usize {
        self.edge_size
    }

    /// Returns the total number of elements (`n²`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the matrix has zero elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a shared slice over the flat, row-major element buffer.
    pub fn data(&self) -> &[Complex] {
        &self.data
    }

    /// Returns the raw bytes of the element buffer.
    ///
    /// Useful for content-based hashing and deduplication keys.
    pub fn as_bytes(&self) -> &[u8] {
        let ptr = self.data.as_ptr() as *const u8;
        let len = self.data.len() * std::mem::size_of::<Complex>();
        // SAFETY: Complex64 is repr(C) with two f64 fields; we view the
        // existing slice as bytes within its lifetime.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Returns a mutable slice over the flat, row-major element buffer.
    pub fn data_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }

    /// Returns a view of the row-th row as a slice.
    pub fn row(&self, row: usize) -> &[Complex] {
        assert!(row < self.edge_size, "row index out of bounds");
        let start = row * self.edge_size;
        &self.data[start..start + self.edge_size]
    }

    /// Returns a mutable view of the row-th row as a slice.
    pub fn row_mut(&mut self, row: usize) -> &mut [Complex] {
        assert!(row < self.edge_size, "row index out of bounds");
        let start = row * self.edge_size;
        &mut self.data[start..start + self.edge_size]
    }

    /// Returns a raw const pointer to the flat, row-major element buffer.
    pub fn data_ptr(&self) -> *const Complex {
        self.data.as_ptr()
    }

    /// Returns a raw mut pointer to the flat, row-major element buffer.
    pub fn data_ptr_mut(&mut self) -> *mut Complex {
        self.data.as_mut_ptr()
    }

    /// Sets every element to zero in place.
    pub fn fill_zeros(&mut self) {
        self.data.fill(Complex::default());
    }

    /// Returns the element at `(row, col)`.
    ///
    /// # Panics
    /// Panics if either index is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Complex {
        self.data[self.index(row, col)]
    }

    /// Sets the element at `(row, col)` to `value`.
    ///
    /// # Panics
    /// Panics if either index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: Complex) {
        let idx = self.index(row, col);
        self.data[idx] = value;
    }

    /// Returns `self * other` (standard matrix multiplication).
    ///
    /// # Panics
    /// Panics if the two matrices have different sizes.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.edge_size, other.edge_size, "matrix size mismatch");
        let n = self.edge_size;
        // SAFETY: every element [row * n + col] is written exactly once below.
        let mut out = unsafe { Self::uninit(n) };
        let ptr = out.data_ptr_mut();

        for row in 0..n {
            for col in 0..n {
                let mut acc = Complex::default();
                for k in 0..n {
                    acc += self.get(row, k) * other.get(k, col);
                }
                unsafe { ptr.add(row * n + col).write(acc) };
            }
        }

        out
    }

    /// Returns the conjugate-transpose (adjoint) `M†`:  `out[i,j] = conj(self[j,i])`.
    pub fn adjoint(&self) -> Self {
        let n = self.edge_size;
        // SAFETY: every element [col * n + row] is written exactly once below.
        let mut out = unsafe { Self::uninit(n) };
        let ptr = out.data_ptr_mut();
        for row in 0..n {
            for col in 0..n {
                unsafe { ptr.add(col * n + row).write(self.get(row, col).conj()) };
            }
        }
        out
    }

    /// Returns `max_{i,j} |self[i,j] - other[i,j]|` (element-wise complex norm, then max).
    ///
    /// # Panics
    /// Panics if the two matrices have different sizes.
    pub fn maximum_norm_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.edge_size, other.edge_size, "matrix size mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .fold(0.0, f64::max)
    }

    /// Computes the flat buffer index for `(row, col)`, with bounds checking.
    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.edge_size, "row index out of bounds");
        assert!(col < self.edge_size, "column index out of bounds");
        row * self.edge_size + col
    }
}

/// Computes the Hermitian inner product `⟨lhs|rhs⟩ = Σ conj(lhs[i]) * rhs[i]`.
fn inner_product(lhs: &[Complex], rhs: &[Complex]) -> Complex {
    assert_eq!(lhs.len(), rhs.len(), "vector size mismatch");
    lhs.iter()
        .zip(rhs.iter())
        .fold(Complex::default(), |acc, (lhs, rhs)| {
            acc + lhs.conj() * *rhs
        })
}

/// Computes the L2 (Euclidean) norm `sqrt(Σ |v[i]|²)`.
fn l2_norm(values: &[Complex]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

impl Add<&ComplexSquareMatrix> for &ComplexSquareMatrix {
    type Output = ComplexSquareMatrix;

    fn add(self, rhs: &ComplexSquareMatrix) -> Self::Output {
        assert_eq!(self.edge_size, rhs.edge_size, "matrix size mismatch");
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(lhs, rhs)| *lhs + *rhs)
            .collect();
        ComplexSquareMatrix::from_vec(self.edge_size, data)
    }
}

impl AddAssign<&ComplexSquareMatrix> for ComplexSquareMatrix {
    fn add_assign(&mut self, rhs: &ComplexSquareMatrix) {
        assert_eq!(self.edge_size, rhs.edge_size, "matrix size mismatch");
        for (lhs, rhs) in self.data.iter_mut().zip(rhs.data.iter()) {
            *lhs += *rhs;
        }
    }
}

impl Mul<f64> for &ComplexSquareMatrix {
    type Output = ComplexSquareMatrix;

    fn mul(self, rhs: f64) -> Self::Output {
        let data = self.data.iter().map(|value| *value * rhs).collect();
        ComplexSquareMatrix::from_vec(self.edge_size, data)
    }
}

impl Mul<Complex> for &ComplexSquareMatrix {
    type Output = ComplexSquareMatrix;

    fn mul(self, rhs: Complex) -> Self::Output {
        let data = self.data.iter().map(|value| *value * rhs).collect();
        ComplexSquareMatrix::from_vec(self.edge_size, data)
    }
}

impl MulAssign<f64> for ComplexSquareMatrix {
    fn mul_assign(&mut self, rhs: f64) {
        for value in &mut self.data {
            *value *= rhs;
        }
    }
}

impl MulAssign<Complex> for ComplexSquareMatrix {
    fn mul_assign(&mut self, rhs: Complex) {
        for value in &mut self.data {
            *value *= rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ComplexSquareMatrix;
    use crate::types::Complex;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn constructs_identity_matrix() {
        let matrix = ComplexSquareMatrix::eye(3);

        assert_eq!(matrix.edge_size(), 3);
        assert_eq!(matrix.get(0, 0), Complex::new(1.0, 0.0));
        assert_eq!(matrix.get(1, 1), Complex::new(1.0, 0.0));
        assert_eq!(matrix.get(2, 2), Complex::new(1.0, 0.0));
        assert_eq!(matrix.get(0, 1), Complex::new(0.0, 0.0));
    }

    #[test]
    fn sets_and_adds_entries() {
        let mut lhs = ComplexSquareMatrix::zeros(2);
        lhs.set(0, 0, Complex::new(1.0, 2.0));
        lhs.set(1, 1, Complex::new(3.0, -1.0));

        let mut rhs = ComplexSquareMatrix::zeros(2);
        rhs.set(0, 0, Complex::new(-1.0, 1.0));
        rhs.set(0, 1, Complex::new(4.0, 0.5));

        let sum = &lhs + &rhs;

        assert_eq!(sum.get(0, 0), Complex::new(0.0, 3.0));
        assert_eq!(sum.get(0, 1), Complex::new(4.0, 0.5));
        assert_eq!(sum.get(1, 1), Complex::new(3.0, -1.0));
    }

    #[test]
    fn multiplies_by_scalar_and_matrix() {
        let lhs = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
        );
        let rhs = ComplexSquareMatrix::eye(2);

        let scaled = &lhs * 2.0;
        let product = lhs.matmul(&rhs);

        assert_eq!(scaled.get(0, 1), Complex::new(4.0, 0.0));
        assert_eq!(scaled.get(1, 0), Complex::new(0.0, 2.0));
        assert_eq!(product, lhs);
    }

    #[test]
    fn computes_maximum_norm_difference() {
        let lhs = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        );
        let rhs = ComplexSquareMatrix::from_vec(
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 1.0),
            ],
        );

        let diff = lhs.maximum_norm_diff(&rhs);

        assert!((diff - 2_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn random_unitary_is_approximately_unitary() {
        let matrix = ComplexSquareMatrix::random_unitary(4);
        let matrix_dag = matrix.adjoint();
        let gram = matrix.matmul(&matrix_dag);
        let identity = ComplexSquareMatrix::eye(4);

        assert!(gram.maximum_norm_diff(&identity) < 1e-10);
    }

    #[test]
    fn random_unitary_with_rng_is_reproducible() {
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);

        let a = ComplexSquareMatrix::random_unitary_with_rng(3, &mut rng_a);
        let b = ComplexSquareMatrix::random_unitary_with_rng(3, &mut rng_b);

        assert_eq!(a, b);
    }

    #[test]
    fn unsafe_uninit_can_be_initialized_via_raw_pointer() {
        let mut matrix = unsafe { ComplexSquareMatrix::uninit(2) };
        let ptr = matrix.data_ptr_mut();

        unsafe {
            ptr.add(0).write(Complex::new(1.0, 0.0));
            ptr.add(1).write(Complex::new(2.0, 0.0));
            ptr.add(2).write(Complex::new(3.0, 0.0));
            ptr.add(3).write(Complex::new(4.0, 0.0));
        }

        assert_eq!(matrix.get(0, 0), Complex::new(1.0, 0.0));
        assert_eq!(matrix.get(0, 1), Complex::new(2.0, 0.0));
        assert_eq!(matrix.get(1, 0), Complex::new(3.0, 0.0));
        assert_eq!(matrix.get(1, 1), Complex::new(4.0, 0.0));
    }
}

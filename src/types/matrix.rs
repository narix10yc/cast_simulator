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
    /// Creates an `n×n` zero matrix.
    pub fn zeros(edge_size: usize) -> Self {
        Self {
            data: vec![Complex::default(); edge_size * edge_size],
            edge_size,
        }
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
                if norm > 1e-12 {
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

    /// Creates a matrix from a flat, row-major `Vec` of length `edge_size²`.
    ///
    /// # Panics
    /// Panics if `data.len() != edge_size * edge_size`.
    pub fn from_vec(edge_size: usize, data: Vec<Complex>) -> Self {
        assert_eq!(
            data.len(),
            edge_size * edge_size,
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

    /// Returns a shared slice over the flat, row-major element buffer.
    pub fn data(&self) -> &[Complex] {
        &self.data
    }

    /// Returns a mutable slice over the flat, row-major element buffer.
    pub fn data_mut(&mut self) -> &mut [Complex] {
        &mut self.data
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
        let mut out = Self::zeros(n);

        for row in 0..n {
            for col in 0..n {
                let mut acc = Complex::default();
                for k in 0..n {
                    acc += self.get(row, k) * other.get(k, col);
                }
                out.set(row, col, acc);
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

    /// Returns a shared slice for the given row.
    fn row(&self, row: usize) -> &[Complex] {
        assert!(row < self.edge_size, "row index out of bounds");
        let start = row * self.edge_size;
        &self.data[start..start + self.edge_size]
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

    fn conjugate_transpose(matrix: &ComplexSquareMatrix) -> ComplexSquareMatrix {
        let n = matrix.edge_size();
        let mut out = ComplexSquareMatrix::zeros(n);
        for row in 0..n {
            for col in 0..n {
                out.set(col, row, matrix.get(row, col).conj());
            }
        }
        out
    }

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
        let matrix_dag = conjugate_transpose(&matrix);
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
}

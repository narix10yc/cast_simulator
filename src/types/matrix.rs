use super::Complex;
use std::ops::{Index, IndexMut};

// A generic complex square matrix.
pub struct Matrix {
    pub a: Vec<Vec<Complex>>,
    pub edgesize: usize,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixN<const N: usize> {
    pub a: [[Complex; N]; N],
}

impl<const N: usize> Index<usize> for MatrixN<N> {
    type Output = [Complex; N];

    fn index(&self, index: usize) -> &Self::Output {
        &self.a[index]
    }
}

impl<const N: usize> IndexMut<usize> for MatrixN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.a[index]
    }
}

pub type Matrix2 = MatrixN<2>;
pub type Matrix4 = MatrixN<4>;

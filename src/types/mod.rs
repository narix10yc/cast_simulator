/// All complex numbers are default to take 128 bits (64 bits each for re and im parts).
pub use num_complex::Complex64 as Complex;

mod rational;
pub use rational::Rational;

mod matrix;
pub use matrix::{Matrix, Matrix2, Matrix4};

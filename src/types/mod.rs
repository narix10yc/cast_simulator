//! Core types for quantum circuit simulation.
//!
//! - [`Complex`] — 128-bit complex number (64-bit real + 64-bit imaginary).
//! - [`Rational`] — exact rational arithmetic over `i32`, always stored in reduced form.
//! - [`ComplexSquareMatrix`] — dense, row-major complex square matrix.
//! - [`QuantumGate`] — a unitary matrix paired with its target qubit indices.

/// 128-bit complex number (`f64` real and imaginary parts) re-exported from `num_complex`.
pub use num_complex::Complex64 as Complex;

mod rational;
pub use rational::Rational;

mod matrix;
pub use matrix::ComplexSquareMatrix;

mod gate;
pub use gate::QuantumGate;

mod precision;
pub use precision::Precision;

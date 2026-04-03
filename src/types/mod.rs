//! Core types for quantum circuit simulation.
//!
//! - [`Complex`] — 128-bit complex number (64-bit real + 64-bit imaginary).
//! - [`Rational`] — exact rational arithmetic over `i32`, always stored in reduced form.
//! - [`ComplexSquareMatrix`] — dense, row-major complex square matrix.
//! - [`QuantumGate`] — a unitary matrix paired with its target qubit indices,
//!   optionally carrying a noise model as probability-weighted unitary branches.

/// 128-bit complex number (`f64` real and imaginary parts) re-exported from `num_complex`.
pub use num_complex::Complex64 as Complex;
pub use num_complex::ComplexFloat;

mod rational;
pub use rational::Rational;

mod matrix;
pub use matrix::ComplexSquareMatrix;

mod noise;
pub use noise::NoiseModel;

mod gate;
pub(crate) use gate::compress_bits;
pub use gate::QuantumGate;

mod circuit;
pub use circuit::QuantumCircuit;

mod precision;
pub use precision::Precision;

//! Core types for quantum circuit simulation.
//!
//! - [`Complex`] — 128-bit complex number (re/im `f64`).
//! - [`Rational`] — exact rational arithmetic over `i32`, auto-reduced.
//! - [`ComplexSquareMatrix`] — dense, row-major complex square matrix.
//! - [`NoiseModel`] — probability-weighted Kraus operators for noisy channels.
//! - [`QuantumGate`] — unitary matrix + target qubits, optionally with a [`NoiseModel`].
//! - [`QuantumCircuit`] — ordered gate sequence with qubit count and measurement spec.

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

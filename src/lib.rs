pub mod types;

pub mod circuit;

pub mod fusion;

pub mod openqasm;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use circuit::{CircuitGraph, CircuitRow, GateId};

pub fn project_layout() -> &'static str {
    "rust-root-cpp-bindings"
}

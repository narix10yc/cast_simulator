/// OpenQASM parser and circuit representation.
/// Usage:
/// ```
/// use cast::openqasm::{Circuit, Gate, parse_qasm};
/// let qasm = "OPENQASM 2.0; h q[0]; x q[3];";
/// let circuit = parse_qasm(qasm).unwrap();
/// assert_eq!(circuit.gates.len(), 2);
/// assert!(matches!(circuit.gates[0], Gate::H(0)));
/// assert!(matches!(circuit.gates[1], Gate::X(3)));
/// ```
mod parser;
pub use parser::parse_qasm;

mod circuit;
pub use circuit::{Angle, Circuit, Gate};

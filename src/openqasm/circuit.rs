use crate::types::Rational;

pub enum Angle {
    RationalPi(Rational),
    Number(f64),
}

pub enum Gate {
    // Single-qubit non-parametrized gates
    X(u32),
    Y(u32),
    Z(u32),
    H(u32),
    S(u32),
    T(u32),

    // Single-qubit parametrized gates
    /// angle, q
    RX(Angle, u32),
    /// angle, q
    RY(Angle, u32),
    /// angle, q
    RZ(Angle, u32),
    /// theta, phi, lambda, q
    U3(Angle, Angle, Angle, u32),

    // Two-qubit gates
    /// ctrl, targ
    CX(u32, u32),
    /// ctrl, targ
    CZ(u32, u32),
    /// q0, q1
    SWAP(u32, u32),

    // Other gates
    /// ctrl1, ctrl2, targ
    CCX(u32, u32, u32),
    // Generic {name: String, params: Vec<Angle>, qubits: Vec<u32>},
}

pub struct Circuit {
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new() -> Self {
        Self { gates: Vec::new() }
    }

    pub fn serialize(&self) -> String {
        let mut out = String::new();
        self.write_qasm_to(&mut out)
            .expect("writing QASM to String should be infallible");
        out
    }

    pub fn write_qasm_to<W: std::fmt::Write>(&self, out: &mut W) -> std::fmt::Result {
        writeln!(out, "OPENQASM 2.0;")?;
        writeln!(out, "qreg q[{}];", self.required_qreg_size())?;
        for gate in &self.gates {
            writeln!(out, "{}", serialize_gate(gate))?;
        }
        Ok(())
    }

    pub fn write_qasm_to_io<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "OPENQASM 2.0;")?;
        writeln!(out, "qreg q[{}];", self.required_qreg_size())?;
        for gate in &self.gates {
            writeln!(out, "{}", serialize_gate(gate))?;
        }
        Ok(())
    }

    fn required_qreg_size(&self) -> u32 {
        let mut max_index: Option<u32> = None;
        for gate in &self.gates {
            let gate_max = max_qubit_in_gate(gate);
            max_index = Some(match (max_index, gate_max) {
                (Some(a), Some(b)) => a.max(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => 0,
            });
        }
        max_index.map_or(0, |idx| idx.saturating_add(1))
    }
}

fn max_qubit_in_gate(gate: &Gate) -> Option<u32> {
    match gate {
        Gate::X(q) | Gate::Y(q) | Gate::Z(q) | Gate::H(q) | Gate::S(q) | Gate::T(q) => Some(*q),
        Gate::RX(_, q) | Gate::RY(_, q) | Gate::RZ(_, q) | Gate::U3(_, _, _, q) => Some(*q),
        Gate::CX(ctrl, targ) | Gate::CZ(ctrl, targ) => Some((*ctrl).max(*targ)),
        Gate::SWAP(q0, q1) => Some((*q0).max(*q1)),
        Gate::CCX(ctrl1, ctrl2, targ) => Some((*ctrl1).max(*ctrl2).max(*targ)),
    }
}

fn qubit_ref(idx: u32) -> String {
    format!("q[{idx}]")
}

fn serialize_gate(gate: &Gate) -> String {
    match gate {
        Gate::X(q) => format!("x {};", qubit_ref(*q)),
        Gate::Y(q) => format!("y {};", qubit_ref(*q)),
        Gate::Z(q) => format!("z {};", qubit_ref(*q)),
        Gate::H(q) => format!("h {};", qubit_ref(*q)),
        Gate::S(q) => format!("s {};", qubit_ref(*q)),
        Gate::T(q) => format!("t {};", qubit_ref(*q)),
        Gate::RX(angle, q) => format!("rx({}) {};", serialize_angle(angle), qubit_ref(*q)),
        Gate::RY(angle, q) => format!("ry({}) {};", serialize_angle(angle), qubit_ref(*q)),
        Gate::RZ(angle, q) => format!("rz({}) {};", serialize_angle(angle), qubit_ref(*q)),
        Gate::U3(theta, phi, lambda, q) => format!(
            "u3({},{},{}) {};",
            serialize_angle(theta),
            serialize_angle(phi),
            serialize_angle(lambda),
            qubit_ref(*q)
        ),
        Gate::CX(ctrl, targ) => format!("cx {},{};", qubit_ref(*ctrl), qubit_ref(*targ)),
        Gate::CZ(ctrl, targ) => format!("cz {},{};", qubit_ref(*ctrl), qubit_ref(*targ)),
        Gate::SWAP(q0, q1) => format!("swap {},{};", qubit_ref(*q0), qubit_ref(*q1)),
        Gate::CCX(ctrl1, ctrl2, targ) => format!(
            "ccx {},{},{};",
            qubit_ref(*ctrl1),
            qubit_ref(*ctrl2),
            qubit_ref(*targ)
        ),
    }
}

fn serialize_angle(angle: &Angle) -> String {
    match angle {
        Angle::RationalPi(r) => {
            let n = r.numerator;
            let d = r.denominator;
            if n == 0 {
                return "0".to_string();
            }

            if d == 1 {
                return match n {
                    1 => "pi".to_string(),
                    -1 => "-pi".to_string(),
                    _ => format!("{n}*pi"),
                };
            }

            let abs_n = n.abs();
            let magnitude = if abs_n == 1 {
                format!("pi/{d}")
            } else {
                format!("{abs_n}*pi/{d}")
            };
            if n < 0 {
                format!("-{magnitude}")
            } else {
                magnitude
            }
        }
        Angle::Number(v) => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openqasm::parser::parse_qasm;

    #[test]
    fn serializes_with_header_and_inferred_qreg_size() {
        let circuit = Circuit {
            gates: vec![Gate::X(3), Gate::H(0)],
        };
        let qasm = circuit.serialize();
        assert!(qasm.starts_with("OPENQASM 2.0;\nqreg q[4];\n"));
        assert!(qasm.contains("x q[3];"));
        assert!(qasm.contains("h q[0];"));
    }

    #[test]
    fn serializes_and_round_trips_through_parser() {
        let circuit = Circuit {
            gates: vec![
                Gate::RX(Angle::RationalPi(Rational::new(2, 3)), 0),
                Gate::CX(0, 1),
                Gate::U3(
                    Angle::RationalPi(Rational::new(1, 2)),
                    Angle::Number(0.25),
                    Angle::Number(1e-6),
                    1,
                ),
            ],
        };

        let serialized = circuit.serialize();
        let parsed = parse_qasm(&serialized).expect("round-trip parse should succeed");
        assert_eq!(parsed.gates.len(), 3);
        match &parsed.gates[0] {
            Gate::RX(_, q) => assert_eq!(*q, 0),
            _ => panic!("expected RX gate"),
        }
        match &parsed.gates[1] {
            Gate::CX(ctrl, targ) => {
                assert_eq!(*ctrl, 0);
                assert_eq!(*targ, 1);
            }
            _ => panic!("expected CX gate"),
        }
        match &parsed.gates[2] {
            Gate::U3(_, _, _, q) => assert_eq!(*q, 1),
            _ => panic!("expected U3 gate"),
        }
    }

    #[test]
    fn writes_qasm_to_fmt_stream() {
        let circuit = Circuit {
            gates: vec![Gate::X(0), Gate::CX(0, 1)],
        };
        let mut out = String::new();
        circuit
            .write_qasm_to(&mut out)
            .expect("writing to fmt stream should succeed");
        assert!(out.starts_with("OPENQASM 2.0;\nqreg q[2];\n"));
        assert!(out.contains("x q[0];\n"));
        assert!(out.contains("cx q[0],q[1];\n"));
    }

    #[test]
    fn writes_qasm_to_io_stream() {
        let circuit = Circuit {
            gates: vec![Gate::RZ(Angle::Number(1e-6), 2)],
        };
        let mut out = Vec::new();
        circuit
            .write_qasm_to_io(&mut out)
            .expect("writing to io stream should succeed");
        let text = String::from_utf8(out).expect("serialized qasm should be utf-8");
        assert!(text.starts_with("OPENQASM 2.0;\nqreg q[3];\n"));
        assert!(text.contains("rz(0.000001) q[2];\n"));
    }
}

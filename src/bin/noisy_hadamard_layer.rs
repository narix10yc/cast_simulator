//! Example: layers of noisy Hadamard gates on 10 qubits.
//!
//! Usage: noisy_hadamard_layer [layers] [max_ensemble]
//!
//! Each layer applies H + depolarizing(p=0.01) to every qubit.
//! Prints explored weight vs. layer count.

use anyhow::Result;
use cast::simulator::{Cpu, SimulationMode, Simulator};
use cast::types::{QuantumCircuit, QuantumGate};

fn run_layers(n_qubits: u32, layers: u32, noise_p: f64, max_ensemble: usize) -> Result<f64> {
    let mut circuit = QuantumCircuit::new(n_qubits);
    for _ in 0..layers {
        for q in 0..n_qubits {
            circuit.add(QuantumGate::h(q));
            circuit.add(QuantumGate::depolarizing(q, noise_p));
        }
    }
    let measured_qubits: Vec<u32> = (0..n_qubits).collect();
    circuit.measure(&measured_qubits);

    let sim = Simulator::<Cpu>::f32().with_mode(SimulationMode::Trajectory {
        n_samples: 1000,
        seed: Some(42),
        max_ensemble: Some(max_ensemble),
    });

    let result = sim.run(&circuit)?;
    Ok(result.trajectory_data.unwrap().explored_weight)
}

fn main() -> Result<()> {
    let max_ensemble: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let n_qubits: u32 = 10;
    let noise_p = 0.01;

    // If a single layer count is given, run just that.
    if let Some(layers) = std::env::args().nth(1).and_then(|s| s.parse::<u32>().ok()) {
        let weight = run_layers(n_qubits, layers, noise_p, max_ensemble)?;
        eprintln!(
            "layers={layers:3}  noisy_gates={:4}  M={max_ensemble:3}  explored_weight={weight:.4}",
            layers * n_qubits
        );
        return Ok(());
    }

    // Otherwise, sweep across layer counts.
    eprintln!("{n_qubits} qubits, depolarizing p={noise_p}, max_ensemble={max_ensemble}\n");
    eprintln!("layers  noisy_gates  explored_weight");
    eprintln!("------  -----------  ---------------");

    for layers in [1, 2, 5, 10, 20, 50, 100] {
        let weight = run_layers(n_qubits, layers, noise_p, max_ensemble)?;
        eprintln!("{layers:6}  {:11}  {weight:.4}", layers * n_qubits);
    }

    Ok(())
}

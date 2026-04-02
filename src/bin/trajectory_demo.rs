//! Quick trajectory simulation experiment for profiling/review.

use std::time::Instant;

use anyhow::Result;
use cast::simulator::{Cpu, SimulationMode, Simulator};
use cast::types::{QuantumCircuit, QuantumGate};

fn build_noisy_qft(n: u32, noise_p: f64) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n);

    // QFT gates: H + controlled-phase + bit-reversal SWAPs
    for q in 0..n {
        circuit.add(QuantumGate::h(q));
        circuit.add(QuantumGate::depolarizing(q, noise_p));
        for k in 1..(n - q) {
            let theta = std::f64::consts::PI / (1u64 << k) as f64;
            circuit.add(QuantumGate::cp(theta, q, q + k));
            circuit.add(QuantumGate::depolarizing(q, noise_p));
            circuit.add(QuantumGate::depolarizing(q + k, noise_p));
        }
    }
    for q in 0..(n / 2) {
        circuit.add(QuantumGate::swap(q, n - 1 - q));
        circuit.add(QuantumGate::depolarizing(q, noise_p));
        circuit.add(QuantumGate::depolarizing(n - 1 - q, noise_p));
    }

    circuit
}

fn main() -> Result<()> {
    let n_qubits: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let n_samples: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);
    let max_ensemble: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let noise_p = 0.005;

    let mut circuit = build_noisy_qft(n_qubits, noise_p);
    let measured_qubits: Vec<u32> = (0..n_qubits).collect();
    circuit.measure(&measured_qubits);

    let n_noisy = circuit.gates().iter().filter(|g| !g.is_unitary()).count();
    let n_unitary = circuit.len() - n_noisy;

    eprintln!("Circuit: {n_qubits}-qubit noisy QFT");
    eprintln!(
        "  Gates: {} total ({} unitary, {} noisy)",
        circuit.len(),
        n_unitary,
        n_noisy
    );
    eprintln!("  Samples: {n_samples}");
    eprintln!("  Max ensemble: {max_ensemble}");
    eprintln!("  Measured qubits: all ({n_qubits})");
    eprintln!("  Precision: F32");
    eprintln!();

    let t_total = Instant::now();

    let sim = Simulator::<Cpu>::f32().with_mode(SimulationMode::Trajectory {
        n_samples,
        seed: Some(42),
        max_ensemble: Some(max_ensemble),
    });

    let result = sim.run(&circuit)?;

    let total_s = t_total.elapsed().as_secs_f64();

    eprintln!("Results:");
    eprintln!("  Compile time:     {:.3} s", result.compile_time_s);

    if let Some(ref traj) = result.trajectory_data {
        eprintln!("  Exec time:        {}", result.timing);
        eprintln!("  Explored weight:  {:.6}", traj.explored_weight);
        eprintln!("  Branches:         {}", traj.branches.len());
        eprintln!("  Histogram entries: {}", traj.histogram.len());
        eprintln!("  Total samples:    {}", traj.n_samples);

        for (i, branch) in traj.branches.iter().enumerate() {
            let n_identity = branch.noise_path.iter().filter(|&&op| op == 0).count();
            let n_total = branch.noise_path.len();
            eprintln!(
                "  Branch {i}: weight={:.6}  identity={n_identity}/{n_total}  samples={}",
                branch.weight, branch.n_samples
            );
        }
    }

    eprintln!("  Total wall time:  {:.3} s", total_s);

    Ok(())
}

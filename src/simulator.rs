//! Generic simulator that unifies statevector, density-matrix, and trajectory
//! simulation modes across CPU and CUDA backends.
//!
//! # Usage
//!
//! ```no_run
//! use cast::simulator::{Simulator, SimulationMode};
//! use cast::cpu::CPUKernelGenSpec;
//! use cast::openqasm::parse_qasm;
//!
//! let sim = Simulator::cpu(CPUKernelGenSpec::f64())
//!     .with_mode(SimulationMode::StateVector);
//!
//! let circuit = parse_qasm("qreg q[4]; h q[0]; cx q[0],q[1];").unwrap();
//! let result = sim.run(&circuit).unwrap();
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::cost_model::FusionConfig;
use crate::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};
use crate::fusion;
use crate::timing::TimingStats;
use crate::types::QuantumGate;
use crate::CircuitGraph;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaKernelGenSpec, CudaKernelManager, CudaStatevector};

// ── Public types ─────────────────────────────────────────────────────────────

/// Simulation mode.
#[derive(Clone, Debug)]
pub enum SimulationMode {
    /// Pure-state simulation on n qubits. Errors if the circuit contains
    /// noise channels.
    StateVector,
    /// Density-matrix simulation on 2n virtual qubits. Supports noise channels
    /// via superoperator lifting.
    DensityMatrix,
    /// Monte Carlo wavefunction: pure-state simulation with stochastic noise
    /// branch sampling at each noisy gate. Memory cost is O(2^n) per trajectory.
    Trajectory {
        n_trajectories: usize,
        seed: Option<u64>,
    },
}

/// Result of a simulation run.
pub struct SimulationResult {
    /// Execution time statistics.
    pub timing: TimingStats,
    /// Kernel compilation wall time (seconds).
    pub compile_time_s: f64,
    /// Per-trajectory data (only for [`SimulationMode::Trajectory`]).
    pub trajectory_data: Option<Vec<TrajectoryResult>>,
}

/// Per-trajectory outcome data.
pub struct TrajectoryResult {
    /// Index of the noise branch sampled at each noisy gate, in circuit
    /// gate order. Only noisy gates contribute entries.
    pub sampled_operators: Vec<usize>,
    /// Wall time for this trajectory (seconds).
    pub time_s: f64,
}

/// Backend-agnostic kernel ID.
#[derive(Clone, Copy)]
enum KernelId {
    Cpu(crate::cpu::KernelId),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaKernelId),
}

impl KernelId {
    fn as_cpu(self) -> crate::cpu::KernelId {
        match self {
            KernelId::Cpu(id) => id,
            #[cfg(feature = "cuda")]
            _ => panic!("expected CPU kernel ID"),
        }
    }

    #[cfg(feature = "cuda")]
    fn as_cuda(self) -> crate::cuda::CudaKernelId {
        match self {
            KernelId::Cuda(id) => id,
            _ => panic!("expected CUDA kernel ID"),
        }
    }
}

/// Internal backend state. Created lazily during `run()`.
enum Backend {
    Cpu {
        mgr: CpuKernelManager,
        spec: CPUKernelGenSpec,
        n_threads: u32,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        mgr: CudaKernelManager,
        spec: CudaKernelGenSpec,
    },
}

// ── Simulator ────────────────────────────────────────────────────────────────

/// Unified quantum circuit simulator.
pub struct Simulator {
    mode: SimulationMode,
    fusion_config: Option<FusionConfig>,
    backend: Backend,
}

impl Simulator {
    /// Create a simulator using the CPU backend.
    pub fn cpu(spec: CPUKernelGenSpec) -> Self {
        Self {
            mode: SimulationMode::StateVector,
            fusion_config: None,
            backend: Backend::Cpu {
                mgr: CpuKernelManager::new(),
                spec,
                n_threads: get_num_threads(),
            },
        }
    }

    /// Create a simulator using the CUDA backend.
    #[cfg(feature = "cuda")]
    pub fn cuda(spec: CudaKernelGenSpec) -> Self {
        Self {
            mode: SimulationMode::StateVector,
            fusion_config: None,
            backend: Backend::Cuda {
                mgr: CudaKernelManager::new(),
                spec,
            },
        }
    }

    /// Set the simulation mode.
    pub fn with_mode(mut self, mode: SimulationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the fusion configuration. If not set, no fusion is applied.
    pub fn with_fusion(mut self, config: FusionConfig) -> Self {
        self.fusion_config = Some(config);
        self
    }

    /// Run the simulation on an OpenQASM circuit.
    ///
    /// Internally builds a [`CircuitGraph`], applies fusion (if configured),
    /// compiles kernels, and executes. Use [`run_graph`] if you already have
    /// a `CircuitGraph`.
    pub fn run(&self, circuit: &crate::openqasm::Circuit) -> Result<SimulationResult> {
        let graph = CircuitGraph::from_qasm_circuit(circuit);
        self.run_graph(&graph)
    }

    /// Run the simulation on a pre-built circuit graph.
    pub fn run_graph(&self, circuit: &CircuitGraph) -> Result<SimulationResult> {
        let n_physical = circuit.n_qubits() as u32;

        // 1. Transform circuit for the simulation mode.
        let (gates, n_sv) = self.prepare_gates(circuit, n_physical)?;

        // 2. Compile kernels for all gates.
        let skip_channels = matches!(self.mode, SimulationMode::Trajectory { .. });
        let t0 = Instant::now();
        let kernel_ids = self.compile_gates(&gates, skip_channels)?;
        let compile_time_s = t0.elapsed().as_secs_f64();

        // 3. Execute based on mode.
        match &self.mode {
            SimulationMode::StateVector | SimulationMode::DensityMatrix => {
                let ids: Vec<KernelId> =
                    kernel_ids.iter().map(|k| k.unwrap()).collect();
                let timing = self.execute_standard(&ids, n_sv)?;
                Ok(SimulationResult {
                    timing,
                    compile_time_s,
                    trajectory_data: None,
                })
            }
            SimulationMode::Trajectory {
                n_trajectories,
                seed,
            } => self.execute_trajectory(
                &gates,
                &kernel_ids,
                n_sv,
                *n_trajectories,
                *seed,
                compile_time_s,
            ),
        }
    }

    // ── Circuit preparation ──────────────────────────────────────────────

    /// Transform the circuit for the current simulation mode and apply fusion.
    /// Returns the ordered gate list and the statevector qubit count.
    fn prepare_gates(
        &self,
        circuit: &CircuitGraph,
        n_physical: u32,
    ) -> Result<(Vec<Arc<QuantumGate>>, u32)> {
        let mut graph = circuit.clone();

        // Apply fusion if configured.
        if let Some(ref config) = self.fusion_config {
            fusion::optimize(&mut graph, config);
        }

        let gates = graph.gates_in_row_order();

        match &self.mode {
            SimulationMode::StateVector => {
                // Validate: no channel gates allowed.
                for gate in &gates {
                    if !gate.is_unitary() {
                        anyhow::bail!(
                            "StateVector mode does not support noise channels; \
                             use DensityMatrix or Trajectory mode"
                        );
                    }
                }
                Ok((gates, n_physical))
            }
            SimulationMode::DensityMatrix => {
                // Lift all gates to superoperators on 2n virtual qubits.
                let dm_gates: Vec<Arc<QuantumGate>> = gates
                    .iter()
                    .map(|g| Arc::new(g.to_density_matrix_gate(n_physical as usize)))
                    .collect();
                Ok((dm_gates, 2 * n_physical))
            }
            SimulationMode::Trajectory { .. } => {
                // Gates used as-is; channels handled during execution.
                Ok((gates, n_physical))
            }
        }
    }

    // ── Kernel compilation ───────────────────────────────────────────────

    /// Compile kernels for all gates. For Trajectory mode, channel gates are
    /// skipped (their noise-branch kernels are compiled separately).
    fn compile_gates(
        &self,
        gates: &[Arc<QuantumGate>],
        skip_channels: bool,
    ) -> Result<Vec<Option<KernelId>>> {
        let mut ids = Vec::with_capacity(gates.len());
        for gate in gates {
            if skip_channels && !gate.is_unitary() {
                ids.push(None);
            } else {
                ids.push(Some(self.compile_one(gate)?));
            }
        }
        Ok(ids)
    }

    fn compile_one(&self, gate: &Arc<QuantumGate>) -> Result<KernelId> {
        match &self.backend {
            Backend::Cpu { mgr, spec, .. } => {
                let kid = mgr.generate(spec, gate)?;
                Ok(KernelId::Cpu(kid))
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda { mgr, spec } => {
                let kid = mgr.generate(gate, *spec)?;
                Ok(KernelId::Cuda(kid))
            }
        }
    }

    // ── Standard execution (StateVector / DensityMatrix) ─────────────────

    fn execute_standard(&self, kernel_ids: &[KernelId], n_sv: u32) -> Result<TimingStats> {
        match &self.backend {
            Backend::Cpu {
                mgr, n_threads, spec, ..
            } => {
                let mut sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
                let timing = crate::timing::time_adaptive(
                    || -> Result<()> {
                        sv.initialize();
                        for kid in kernel_ids {
                            mgr.apply(kid.as_cpu(), &mut sv, *n_threads)?;
                        }
                        Ok(())
                    },
                    5.0,
                )?;
                Ok(timing)
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda { mgr, spec } => {
                let mut sv = CudaStatevector::new(n_sv, spec.precision)?;
                let timing = crate::timing::time_adaptive_with(
                    || -> Result<std::time::Duration> {
                        sv.zero()?;
                        for kid in kernel_ids {
                            mgr.apply(kid.as_cuda(), &mut sv)?;
                        }
                        let stats = mgr.sync()?;
                        Ok(stats.kernels.iter().map(|k| k.gpu_time).sum())
                    },
                    5.0,
                )?;
                Ok(timing)
            }
        }
    }

    // ── Trajectory execution ─────────────────────────────────────────────

    fn execute_trajectory(
        &self,
        gates: &[Arc<QuantumGate>],
        gate_kernel_ids: &[Option<KernelId>],
        n_sv: u32,
        n_trajectories: usize,
        seed: Option<u64>,
        compile_time_s: f64,
    ) -> Result<SimulationResult> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Pre-compile noise-branch kernels and sampling distributions.
        let mut noise_kernels: Vec<Option<Vec<KernelId>>> = Vec::with_capacity(gates.len());
        let mut noise_dists: Vec<Option<WeightedIndex<f64>>> = Vec::with_capacity(gates.len());
        for gate in gates {
            if !gate.is_unitary() {
                let mut kids = Vec::new();
                for (_, u_noise) in gate.noise() {
                    let composed = u_noise.matmul(gate.matrix());
                    let noise_gate =
                        Arc::new(QuantumGate::new(composed, gate.qubits().to_vec()));
                    kids.push(self.compile_one(&noise_gate)?);
                }
                noise_kernels.push(Some(kids));

                let weights: Vec<f64> = gate.noise().iter().map(|(p, _)| *p).collect();
                let dist = WeightedIndex::new(&weights)
                    .context("invalid noise probabilities")?;
                noise_dists.push(Some(dist));
            } else {
                noise_kernels.push(None);
                noise_dists.push(None);
            }
        }

        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(StdRng::from_entropy()),
        };

        let mut trajectory_results = Vec::with_capacity(n_trajectories);

        for _ in 0..n_trajectories {
            let t0 = Instant::now();
            let sampled = self.run_one_trajectory(
                gates,
                gate_kernel_ids,
                &noise_kernels,
                &noise_dists,
                n_sv,
                &mut *rng,
            )?;
            let time_s = t0.elapsed().as_secs_f64();
            trajectory_results.push(TrajectoryResult {
                sampled_operators: sampled,
                time_s,
            });
        }

        let samples: Vec<f64> = trajectory_results.iter().map(|t| t.time_s).collect();
        let timing = crate::timing::stats_from_samples(&samples);

        Ok(SimulationResult {
            timing,
            compile_time_s,
            trajectory_data: Some(trajectory_results),
        })
    }

    fn run_one_trajectory(
        &self,
        gates: &[Arc<QuantumGate>],
        gate_kernel_ids: &[Option<KernelId>],
        noise_kernels: &[Option<Vec<KernelId>>],
        noise_dists: &[Option<rand::distributions::WeightedIndex<f64>>],
        n_sv: u32,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<usize>> {
        match &self.backend {
            Backend::Cpu {
                mgr, spec, n_threads, ..
            } => self.trajectory_cpu(
                mgr,
                spec,
                *n_threads,
                gates,
                gate_kernel_ids,
                noise_kernels,
                noise_dists,
                n_sv,
                rng,
            ),
            #[cfg(feature = "cuda")]
            Backend::Cuda { mgr, spec } => self.trajectory_cuda(
                mgr,
                spec,
                gates,
                gate_kernel_ids,
                noise_kernels,
                noise_dists,
                n_sv,
                rng,
            ),
        }
    }

    fn trajectory_cpu(
        &self,
        mgr: &CpuKernelManager,
        spec: &CPUKernelGenSpec,
        n_threads: u32,
        gates: &[Arc<QuantumGate>],
        gate_kernel_ids: &[Option<KernelId>],
        noise_kernels: &[Option<Vec<KernelId>>],
        noise_dists: &[Option<rand::distributions::WeightedIndex<f64>>],
        n_sv: u32,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<usize>> {
        use rand::distributions::Distribution;

        let mut sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
        sv.initialize();

        let mut sampled_ops = Vec::new();

        for (i, gate) in gates.iter().enumerate() {
            if gate.is_unitary() {
                mgr.apply(gate_kernel_ids[i].unwrap().as_cpu(), &mut sv, n_threads)?;
            } else {
                let chosen = noise_dists[i].as_ref().unwrap().sample(rng);
                sampled_ops.push(chosen);
                mgr.apply(noise_kernels[i].as_ref().unwrap()[chosen].as_cpu(), &mut sv, n_threads)?;
            }
        }

        Ok(sampled_ops)
    }

    #[cfg(feature = "cuda")]
    fn trajectory_cuda(
        &self,
        mgr: &CudaKernelManager,
        spec: &CudaKernelGenSpec,
        gates: &[Arc<QuantumGate>],
        gate_kernel_ids: &[Option<KernelId>],
        noise_kernels: &[Option<Vec<KernelId>>],
        noise_dists: &[Option<rand::distributions::WeightedIndex<f64>>],
        n_sv: u32,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<usize>> {
        use rand::distributions::Distribution;

        let mut sv = CudaStatevector::new(n_sv, spec.precision)?;
        sv.zero()?;

        let mut sampled_ops = Vec::new();

        for (i, gate) in gates.iter().enumerate() {
            if gate.is_unitary() {
                mgr.apply(gate_kernel_ids[i].unwrap().as_cuda(), &mut sv)?;
            } else {
                mgr.sync()?;
                let chosen = noise_dists[i].as_ref().unwrap().sample(rng);
                sampled_ops.push(chosen);
                mgr.apply(noise_kernels[i].as_ref().unwrap()[chosen].as_cuda(), &mut sv)?;
            }
        }

        mgr.sync()?;
        Ok(sampled_ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantumGate;

    // Minimum SV qubits for F64+W256 SIMD (simd_s=2): n_gate_qubits + 2.
    // Use 4 qubits to safely handle 1- and 2-qubit gates.
    const TEST_QUBITS: u32 = 4;

    fn test_graph_with_padding(gates: Vec<QuantumGate>) -> CircuitGraph {
        let mut graph = CircuitGraph::new();
        // Insert a dummy gate on a high qubit to force the graph width.
        graph.insert_gate(QuantumGate::rz(0.0, TEST_QUBITS - 1));
        for g in gates {
            graph.insert_gate(g);
        }
        graph
    }

    /// StateVector mode runs without error.
    #[test]
    fn statevector_h_gate() {
        let sim = Simulator::cpu(CPUKernelGenSpec::f64())
            .with_mode(SimulationMode::StateVector);

        let graph = test_graph_with_padding(vec![QuantumGate::h(0)]);
        let result = sim.run_graph(&graph).unwrap();
        assert!(result.timing.n_iters > 0);
    }

    /// StateVector mode rejects channel gates.
    #[test]
    fn statevector_rejects_channels() {
        let sim = Simulator::cpu(CPUKernelGenSpec::f64())
            .with_mode(SimulationMode::StateVector);

        let graph = test_graph_with_padding(vec![
            QuantumGate::h(0),
            QuantumGate::depolarizing(0, 0.1),
        ]);
        assert!(sim.run_graph(&graph).is_err());
    }

    /// DensityMatrix mode preserves trace after depolarizing noise.
    /// Uses the Simulator's internal DM lifting; reads the final state back
    /// manually to verify Tr(ρ) = 1.
    #[test]
    fn density_matrix_trace_preserved() {
        let spec = CPUKernelGenSpec::f64();
        let n = TEST_QUBITS as usize;

        let graph = test_graph_with_padding(vec![
            QuantumGate::h(0),
            QuantumGate::cx(0, 1),
            QuantumGate::depolarizing(0, 0.05),
        ]);

        // Use Simulator's prepare_gates to get the DM-lifted gate list.
        let sim = Simulator::cpu(spec.clone())
            .with_mode(SimulationMode::DensityMatrix);
        let (gates, n_sv) = sim
            .prepare_gates(&graph, TEST_QUBITS)
            .unwrap();

        // Execute manually to read back the final state.
        let mgr = CpuKernelManager::new();
        let mut sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
        sv.set_amp(0, crate::types::Complex::new(1.0, 0.0));

        let n_threads = get_num_threads();
        for gate in &gates {
            let kid = mgr.generate(&spec, gate).unwrap();
            mgr.apply(kid, &mut sv, n_threads).unwrap();
        }

        let dim = 1usize << n;
        let trace: f64 = (0..dim).map(|i| sv.amp(i | (i << n)).re).sum();
        assert!(
            (trace - 1.0).abs() < 1e-10,
            "trace should be 1.0, got {trace}"
        );
    }

    /// Trajectory mode: many trajectories with depolarizing noise should
    /// produce the correct sampling distribution.
    #[test]
    fn trajectory_converges_to_dm() {
        let spec = CPUKernelGenSpec::f64();

        // Circuit: H(0) + depolarizing(0, 0.1), padded to TEST_QUBITS.
        let graph = test_graph_with_padding(vec![
            QuantumGate::h(0),
            QuantumGate::depolarizing(0, 0.1),
        ]);

        let sim = Simulator::cpu(spec)
            .with_mode(SimulationMode::Trajectory {
                n_trajectories: 2000,
                seed: Some(42),
            });
        let result = sim.run_graph(&graph).unwrap();
        let traj_data = result.trajectory_data.unwrap();
        assert_eq!(traj_data.len(), 2000);

        // Depolarizing(p=0.1) has 4 noise branches.
        // K_0 = √(1-p)·I with probability (1-p) = 0.9.
        let n_identity: usize = traj_data
            .iter()
            .filter(|t| t.sampled_operators[0] == 0)
            .count();
        let frac = n_identity as f64 / 2000.0;
        assert!(
            (frac - 0.9).abs() < 0.05,
            "identity operator fraction should be ~0.9, got {frac}"
        );
    }

    /// CUDA: norm_squared and normalize work on device.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_norm_squared_and_normalize() {
        use crate::cuda::{CudaPrecision, CudaStatevector};

        // Create a 4-qubit SV = |0⟩ → norm² should be 1.0.
        let mut sv = CudaStatevector::new(4, CudaPrecision::F64).unwrap();
        sv.zero().unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!(
            (ns - 1.0).abs() < 1e-12,
            "|0⟩ norm² should be 1.0, got {ns}"
        );

        // Upload a custom state: (0.6, 0.8, 0, 0, ...) → norm² = 1.0
        let n = 1usize << 4; // 16 amplitudes
        let mut data = vec![(0.0, 0.0); n];
        data[0] = (0.6, 0.0);
        data[1] = (0.0, 0.8); // amplitude 1 is purely imaginary
        sv.upload(&data).unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!(
            (ns - 1.0).abs() < 1e-12,
            "custom state norm² should be 1.0, got {ns}"
        );

        // Upload unnormalized state: (3, 4, 0, ...) → norm² = 25
        data[0] = (3.0, 0.0);
        data[1] = (4.0, 0.0);
        sv.upload(&data).unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!(
            (ns - 25.0).abs() < 1e-10,
            "unnormalized norm² should be 25.0, got {ns}"
        );

        // Normalize → norm² = 1.0.
        sv.normalize().unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!(
            (ns - 1.0).abs() < 1e-10,
            "after normalize, norm² should be 1.0, got {ns}"
        );
    }
}

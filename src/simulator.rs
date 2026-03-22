//! Generic simulator parameterized by backend.
//!
//! # Example
//!
//! ```no_run
//! use cast::simulator::{Simulator, Cpu, SimulationMode, QuantumCircuit};
//! use cast::types::QuantumGate;
//!
//! let mut circuit = QuantumCircuit::new(4);
//! circuit.add(QuantumGate::h(0));
//! circuit.add(QuantumGate::cx(0, 1));
//!
//! let sim = Simulator::<Cpu>::f64();
//! let result = sim.run(&circuit).unwrap();
//! println!("amplitudes: {:?}", result.state.amplitudes());
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::cost_model::FusionConfig;
use crate::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};
use crate::fusion;
use crate::timing::TimingStats;
use crate::types::{Complex, QuantumGate};
use crate::CircuitGraph;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaKernelGenSpec, CudaKernelManager, CudaStatevector};

// ── QuantumCircuit ───────────────────────────────────────────────────────────

/// A sequence of quantum gates forming a circuit.
///
/// This is the primary user-facing circuit type. Build circuits programmatically
/// or parse from OpenQASM, then pass to [`Simulator::run`]. Internally converted
/// to a [`CircuitGraph`] for scheduling and fusion before simulation.
#[derive(Clone, Debug)]
pub struct QuantumCircuit {
    gates: Vec<Arc<QuantumGate>>,
    n_qubits: u32,
}

impl QuantumCircuit {
    /// Create an empty circuit on `n_qubits` qubits.
    pub fn new(n_qubits: u32) -> Self {
        Self {
            gates: Vec::new(),
            n_qubits,
        }
    }

    /// Add a gate to the circuit. Returns `&mut Self` for chaining.
    ///
    /// # Panics
    /// Panics if any qubit index in the gate is ≥ `n_qubits`.
    pub fn add(&mut self, gate: QuantumGate) -> &mut Self {
        for &q in gate.qubits() {
            assert!(
                q < self.n_qubits,
                "qubit {q} out of range for {}-qubit circuit",
                self.n_qubits
            );
        }
        self.gates.push(Arc::new(gate));
        self
    }

    /// Parse an OpenQASM 2.0 string into a circuit.
    pub fn from_qasm(qasm: &str) -> Result<Self> {
        let parsed = crate::openqasm::parse_qasm(qasm)?;
        Ok(Self::from_openqasm(&parsed))
    }

    /// Convert from a parsed OpenQASM circuit.
    pub fn from_openqasm(qasm_circuit: &crate::openqasm::Circuit) -> Self {
        use crate::circuit::quantum_gate_from_qasm_gate;
        let n_qubits = qasm_circuit.required_qreg_size();
        let gates: Vec<Arc<QuantumGate>> = qasm_circuit
            .gates
            .iter()
            .map(|g| Arc::new(quantum_gate_from_qasm_gate(g)))
            .collect();
        Self { gates, n_qubits }
    }

    /// Number of qubits in the circuit.
    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    /// Number of gates in the circuit.
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Whether the circuit has no gates.
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// The gates in circuit order.
    pub fn gates(&self) -> &[Arc<QuantumGate>] {
        &self.gates
    }
}

// ── Backend trait ────────────────────────────────────────────────────────────

/// Sealed trait mapping a backend marker to its concrete types and operations.
pub trait Backend: sealed::Sealed + Sized {
    #[doc(hidden)]
    type Sv;
    #[doc(hidden)]
    type KernelId: Copy;
    #[doc(hidden)]
    type Mgr;
    #[doc(hidden)]
    type Spec: Copy;

    #[doc(hidden)]
    fn new_manager() -> Self::Mgr;
    #[doc(hidden)]
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv>;
    #[doc(hidden)]
    fn init_sv(sv: &mut Self::Sv) -> Result<()>;
    #[doc(hidden)]
    fn generate(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        gate: &Arc<QuantumGate>,
    ) -> Result<Self::KernelId>;
    /// Extra per-simulator state needed for apply (e.g., CPU thread count).
    #[doc(hidden)]
    type Extra;
    #[doc(hidden)]
    fn default_extra() -> Self::Extra;
    #[doc(hidden)]
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        extra: &Self::Extra,
    ) -> Result<()>;
    #[doc(hidden)]
    fn sv_n_qubits(sv: &Self::Sv) -> u32;
    #[doc(hidden)]
    fn flush(mgr: &Self::Mgr) -> Result<()>;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Cpu {}
    #[cfg(feature = "cuda")]
    impl Sealed for super::Cuda {}
}

// ── CPU backend ──────────────────────────────────────────────────────────────

/// CPU simulation backend.
pub struct Cpu;

impl Backend for Cpu {
    type Sv = CPUStatevector;
    type KernelId = crate::cpu::KernelId;
    type Mgr = CpuKernelManager;
    type Spec = CPUKernelGenSpec;
    type Extra = u32; // n_threads

    fn new_manager() -> Self::Mgr {
        CpuKernelManager::new()
    }
    fn default_extra() -> Self::Extra {
        get_num_threads()
    }
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv> {
        Ok(CPUStatevector::new(
            n_qubits,
            spec.precision,
            spec.simd_width,
        ))
    }
    fn init_sv(sv: &mut Self::Sv) -> Result<()> {
        sv.initialize();
        Ok(())
    }
    fn generate(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        gate: &Arc<QuantumGate>,
    ) -> Result<Self::KernelId> {
        mgr.generate(spec, gate)
    }
    fn sv_n_qubits(sv: &Self::Sv) -> u32 {
        sv.n_qubits()
    }
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        n_threads: &Self::Extra,
    ) -> Result<()> {
        mgr.apply(id, sv, *n_threads)
    }
    fn flush(_mgr: &Self::Mgr) -> Result<()> {
        Ok(())
    }
}

// ── CUDA backend ─────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
/// CUDA simulation backend.
pub struct Cuda;

#[cfg(feature = "cuda")]
impl Backend for Cuda {
    type Sv = CudaStatevector;
    type KernelId = crate::cuda::CudaKernelId;
    type Mgr = CudaKernelManager;
    type Spec = CudaKernelGenSpec;
    type Extra = (); // no extra state needed

    fn new_manager() -> Self::Mgr {
        CudaKernelManager::new()
    }
    fn default_extra() -> Self::Extra {}
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv> {
        CudaStatevector::new(n_qubits, spec.precision)
    }
    fn init_sv(sv: &mut Self::Sv) -> Result<()> {
        sv.zero()
    }
    fn generate(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        gate: &Arc<QuantumGate>,
    ) -> Result<Self::KernelId> {
        mgr.generate(gate, *spec)
    }
    fn sv_n_qubits(sv: &Self::Sv) -> u32 {
        sv.n_qubits()
    }
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        _: &Self::Extra,
    ) -> Result<()> {
        mgr.apply(id, sv)
    }
    fn flush(mgr: &Self::Mgr) -> Result<()> {
        mgr.sync()?;
        Ok(())
    }
}

// ── QuantumState ─────────────────────────────────────────────────────────────

/// Quantum state after simulation — either a pure statevector or a density matrix.
pub struct QuantumState<B: Backend> {
    repr: StateRepr<B>,
}

enum StateRepr<B: Backend> {
    Pure(B::Sv),
    DensityMatrix { sv: B::Sv, n_physical: u32 },
}

impl<B: Backend> QuantumState<B> {
    /// Number of physical qubits.
    pub fn n_qubits(&self) -> u32 {
        match &self.repr {
            StateRepr::Pure(sv) => B::sv_n_qubits(sv),
            StateRepr::DensityMatrix { n_physical, .. } => *n_physical,
        }
    }

    /// Whether this is a pure state (vs density matrix).
    pub fn is_pure(&self) -> bool {
        matches!(self.repr, StateRepr::Pure(_))
    }

    fn sv(&self) -> &B::Sv {
        match &self.repr {
            StateRepr::Pure(sv) | StateRepr::DensityMatrix { sv, .. } => sv,
        }
    }
}

// CPU-specific inspection methods.
impl QuantumState<Cpu> {
    /// All amplitudes of the underlying statevector.
    pub fn amplitudes(&self) -> Vec<Complex> {
        self.sv().amplitudes()
    }

    /// Single amplitude by index.
    pub fn amp(&self, idx: usize) -> Complex {
        self.sv().amp(idx)
    }

    /// Diagonal populations: `|a_i|²` for pure states, `ρ[i,i]` for DM.
    pub fn populations(&self) -> Vec<f64> {
        match &self.repr {
            StateRepr::Pure(sv) => sv
                .amplitudes()
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect(),
            StateRepr::DensityMatrix { sv, n_physical } => {
                let n = *n_physical as usize;
                let dim = 1usize << n;
                (0..dim).map(|i| sv.amp(i | (i << n)).re).collect()
            }
        }
    }

    /// Trace (squared norm for pure states, Tr(ρ) for density matrix).
    pub fn trace(&self) -> f64 {
        match &self.repr {
            StateRepr::Pure(sv) => sv.norm_squared(),
            StateRepr::DensityMatrix { sv, n_physical } => {
                let n = *n_physical as usize;
                let dim = 1usize << n;
                (0..dim).map(|i| sv.amp(i | (i << n)).re).sum()
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl QuantumState<Cuda> {
    /// Download all amplitudes from GPU to host.
    pub fn download_amplitudes(&self) -> Result<Vec<(f64, f64)>> {
        self.sv().download()
    }

    /// Download and compute populations.
    pub fn populations(&self) -> Result<Vec<f64>> {
        match &self.repr {
            StateRepr::Pure(sv) => {
                let amps = sv.download()?;
                Ok(amps.iter().map(|(re, im)| re * re + im * im).collect())
            }
            StateRepr::DensityMatrix { sv, n_physical } => {
                let n = *n_physical as usize;
                let dim = 1usize << n;
                let amps = sv.download()?;
                Ok((0..dim).map(|i| amps[i | (i << n)].0).collect())
            }
        }
    }

    /// Trace (computed on GPU for pure states, downloaded for DM).
    pub fn trace(&self) -> Result<f64> {
        match &self.repr {
            StateRepr::Pure(sv) => sv.norm_squared(),
            StateRepr::DensityMatrix { sv, n_physical } => {
                let n = *n_physical as usize;
                let dim = 1usize << n;
                let amps = sv.download()?;
                Ok((0..dim).map(|i| amps[i | (i << n)].0).sum())
            }
        }
    }
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Simulation mode.
#[derive(Clone, Debug)]
pub enum SimulationMode {
    /// Pure-state simulation on n qubits. Errors if the circuit contains
    /// noisy gates.
    StateVector,
    /// Density-matrix simulation on 2n virtual qubits. Supports noisy gates
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
pub struct SimulationResult<B: Backend> {
    /// Final quantum state.
    pub state: QuantumState<B>,
    /// Execution time statistics.
    pub timing: TimingStats,
    /// Kernel compilation wall time (seconds).
    pub compile_time_s: f64,
    /// Per-trajectory data (only for [`SimulationMode::Trajectory`]).
    pub trajectory_data: Option<Vec<TrajectoryResult>>,
}

/// Per-trajectory outcome data.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    /// Index of the noise branch sampled at each noisy gate.
    pub sampled_operators: Vec<usize>,
    /// Wall time for this trajectory (seconds).
    pub time_s: f64,
}

// ── Simulator ────────────────────────────────────────────────────────────────

/// Unified quantum circuit simulator, generic over backend.
pub struct Simulator<B: Backend> {
    mode: SimulationMode,
    fusion_config: Option<FusionConfig>,
    mgr: B::Mgr,
    spec: B::Spec,
    extra: B::Extra,
}

// CPU constructors.
impl Simulator<Cpu> {
    /// Create a CPU simulator with the given spec.
    pub fn new(spec: CPUKernelGenSpec) -> Self {
        Self {
            mode: SimulationMode::StateVector,
            fusion_config: None,
            mgr: Cpu::new_manager(),
            spec,
            extra: Cpu::default_extra(),
        }
    }

    /// CPU simulator with F64 precision defaults.
    pub fn f64() -> Self {
        Self::new(CPUKernelGenSpec::f64())
    }

    /// CPU simulator with F32 precision defaults.
    pub fn f32() -> Self {
        Self::new(CPUKernelGenSpec::f32())
    }
}

// CUDA constructors.
#[cfg(feature = "cuda")]
impl Simulator<Cuda> {
    /// Create a CUDA simulator with the given spec.
    pub fn new(spec: CudaKernelGenSpec) -> Self {
        Self {
            mode: SimulationMode::StateVector,
            fusion_config: None,
            mgr: Cuda::new_manager(),
            spec,
            extra: (),
        }
    }

    /// CUDA simulator with F64 precision defaults.
    pub fn f64() -> Self {
        Self::new(CudaKernelGenSpec::f64())
    }

    /// CUDA simulator with F32 precision defaults.
    pub fn f32() -> Self {
        Self::new(CudaKernelGenSpec::f32())
    }
}

// Generic methods.
impl<B: Backend> Simulator<B> {
    /// Set the simulation mode.
    pub fn with_mode(mut self, mode: SimulationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the fusion configuration. Applied only by [`run`](Self::run), not
    /// [`run_graph`](Self::run_graph).
    pub fn with_fusion(mut self, config: FusionConfig) -> Self {
        self.fusion_config = Some(config);
        self
    }

    /// Run simulation on a [`QuantumCircuit`]. Builds an internal
    /// [`CircuitGraph`], applies fusion (if configured), and simulates.
    pub fn run(&self, circuit: &QuantumCircuit) -> Result<SimulationResult<B>> {
        let mut graph = CircuitGraph::new();
        graph.ensure_n_qubits(circuit.n_qubits() as usize);
        for gate in circuit.gates() {
            graph.insert_gate(Arc::clone(gate));
        }
        if let Some(ref config) = self.fusion_config {
            fusion::optimize(&mut graph, config);
        }
        self.run_graph(&graph)
    }

    /// Run simulation on a pre-built circuit graph (no fusion applied).
    pub fn run_graph(&self, circuit: &CircuitGraph) -> Result<SimulationResult<B>> {
        let n_physical = circuit.n_qubits() as u32;
        let (gates, n_sv) = self.prepare_gates(circuit, n_physical)?;

        let skip_noisy = matches!(self.mode, SimulationMode::Trajectory { .. });
        let t0 = Instant::now();
        let kernel_ids = self.compile_gates(&gates, skip_noisy)?;
        let compile_time_s = t0.elapsed().as_secs_f64();

        match &self.mode {
            SimulationMode::StateVector | SimulationMode::DensityMatrix => {
                let ids: Vec<B::KernelId> = kernel_ids
                    .iter()
                    .map(|k| k.expect("non-trajectory mode must compile all gates"))
                    .collect();

                let t_exec = Instant::now();
                let sv = self.execute_standard(&ids, n_sv)?;
                let exec_s = t_exec.elapsed().as_secs_f64();

                let state = match &self.mode {
                    SimulationMode::DensityMatrix => QuantumState {
                        repr: StateRepr::DensityMatrix { sv, n_physical },
                    },
                    _ => QuantumState {
                        repr: StateRepr::Pure(sv),
                    },
                };

                Ok(SimulationResult {
                    state,
                    timing: crate::timing::stats_from_samples(&[exec_s]),
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

    fn prepare_gates(
        &self,
        circuit: &CircuitGraph,
        n_physical: u32,
    ) -> Result<(Vec<Arc<QuantumGate>>, u32)> {
        let gates = circuit.gates_in_row_order();
        match &self.mode {
            SimulationMode::StateVector => {
                for gate in &gates {
                    if !gate.is_unitary() {
                        anyhow::bail!(
                            "StateVector mode does not support noisy gates; \
                             use DensityMatrix or Trajectory mode"
                        );
                    }
                }
                Ok((gates, n_physical))
            }
            SimulationMode::DensityMatrix => {
                let dm_gates: Vec<Arc<QuantumGate>> = gates
                    .iter()
                    .map(|g| Arc::new(g.to_density_matrix_gate(n_physical as usize)))
                    .collect();
                Ok((dm_gates, 2 * n_physical))
            }
            SimulationMode::Trajectory { .. } => Ok((gates, n_physical)),
        }
    }

    fn compile_gates(
        &self,
        gates: &[Arc<QuantumGate>],
        skip_noisy: bool,
    ) -> Result<Vec<Option<B::KernelId>>> {
        let mut ids = Vec::with_capacity(gates.len());
        for gate in gates {
            if skip_noisy && !gate.is_unitary() {
                ids.push(None);
            } else {
                ids.push(Some(B::generate(&self.mgr, &self.spec, gate)?));
            }
        }
        Ok(ids)
    }

    fn execute_standard(&self, kernel_ids: &[B::KernelId], n_sv: u32) -> Result<B::Sv> {
        let mut sv = B::new_sv(n_sv, &self.spec)?;
        B::init_sv(&mut sv)?;
        for &kid in kernel_ids {
            self.apply_one(kid, &mut sv)?;
        }
        B::flush(&self.mgr)?;
        Ok(sv)
    }

    fn execute_trajectory(
        &self,
        gates: &[Arc<QuantumGate>],
        gate_kernel_ids: &[Option<B::KernelId>],
        n_sv: u32,
        n_trajectories: usize,
        seed: Option<u64>,
        compile_time_s: f64,
    ) -> Result<SimulationResult<B>> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // Pre-compile noise-branch kernels and sampling distributions.
        let mut noise_kernels: Vec<Option<Vec<B::KernelId>>> = Vec::with_capacity(gates.len());
        let mut noise_dists: Vec<Option<WeightedIndex<f64>>> = Vec::with_capacity(gates.len());
        for gate in gates {
            if !gate.is_unitary() {
                let mut kids = Vec::new();
                for (_, u_noise) in gate.noise() {
                    let composed = u_noise.matmul(gate.matrix());
                    let g = Arc::new(QuantumGate::new(composed, gate.qubits().to_vec()));
                    kids.push(B::generate(&self.mgr, &self.spec, &g)?);
                }
                noise_kernels.push(Some(kids));
                let weights: Vec<f64> = gate.noise().iter().map(|(p, _)| *p).collect();
                noise_dists.push(Some(
                    WeightedIndex::new(&weights).context("invalid noise probabilities")?,
                ));
            } else {
                noise_kernels.push(None);
                noise_dists.push(None);
            }
        }

        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(StdRng::from_entropy()),
        };

        let n_noisy_gates = noise_kernels.iter().filter(|k| k.is_some()).count();
        let mut trajectory_results = Vec::with_capacity(n_trajectories);
        let mut sv = B::new_sv(n_sv, &self.spec)?;

        for _ in 0..n_trajectories {
            let t0 = Instant::now();
            B::init_sv(&mut sv)?;

            let mut sampled_ops = Vec::with_capacity(n_noisy_gates);
            for (i, gate) in gates.iter().enumerate() {
                if gate.is_unitary() {
                    self.apply_one(gate_kernel_ids[i].unwrap(), &mut sv)?;
                } else {
                    let chosen = noise_dists[i].as_ref().unwrap().sample(&mut *rng);
                    sampled_ops.push(chosen);
                    self.apply_one(noise_kernels[i].as_ref().unwrap()[chosen], &mut sv)?;
                }
            }
            B::flush(&self.mgr)?;

            let time_s = t0.elapsed().as_secs_f64();
            trajectory_results.push(TrajectoryResult {
                sampled_operators: sampled_ops,
                time_s,
            });
        }

        let samples: Vec<f64> = trajectory_results.iter().map(|t| t.time_s).collect();
        let timing = crate::timing::stats_from_samples(&samples);

        let state = QuantumState {
            repr: StateRepr::Pure(sv),
        };

        Ok(SimulationResult {
            state,
            timing,
            compile_time_s,
            trajectory_data: Some(trajectory_results),
        })
    }

    fn apply_one(&self, kid: B::KernelId, sv: &mut B::Sv) -> Result<()> {
        B::apply(&self.mgr, kid, sv, &self.extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_QUBITS: u32 = 4;

    fn test_circuit(gates: Vec<QuantumGate>) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(TEST_QUBITS);
        for g in gates {
            circuit.add(g);
        }
        circuit
    }

    #[test]
    fn statevector_h_gate() {
        let sim = Simulator::<Cpu>::f64();
        let circuit = test_circuit(vec![QuantumGate::h(0)]);
        let result = sim.run(&circuit).unwrap();
        assert!(result.state.is_pure());
        let pops = result.state.populations();
        assert!((pops[0] - 0.5).abs() < 1e-10);
        assert!((pops[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn statevector_rejects_noisy_gates() {
        let sim = Simulator::<Cpu>::f64();
        let circuit = test_circuit(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);
        assert!(sim.run(&circuit).is_err());
    }

    #[test]
    fn density_matrix_trace_preserved() {
        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::DensityMatrix);
        let circuit = test_circuit(vec![
            QuantumGate::h(0),
            QuantumGate::cx(0, 1),
            QuantumGate::depolarizing(0, 0.05),
        ]);
        let result = sim.run(&circuit).unwrap();
        assert!(!result.state.is_pure());
        let trace = result.state.trace();
        assert!(
            (trace - 1.0).abs() < 1e-10,
            "trace should be 1.0, got {trace}"
        );
    }

    #[test]
    fn trajectory_sampling_distribution() {
        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_trajectories: 2000,
            seed: Some(42),
        });
        let circuit = test_circuit(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);
        let result = sim.run(&circuit).unwrap();
        let traj_data = result.trajectory_data.unwrap();
        assert_eq!(traj_data.len(), 2000);

        let n_identity: usize = traj_data
            .iter()
            .filter(|t| t.sampled_operators[0] == 0)
            .count();
        let frac = n_identity as f64 / 2000.0;
        assert!(
            (frac - 0.9).abs() < 0.05,
            "identity fraction should be ~0.9, got {frac}"
        );
    }

    #[test]
    fn from_qasm() {
        let circuit = QuantumCircuit::from_qasm(
            "OPENQASM 2.0; qreg q[4]; h q[0]; cx q[0],q[1]; cx q[2],q[3];",
        )
        .unwrap();
        assert_eq!(circuit.n_qubits(), 4);
        assert_eq!(circuit.len(), 3);

        let sim = Simulator::<Cpu>::f64();
        let result = sim.run(&circuit).unwrap();
        assert!(result.state.is_pure());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_norm_squared_and_normalize() {
        use crate::cuda::{CudaPrecision, CudaStatevector};

        let mut sv = CudaStatevector::new(4, CudaPrecision::F64).unwrap();
        sv.zero().unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!((ns - 1.0).abs() < 1e-12);

        let n = 1usize << 4;
        let mut data = vec![(0.0, 0.0); n];
        data[0] = (3.0, 0.0);
        data[1] = (4.0, 0.0);
        sv.upload(&data).unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!((ns - 25.0).abs() < 1e-10);

        sv.normalize().unwrap();
        let ns = sv.norm_squared().unwrap();
        assert!((ns - 1.0).abs() < 1e-10);
    }
}

//! Generic simulator parameterized by backend.
//!
//! # Example
//!
//! ```no_run
//! use cast::simulator::{Simulator, Cpu, SimulationMode};
//! use cast::types::{QuantumGate, QuantumCircuit};
//!
//! let mut circuit = QuantumCircuit::new(4);
//! circuit.add(QuantumGate::h(0));
//! circuit.add(QuantumGate::cx(0, 1));
//!
//! let sim = Simulator::<Cpu>::f64();
//! let result = sim.run(&circuit).unwrap();
//! println!("amplitudes: {:?}", result.state.unwrap().amplitudes());
//! ```

mod measure;
mod trajectory;

pub use trajectory::{ExploredBranch, TrajectoryResult};

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use crate::cost_model::FusionConfig;
use crate::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};
use crate::fusion;
use crate::timing::TimingStats;
use crate::types::{Complex, QuantumCircuit, QuantumGate};
use crate::CircuitGraph;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaKernelGenSpec, CudaKernelManager, CudaStatevector};

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
    fn new_manager(spec: Self::Spec) -> Self::Mgr;
    #[doc(hidden)]
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv>;
    #[doc(hidden)]
    fn init_sv(sv: &mut Self::Sv) -> Result<()>;
    #[doc(hidden)]
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId>;
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
    /// Compute marginal measurement probabilities for trajectory simulation.
    #[doc(hidden)]
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>>;
    /// Clone a statevector (deep copy). Used for ensemble branching.
    #[doc(hidden)]
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv>;

    /// Compile and execute `gates` on a freshly initialised statevector, using
    /// the backend-optimal pipelining strategy.
    ///
    /// - **CPU**: compile all gates (batch JIT), then execute all sequentially.
    ///   `compile_s` and `exec_s` are measured independently.
    /// - **CUDA**: compile in a thread pool while launching in a windowed
    ///   pipeline on the GPU stream.  Compile and execute overlap, so
    ///   `compile_s = 0` and `exec_s = total_wall`; per-kernel CPU compile
    ///   times are still available via the manager's stats if needed.
    #[doc(hidden)]
    fn execute_pipelined(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        extra: &Self::Extra,
        gates: &[Arc<QuantumGate>],
        n_sv: u32,
    ) -> Result<(Self::Sv, PipelineTimings)>;
}

/// Wall-time split returned by [`Backend::execute_pipelined`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineTimings {
    pub compile_s: f64,
    pub exec_s: f64,
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

    fn new_manager(spec: Self::Spec) -> Self::Mgr {
        CpuKernelManager::new(spec)
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
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId> {
        mgr.generate(gate)
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
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>> {
        Ok(measure::marginal_probabilities_cpu(sv, measured_qubits))
    }
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        Ok(sv.clone())
    }
    fn execute_pipelined(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        extra: &Self::Extra,
        gates: &[Arc<QuantumGate>],
        n_sv: u32,
    ) -> Result<(Self::Sv, PipelineTimings)> {
        // CPU: compile all gates first (batch JIT), then execute all.
        // Explicitly finalize the batch so the compile cost is attributed
        // to compile_s rather than to the first apply.
        let compile_t0 = Instant::now();
        let ids: Vec<Self::KernelId> = gates
            .iter()
            .map(|g| Self::generate(mgr, g))
            .collect::<Result<_>>()?;
        mgr.finalize()?;
        let compile_s = compile_t0.elapsed().as_secs_f64();

        let exec_t0 = Instant::now();
        let mut sv = Self::new_sv(n_sv, spec)?;
        Self::init_sv(&mut sv)?;
        for &kid in &ids {
            Self::apply(mgr, kid, &mut sv, extra)?;
        }
        Self::flush(mgr)?;
        let exec_s = exec_t0.elapsed().as_secs_f64();

        Ok((sv, PipelineTimings { compile_s, exec_s }))
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

    fn new_manager(spec: Self::Spec) -> Self::Mgr {
        CudaKernelManager::new(spec)
    }
    fn default_extra() -> Self::Extra {}
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv> {
        CudaStatevector::new(n_qubits, spec.precision)
    }
    fn init_sv(sv: &mut Self::Sv) -> Result<()> {
        sv.zero()
    }
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId> {
        mgr.generate(gate)
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
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>> {
        let amps = sv.download()?;
        let positions = measure::qubit_positions(measured_qubits);
        let n_bins = 1usize << measured_qubits.len();
        Ok(measure::accumulate_marginal_probs(
            amps.into_iter(),
            &positions,
            n_bins,
        ))
    }
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        sv.clone_device()
    }
    fn execute_pipelined(
        mgr: &Self::Mgr,
        spec: &Self::Spec,
        _extra: &Self::Extra,
        gates: &[Arc<QuantumGate>],
        n_sv: u32,
    ) -> Result<(Self::Sv, PipelineTimings)> {
        let mut sv = Self::new_sv(n_sv, spec)?;
        Self::init_sv(&mut sv)?;
        // Default window = LRU cache size, compile threads = 4.
        let window_size = 4;
        let n_compile_threads = 4;
        let stats = mgr.execute_pipelined(gates, &mut sv, window_size, n_compile_threads)?;
        // Compile and execute overlap in the pipelined CUDA path, so the
        // entire wall is reported as exec_s.  Per-kernel CPU compile times
        // remain available via `stats.kernels`.
        let exec_s = stats.wall_time.as_secs_f64();
        Ok((
            sv,
            PipelineTimings {
                compile_s: 0.0,
                exec_s,
            },
        ))
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
    /// Trajectory-based noisy simulation: deterministic ensemble branching
    /// with batch measurement sampling. Produces a measurement histogram
    /// over the circuit's `measured_qubits`.
    Trajectory {
        /// Total number of measurement samples to generate.
        n_samples: u64,
        /// RNG seed for measurement sampling (None = entropy).
        seed: Option<u64>,
        /// Maximum number of concurrent statevectors in the ensemble.
        /// `None` = auto-detect from available memory. `Some(1)` = single
        /// deterministic trajectory (highest-probability branch only).
        max_ensemble: Option<usize>,
    },
}

/// Result of a simulation run.
pub struct SimulationResult<B: Backend> {
    /// Final quantum state (`None` for trajectory mode).
    pub state: Option<QuantumState<B>>,
    /// Execution time statistics.
    pub timing: TimingStats,
    /// Kernel compilation wall time (seconds).
    pub compile_time_s: f64,
    /// Trajectory simulation data (only for [`SimulationMode::Trajectory`]).
    pub trajectory_data: Option<TrajectoryResult>,
}

// ── Simulator ────────────────────────────────────────────────────────────────

/// Unified quantum circuit simulator, generic over backend.
pub struct Simulator<B: Backend> {
    mode: SimulationMode,
    fusion_config: Option<FusionConfig>,
    pub(crate) mgr: B::Mgr,
    pub(crate) spec: B::Spec,
    pub(crate) extra: B::Extra,
}

// CPU constructors.
impl Simulator<Cpu> {
    /// Create a CPU simulator with the given spec.
    pub fn new(spec: CPUKernelGenSpec) -> Self {
        Self {
            mode: SimulationMode::StateVector,
            fusion_config: None,
            mgr: Cpu::new_manager(spec),
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

    /// Override the number of worker threads used for kernel application.
    pub fn with_threads(mut self, n_threads: u32) -> Self {
        self.extra = n_threads;
        self
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
            mgr: Cuda::new_manager(spec),
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
    ///
    /// For trajectory mode, dead-gate elimination is applied automatically
    /// based on [`QuantumCircuit::measured_qubits`].
    pub fn run(&self, circuit: &QuantumCircuit) -> Result<SimulationResult<B>> {
        // For trajectory mode, eliminate gates that can't influence measured qubits.
        let circuit = if matches!(self.mode, SimulationMode::Trajectory { .. }) {
            circuit.eliminate_dead_gates()
        } else {
            circuit.clone()
        };

        let mut graph = CircuitGraph::new();
        graph.ensure_n_qubits(circuit.n_qubits() as usize);
        for gate in circuit.gates() {
            graph.insert_gate(Arc::clone(gate));
        }
        if let Some(ref config) = self.fusion_config {
            fusion::optimize(&mut graph, config);
        }
        self.run_graph(&graph, circuit.measured_qubits())
    }

    /// Run simulation on a pre-built circuit graph (no fusion applied).
    ///
    /// `measured_qubits` is used only for trajectory mode — it specifies
    /// which qubits to measure at the end of the circuit.
    pub fn run_graph(
        &self,
        circuit: &CircuitGraph,
        measured_qubits: &[u32],
    ) -> Result<SimulationResult<B>> {
        let n_physical = circuit.n_qubits() as u32;
        let (gates, n_sv) = self.prepare_gates(circuit, n_physical)?;

        match &self.mode {
            SimulationMode::StateVector | SimulationMode::DensityMatrix => {
                let (sv, timings) =
                    B::execute_pipelined(&self.mgr, &self.spec, &self.extra, &gates, n_sv)?;

                let state = match &self.mode {
                    SimulationMode::DensityMatrix => QuantumState {
                        repr: StateRepr::DensityMatrix { sv, n_physical },
                    },
                    _ => QuantumState {
                        repr: StateRepr::Pure(sv),
                    },
                };

                Ok(SimulationResult {
                    state: Some(state),
                    timing: crate::timing::stats_from_samples(&[timings.exec_s]),
                    compile_time_s: timings.compile_s,
                    trajectory_data: None,
                })
            }
            SimulationMode::Trajectory { .. } => {
                let t0 = Instant::now();
                let kernel_ids = self.compile_gates(&gates)?;
                let compile_time_s = t0.elapsed().as_secs_f64();
                let prepared = trajectory::PreparedCircuit {
                    gates,
                    kernel_ids,
                    n_sv,
                };
                self.execute_trajectory(&prepared, measured_qubits, compile_time_s)
            }
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

    fn compile_gates(&self, gates: &[Arc<QuantumGate>]) -> Result<Vec<Option<B::KernelId>>> {
        let skip_noisy = matches!(self.mode, SimulationMode::Trajectory { .. });
        let mut ids = Vec::with_capacity(gates.len());
        for gate in gates {
            if skip_noisy && !gate.is_unitary() {
                ids.push(None);
            } else {
                ids.push(Some(B::generate(&self.mgr, gate)?));
            }
        }
        Ok(ids)
    }

    pub(crate) fn apply_one(&self, kid: B::KernelId, sv: &mut B::Sv) -> Result<()> {
        B::apply(&self.mgr, kid, sv, &self.extra)
    }
}

#[cfg(test)]
mod tests;

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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use crate::cost_model::FusionConfig;
use crate::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};
use crate::fusion;
use crate::timing::TimingStats;
use crate::types::{compress_bits, Complex, QuantumCircuit, QuantumGate};
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
    /// Compute marginal measurement probabilities for trajectory simulation.
    #[doc(hidden)]
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>>;
    /// Clone a statevector (deep copy). Used for ensemble branching.
    #[doc(hidden)]
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv>;
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
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>> {
        Ok(marginal_probabilities_cpu(sv, measured_qubits))
    }
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        Ok(sv.clone())
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
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>> {
        let amps = sv.download()?;
        let positions = qubit_positions(measured_qubits);
        let n_bins = 1usize << measured_qubits.len();
        Ok(accumulate_marginal_probs(
            amps.into_iter(),
            &positions,
            n_bins,
        ))
    }
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        sv.clone_device()
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
    /// over the specified qubits.
    Trajectory {
        /// Total number of measurement samples to generate.
        n_samples: u64,
        /// Which qubits to measure at the end of the circuit.
        measured_qubits: Vec<u32>,
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

/// A single explored noise branch and its contribution to the histogram.
#[derive(Clone, Debug)]
pub struct ExploredBranch {
    /// Probability weight of this branch (product of noise-branch
    /// probabilities along the path).
    pub weight: f64,
    /// Noise branch index chosen at each noisy gate in circuit order.
    pub noise_path: Vec<usize>,
    /// Number of measurement samples generated from this branch's
    /// final statevector.
    pub n_samples: u64,
}

/// Result of a trajectory / ensemble simulation.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    /// Measurement histogram: bitstring → count.
    pub histogram: HashMap<u64, u64>,
    /// Total measurement samples generated.
    pub n_samples: u64,
    /// Branches explored, sorted by weight descending.
    pub branches: Vec<ExploredBranch>,
    /// Sum of explored branch weights (≤ 1.0). The gap
    /// `1.0 − explored_weight` bounds the approximation error.
    pub explored_weight: f64,
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
        use std::borrow::Cow;

        // For trajectory mode, eliminate gates that can't influence measured qubits.
        let gates: Cow<[Arc<QuantumGate>]> = match &self.mode {
            SimulationMode::Trajectory {
                ref measured_qubits,
                ..
            } => Cow::Owned(eliminate_dead_gates(circuit.gates(), measured_qubits)),
            _ => Cow::Borrowed(circuit.gates()),
        };

        let mut graph = CircuitGraph::new();
        graph.ensure_n_qubits(circuit.n_qubits() as usize);
        for gate in gates.iter() {
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
                    state: Some(state),
                    timing: crate::timing::stats_from_samples(&[exec_s]),
                    compile_time_s,
                    trajectory_data: None,
                })
            }
            SimulationMode::Trajectory {
                n_samples,
                ref measured_qubits,
                seed,
                ref max_ensemble,
            } => self.execute_trajectory(
                &gates,
                &kernel_ids,
                n_sv,
                *n_samples,
                measured_qubits,
                *seed,
                max_ensemble.unwrap_or(1).max(1),
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
        n_samples: u64,
        measured_qubits: &[u32],
        seed: Option<u64>,
        max_ensemble: usize,
        compile_time_s: f64,
    ) -> Result<SimulationResult<B>> {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        // ── Pre-compile noise-branch kernels (deduplicated) ────────────────
        let t_noise_compile = Instant::now();
        let mut noise_kernels: Vec<Option<Vec<B::KernelId>>> = Vec::with_capacity(gates.len());
        let mut noise_branch_weights: Vec<Option<Vec<f64>>> = Vec::with_capacity(gates.len());
        // Cache: (qubits, matrix_bytes) → KernelId. Bit-identical composed
        // matrices share kernels — avoids recompiling e.g. 812 identical
        // depolarizing noise branches.
        let mut kernel_cache: HashMap<(Vec<u32>, Vec<u8>), B::KernelId> = HashMap::new();
        for gate in gates {
            if !gate.is_unitary() {
                let mut kids = Vec::new();
                for (_, u_noise) in gate.noise() {
                    let composed = u_noise.matmul(gate.matrix());
                    let key = (gate.qubits().to_vec(), matrix_to_bytes(&composed));
                    let kid = match kernel_cache.get(&key) {
                        Some(&cached) => cached,
                        None => {
                            let g = Arc::new(QuantumGate::new(composed, gate.qubits().to_vec()));
                            let kid = B::generate(&self.mgr, &self.spec, &g)?;
                            kernel_cache.insert(key, kid);
                            kid
                        }
                    };
                    kids.push(kid);
                }
                noise_kernels.push(Some(kids));
                noise_branch_weights.push(Some(gate.noise().iter().map(|(p, _)| *p).collect()));
            } else {
                noise_kernels.push(None);
                noise_branch_weights.push(None);
            }
        }
        let total_compile_s = compile_time_s + t_noise_compile.elapsed().as_secs_f64();

        // ── Ensemble simulation ─────────────────────────────────────────────
        struct Member<Sv> {
            sv: Sv,
            weight: f64,
            noise_path: Vec<usize>,
        }

        let t_sim = Instant::now();

        let mut sv0 = B::new_sv(n_sv, &self.spec)?;
        B::init_sv(&mut sv0)?;
        let mut ensemble: Vec<Member<B::Sv>> = vec![Member {
            sv: sv0,
            weight: 1.0,
            noise_path: Vec::new(),
        }];

        for (i, gate) in gates.iter().enumerate() {
            if gate.is_unitary() {
                for member in &mut ensemble {
                    self.apply_one(gate_kernel_ids[i].unwrap(), &mut member.sv)?;
                }
            } else {
                let branch_kernels = noise_kernels[i].as_ref().unwrap();
                let branch_weights = noise_branch_weights[i].as_ref().unwrap();
                let n_branches = branch_weights.len();

                // Compute all candidate (member_idx, branch_idx, weight).
                let mut candidates: Vec<(usize, usize, f64)> =
                    Vec::with_capacity(ensemble.len() * n_branches);
                for (m_idx, member) in ensemble.iter().enumerate() {
                    for (b_idx, &bp) in branch_weights.iter().enumerate() {
                        candidates.push((m_idx, b_idx, member.weight * bp));
                    }
                }

                // Sort by weight descending, keep top M.
                candidates.sort_unstable_by(|a, b| {
                    b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                });
                let kept = candidates.len().min(max_ensemble);
                let candidates = &candidates[..kept];

                if ensemble.len() == 1 && kept == 1 {
                    // Fast path: single member stays single — no clone.
                    let (_, b_idx, new_weight) = candidates[0];
                    ensemble[0].weight = new_weight;
                    ensemble[0].noise_path.push(b_idx);
                    self.apply_one(branch_kernels[b_idx], &mut ensemble[0].sv)?;
                } else {
                    // General path: build new ensemble from kept candidates.
                    // Flush to ensure pending kernels complete before cloning.
                    B::flush(&self.mgr)?;

                    // Count references to each parent so the last use can
                    // move the SV instead of cloning.
                    let mut parent_refs = vec![0usize; ensemble.len()];
                    for &(m_idx, _, _) in candidates {
                        parent_refs[m_idx] += 1;
                    }

                    // Wrap parents in Option so we can take() on last use.
                    let mut parents: Vec<Option<Member<B::Sv>>> =
                        ensemble.into_iter().map(Some).collect();

                    let mut new_ensemble: Vec<Member<B::Sv>> = Vec::with_capacity(kept);
                    for &(m_idx, b_idx, new_weight) in candidates {
                        parent_refs[m_idx] -= 1;
                        let is_last_ref = parent_refs[m_idx] == 0;

                        let (mut sv, mut noise_path) = if is_last_ref {
                            // Move SV + noise_path from parent (no clone).
                            let m = parents[m_idx].take().unwrap();
                            (m.sv, m.noise_path)
                        } else {
                            // Clone SV + noise_path from parent.
                            let m = parents[m_idx].as_ref().unwrap();
                            (B::clone_sv(&m.sv)?, m.noise_path.clone())
                        };

                        self.apply_one(branch_kernels[b_idx], &mut sv)?;
                        noise_path.push(b_idx);

                        new_ensemble.push(Member {
                            sv,
                            weight: new_weight,
                            noise_path,
                        });
                    }
                    ensemble = new_ensemble;
                }
            }
        }
        B::flush(&self.mgr)?;
        let simulation_time_s = t_sim.elapsed().as_secs_f64();

        // ── Batch measurement sampling ──────────────────────────────────────
        let t_sample = Instant::now();
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Sum of kept branch weights. Gap (1.0 - total_weight) bounds error.
        let total_weight: f64 = ensemble.iter().map(|m| m.weight).sum();
        let mut histogram: HashMap<u64, u64> = HashMap::new();
        let mut branches: Vec<ExploredBranch> = Vec::with_capacity(ensemble.len());
        let mut samples_remaining = n_samples;

        for (idx, member) in ensemble.iter().enumerate() {
            let member_samples = if idx == ensemble.len() - 1 {
                samples_remaining
            } else {
                let s = ((member.weight / total_weight) * n_samples as f64).floor() as u64;
                s.min(samples_remaining)
            };
            samples_remaining = samples_remaining.saturating_sub(member_samples);

            if member_samples > 0 {
                let probs = B::marginal_probabilities(&member.sv, measured_qubits)?;
                let member_hist = batch_sample(&probs, member_samples, &mut rng);
                for (outcome, count) in member_hist {
                    *histogram.entry(outcome).or_insert(0) += count;
                }
            }

            branches.push(ExploredBranch {
                weight: member.weight,
                noise_path: member.noise_path.clone(),
                n_samples: member_samples,
            });
        }
        let sampling_time_s = t_sample.elapsed().as_secs_f64();

        let total_time_s = simulation_time_s + sampling_time_s;
        let timing = crate::timing::stats_from_samples(&[total_time_s]);

        Ok(SimulationResult {
            state: None,
            timing,
            compile_time_s: total_compile_s,
            trajectory_data: Some(TrajectoryResult {
                histogram,
                n_samples,
                branches,
                explored_weight: total_weight,
            }),
        })
    }

    fn apply_one(&self, kid: B::KernelId, sv: &mut B::Sv) -> Result<()> {
        B::apply(&self.mgr, kid, sv, &self.extra)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Reinterpret a complex matrix's data as raw bytes for deduplication keying.
fn matrix_to_bytes(m: &crate::types::ComplexSquareMatrix) -> Vec<u8> {
    let data = m.data();
    let ptr = data.as_ptr() as *const u8;
    let len = data.len() * std::mem::size_of::<crate::types::Complex>();
    // SAFETY: Complex64 is repr(C) with two f64 fields; we're reading the
    // existing slice as bytes within its lifetime.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

// ── Dead gate elimination ────────────────────────────────────────────────────

/// Remove gates that cannot influence any measured qubit.
///
/// Performs backward liveness analysis: starting from `measured_qubits`, any
/// gate that touches a live qubit makes all its qubits live. Gates touching
/// only dead qubits are eliminated.
fn eliminate_dead_gates(
    gates: &[Arc<QuantumGate>],
    measured_qubits: &[u32],
) -> Vec<Arc<QuantumGate>> {
    let mut live = 0u64;
    for &q in measured_qubits {
        live |= 1u64 << q;
    }

    // Mark each gate as live/dead in reverse order.
    let mut is_live: Vec<bool> = vec![false; gates.len()];
    for (i, gate) in gates.iter().enumerate().rev() {
        let gate_mask: u64 = gate.qubits().iter().fold(0u64, |m, &q| m | (1u64 << q));
        if live & gate_mask != 0 {
            is_live[i] = true;
            live |= gate_mask;
        }
    }

    // Collect live gates in forward order.
    gates
        .iter()
        .zip(is_live.iter())
        .filter(|(_, &live)| live)
        .map(|(g, _)| Arc::clone(g))
        .collect()
}

// ── Measurement sampling ─────────────────────────────────────────────────────

/// Convert measured qubit indices (u32) to usize positions for `compress_bits`.
fn qubit_positions(measured_qubits: &[u32]) -> Vec<usize> {
    measured_qubits.iter().map(|&q| q as usize).collect()
}

/// Accumulate marginal probabilities from an iterator of `(re, im)` amplitude
/// pairs into a histogram over the measured qubit positions.
fn accumulate_marginal_probs(
    amps: impl Iterator<Item = (f64, f64)>,
    positions: &[usize],
    n_bins: usize,
) -> Vec<f64> {
    let mut probs = vec![0.0f64; n_bins];
    for (j, (re, im)) in amps.enumerate() {
        probs[compress_bits(j, positions)] += re * re + im * im;
    }
    probs
}

/// Compute marginal measurement probabilities over `measured_qubits` from a
/// CPU statevector.
fn marginal_probabilities_cpu(sv: &CPUStatevector, measured_qubits: &[u32]) -> Vec<f64> {
    let positions = qubit_positions(measured_qubits);
    let n_bins = 1usize << measured_qubits.len();
    let n = 1usize << sv.n_qubits();
    accumulate_marginal_probs(
        (0..n).map(|j| {
            let a = sv.amp(j);
            (a.re, a.im)
        }),
        &positions,
        n_bins,
    )
}

/// Batch-sample `n_samples` bitstrings from a discrete probability distribution.
///
/// Uses the inverse-CDF method: build a CDF of size D (= `probs.len()`), then
/// binary-search for each sample. Total cost: O(D + N log D) with O(D) memory.
///
/// Returns a histogram mapping outcome bitstring → count.
fn batch_sample(probs: &[f64], n_samples: u64, rng: &mut impl rand::Rng) -> HashMap<u64, u64> {
    if n_samples == 0 || probs.is_empty() {
        return HashMap::new();
    }

    // Build CDF (O(D) memory).
    let cdf: Vec<f64> = probs
        .iter()
        .scan(0.0f64, |acc, &p| {
            *acc += p;
            Some(*acc)
        })
        .collect();

    let mut histogram = HashMap::new();
    let last = probs.len() - 1;
    for _ in 0..n_samples {
        let u: f64 = rng.gen();
        let outcome = cdf.partition_point(|&c| c <= u).min(last);
        *histogram.entry(outcome as u64).or_insert(0u64) += 1;
    }

    histogram
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
        let state = result.state.unwrap();
        assert!(state.is_pure());
        let pops = state.populations();
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
        let state = result.state.unwrap();
        assert!(!state.is_pure());
        let trace = state.trace();
        assert!(
            (trace - 1.0).abs() < 1e-10,
            "trace should be 1.0, got {trace}"
        );
    }

    #[test]
    fn trajectory_h_gate_histogram() {
        // H(0) on 4-qubit circuit, no noise. Measure qubit 0.
        // Should get ~50/50 histogram for outcomes 0 and 1.
        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 10000,
            measured_qubits: vec![0],
            seed: Some(42),
            max_ensemble: Some(1),
        });
        let circuit = test_circuit(vec![QuantumGate::h(0)]);
        let result = sim.run(&circuit).unwrap();
        assert!(result.state.is_none());
        let traj = result.trajectory_data.unwrap();
        assert_eq!(traj.n_samples, 10000);
        assert_eq!(traj.histogram.values().sum::<u64>(), 10000);
        // Should be ~50/50
        let count_0 = traj.histogram.get(&0).copied().unwrap_or(0);
        let count_1 = traj.histogram.get(&1).copied().unwrap_or(0);
        assert!(
            (count_0 as f64 - 5000.0).abs() < 500.0,
            "expected ~5000 for |0⟩, got {count_0}"
        );
        assert!(
            (count_1 as f64 - 5000.0).abs() < 500.0,
            "expected ~5000 for |1⟩, got {count_1}"
        );
        // No noise → explored_weight = 1.0
        assert!((traj.explored_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trajectory_deterministic_branch() {
        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 100,
            measured_qubits: vec![0],
            seed: Some(42),
            max_ensemble: Some(1),
        });
        let circuit = test_circuit(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);
        let result = sim.run(&circuit).unwrap();
        assert!(result.state.is_none());
        let traj = result.trajectory_data.unwrap();
        // M=1 deterministic: always picks branch 0 (identity with prob 0.9)
        assert_eq!(traj.branches.len(), 1);
        assert_eq!(traj.branches[0].noise_path, vec![0]);
        assert!((traj.branches[0].weight - 0.9).abs() < 1e-10);
        assert!((traj.explored_weight - 0.9).abs() < 1e-10);
    }

    #[test]
    fn ensemble_m4_explores_more_weight() {
        // M=4 should explore more weight than M=1 on a noisy circuit.
        let circuit = test_circuit(vec![QuantumGate::h(0), QuantumGate::depolarizing(0, 0.1)]);

        let sim_m1 = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 1000,
            measured_qubits: vec![0],
            seed: Some(42),
            max_ensemble: Some(1),
        });
        let r1 = sim_m1.run(&circuit).unwrap();
        let t1 = r1.trajectory_data.unwrap();

        let sim_m4 = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 1000,
            measured_qubits: vec![0],
            seed: Some(42),
            max_ensemble: Some(4),
        });
        let r4 = sim_m4.run(&circuit).unwrap();
        let t4 = r4.trajectory_data.unwrap();

        assert_eq!(t1.branches.len(), 1);
        assert_eq!(t4.branches.len(), 4); // depolarizing has 4 branches
        assert!(
            t4.explored_weight > t1.explored_weight,
            "M=4 ({:.4}) should explore more than M=1 ({:.4})",
            t4.explored_weight,
            t1.explored_weight,
        );
        // With all 4 branches explored, weight should be 1.0
        assert!(
            (t4.explored_weight - 1.0).abs() < 1e-10,
            "all branches explored: weight should be 1.0, got {}",
            t4.explored_weight,
        );
    }

    #[test]
    fn ensemble_matches_dm_populations() {
        // On a small noisy circuit, ensemble with enough capacity to keep all
        // branches should produce a histogram matching exact DM populations.
        let circuit = test_circuit(vec![
            QuantumGate::h(0),
            QuantumGate::depolarizing(0, 0.1),
            QuantumGate::cx(0, 1),
        ]);

        // Exact DM populations
        let sim_dm = Simulator::<Cpu>::f64().with_mode(SimulationMode::DensityMatrix);
        let dm_result = sim_dm.run(&circuit).unwrap();
        let dm_pops = dm_result.state.unwrap().populations();

        // Ensemble with large M (enough to keep all branches)
        let sim_traj = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 100_000,
            measured_qubits: vec![0, 1],
            seed: Some(99),
            max_ensemble: Some(64),
        });
        let traj_result = sim_traj.run(&circuit).unwrap();
        let traj = traj_result.trajectory_data.unwrap();

        // All branches should be explored (1 noise gate × 4 branches = 4)
        assert!(
            (traj.explored_weight - 1.0).abs() < 1e-10,
            "expected full weight coverage, got {}",
            traj.explored_weight,
        );

        // Compare histogram frequencies to DM populations
        let total = traj.n_samples as f64;
        for (outcome, &expected_pop) in dm_pops.iter().enumerate() {
            let count = traj.histogram.get(&(outcome as u64)).copied().unwrap_or(0);
            let freq = count as f64 / total;
            assert!(
                (freq - expected_pop).abs() < 0.02,
                "outcome {outcome}: freq={freq:.4}, expected={expected_pop:.4}"
            );
        }
    }

    #[test]
    fn ensemble_multi_noise_pruning() {
        // Circuit with 2 noise gates: M=2 should prune at each gate,
        // keeping the 2 highest-weight branches out of 4^2=16 total paths.
        let circuit = test_circuit(vec![
            QuantumGate::h(0),
            QuantumGate::depolarizing(0, 0.1),
            QuantumGate::cx(0, 1),
            QuantumGate::depolarizing(1, 0.1),
        ]);

        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 10000,
            measured_qubits: vec![0, 1],
            seed: Some(42),
            max_ensemble: Some(2),
        });
        let result = sim.run(&circuit).unwrap();
        let traj = result.trajectory_data.unwrap();

        // M=2: should have exactly 2 branches after pruning.
        assert_eq!(traj.branches.len(), 2);
        // Branches should be sorted by weight (heaviest first).
        assert!(traj.branches[0].weight >= traj.branches[1].weight);
        // Total explored weight < 1.0 (some branches pruned).
        assert!(
            traj.explored_weight < 1.0,
            "with pruning, explored_weight should be < 1.0, got {}",
            traj.explored_weight,
        );
        // But should be high (dominant branches kept).
        assert!(
            traj.explored_weight > 0.7,
            "explored_weight too low: {}",
            traj.explored_weight,
        );
        // Samples should sum to n_samples.
        assert_eq!(traj.histogram.values().sum::<u64>(), 10000);
    }

    #[test]
    fn ensemble_with_dce_partial_measure() {
        // Circuit on 4 qubits, only measure qubit 0. DCE should eliminate
        // gates that only affect qubits 2,3. Ensemble should still work.
        let mut circuit = QuantumCircuit::new(TEST_QUBITS);
        circuit.add(QuantumGate::h(0));
        circuit.add(QuantumGate::depolarizing(0, 0.1));
        circuit.add(QuantumGate::h(2)); // dead: only affects q2
        circuit.add(QuantumGate::cx(2, 3)); // dead: only affects q2,q3

        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 5000,
            measured_qubits: vec![0],
            seed: Some(42),
            max_ensemble: Some(4),
        });
        let result = sim.run(&circuit).unwrap();
        let traj = result.trajectory_data.unwrap();

        // After DCE: only H(0) + depolarizing(0) survive → 4 branches.
        assert_eq!(traj.branches.len(), 4);
        assert!((traj.explored_weight - 1.0).abs() < 1e-10);

        // Measurement on qubit 0 after H: should be ~50/50 (noise is
        // small perturbation from identity).
        let total = traj.n_samples as f64;
        let count_0 = traj.histogram.get(&0).copied().unwrap_or(0);
        let count_1 = traj.histogram.get(&1).copied().unwrap_or(0);
        assert!(
            (count_0 as f64 / total - 0.5).abs() < 0.05,
            "expected ~50% for |0⟩, got {:.1}%",
            100.0 * count_0 as f64 / total,
        );
        assert!(
            (count_1 as f64 / total - 0.5).abs() < 0.05,
            "expected ~50% for |1⟩, got {:.1}%",
            100.0 * count_1 as f64 / total,
        );
    }

    #[test]
    fn ensemble_noiseless_circuit() {
        // Noiseless circuit: no branching needed. M has no effect.
        // ensemble should have exactly 1 branch with weight 1.0.
        let circuit = test_circuit(vec![QuantumGate::h(0), QuantumGate::cx(0, 1)]);

        let sim = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 1000,
            measured_qubits: vec![0, 1],
            seed: Some(42),
            max_ensemble: Some(10),
        });
        let result = sim.run(&circuit).unwrap();
        let traj = result.trajectory_data.unwrap();

        assert_eq!(traj.branches.len(), 1);
        assert!((traj.branches[0].weight - 1.0).abs() < 1e-10);
        assert!((traj.explored_weight - 1.0).abs() < 1e-10);
        // Bell state: only |00⟩ and |11⟩ should appear.
        assert_eq!(traj.histogram.len(), 2);
        assert!(traj.histogram.contains_key(&0)); // |00⟩
        assert!(traj.histogram.contains_key(&3)); // |11⟩
    }

    #[test]
    fn ensemble_multi_noise_matches_dm() {
        // Deeper noisy circuit with 3 noise gates. With M large enough to
        // hold all 4^3=64 branches, ensemble should match DM exactly.
        let circuit = test_circuit(vec![
            QuantumGate::h(0),
            QuantumGate::depolarizing(0, 0.05),
            QuantumGate::cx(0, 1),
            QuantumGate::depolarizing(0, 0.05),
            QuantumGate::depolarizing(1, 0.05),
        ]);

        // Exact DM
        let sim_dm = Simulator::<Cpu>::f64().with_mode(SimulationMode::DensityMatrix);
        let dm_pops = sim_dm.run(&circuit).unwrap().state.unwrap().populations();

        // Ensemble with M=64 (enough for all 4^3=64 branches)
        let sim_traj = Simulator::<Cpu>::f64().with_mode(SimulationMode::Trajectory {
            n_samples: 200_000,
            measured_qubits: vec![0, 1],
            seed: Some(123),
            max_ensemble: Some(64),
        });
        let traj = sim_traj.run(&circuit).unwrap().trajectory_data.unwrap();

        assert!(
            (traj.explored_weight - 1.0).abs() < 1e-10,
            "all branches should be explored, got {}",
            traj.explored_weight,
        );

        let total = traj.n_samples as f64;
        for (outcome, &expected) in dm_pops.iter().enumerate() {
            let count = traj.histogram.get(&(outcome as u64)).copied().unwrap_or(0);
            let freq = count as f64 / total;
            assert!(
                (freq - expected).abs() < 0.01,
                "outcome {outcome}: freq={freq:.4}, dm={expected:.4}"
            );
        }
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
        assert!(result.state.unwrap().is_pure());
    }

    // ── Measurement sampling tests ─────────────────────────────────────────

    /// Helper: build a 4-qubit SV with Bell state on qubits 0,1:
    /// (|00⟩+|11⟩)/√2 ⊗ |00⟩ via H(0) + CX(0,1).
    /// Uses 4 qubits to satisfy SIMD kernel constraints (n_sv >= n_gate + simd_s).
    fn bell_state_sv() -> CPUStatevector {
        use crate::cpu::{CPUKernelGenSpec, CpuKernelManager};
        let spec = CPUKernelGenSpec::f64();
        let mgr = CpuKernelManager::new();
        let mut sv = CPUStatevector::new(4, spec.precision, spec.simd_width);
        sv.initialize();
        let h = Arc::new(QuantumGate::h(0));
        let cx = Arc::new(QuantumGate::cx(0, 1));
        let kid_h = mgr.generate(&spec, &h).unwrap();
        let kid_cx = mgr.generate(&spec, &cx).unwrap();
        let n_threads = crate::cpu::get_num_threads();
        mgr.apply(kid_h, &mut sv, n_threads).unwrap();
        mgr.apply(kid_cx, &mut sv, n_threads).unwrap();
        sv
    }

    #[test]
    fn marginal_probs_bell_state_measure_q0() {
        let sv = bell_state_sv();
        // 4-qubit SV, Bell on q0,q1. Measuring qubit 0 → 50/50.
        let probs = marginal_probabilities_cpu(&sv, &[0]);
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn marginal_probs_bell_state_measure_q0_q1() {
        let sv = bell_state_sv();
        // 4-qubit SV, Bell on q0,q1. Measuring qubits 0,1:
        // |00⟩=50%, |01⟩=0%, |10⟩=0%, |11⟩=50%
        let probs = marginal_probabilities_cpu(&sv, &[0, 1]);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!(probs[1].abs() < 1e-10); // |01⟩
        assert!(probs[2].abs() < 1e-10); // |10⟩
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn batch_sample_delta() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        let hist = batch_sample(&probs, 1000, &mut rng);
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[&2], 1000);
    }

    #[test]
    fn batch_sample_uniform() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let hist = batch_sample(&probs, 10000, &mut rng);
        assert_eq!(hist.values().sum::<u64>(), 10000);
        for outcome in 0u64..4 {
            let count = hist.get(&outcome).copied().unwrap_or(0);
            assert!(
                (count as f64 - 2500.0).abs() < 300.0,
                "outcome {outcome}: expected ~2500, got {count}"
            );
        }
    }

    // ── Dead gate elimination tests ────────────────────────────────────────

    #[test]
    fn dce_eliminates_independent_qubits() {
        // H(0), H(2), CX(2,3). Measure [0]. H(2) and CX(2,3) are dead.
        let gates: Vec<Arc<QuantumGate>> = vec![
            Arc::new(QuantumGate::h(0)),
            Arc::new(QuantumGate::h(2)),
            Arc::new(QuantumGate::cx(2, 3)),
        ];
        let live = eliminate_dead_gates(&gates, &[0]);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].qubits(), &[0]);
    }

    #[test]
    fn dce_keeps_entangled_chain() {
        // H(0), CX(0,1), H(2). Measure [1].
        // q1 live → CX(0,1) touches q1 → q0 becomes live → H(0) kept.
        // H(2) dead.
        let gates: Vec<Arc<QuantumGate>> = vec![
            Arc::new(QuantumGate::h(0)),
            Arc::new(QuantumGate::cx(0, 1)),
            Arc::new(QuantumGate::h(2)),
        ];
        let live = eliminate_dead_gates(&gates, &[1]);
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].qubits(), &[0]); // H(0)
        assert_eq!(live[1].qubits(), &[0, 1]); // CX(0,1)
    }

    #[test]
    fn dce_keeps_all_when_measuring_all() {
        let gates: Vec<Arc<QuantumGate>> = vec![
            Arc::new(QuantumGate::h(0)),
            Arc::new(QuantumGate::h(1)),
            Arc::new(QuantumGate::cx(0, 1)),
        ];
        let live = eliminate_dead_gates(&gates, &[0, 1]);
        assert_eq!(live.len(), 3);
    }

    #[test]
    fn dce_eliminates_noisy_gates() {
        // H(0), depolarizing(2), measure [0]. Noise on q2 is dead.
        let gates: Vec<Arc<QuantumGate>> = vec![
            Arc::new(QuantumGate::h(0)),
            Arc::new(QuantumGate::depolarizing(2, 0.01)),
        ];
        let live = eliminate_dead_gates(&gates, &[0]);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].qubits(), &[0]);
    }

    #[test]
    fn compress_bits_for_measurement() {
        // 0b1010: bit 0=0, bit 1=1, bit 2=0, bit 3=1
        assert_eq!(compress_bits(0b1010, &[1, 3]), 0b11);
        assert_eq!(compress_bits(0b1010, &[0, 2]), 0b00);
        assert_eq!(compress_bits(0b1010, &[0, 1, 2, 3]), 0b1010);
        assert_eq!(compress_bits(0b1111, &[0, 2]), 0b11);
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

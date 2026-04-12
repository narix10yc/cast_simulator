//! Backend-generic circuit simulator.
//!
//! [`Simulator`] owns long-lived backend state (kernel manager, kernel-gen
//! spec, thread count) and exposes three per-run methods, each returning a
//! distinct result type:
//!
//! - [`Simulator::simulate`] — run once, return the final quantum state.
//! - [`Simulator::sample_trajectory`] — ensemble-sample a noisy circuit,
//!   return a measurement histogram.
//! - [`Simulator::bench`] — compile once, execute adaptively within a time
//!   budget, return raw per-iter timing samples.
//!
//! One simulator can handle many runs; the kernel cache persists across
//! calls so duplicate gates across runs are deduplicated.
//!
//! # Explicit optimization
//!
//! The simulator does not apply fusion. Callers build a [`CircuitGraph`],
//! optionally call [`fusion::optimize`](crate::fusion::optimize), then pass
//! the graph to one of the run methods. This keeps the optimization step
//! visible at the call site.
//!
//! # Example
//!
//! ```no_run
//! use cast::simulator::{Simulator, Cpu, Representation};
//! use cast::types::QuantumCircuit;
//! use cast::CircuitGraph;
//!
//! let circuit = QuantumCircuit::from_qasm(
//!     "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];",
//! ).unwrap();
//! let graph = CircuitGraph::from_circuit(&circuit);
//!
//! let sim = Simulator::<Cpu>::f64();
//! let state = sim.simulate(&graph, Representation::StateVector).unwrap();
//! println!("amplitudes: {:?}", state.amplitudes());
//! ```

mod bench;
mod measure;
mod trajectory;

pub use bench::{PhaseTiming, RunTiming, TimingSource};
pub use trajectory::{ExploredBranch, TrajectoryOpts, TrajectoryResult};

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};
use crate::types::{Complex, QuantumGate};
use crate::CircuitGraph;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaKernelGenSpec, CudaKernelManager, CudaStatevector};

// ── Backend trait ────────────────────────────────────────────────────────────

/// Sealed trait mapping a backend marker to its concrete types and operations.
///
/// Every method is `#[doc(hidden)]`: the Backend trait is an implementation
/// detail of [`Simulator`], not a public extension point.
pub trait Backend: sealed::Sealed + Sized {
    #[doc(hidden)]
    type Sv;
    #[doc(hidden)]
    type KernelId: Copy;
    #[doc(hidden)]
    type Mgr;
    #[doc(hidden)]
    type Spec: Copy;
    /// Per-simulator state needed for apply (e.g., CPU thread count).
    #[doc(hidden)]
    type Extra;

    #[doc(hidden)]
    fn new_manager(spec: Self::Spec) -> Self::Mgr;
    #[doc(hidden)]
    fn default_extra() -> Self::Extra;

    #[doc(hidden)]
    fn new_sv(n_qubits: u32, spec: &Self::Spec) -> Result<Self::Sv>;
    #[doc(hidden)]
    fn init_sv(sv: &mut Self::Sv) -> Result<()>;
    #[doc(hidden)]
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv>;

    /// Record a kernel for `gate` in the manager. Compilation may be lazy
    /// (CPU batch JIT, finalized by [`finalize_compile`](Self::finalize_compile))
    /// or eager (CUDA PTX + cubin).
    #[doc(hidden)]
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId>;

    /// Apply a compiled kernel to a statevector.
    #[doc(hidden)]
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        extra: &Self::Extra,
    ) -> Result<()>;

    /// Drain any pending compile-phase work (CPU batch JIT). CUDA is a no-op.
    #[doc(hidden)]
    fn finalize_compile(mgr: &Self::Mgr) -> Result<()>;

    /// Ensure all queued `apply` calls have completed and the statevector is
    /// readable. CPU is a no-op (apply is synchronous); CUDA calls `sync()`.
    #[doc(hidden)]
    fn sync(mgr: &Self::Mgr) -> Result<()>;

    /// Compute marginal measurement probabilities over `measured_qubits`.
    /// Used by trajectory sampling.
    #[doc(hidden)]
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>>;

    /// Which timing source [`time_one_exec`](Self::time_one_exec) reports
    /// samples from. CPU = wall-clock; CUDA = GPU events.
    #[doc(hidden)]
    const EXEC_TIMING_SOURCE: TimingSource;

    /// Run one iteration of "reset statevector and apply all kernels in
    /// order", returning the measured iteration duration.
    ///
    /// CPU reports wall time; CUDA reports the sum of per-kernel GPU event
    /// times (excluding host launch overhead). Used by [`Simulator::bench`]
    /// inside the adaptive timing loop.
    #[doc(hidden)]
    fn time_one_exec(
        mgr: &Self::Mgr,
        sv: &mut Self::Sv,
        kernel_ids: &[Self::KernelId],
        extra: &Self::Extra,
    ) -> Result<Duration>;
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
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        Ok(sv.clone())
    }
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId> {
        mgr.generate(gate)
    }
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        n_threads: &Self::Extra,
    ) -> Result<()> {
        mgr.apply(id, sv, *n_threads)
    }
    fn finalize_compile(mgr: &Self::Mgr) -> Result<()> {
        mgr.finalize()
    }
    fn sync(_mgr: &Self::Mgr) -> Result<()> {
        Ok(())
    }
    fn marginal_probabilities(sv: &Self::Sv, measured_qubits: &[u32]) -> Result<Vec<f64>> {
        Ok(measure::marginal_probabilities_cpu(sv, measured_qubits))
    }

    const EXEC_TIMING_SOURCE: TimingSource = TimingSource::Wall;

    fn time_one_exec(
        mgr: &Self::Mgr,
        sv: &mut Self::Sv,
        kernel_ids: &[Self::KernelId],
        n_threads: &Self::Extra,
    ) -> Result<Duration> {
        let t0 = Instant::now();
        Self::init_sv(sv)?;
        for &kid in kernel_ids {
            mgr.apply(kid, sv, *n_threads)?;
        }
        Ok(t0.elapsed())
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
    fn clone_sv(sv: &Self::Sv) -> Result<Self::Sv> {
        sv.clone_device()
    }
    fn generate(mgr: &Self::Mgr, gate: &Arc<QuantumGate>) -> Result<Self::KernelId> {
        mgr.generate(gate)
    }
    fn apply(
        mgr: &Self::Mgr,
        id: Self::KernelId,
        sv: &mut Self::Sv,
        _: &Self::Extra,
    ) -> Result<()> {
        mgr.apply(id, sv)
    }
    fn finalize_compile(_mgr: &Self::Mgr) -> Result<()> {
        // CUDA's `generate` compiles eagerly (PTX + cubin JIT). No batch.
        Ok(())
    }
    fn sync(mgr: &Self::Mgr) -> Result<()> {
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

    const EXEC_TIMING_SOURCE: TimingSource = TimingSource::GpuEvents;

    fn time_one_exec(
        mgr: &Self::Mgr,
        sv: &mut Self::Sv,
        kernel_ids: &[Self::KernelId],
        _extra: &Self::Extra,
    ) -> Result<Duration> {
        Self::init_sv(sv)?;
        for &kid in kernel_ids {
            mgr.apply(kid, sv)?;
        }
        let stats = mgr.sync()?;
        Ok(stats.kernels.iter().map(|k| k.gpu_time).sum())
    }
}

// ── QuantumState ─────────────────────────────────────────────────────────────

/// Quantum state returned by [`Simulator::simulate`]: a pure statevector or
/// a density matrix.
///
/// For a density matrix the underlying statevector has `2 * n_physical`
/// qubits (a vectorised ρ), and `self.n_physical` records the logical
/// (physical) qubit count.
pub struct QuantumState<B: Backend> {
    sv: B::Sv,
    n_physical: u32,
    repr: Representation,
}

impl<B: Backend> QuantumState<B> {
    /// Number of physical qubits.
    pub fn n_qubits(&self) -> u32 {
        self.n_physical
    }

    /// Whether this is a pure state (vs density matrix).
    pub fn is_pure(&self) -> bool {
        matches!(self.repr, Representation::StateVector)
    }
}

// CPU-specific inspection methods.
impl QuantumState<Cpu> {
    /// All amplitudes of the underlying statevector.
    pub fn amplitudes(&self) -> Vec<Complex> {
        self.sv.amplitudes()
    }

    /// Single amplitude by index.
    pub fn amp(&self, idx: usize) -> Complex {
        self.sv.amp(idx)
    }

    /// Diagonal populations: `|a_i|²` for pure states, `ρ[i,i]` for DM.
    pub fn populations(&self) -> Vec<f64> {
        match self.repr {
            Representation::StateVector => self
                .sv
                .amplitudes()
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect(),
            Representation::DensityMatrix => {
                let n = self.n_physical as usize;
                let dim = 1usize << n;
                (0..dim).map(|i| self.sv.amp(i | (i << n)).re).collect()
            }
        }
    }

    /// Trace (squared norm for pure states, Tr(ρ) for density matrix).
    pub fn trace(&self) -> f64 {
        match self.repr {
            Representation::StateVector => self.sv.norm_squared(),
            Representation::DensityMatrix => {
                let n = self.n_physical as usize;
                let dim = 1usize << n;
                (0..dim).map(|i| self.sv.amp(i | (i << n)).re).sum()
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl QuantumState<Cuda> {
    /// Download all amplitudes from GPU to host.
    pub fn download_amplitudes(&self) -> Result<Vec<(f64, f64)>> {
        self.sv.download()
    }

    /// Download and compute populations.
    pub fn populations(&self) -> Result<Vec<f64>> {
        let amps = self.sv.download()?;
        match self.repr {
            Representation::StateVector => {
                Ok(amps.iter().map(|(re, im)| re * re + im * im).collect())
            }
            Representation::DensityMatrix => {
                let n = self.n_physical as usize;
                let dim = 1usize << n;
                Ok((0..dim).map(|i| amps[i | (i << n)].0).collect())
            }
        }
    }

    /// Trace (computed on GPU for pure states, downloaded for DM).
    pub fn trace(&self) -> Result<f64> {
        match self.repr {
            Representation::StateVector => self.sv.norm_squared(),
            Representation::DensityMatrix => {
                let n = self.n_physical as usize;
                let dim = 1usize << n;
                let amps = self.sv.download()?;
                Ok((0..dim).map(|i| amps[i | (i << n)].0).sum())
            }
        }
    }
}

// ── Representation ───────────────────────────────────────────────────────────

/// State representation: pure statevector or density matrix. Orthogonal to
/// sampling strategy — [`Simulator::sample_trajectory`] has its own options
/// struct and always operates on statevectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    /// Pure-state simulation on n qubits. Errors if the circuit contains
    /// non-unitary gates.
    StateVector,
    /// Density-matrix simulation on 2n virtual qubits. Supports noisy gates
    /// via superoperator lifting (each gate is transformed into a
    /// superoperator acting on the doubled state space).
    DensityMatrix,
}

// ── Simulator ────────────────────────────────────────────────────────────────

/// Backend-generic simulator. Owns long-lived backend state (kernel manager,
/// spec, thread count) and provides three per-run methods, each returning a
/// distinct result type.
///
/// A single simulator can handle many runs; the kernel cache persists across
/// calls, so duplicate gates are deduplicated across invocations.
pub struct Simulator<B: Backend> {
    pub(crate) mgr: B::Mgr,
    pub(crate) spec: B::Spec,
    pub(crate) extra: B::Extra,
}

// CPU constructors.
impl Simulator<Cpu> {
    /// Create a CPU simulator with the given spec.
    pub fn new(spec: CPUKernelGenSpec) -> Self {
        Self {
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

    /// Per-kernel compilation stats for debugging.
    /// Returns `(id, n_gate_qubits, precision, ptx_virtual_regs, ptx_lines, compile_ms)`.
    pub fn kernel_stats(
        &self,
    ) -> Vec<(
        crate::cuda::CudaKernelId,
        u32,
        crate::cuda::CudaPrecision,
        u32,
        usize,
        f64,
    )> {
        self.mgr.kernel_stats()
    }
}

// Generic methods.
impl<B: Backend> Simulator<B> {
    /// Run `graph` once and return the final quantum state.
    ///
    /// For `Representation::StateVector`, every gate must be unitary.
    /// For `Representation::DensityMatrix`, gates are lifted to superoperators
    /// acting on a 2n-qubit virtual statevector, so noisy gates are supported.
    pub fn simulate(
        &self,
        graph: &CircuitGraph,
        repr: Representation,
    ) -> Result<QuantumState<B>> {
        let n_physical = graph.n_qubits() as u32;
        let (gates, n_sv) = prepare_gates(graph, n_physical, repr)?;

        let kernel_ids = self.compile_batch(&gates)?;

        let mut sv = B::new_sv(n_sv, &self.spec)
            .with_context(|| format!("allocating {n_sv}-qubit statevector"))?;
        B::init_sv(&mut sv)?;
        for (i, &kid) in kernel_ids.iter().enumerate() {
            B::apply(&self.mgr, kid, &mut sv, &self.extra)
                .with_context(|| format!("applying kernel at gate index {i}"))?;
        }
        B::sync(&self.mgr)?;

        Ok(QuantumState {
            sv,
            n_physical,
            repr,
        })
    }

    /// Compile and finalize all kernels for `gates`, returning their ids in
    /// gate order. Shared helper for `simulate` and `bench`; callers that
    /// need compile-time measurement take it themselves.
    pub(crate) fn compile_batch(
        &self,
        gates: &[Arc<QuantumGate>],
    ) -> Result<Vec<B::KernelId>> {
        let mut ids: Vec<B::KernelId> = Vec::with_capacity(gates.len());
        for (i, gate) in gates.iter().enumerate() {
            ids.push(self.compile_one_gate(i, gate)?);
        }
        B::finalize_compile(&self.mgr)?;
        Ok(ids)
    }

    /// Generate a kernel for a single gate, wrapping any error with the
    /// gate's circuit position. Shared between [`compile_batch`] and the
    /// trajectory compile paths.
    pub(crate) fn compile_one_gate(
        &self,
        gate_index: usize,
        gate: &Arc<QuantumGate>,
    ) -> Result<B::KernelId> {
        B::generate(&self.mgr, gate)
            .with_context(|| format!("generating kernel for gate index {gate_index}"))
    }

    pub(crate) fn apply_one(&self, kid: B::KernelId, sv: &mut B::Sv) -> Result<()> {
        B::apply(&self.mgr, kid, sv, &self.extra)
    }
}

/// Extract gates from the graph and perform any representation-specific
/// transformation (e.g., DM superoperator lifting). Returns `(gates, n_sv)`.
///
/// Shared by [`Simulator::simulate`] and [`Simulator::bench`]. Trajectory
/// simulation has its own gate-preparation path in [`trajectory`].
pub(crate) fn prepare_gates(
    graph: &CircuitGraph,
    n_physical: u32,
    repr: Representation,
) -> Result<(Vec<Arc<QuantumGate>>, u32)> {
    let gates = graph.gates_in_row_order();
    match repr {
        Representation::StateVector => {
            for (i, gate) in gates.iter().enumerate() {
                if !gate.is_unitary() {
                    anyhow::bail!(
                        "gate index {i} is non-unitary; Representation::StateVector \
                         requires unitary gates (use DensityMatrix or sample_trajectory \
                         for noisy circuits)"
                    );
                }
            }
            Ok((gates, n_physical))
        }
        Representation::DensityMatrix => {
            let dm_gates: Vec<Arc<QuantumGate>> = gates
                .iter()
                .map(|g| Arc::new(g.to_density_matrix_gate(n_physical as usize)))
                .collect();
            Ok((dm_gates, 2 * n_physical))
        }
    }
}

#[cfg(test)]
mod tests;

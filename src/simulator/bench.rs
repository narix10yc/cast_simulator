//! Benchmarking entry point for [`Simulator`](super::Simulator).
//!
//! [`Simulator::bench`](super::Simulator::bench) compiles a circuit once and
//! adaptively executes it many times, returning raw per-iter samples packaged
//! in a [`RunTiming`]. Unlike [`Simulator::simulate`](super::Simulator::simulate),
//! which runs once and returns a quantum state, `bench` is strictly for
//! performance measurement and does not build a [`QuantumState`](super::QuantumState).

use std::time::Instant;

use anyhow::{Context, Result};

use super::{prepare_gates, Backend, Representation, Simulator};
use crate::timing::{stats_from_samples, time_adaptive_samples_with, TimingStats};
use crate::CircuitGraph;

// ── Timing sample types ──────────────────────────────────────────────────────

/// Where a timing sample came from: wall-clock or hardware events.
///
/// - [`Wall`](TimingSource::Wall): samples measured with `Instant::now()`
///   brackets around the operation. Includes host-side overhead.
/// - [`GpuEvents`](TimingSource::GpuEvents): samples measured via CUDA event
///   elapsed times. Excludes host launch and synchronization overhead —
///   reports the pure device time for the kernel(s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingSource {
    Wall,
    GpuEvents,
}

/// Raw timing samples for one phase of a benchmark run (compile or exec),
/// plus metadata about where they came from.
///
/// Unlike [`TimingStats`], `PhaseTiming` keeps the *raw samples* rather than
/// summary statistics, so downstream consumers (reproducibility scripts,
/// JSON writers) can compute any statistic they want without information
/// loss. A 1-sample phase is represented honestly as a length-1 vector.
#[derive(Clone, Debug)]
pub struct PhaseTiming {
    pub samples_s: Vec<f64>,
    pub source: TimingSource,
}

impl PhaseTiming {
    /// Single-sample phase (e.g., a one-shot compile measurement).
    pub fn single(sample_s: f64, source: TimingSource) -> Self {
        Self {
            samples_s: vec![sample_s],
            source,
        }
    }

    /// Build from a vector of raw samples.
    pub fn from_samples(samples_s: Vec<f64>, source: TimingSource) -> Self {
        Self { samples_s, source }
    }

    /// Number of samples.
    pub fn n(&self) -> usize {
        self.samples_s.len()
    }

    /// Mean of the samples (panics on empty).
    pub fn mean_s(&self) -> f64 {
        debug_assert!(!self.samples_s.is_empty(), "PhaseTiming has no samples");
        let n = self.samples_s.len() as f64;
        self.samples_s.iter().copied().sum::<f64>() / n
    }

    /// Population standard deviation of the samples (0 for 1-sample phases).
    pub fn stddev_s(&self) -> f64 {
        if self.samples_s.len() <= 1 {
            return 0.0;
        }
        let mean = self.mean_s();
        let n = self.samples_s.len() as f64;
        let var = self
            .samples_s
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / n;
        var.sqrt()
    }

    /// Minimum sample (panics on empty).
    pub fn min_s(&self) -> f64 {
        self.samples_s.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Maximum sample (panics on empty).
    pub fn max_s(&self) -> f64 {
        self.samples_s
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Lossy conversion to [`TimingStats`] for display / legacy call sites.
    pub fn to_stats(&self) -> TimingStats {
        stats_from_samples(&self.samples_s)
    }
}

/// Full timing record for one [`Simulator::bench`](super::Simulator::bench)
/// invocation: compile phase (always 1 sample) plus exec phase (N samples
/// from adaptive timing).
#[derive(Clone, Debug)]
pub struct RunTiming {
    pub compile: PhaseTiming,
    pub exec: PhaseTiming,
}

// ── Simulator::bench ─────────────────────────────────────────────────────────

impl<B: Backend> Simulator<B> {
    /// Compile the circuit once, then adaptively execute it within
    /// `exec_budget_s` wall seconds, returning raw per-iter samples.
    ///
    /// The compile phase always produces exactly one sample (wall time of
    /// `generate` + `finalize_compile`). The exec phase produces one or
    /// more samples: CPU backends report wall-clock per-iter durations,
    /// CUDA backends report CUDA-event GPU time summed across all kernels
    /// in the iteration.
    ///
    /// Unlike [`Simulator::simulate`](Self::simulate), `bench` does not
    /// return a [`QuantumState`](super::QuantumState) — it is intended
    /// strictly for performance measurement. For state output, use
    /// `simulate`; for noisy simulation, use
    /// [`sample_trajectory`](Self::sample_trajectory).
    pub fn bench(
        &self,
        graph: &CircuitGraph,
        repr: Representation,
        exec_budget_s: f64,
    ) -> Result<RunTiming> {
        let n_physical = graph.n_qubits() as u32;
        let (gates, n_sv) = prepare_gates(graph, n_physical, repr)?;

        let t0 = Instant::now();
        let kernel_ids = self.compile_batch(&gates)?;
        let compile_s = t0.elapsed().as_secs_f64();

        let mut sv = B::new_sv(n_sv, &self.spec)
            .with_context(|| format!("allocating {n_sv}-qubit statevector"))?;
        let samples_s = time_adaptive_samples_with(
            || B::time_one_exec(&self.mgr, &mut sv, &kernel_ids, &self.extra),
            exec_budget_s,
        )
        .with_context(|| {
            format!(
                "exec phase failed ({} kernels, budget {:.3}s)",
                kernel_ids.len(),
                exec_budget_s
            )
        })?;

        Ok(RunTiming {
            compile: PhaseTiming::single(compile_s, TimingSource::Wall),
            exec: PhaseTiming::from_samples(samples_s, B::EXEC_TIMING_SOURCE),
        })
    }
}

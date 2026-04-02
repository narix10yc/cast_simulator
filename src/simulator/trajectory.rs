//! Trajectory (ensemble) simulation: deterministic branching over noise paths
//! with batch measurement sampling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use super::measure::batch_sample;
use super::{Backend, SimulationResult};
use crate::types::{ComplexSquareMatrix, QuantumGate};

// ── Public types ─────────────────────────────────────────────────────────────

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

// ── Internal types ───────────────────────────────────────────────────────────

/// Pre-compiled noise kernels and branch weights for each gate in the circuit.
/// Unitary gates have `None` entries.
struct NoiseKernelTable<K> {
    kernels: Vec<Option<Vec<K>>>,
    weights: Vec<Option<Vec<f64>>>,
}

/// One member of the ensemble: a statevector, its accumulated probability
/// weight, and the noise branch indices chosen so far.
struct Member<Sv> {
    sv: Sv,
    weight: f64,
    noise_path: Vec<usize>,
}

/// Prepared circuit data ready for trajectory simulation.
pub(crate) struct PreparedCircuit<K> {
    pub gates: Vec<Arc<QuantumGate>>,
    pub kernel_ids: Vec<Option<K>>,
    pub n_sv: u32,
}

// ── Simulator extension ──────────────────────────────────────────────────────

impl<B: Backend> super::Simulator<B> {
    pub(crate) fn execute_trajectory(
        &self,
        prepared: &PreparedCircuit<B::KernelId>,
        measured_qubits: &[u32],
        compile_time_s: f64,
    ) -> Result<SimulationResult<B>> {
        let (n_samples, _, _) = self.trajectory_params();

        let t_noise = Instant::now();
        let noise_table = self.compile_noise_kernels(&prepared.gates)?;
        let total_compile_s = compile_time_s + t_noise.elapsed().as_secs_f64();

        let t_sim = Instant::now();
        let ensemble = self.simulate_ensemble(prepared, &noise_table)?;
        let sim_s = t_sim.elapsed().as_secs_f64();

        let t_sample = Instant::now();
        let (histogram, branches, explored_weight) =
            self.sample_ensemble(&ensemble, measured_qubits)?;
        let sample_s = t_sample.elapsed().as_secs_f64();

        Ok(SimulationResult {
            state: None,
            timing: crate::timing::stats_from_samples(&[sim_s + sample_s]),
            compile_time_s: total_compile_s,
            trajectory_data: Some(TrajectoryResult {
                histogram,
                n_samples,
                branches,
                explored_weight,
            }),
        })
    }

    /// Extract trajectory parameters from `self.mode`.
    /// Returns `(n_samples, seed, max_ensemble)`.
    fn trajectory_params(&self) -> (u64, Option<u64>, usize) {
        match &self.mode {
            super::SimulationMode::Trajectory {
                n_samples,
                seed,
                max_ensemble,
            } => (*n_samples, *seed, max_ensemble.unwrap_or(1).max(1)),
            _ => unreachable!("called in non-trajectory mode"),
        }
    }

    /// Compile deduplicated kernels for every noise branch in the circuit.
    fn compile_noise_kernels(
        &self,
        gates: &[Arc<QuantumGate>],
    ) -> Result<NoiseKernelTable<B::KernelId>> {
        let mut kernels: Vec<Option<Vec<B::KernelId>>> = Vec::with_capacity(gates.len());
        let mut weights: Vec<Option<Vec<f64>>> = Vec::with_capacity(gates.len());
        let mut cache: HashMap<(Vec<u32>, Vec<u8>), B::KernelId> = HashMap::new();

        for gate in gates {
            if let Some(noise) = gate.noise_model() {
                let mut kids = Vec::new();
                for (_, u_noise) in noise.branches() {
                    let composed = u_noise.matmul(gate.matrix());
                    let key = (gate.qubits().to_vec(), matrix_to_bytes(&composed));
                    let kid = match cache.get(&key) {
                        Some(&cached) => cached,
                        None => {
                            let g = Arc::new(QuantumGate::new(composed, gate.qubits().to_vec()));
                            let kid = B::generate(&self.mgr, &self.spec, &g)?;
                            cache.insert(key, kid);
                            kid
                        }
                    };
                    kids.push(kid);
                }
                kernels.push(Some(kids));
                weights.push(Some(noise.branches().iter().map(|(p, _)| *p).collect()));
            } else {
                kernels.push(None);
                weights.push(None);
            }
        }

        Ok(NoiseKernelTable { kernels, weights })
    }

    /// Run the ensemble simulation: apply gates, branch at noise points, and
    /// prune to `max_ensemble` members by weight at each noise gate.
    fn simulate_ensemble(
        &self,
        prepared: &PreparedCircuit<B::KernelId>,
        noise_table: &NoiseKernelTable<B::KernelId>,
    ) -> Result<Vec<Member<B::Sv>>> {
        let (_, _, max_ensemble) = self.trajectory_params();

        let mut sv0 = B::new_sv(prepared.n_sv, &self.spec)?;
        B::init_sv(&mut sv0)?;
        let mut ensemble: Vec<Member<B::Sv>> = vec![Member {
            sv: sv0,
            weight: 1.0,
            noise_path: Vec::new(),
        }];

        for (i, gate) in prepared.gates.iter().enumerate() {
            if gate.is_unitary() {
                for member in &mut ensemble {
                    self.apply_one(prepared.kernel_ids[i].unwrap(), &mut member.sv)?;
                }
            } else {
                let branch_kernels = noise_table.kernels[i].as_ref().unwrap();
                let branch_weights = noise_table.weights[i].as_ref().unwrap();
                let n_branches = branch_weights.len();

                let mut candidates: Vec<(usize, usize, f64)> =
                    Vec::with_capacity(ensemble.len() * n_branches);
                for (m_idx, member) in ensemble.iter().enumerate() {
                    for (b_idx, &bp) in branch_weights.iter().enumerate() {
                        candidates.push((m_idx, b_idx, member.weight * bp));
                    }
                }

                candidates.sort_unstable_by(|a, b| {
                    b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                });
                let kept = candidates.len().min(max_ensemble);
                let candidates = &candidates[..kept];

                if ensemble.len() == 1 && kept == 1 {
                    let (_, b_idx, new_weight) = candidates[0];
                    ensemble[0].weight = new_weight;
                    ensemble[0].noise_path.push(b_idx);
                    self.apply_one(branch_kernels[b_idx], &mut ensemble[0].sv)?;
                } else {
                    B::flush(&self.mgr)?;

                    let mut parent_refs = vec![0usize; ensemble.len()];
                    for &(m_idx, _, _) in candidates {
                        parent_refs[m_idx] += 1;
                    }

                    let mut parents: Vec<Option<Member<B::Sv>>> =
                        ensemble.into_iter().map(Some).collect();

                    let mut new_ensemble: Vec<Member<B::Sv>> = Vec::with_capacity(kept);
                    for &(m_idx, b_idx, new_weight) in candidates {
                        parent_refs[m_idx] -= 1;
                        let is_last_ref = parent_refs[m_idx] == 0;

                        let (mut sv, mut noise_path) = if is_last_ref {
                            let m = parents[m_idx].take().unwrap();
                            (m.sv, m.noise_path)
                        } else {
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
        Ok(ensemble)
    }

    /// Sample measurement outcomes from the final ensemble, distributing
    /// samples across members proportional to their weights.
    fn sample_ensemble(
        &self,
        ensemble: &[Member<B::Sv>],
        measured_qubits: &[u32],
    ) -> Result<(HashMap<u64, u64>, Vec<ExploredBranch>, f64)> {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        let (n_samples, seed, _) = self.trajectory_params();

        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

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

        Ok((histogram, branches, total_weight))
    }
}

/// Reinterpret a complex matrix's data as raw bytes for deduplication keying.
fn matrix_to_bytes(m: &ComplexSquareMatrix) -> Vec<u8> {
    let data = m.data();
    let ptr = data.as_ptr() as *const u8;
    let len = data.len() * std::mem::size_of::<crate::types::Complex>();
    // SAFETY: Complex64 is repr(C) with two f64 fields; we're reading the
    // existing slice as bytes within its lifetime.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

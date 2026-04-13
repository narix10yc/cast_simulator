//! Trajectory (ensemble) simulation: deterministic branching over noise paths
//! with batch measurement sampling.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};

use super::measure::batch_sample;
use super::Backend;
use crate::types::QuantumGate;
use crate::CircuitGraph;

// ── Public types ─────────────────────────────────────────────────────────────

/// Options for [`Simulator::sample_trajectory`](super::Simulator::sample_trajectory).
#[derive(Clone, Debug)]
pub struct TrajectoryOpts {
    /// Qubits to measure at the end of the circuit. The histogram keys are
    /// bitstrings over these qubits in the order given.
    pub measured_qubits: Vec<u32>,
    /// Total number of measurement samples to generate across the ensemble.
    pub n_samples: u64,
    /// RNG seed for measurement sampling. `None` uses system entropy.
    pub seed: Option<u64>,
    /// Maximum number of concurrent statevectors in the ensemble.
    /// `None` defaults to `Some(1)` (single deterministic trajectory —
    /// highest-probability branch only). Larger values explore more noise
    /// paths at the cost of more memory.
    pub max_ensemble: Option<usize>,
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

// ── Simulator::sample_trajectory ─────────────────────────────────────────────

impl<B: Backend> super::Simulator<B> {
    /// Ensemble-sample a noisy circuit.
    ///
    /// Runs the circuit deterministically, branching at each noisy gate into
    /// the top `max_ensemble` highest-weight continuations (pruning the
    /// tail). At the end of the circuit, measurement samples are drawn from
    /// each surviving branch proportionally to its weight.
    ///
    /// There is no [`Representation`](super::Representation) parameter:
    /// trajectory sampling *is* the alternative to density-matrix simulation
    /// for noisy circuits. It always acts on pure statevectors and handles
    /// noise via ensemble branching, so combining it with `DensityMatrix`
    /// would be meaningless.
    ///
    /// The caller is responsible for any upstream optimization
    /// (fusion, dead-gate elimination). For trajectory simulation specifically,
    /// [`QuantumCircuit::eliminate_dead_gates`](crate::types::QuantumCircuit::eliminate_dead_gates)
    /// on the source circuit is recommended — it removes gates that cannot
    /// influence the measured qubits.
    pub fn sample_trajectory(
        &self,
        graph: &CircuitGraph,
        opts: &TrajectoryOpts,
    ) -> Result<TrajectoryResult> {
        let n_physical = graph.n_qubits() as u32;
        let gates = graph.gates_in_row_order();

        // Compile unitary (base) kernels; noisy gates have None entries because
        // their work is expanded into per-branch kernels below.
        let base_ids = self.compile_base_kernels(&gates)?;
        let noise_table = self.compile_noise_kernels(&gates)?;
        B::finalize_compile(&self.mgr)?;

        let max_ensemble = opts.max_ensemble.unwrap_or(1).max(1);
        let ensemble =
            self.simulate_ensemble(n_physical, &gates, &base_ids, &noise_table, max_ensemble)?;

        let (histogram, branches, explored_weight) = self.sample_ensemble(&ensemble, opts)?;

        Ok(TrajectoryResult {
            histogram,
            n_samples: opts.n_samples,
            branches,
            explored_weight,
        })
    }

    /// Compile the unitary gates; noisy gates get `None` entries.
    fn compile_base_kernels(&self, gates: &[Arc<QuantumGate>]) -> Result<Vec<Option<B::KernelId>>> {
        let mut ids = Vec::with_capacity(gates.len());
        for (i, gate) in gates.iter().enumerate() {
            if gate.is_unitary() {
                ids.push(Some(self.compile_one_gate(i, gate)?));
            } else {
                ids.push(None);
            }
        }
        Ok(ids)
    }

    /// Compile kernels for every noise branch in the circuit.
    /// Deduplication is handled by the backend's kernel manager.
    fn compile_noise_kernels(
        &self,
        gates: &[Arc<QuantumGate>],
    ) -> Result<NoiseKernelTable<B::KernelId>> {
        let mut kernels: Vec<Option<Vec<B::KernelId>>> = Vec::with_capacity(gates.len());
        let mut weights: Vec<Option<Vec<f64>>> = Vec::with_capacity(gates.len());

        for (i, gate) in gates.iter().enumerate() {
            if let Some(noise) = gate.noise_model() {
                let mut kids = Vec::with_capacity(noise.branches().len());
                for (branch_idx, (_, kraus)) in noise.branches().iter().enumerate() {
                    let g = Arc::new(QuantumGate::new(kraus.clone(), gate.qubits().to_vec()));
                    let id = self
                        .compile_one_gate(i, &g)
                        .with_context(|| format!("noise branch {branch_idx} for gate index {i}"))?;
                    kids.push(id);
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
        n_physical: u32,
        gates: &[Arc<QuantumGate>],
        base_ids: &[Option<B::KernelId>],
        noise_table: &NoiseKernelTable<B::KernelId>,
        max_ensemble: usize,
    ) -> Result<Vec<Member<B::Sv>>> {
        let mut sv0 = B::new_sv(n_physical, &self.spec)?;
        B::init_sv(&mut sv0)?;
        let mut ensemble: Vec<Member<B::Sv>> = vec![Member {
            sv: sv0,
            weight: 1.0,
            noise_path: Vec::new(),
        }];

        for (i, gate) in gates.iter().enumerate() {
            if gate.is_unitary() {
                let kid = base_ids[i].expect("unitary gate should have a base kernel id");
                for member in &mut ensemble {
                    self.apply_one(kid, &mut member.sv)?;
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
                    B::sync(&self.mgr)?;

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
        B::sync(&self.mgr)?;
        Ok(ensemble)
    }

    /// Sample measurement outcomes from the final ensemble, distributing
    /// samples across members proportional to their weights.
    fn sample_ensemble(
        &self,
        ensemble: &[Member<B::Sv>],
        opts: &TrajectoryOpts,
    ) -> Result<(HashMap<u64, u64>, Vec<ExploredBranch>, f64)> {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        let mut rng: StdRng = match opts.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let total_weight: f64 = ensemble.iter().map(|m| m.weight).sum();
        let mut histogram: HashMap<u64, u64> = HashMap::new();
        let mut branches: Vec<ExploredBranch> = Vec::with_capacity(ensemble.len());
        let mut samples_remaining = opts.n_samples;

        for (idx, member) in ensemble.iter().enumerate() {
            let member_samples = if idx == ensemble.len() - 1 {
                samples_remaining
            } else {
                let s = ((member.weight / total_weight) * opts.n_samples as f64).floor() as u64;
                s.min(samples_remaining)
            };
            samples_remaining = samples_remaining.saturating_sub(member_samples);

            if member_samples > 0 {
                let probs = B::marginal_probabilities(&member.sv, &opts.measured_qubits)?;
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

//! Measurement helpers: marginal probabilities and batch sampling.

use std::collections::HashMap;

use crate::cpu::CPUStatevector;
use crate::types::compress_bits;

/// Convert measured qubit indices (u32) to usize positions for `compress_bits`.
pub(crate) fn qubit_positions(measured_qubits: &[u32]) -> Vec<usize> {
    measured_qubits.iter().map(|&q| q as usize).collect()
}

/// Accumulate marginal probabilities from an iterator of `(re, im)` amplitude
/// pairs into a histogram over the measured qubit positions.
pub(crate) fn accumulate_marginal_probs(
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
pub(crate) fn marginal_probabilities_cpu(sv: &CPUStatevector, measured_qubits: &[u32]) -> Vec<f64> {
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
pub(crate) fn batch_sample(
    probs: &[f64],
    n_samples: u64,
    rng: &mut impl rand::Rng,
) -> HashMap<u64, u64> {
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
    use crate::cpu::{CPUKernelGenSpec, CpuKernelManager};
    use crate::types::QuantumGate;
    use std::sync::Arc;

    /// Helper: build a 6-qubit SV with Bell state on qubits 0,1:
    /// (|00⟩+|11⟩)/√2 ⊗ |0000⟩ via H(0) + CX(0,1).
    fn bell_state_sv() -> CPUStatevector {
        let spec = CPUKernelGenSpec::f64();
        let mgr = CpuKernelManager::new();
        let mut sv = CPUStatevector::new(6, spec.precision, spec.simd_width);
        sv.initialize();
        let h = Arc::new(QuantumGate::h(0));
        let cx = Arc::new(QuantumGate::cx(0, 1));
        let kid_h = mgr.generate_gate(spec, &h).unwrap();
        let kid_cx = mgr.generate_gate(spec, &cx).unwrap();
        let n_threads = crate::cpu::get_num_threads();
        mgr.apply(kid_h, &mut sv, n_threads).unwrap();
        mgr.apply(kid_cx, &mut sv, n_threads).unwrap();
        sv
    }

    #[test]
    fn marginal_probs_bell_state_measure_q0() {
        let sv = bell_state_sv();
        let probs = marginal_probabilities_cpu(&sv, &[0]);
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn marginal_probs_bell_state_measure_q0_q1() {
        let sv = bell_state_sv();
        let probs = marginal_probabilities_cpu(&sv, &[0, 1]);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!(probs[1].abs() < 1e-10); // |01⟩
        assert!(probs[2].abs() < 1e-10); // |10⟩
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
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
}

// ── Hardware profiling ────────────────────────────────────────────────────────

use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};

use crate::cost_model::HardwareProfile;
use crate::types::QuantumGate;

use super::*;

const PROFILE_SEED: u64 = 7;

/// Profiles the hardware by timing JIT gate kernels across a range of
/// arithmetic intensities and fitting a roofline model.
///
/// The resulting [`HardwareProfile`] encodes the knee arithmetic intensity — the boundary
/// between memory-bound and compute-bound gate simulation — suitable for use
/// with [`crate::cost_model::FusionConfig::hardware_adaptive`].
///
/// Thread count is read from `CAST_NUM_THREADS` (see [`get_num_threads`]).
/// **Profile at the thread count you will simulate with**: the compute side of
/// the roofline scales with threads while bandwidth does not, so the knee
/// shifts accordingly.
///
/// The sweep starts with 6 seed points at arithmetic intensity 1, 2, 4, 8, 16, 32. Up to
/// two refinement rounds add at most 3 points each near the estimated knee
/// until the roofline R² exceeds 0.95.
pub fn measure_cpu_profile(spec: &CPUKernelGenSpec) -> anyhow::Result<HardwareProfile> {
    // A 20-qubit statevector is 16 MiB for f64 — large enough to stress DRAM
    // on most systems (typical L3 cache is 8–36 MiB).
    const N_QUBITS_SV: u32 = 28;
    const N_WARMUP: u32 = 3;
    const N_ITERS: u32 = 3;
    const R2_THRESHOLD: f64 = 0.95;
    const MAX_REFINEMENT_ROUNDS: u32 = 2;
    const MAX_NEW_POINTS_PER_ROUND: usize = 3;

    let n_threads = get_num_threads();

    let seed_ais: &[u32] = &[1, 2, 4, 8, 16, 32];
    let mut sweep: Vec<(f64, f64, f64)> = Vec::new(); // (ai, gflops_s, gib_s)
    for &target_ai in seed_ais {
        sweep.push(probe_ai(
            target_ai,
            spec,
            N_QUBITS_SV,
            n_threads,
            N_WARMUP,
            N_ITERS,
        )?);
    }
    sweep.sort_by(|a, b| a.0.total_cmp(&b.0));

    for _ in 0..MAX_REFINEMENT_ROUNDS {
        let (knee, c_slope) = fit_knee(&sweep);
        if roofline_r2(&sweep, knee, c_slope) >= R2_THRESHOLD {
            break;
        }
        // Candidate arithmetic intensities bracketing [knee/2, knee*1.5]; skip duplicates.
        let candidates: Vec<u32> = [knee * 0.5, knee, knee * 1.5]
            .iter()
            .map(|&v| (v.round() as u32).max(1))
            .filter(|&ai| {
                !sweep
                    .iter()
                    .any(|(measured_ai, _, _)| (*measured_ai - ai as f64).abs() < 0.5)
            })
            .take(MAX_NEW_POINTS_PER_ROUND)
            .collect();
        if candidates.is_empty() {
            break;
        }
        for target_ai in candidates {
            sweep.push(probe_ai(
                target_ai,
                spec,
                N_QUBITS_SV,
                n_threads,
                N_WARMUP,
                N_ITERS,
            )?);
        }
        sweep.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    let (knee_ai, _) = fit_knee(&sweep);
    let peak_bw_gib_s = sweep.iter().map(|(_, _, g)| *g).fold(0.0_f64, f64::max);
    Ok(HardwareProfile::from_fit(knee_ai, peak_bw_gib_s))
}

/// Times a JIT kernel for a synthetic gate with the given target arithmetic
/// intensity.
///
/// Returns `(actual_ai, gflops_s, gib_s)`.
fn probe_ai(
    target_ai: u32,
    spec: &CPUKernelGenSpec,
    n_qubits_sv: u32,
    n_threads: u32,
    n_warmup: u32,
    n_iters: u32,
) -> anyhow::Result<(f64, f64, f64)> {
    let k = k_for_target_ai(target_ai);
    let qubits: Vec<u32> = (0..k).collect();
    let max_ai = 2.0 * (1usize << k) as f64;
    let sparsity = (target_ai as f64 / max_ai).clamp(0.0, 1.0);
    let mut rng = StdRng::seed_from_u64(PROFILE_SEED ^ target_ai as u64);
    let gate = QuantumGate::random_sparse_with_rng(&qubits, sparsity, &mut rng);
    let actual_ai = gate.arithmatic_intensity(spec.ztol);
    let n_qubits = n_qubits_sv
        .max(gate.n_qubits() as u32 + super::get_simd_s(spec.simd_width, spec.precision) + 1);

    let mut gen = CPUKernelGenerator::new()?;
    let kid = gen.generate(spec, gate.matrix().data(), gate.qubits())?;
    let mut jit = gen.init_jit()?;

    let mut sv = CPUStatevector::new(n_qubits, spec.precision, spec.simd_width);
    sv.initialize();
    for _ in 0..n_warmup {
        jit.apply(kid, &mut sv, n_threads)?;
    }
    let t = Instant::now();
    for _ in 0..n_iters {
        jit.apply(kid, &mut sv, n_threads)?;
    }
    let mean_s = t.elapsed().as_secs_f64() / n_iters as f64;

    let gib_s = 2.0 * sv.byte_len() as f64 / mean_s / (1u64 << 30) as f64;
    let gflops_s = actual_ai * sv.len() as f64 * 2.0 / mean_s / 1e9;
    Ok((actual_ai, gflops_s, gib_s))
}

/// Minimum qubit count `k` such that a random sparse complex gate can reach
/// the target arithmetic intensity with `sparsity <= 1`.
fn k_for_target_ai(target_ai: u32) -> u32 {
    let min_edge_size = ((target_ai as f64) / 2.0).ceil().max(4.0);
    min_edge_size.log2().ceil() as u32
}

/// Fits the roofline model to `(ai, gflops_s, gib_s)` data.
///
/// Returns `(knee_ai, c_slope)` where `c_slope` is GFLOPs/s per unit of arithmetic intensity
/// in the memory-bound regime (slope of the linear region).
fn fit_knee(sweep: &[(f64, f64, f64)]) -> (f64, f64) {
    let n = sweep.len();
    let lower = &sweep[..(n + 1) / 2];
    let mut ratios: Vec<f64> = lower.iter().map(|(ai, g, _)| g / ai).collect();
    ratios.sort_by(|a, b| a.total_cmp(b));
    let c_slope = ratios[ratios.len() / 2];
    let peak_gflops = sweep.iter().map(|(_, g, _)| *g).fold(0.0_f64, f64::max);
    let knee = if c_slope > 0.0 {
        peak_gflops / c_slope
    } else {
        f64::INFINITY
    };
    (knee, c_slope)
}

/// R² of the piecewise roofline fit `gflops_pred = min(c_slope*ai, c_slope*knee)`.
fn roofline_r2(sweep: &[(f64, f64, f64)], knee: f64, c_slope: f64) -> f64 {
    let peak = c_slope * knee;
    let gflops: Vec<f64> = sweep.iter().map(|(_, g, _)| *g).collect();
    let mean = gflops.iter().sum::<f64>() / gflops.len() as f64;
    let ss_tot: f64 = gflops.iter().map(|g| (g - mean).powi(2)).sum();
    if ss_tot < 1e-30 {
        return 1.0;
    }
    let ss_res: f64 = sweep
        .iter()
        .map(|(ai, g, _)| (g - (c_slope * ai).min(peak)).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

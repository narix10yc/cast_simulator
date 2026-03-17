// ── Hardware profiling ────────────────────────────────────────────────────────

use std::time::Instant;

use rand::Rng;

use crate::cost_model::HardwareProfile;
use crate::types::{Complex, ComplexSquareMatrix, QuantumGate};

use super::*;

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
    const N_QUBITS_SV: usize = 20;
    const N_WARMUP: usize = 3;
    const N_ITERS: usize = 15;
    const R2_THRESHOLD: f64 = 0.95;
    const MAX_REFINEMENT_ROUNDS: usize = 2;
    const MAX_NEW_POINTS_PER_ROUND: usize = 3;

    let n_threads = get_num_threads();

    let seed_ais: &[usize] = &[1, 2, 4, 8, 16, 32];
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
        let candidates: Vec<usize> = [knee * 0.5, knee, knee * 1.5]
            .iter()
            .map(|&v| (v.round() as usize).max(1))
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

/// Times a JIT kernel for a synthetic gate with the given sparsity `s`
/// (nonzeros per row → arithmetic intensity = s for real-only matrices).
///
/// Returns `(actual_ai, gflops_s, gib_s)`.
fn probe_ai(
    target_ai: usize,
    spec: &CPUKernelGenSpec,
    n_qubits_sv: usize,
    n_threads: usize,
    n_warmup: usize,
    n_iters: usize,
) -> anyhow::Result<(f64, f64, f64)> {
    let gate = make_sparse_real_gate(k_for_sparsity(target_ai), target_ai);
    let actual_ai = gate.arithmatic_intensity(spec.ztol);
    let n =
        n_qubits_sv.max(gate.n_qubits() + super::get_simd_s(spec.simd_width, spec.precision) + 1);

    let mut gen = CPUKernelGenerator::new()?;
    let kid = gen.generate(spec, gate.matrix().data(), gate.qubits())?;
    let mut jit = gen.init_jit()?;

    let mut sv = CPUStatevector::new(n, spec.precision, spec.simd_width);
    sv.initialize();
    for _ in 0..n_warmup {
        jit.apply(kid, &mut sv, Some(n_threads))?;
    }
    let t = Instant::now();
    for _ in 0..n_iters {
        jit.apply(kid, &mut sv, Some(n_threads))?;
    }
    let mean_s = t.elapsed().as_secs_f64() / n_iters as f64;

    let gib_s = 2.0 * sv.byte_len() as f64 / mean_s / (1u64 << 30) as f64;
    let gflops_s = actual_ai * sv.len() as f64 * 2.0 / mean_s / 1e9;
    Ok((actual_ai, gflops_s, gib_s))
}

/// Minimum qubit count `k` such that a gate with `s` nonzeros per row fits
/// in a `k`-qubit matrix (`2^k >= s`).
fn k_for_sparsity(s: usize) -> usize {
    if s <= 1 {
        return 2;
    }
    ((s as f64).log2().ceil() as usize).max(2)
}

/// Builds a `k`-qubit gate with exactly `s` nonzero real entries per row.
/// Nonzeros at consecutive columns starting at `(row + n/2) % n`, giving
/// `arithmatic_intensity(ztol=0) = s`. Intended for throughput measurement only.
fn make_sparse_real_gate(k: usize, s: usize) -> QuantumGate {
    let n = 1 << k;
    let s = s.min(n);
    let mut rng = rand::thread_rng();
    let mut m = ComplexSquareMatrix::zeros(n);
    for row in 0..n {
        for t in 0..s {
            let col = (row + n / 2 + t) % n;
            let v = rng.gen_range(0.25_f64..1.0);
            m.set(
                row,
                col,
                Complex::new(if rng.gen_bool(0.5) { v } else { -v }, 0.0),
            );
        }
    }
    QuantumGate::new(m, (0..k as u32).collect())
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

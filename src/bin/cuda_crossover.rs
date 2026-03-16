//! Computation-memory balance analysis for CUDA NVPTX gate simulation kernels.
//!
//! Mirrors `cpu_crossover` exactly, substituting the CUDA execution backend for
//! the CPU JIT backend.  Sweeps gate size and density (opcount) from purely
//! memory-bound (sparse permutation gates, opcount = 1) to compute-bound
//! (dense multi-qubit unitaries, opcount up to 32), reporting GiB/s, GFLOPs/s,
//! and timing variance side-by-side so the roofline crossover is visible.
//!
//! ## Metrics
//!
//! ```text
//! opcount  = scalar_nnz(M) / edge_size(M)   (re and im counted separately)
//! FLOPs    = opcount × |ψ| × 2              (2 real FLOPs per nonzero scalar)
//! GFLOPs/s = FLOPs / mean_time
//! GiB/s    = 2 × sv_bytes / mean_time       (read + write every amplitude once)
//! CV       = stddev / mean                  (coefficient of variation)
//! ```
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin cuda_crossover --features cuda --release
//! cargo run --bin cuda_crossover --features cuda --release -- --n-qubits 28 --sm 120
//! cargo run --bin cuda_crossover --features cuda --release -- --help
//! ```

use anyhow::Result;
use cast::cuda::{
    CudaCompilationSession, CudaExecSession, CudaKernelGenSpec, CudaKernelGenerator,
    CudaKernelId, CudaPrecision, CudaStatevector,
};
use cast::types::{Complex, ComplexSquareMatrix, QuantumGate};
use rand::Rng;
use std::time::Instant;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Args {
    n_qubits: usize,
    precision: CudaPrecision,
    sm_major: u32,
    sm_minor: u32,
    /// Total wall-time budget for the entire sweep (seconds).
    budget_secs: f64,
    /// Minimum timed iterations per gate regardless of budget.
    min_iters: usize,
    /// Warmup iterations (not timed, not counted toward budget).
    n_warmup: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            n_qubits: 25,
            precision: CudaPrecision::F64,
            sm_major: 8,
            sm_minor: 0,
            budget_secs: 120.0,
            min_iters: 5,
            n_warmup: 3,
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Self::default();
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || {
                it.next()
                    .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
            };
            match flag.as_str() {
                "--n-qubits" => {
                    a.n_qubits = val()?.parse()?;
                }
                "--n-warmup" => {
                    a.n_warmup = val()?.parse()?;
                }
                "--min-iters" => {
                    a.min_iters = val()?.parse()?;
                }
                "--budget-secs" => {
                    a.budget_secs = val()?.parse()?;
                }
                "--sm" => {
                    let s = val()?;
                    let s = s.trim();
                    if s.len() >= 2 {
                        a.sm_major = s[..s.len() - 1].parse()?;
                        a.sm_minor = s[s.len() - 1..].parse()?;
                    } else {
                        anyhow::bail!("--sm expects a value like 80 or 120");
                    }
                }
                "--precision" => {
                    a.precision = match val()?.as_str() {
                        "f32" => CudaPrecision::F32,
                        "f64" => CudaPrecision::F64,
                        s => anyhow::bail!("unknown precision {s:?}; want f32 or f64"),
                    };
                }
                "--help" | "-h" => {
                    println!("Usage: cuda_crossover --features cuda [OPTIONS]");
                    println!();
                    println!("  --n-qubits N      log2 of statevector length  (default: 25 → 512 MiB for f64)");
                    println!("  --sm XY           CUDA SM target, e.g. 80 or 120  (default: 80)");
                    println!("  --precision P     f32 or f64  (default: f64)");
                    println!(
                        "  --budget-secs T   total wall-time budget in seconds  (default: 120)"
                    );
                    println!("  --min-iters N     minimum timed samples per gate  (default: 5)");
                    println!("  --n-warmup N      warmup iterations before timing  (default: 3)");
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag {other:?}; run with --help"),
            }
        }
        Ok(a)
    }

    fn spec(&self) -> CudaKernelGenSpec {
        CudaKernelGenSpec {
            precision: self.precision,
            ztol: match self.precision {
                CudaPrecision::F32 => 1e-6,
                CudaPrecision::F64 => 1e-12,
            },
            otol: match self.precision {
                CudaPrecision::F32 => 1e-6,
                CudaPrecision::F64 => 1e-12,
            },
            sm_major: self.sm_major,
            sm_minor: self.sm_minor,
        }
    }

    fn scalar_size(&self) -> usize {
        match self.precision {
            CudaPrecision::F32 => 4,
            CudaPrecision::F64 => 8,
        }
    }

    /// Total bytes of device statevector memory: 2 scalars × 2^n_qubits amplitudes.
    fn sv_bytes(&self, n_sv_qubits: usize) -> usize {
        (1usize << n_sv_qubits) * 2 * self.scalar_size()
    }
}

// ── Gate sweep ────────────────────────────────────────────────────────────────

struct Case {
    label: &'static str,
    gate: QuantumGate,
}

fn gate_sweep() -> Vec<Case> {
    vec![
        // ── opcount = 1 ───────────────────────────────────────────────────────
        Case {
            label: "X           [1q dense,  op=1]",
            gate: QuantumGate::x(0),
        },
        Case {
            label: "CX          [2q perm,   op=1]",
            gate: QuantumGate::cx(0, 1),
        },
        Case {
            label: "CCX         [3q perm,   op=1]",
            gate: QuantumGate::ccx(0, 1, 2),
        },
        Case {
            label: "Sparse-4q   [4q sparse, op=1]",
            gate: sparse(4, 1),
        },
        Case {
            label: "Sparse-5q   [5q sparse, op=1]",
            gate: sparse(5, 1),
        },
        // ── opcount = 2 ───────────────────────────────────────────────────────
        Case {
            label: "H           [1q dense,  op=2]",
            gate: QuantumGate::h(0),
        },
        Case {
            label: "Sparse-4q   [4q sparse, op=2]",
            gate: sparse(4, 2),
        },
        Case {
            label: "Sparse-5q   [5q sparse, op=2]",
            gate: sparse(5, 2),
        },
        // ── opcount = 4 ───────────────────────────────────────────────────────
        Case {
            label: "Haar-2q     [2q dense,  op=4]",
            gate: haar(2),
        },
        Case {
            label: "Sparse-4q   [4q sparse, op=4]",
            gate: sparse(4, 4),
        },
        Case {
            label: "Sparse-5q   [5q sparse, op=4]",
            gate: sparse(5, 4),
        },
        // ── opcount = 8 ───────────────────────────────────────────────────────
        Case {
            label: "Haar-3q     [3q dense,  op=8]",
            gate: haar(3),
        },
        Case {
            label: "Sparse-4q   [4q sparse, op=8]",
            gate: sparse(4, 8),
        },
        Case {
            label: "Sparse-5q   [5q sparse, op=8]",
            gate: sparse(5, 8),
        },
        // ── opcount = 16 ──────────────────────────────────────────────────────
        Case {
            label: "Haar-4q     [4q dense, op=16]",
            gate: haar(4),
        },
        Case {
            label: "Sparse-5q   [5q sparse,op=16]",
            gate: sparse(5, 16),
        },
        // ── opcount = 32 ──────────────────────────────────────────────────────
        Case {
            label: "Haar-5q     [5q dense, op=32]",
            gate: haar(5),
        },
    ]
}

fn haar(k: usize) -> QuantumGate {
    let qubits: Vec<u32> = (0..k as u32).collect();
    QuantumGate::new(ComplexSquareMatrix::random_unitary(1 << k), qubits)
}

fn sparse(k: usize, s: usize) -> QuantumGate {
    let n = 1usize << k;
    assert!(s <= n, "s must be ≤ edge_size");
    let stride = n / s;
    let mut rng = rand::thread_rng();
    let mut m = ComplexSquareMatrix::zeros(n);
    for row in 0..n {
        for t in 0..s {
            let col = (row + n / 2 + t * stride) % n;
            let re = rng.gen_range(-1.0_f64..1.0);
            let im = rng.gen_range(-1.0_f64..1.0);
            m.set(row, col, Complex::new(re, im));
        }
    }
    let qubits: Vec<u32> = (0..k as u32).collect();
    QuantumGate::new(m, qubits)
}

// ── Adaptive timing ───────────────────────────────────────────────────────────

struct Timing {
    n_iters: usize,
    mean_s: f64,
    stddev_s: f64,
    cv: f64,
    min_s: f64,
    max_s: f64,
}

/// Runs `n_warmup` un-timed iterations, then collects timed samples until
/// `per_gate_budget_s` is exhausted, with at least `min_iters` samples.
///
/// Each sample times a single `exec.apply()` call which includes a full
/// device synchronisation, so host-side `Instant` is accurate.
fn time_adaptive(
    exec: &CudaExecSession,
    kid: CudaKernelId,
    sv: &mut CudaStatevector,
    n_warmup: usize,
    min_iters: usize,
    per_gate_budget_s: f64,
) -> Timing {
    const N_PROBE: usize = 3;

    for _ in 0..n_warmup {
        exec.apply(kid, sv).unwrap();
    }

    // Phase 1: probe to estimate per-iteration cost.
    let mut samples: Vec<f64> = Vec::new();
    let probe_wall = Instant::now();
    for _ in 0..N_PROBE {
        let t = Instant::now();
        exec.apply(kid, sv).unwrap();
        samples.push(t.elapsed().as_secs_f64());
    }
    let probe_elapsed = probe_wall.elapsed().as_secs_f64();

    // Phase 2: fill remaining budget.
    let est_per_iter = probe_elapsed / N_PROBE as f64;
    let remaining = (per_gate_budget_s - probe_elapsed).max(0.0);
    let n_fill = ((remaining / est_per_iter) as usize).max(min_iters.saturating_sub(N_PROBE));
    for _ in 0..n_fill {
        let t = Instant::now();
        exec.apply(kid, sv).unwrap();
        samples.push(t.elapsed().as_secs_f64());
    }

    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Timing {
        n_iters: samples.len(),
        mean_s: mean,
        stddev_s: stddev,
        cv: stddev / mean,
        min_s: min,
        max_s: max,
    }
}

// ── Row ───────────────────────────────────────────────────────────────────────

struct Row {
    label: String,
    k: usize,
    nnz: usize,
    opcount: f64,
    timing: Timing,
    gib_s: f64,
    gflops_s: f64,
}

fn measure(case: &Case, args: &Args, per_gate_budget_s: f64) -> Result<Row> {
    let spec = args.spec();
    let k = case.gate.n_qubits();
    let opcount = case.gate.opcount(spec.ztol);
    let n = case.gate.matrix().edge_size();
    let scalar_nnz = (opcount * n as f64).round() as usize;

    // CUDA has no SIMD-layout minimum — just ensure the SV is large enough for
    // the gate.
    let n_sv = args.n_qubits.max(k);

    let ffi_matrix: Vec<(f64, f64)> = case
        .gate
        .matrix()
        .data()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();

    let mut gen = CudaKernelGenerator::new()?;
    let kid = gen.generate(&spec, &ffi_matrix, case.gate.qubits())?;
    let session: CudaCompilationSession = gen.compile()?;
    let exec = CudaExecSession::new(&session)?;

    let mut sv = CudaStatevector::new(n_sv as u32, spec.precision)?;
    sv.zero()?;

    let timing = time_adaptive(
        &exec,
        kid,
        &mut sv,
        args.n_warmup,
        args.min_iters,
        per_gate_budget_s,
    );

    let sv_bytes = args.sv_bytes(n_sv);
    let gib_s = 2.0 * sv_bytes as f64 / timing.mean_s / (1u64 << 30) as f64;
    let gflops_s = opcount * (1usize << n_sv) as f64 * 2.0 / timing.mean_s / 1e9;

    Ok(Row {
        label: case.label.to_string(),
        k,
        nnz: scalar_nnz,
        opcount,
        timing,
        gib_s,
        gflops_s,
    })
}

// ── Report ────────────────────────────────────────────────────────────────────

fn print_report(rows: &[Row], args: &Args) {
    let sv_mib = args.sv_bytes(args.n_qubits) / (1 << 20);

    let bar = "═".repeat(100);
    let line = "─".repeat(100);

    println!();
    println!("  {bar}");
    println!("  CUDA Computation-Memory Balance Report");
    println!("  {bar}");
    println!(
        "  statevector  : {} qubits = 2^{} amplitudes = {} MiB  ({:?})",
        args.n_qubits, args.n_qubits, sv_mib, args.precision,
    );
    println!("  target       : sm_{}{}", args.sm_major, args.sm_minor);
    println!(
        "  time budget  : {:.0} s total / {:.1} s per gate",
        args.budget_secs,
        args.budget_secs / rows.len() as f64,
    );
    println!("  FLOPs/scalar : 2 real  (1×mul + 1×add per nonzero scalar component)");
    println!("  opcount      : scalar_nnz(M) / edge_size(M) — re and im parts counted separately");
    println!("  CV           : stddev / mean — scale-free timing noise");
    println!("  note         : each sample includes cuCtxSynchronize; reflects true GPU wall time");
    println!();

    // ── Performance table ─────────────────────────────────────────────────────
    println!(
        "  {:<32}  {:>2}  {:>5}  {:>7}  {:>10}  {:>9}  {:>10}  {:>8}",
        "gate", "k", "snnz", "opcount", "mean (ms)", "GiB/s", "GFLOPs/s", "regime",
    );
    println!("  {line}");

    let first = rows
        .iter()
        .min_by(|a, b| a.opcount.partial_cmp(&b.opcount).unwrap())
        .unwrap();
    let gflops_per_opcount = first.gflops_s / first.opcount;

    let mut prev_regime = "";
    let mut prev_opcount = 0.0_f64;
    let mut crossover: Option<(f64, f64, usize)> = None;

    for row in rows {
        let expected_linear = gflops_per_opcount * row.opcount;
        let regime = if row.gflops_s >= expected_linear * 0.85 {
            "mem-bound"
        } else {
            "compute↑"
        };

        if prev_regime == "mem-bound" && regime == "compute↑" && crossover.is_none() {
            crossover = Some((prev_opcount, row.opcount, row.k));
        }

        println!(
            "  {:<32}  {:>2}  {:>5}  {:>7.1}  {:>10.2}  {:>9.1}  {:>10.1}  {:>8}",
            row.label,
            row.k,
            row.nnz,
            row.opcount,
            row.timing.mean_s * 1e3,
            row.gib_s,
            row.gflops_s,
            regime,
        );
        prev_regime = regime;
        prev_opcount = row.opcount;
    }

    println!("  {line}");

    // ── Variance table ────────────────────────────────────────────────────────
    println!();
    println!(
        "  {:<32}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}  {:>6}",
        "gate", "n_iters", "mean (ms)", "stddev", "min (ms)", "max (ms)", "CV %",
    );
    println!("  {line}");

    for row in rows {
        let t = &row.timing;
        println!(
            "  {:<32}  {:>7}  {:>10.3}  {:>10.3}  {:>10.3}  {:>8.3}  {:>5.2}%",
            row.label,
            t.n_iters,
            t.mean_s * 1e3,
            t.stddev_s * 1e3,
            t.min_s * 1e3,
            t.max_s * 1e3,
            t.cv * 100.0,
        );
    }

    println!("  {line}");

    // ── Summary ───────────────────────────────────────────────────────────────
    let peak_gib = rows.iter().map(|r| r.gib_s).fold(0.0_f64, f64::max);
    let peak_gflops = rows.iter().map(|r| r.gflops_s).fold(0.0_f64, f64::max);
    let knee = peak_gflops / gflops_per_opcount;

    println!();
    println!("  Peak memory bandwidth  : {peak_gib:.1} GiB/s");
    println!("  Peak compute observed  : {peak_gflops:.1} GFLOPs/s");
    println!("  Roofline knee (est.)   : opcount ≈ {knee:.1}");

    if let Some((op_before, op_after, k)) = crossover {
        println!();
        println!(
            "  ⚑ Crossover: mem-bound → compute-bound between opcount {op_before:.0}–{op_after:.0} ({k}-qubit gate)",
        );
    } else {
        println!();
        println!("  ⚑ All measured gates are memory-bound; increase gate size to see crossover.");
    }
    println!();
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse()?;

    let cases = gate_sweep();
    let per_gate_budget = args.budget_secs / cases.len() as f64;
    let sv_mib = args.sv_bytes(args.n_qubits) / (1 << 20);

    eprintln!(
        "CUDA balance sweep: n_qubits={} ({} MiB, {:?}), sm_{}{}",
        args.n_qubits, sv_mib, args.precision, args.sm_major, args.sm_minor,
    );
    eprintln!(
        "Budget: {:.0}s total → {:.1}s per gate  (≥{} iters, {} warmup)",
        args.budget_secs, per_gate_budget, args.min_iters, args.n_warmup,
    );
    eprintln!();

    let mut rows: Vec<Row> = Vec::new();
    for case in &cases {
        eprint!("  {:<40} ", case.label);
        let row = measure(case, &args, per_gate_budget)?;
        eprintln!(
            "{:>6} iters  {:.2} ms ± {:.3} ms  CV={:.2}%  {:>7.1} GiB/s  {:>7.1} GFLOPs/s",
            row.timing.n_iters,
            row.timing.mean_s * 1e3,
            row.timing.stddev_s * 1e3,
            row.timing.cv * 100.0,
            row.gib_s,
            row.gflops_s,
        );
        rows.push(row);
    }

    rows.sort_by(|a, b| a.opcount.partial_cmp(&b.opcount).unwrap());
    print_report(&rows, &args);
    Ok(())
}

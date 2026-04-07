//! Dense vs sparse kernel ablation benchmark.
//!
//! For each input circuit, applies the selected fusion strategy, then runs the
//! fused circuit twice — once with sparsity-aware kernels (default) and once
//! with dense kernels (`ztol=0`).  Reports per-gate sparsity statistics and
//! the speedup from sparsity exploitation.
//!
//! ## Usage
//!
//! ```sh
//! # CUDA backend, hw-adaptive fusion:
//! cargo run --bin bench_ablation --features cuda --release -- \
//!       --backend cuda --profile profiles/cuda_f64.json \
//!       examples/journal_examples/*-30*.qasm
//!
//! # CPU backend, no fusion:
//! cargo run --bin bench_ablation --release -- \
//!       --backend cpu --fusion no-fusion \
//!       examples/journal_examples/qft-cx-24.qasm
//! ```

use anyhow::{Context, Result};
use cast::cost_model::{FusionConfig, HardwareProfile};
use cast::fusion;
use cast::openqasm::parse_qasm;
use cast::timing::{fmt_duration, TimingStats};
use cast::types::QuantumGate;
use cast::CircuitGraph;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "lower")]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "kebab-case")]
enum FusionStrategy {
    NoFusion,
    Default,
    Aggressive,
    HwAdaptive,
}

#[derive(Parser, Debug)]
#[command(
    name = "bench_ablation",
    about = "Dense vs sparse kernel ablation: measures sparsity speedup per circuit."
)]
struct Args {
    /// OpenQASM input files.
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Simulation backend.
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,

    /// Fusion strategy to apply before benchmarking.
    #[arg(long, value_enum, default_value_t = FusionStrategy::HwAdaptive)]
    fusion: FusionStrategy,

    /// Path to a cached HardwareProfile JSON (skips live profiling).
    #[arg(long)]
    profile: Option<PathBuf>,

    /// Number of statevector qubits for profiling (if no --profile is given).
    #[arg(long, default_value_t = 30)]
    profile_qubits: u32,

    /// Profiling time budget in seconds (if no --profile is given).
    #[arg(long, default_value_t = 20.0)]
    profile_budget: f64,

    /// Maximum gate size for hardware-adaptive fusion.
    #[arg(long, default_value_t = 4)]
    max_size: usize,

    /// Time budget per benchmark run in seconds.
    #[arg(long, default_value_t = 5.0)]
    bench_budget: f64,
}

// ── Sparsity statistics ─────────────────────────────────────────────────────

struct SparsityStats {
    n_gates: usize,
    mean_ai: f64,
    min_ai: f64,
    max_ai: f64,
}

fn compute_sparsity_stats(gates: &[Arc<QuantumGate>], ztol: f64) -> SparsityStats {
    let n = gates.len();
    if n == 0 {
        return SparsityStats {
            n_gates: 0,
            mean_ai: 0.0,
            min_ai: 0.0,
            max_ai: 0.0,
        };
    }

    let mut sum_ai = 0.0f64;
    let mut min_ai = f64::MAX;
    let mut max_ai = f64::MIN;

    for gate in gates {
        let edge = gate.matrix().edge_size();
        let nnz = gate.scalar_nnz(ztol);
        let ai = nnz as f64 / edge as f64;
        sum_ai += ai;
        min_ai = min_ai.min(ai);
        max_ai = max_ai.max(ai);
    }

    SparsityStats {
        n_gates: n,
        mean_ai: sum_ai / n as f64,
        min_ai,
        max_ai,
    }
}

// ── CPU runner ───────────────────────────────────────────────────────────────

fn run_cpu(
    gates: &[Arc<QuantumGate>],
    n_qubits: u32,
    budget_s: f64,
    force_dense: bool,
) -> Result<(f64, TimingStats)> {
    use cast::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};

    let mut spec = CPUKernelGenSpec::f64();
    if force_dense {
        spec.ztol = 0.0;
        spec.otol = 0.0;
    }
    let n_threads = get_num_threads();
    let mgr = CpuKernelManager::new(spec);

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in gates {
        kernel_ids.push(mgr.generate(gate)?);
    }
    let compile_s = t0.elapsed().as_secs_f64();

    let mut sv = CPUStatevector::new(n_qubits, spec.precision, spec.simd_width);
    let timing = cast::timing::time_adaptive(
        || -> anyhow::Result<()> {
            sv.initialize();
            for &kid in &kernel_ids {
                mgr.apply(kid, &mut sv, n_threads)?;
            }
            Ok(())
        },
        budget_s,
    )?;

    Ok((compile_s, timing))
}

// ── CUDA runner ──────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn run_cuda(
    gates: &[Arc<QuantumGate>],
    n_qubits: u32,
    budget_s: f64,
    force_dense: bool,
) -> Result<(f64, TimingStats)> {
    use cast::cuda::{
        device_sm, CudaKernelGenSpec, CudaKernelManager, CudaPrecision, CudaStatevector,
    };

    let (sm_major, sm_minor) = device_sm()?;
    let (ztol, otol) = if force_dense {
        (0.0, 0.0)
    } else {
        (1e-12, 1e-12)
    };
    let spec = CudaKernelGenSpec {
        precision: CudaPrecision::F64,
        ztol,
        otol,
        sm_major,
        sm_minor,
    };

    let mgr = CudaKernelManager::new(spec);

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in gates {
        kernel_ids.push(mgr.generate(gate)?);
    }
    let compile_s = t0.elapsed().as_secs_f64();

    let mut sv = CudaStatevector::new(n_qubits, CudaPrecision::F64)?;
    let timing = cast::timing::time_adaptive_with(
        || -> anyhow::Result<std::time::Duration> {
            sv.zero()?;
            for &kid in &kernel_ids {
                mgr.apply(kid, &mut sv)?;
            }
            let stats = mgr.sync()?;
            Ok(stats.kernels.iter().map(|k| k.gpu_time).sum())
        },
        budget_s,
    )?;

    Ok((compile_s, timing))
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(
    _gates: &[Arc<QuantumGate>],
    _n_qubits: u32,
    _budget_s: f64,
    _force_dense: bool,
) -> Result<(f64, TimingStats)> {
    anyhow::bail!("CUDA backend requires the `cuda` feature");
}

fn run_backend(
    gates: &[Arc<QuantumGate>],
    n_qubits: u32,
    backend: Backend,
    budget_s: f64,
    force_dense: bool,
) -> Result<(f64, TimingStats)> {
    match backend {
        Backend::Cpu => run_cpu(gates, n_qubits, budget_s, force_dense),
        Backend::Cuda => run_cuda(gates, n_qubits, budget_s, force_dense),
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Obtain hardware profile ─────────────────────────────────────────────
    let hw_profile = if let Some(ref path) = args.profile {
        eprintln!("Loading cached profile from {}", path.display());
        HardwareProfile::load(path)?
    } else {
        eprintln!("Profiling hardware (budget={:.0}s)...", args.profile_budget);
        match args.backend {
            Backend::Cpu => {
                use cast::cpu::CPUKernelGenSpec;
                cast::profile::measure_cpu(
                    &CPUKernelGenSpec::f64(),
                    args.profile_qubits,
                    args.profile_budget,
                )?
            }
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};
                    let (sm_major, sm_minor) = device_sm()?;
                    cast::profile::measure_cuda(
                        &CudaKernelGenSpec {
                            precision: CudaPrecision::F64,
                            ztol: 1e-12,
                            otol: 1e-12,
                            sm_major,
                            sm_minor,
                        },
                        args.profile_qubits,
                        args.profile_budget,
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                anyhow::bail!("CUDA backend requires the `cuda` feature");
            }
        }
    };

    eprintln!(
        "  Profile: {}  BW={:.1} GiB/s  Compute={:.1} GFLOPs/s  Crossover AI={:.1}",
        hw_profile.config.device,
        hw_profile.peak_bw_gib_s,
        hw_profile.peak_gflops_s,
        hw_profile.crossover_ai,
    );

    let fusion_config = match args.fusion {
        FusionStrategy::NoFusion => FusionConfig::size_only(1),
        FusionStrategy::Default => FusionConfig::default(),
        FusionStrategy::Aggressive => FusionConfig::aggressive(),
        FusionStrategy::HwAdaptive => {
            FusionConfig::hardware_adaptive(&hw_profile, args.max_size, 1e-12)
        }
    };
    eprintln!("  Fusion: {:?}\n", args.fusion);

    // ── Print header ────────────────────────────────────────────────────────
    println!(
        "{:<16} {:>5} {:>5} {:>5}  {:>8} {:>8}  {:>10} {:>10} {:>10}  {:>10} {:>10} {:>8}",
        "Circuit",
        "Qubits",
        "Gates",
        "Depth",
        "AI_mean",
        "AI_range",
        "Sparse",
        "Dense",
        "Speedup",
        "Sparse_JIT",
        "Dense_JIT",
        "JIT_ratio",
    );
    println!("{}", "-".repeat(130));

    // ── Run benchmarks ──────────────────────────────────────────────────────
    for path in &args.files {
        let filename = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        let qasm =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let circuit = parse_qasm(&qasm).with_context(|| format!("parsing {}", path.display()))?;

        let mut graph = CircuitGraph::from_qasm_circuit(&circuit);
        fusion::optimize(&mut graph, &fusion_config);

        let gates = graph.gates_in_row_order();
        let n_qubits = graph.n_qubits() as u32;
        let n_rows = graph.n_rows();

        let stats = compute_sparsity_stats(&gates, 1e-12);

        // Run sparse (default) then dense
        let (sparse_compile, sparse_timing) =
            run_backend(&gates, n_qubits, args.backend, args.bench_budget, false)?;
        let (dense_compile, dense_timing) =
            run_backend(&gates, n_qubits, args.backend, args.bench_budget, true)?;

        let speedup = dense_timing.mean_s / sparse_timing.mean_s;
        let jit_ratio = if sparse_compile > 1e-15 {
            dense_compile / sparse_compile
        } else {
            f64::NAN
        };

        println!(
            "{:<16} {:>5} {:>5} {:>5}  {:>8.2} {:>8}  {:>10} {:>10} {:>10.2}  {:>10} {:>10} {:>8.2}",
            filename,
            n_qubits,
            stats.n_gates,
            n_rows,
            stats.mean_ai,
            format!("{:.1}-{:.1}", stats.min_ai, stats.max_ai),
            sparse_timing,
            dense_timing,
            speedup,
            fmt_duration(sparse_compile),
            fmt_duration(dense_compile),
            jit_ratio,
        );
    }

    Ok(())
}

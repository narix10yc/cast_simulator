//! Benchmarks circuit simulation under different fusion strategies.
//!
//! Loads one or more OpenQASM files, applies fusion with each configured cost
//! model, then runs the fused circuit on the selected backend.  Reports gate
//! counts, fusion depth, compilation time, and execution time for each config.
//!
//! ## Usage
//!
//! ```sh
//! # CUDA backend, all 30-qubit circuits:
//! cargo run --bin bench_fusion --features cuda --release -- \
//!       --backend cuda examples/journal_examples/*-30*.qasm
//!
//! # CPU backend, single circuit:
//! cargo run --bin bench_fusion --release -- \
//!       --backend cpu examples/journal_examples/qft-cx-30.qasm
//!
//! # Load a cached hardware profile instead of re-profiling:
//! cargo run --bin bench_fusion --features cuda --release -- \
//!       --backend cuda --profile profiles/cuda_f64.json \
//!       examples/journal_examples/ala-30.qasm
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

#[derive(Parser, Debug)]
#[command(
    name = "bench_fusion",
    about = "Benchmark circuit simulation with size-only vs hardware-adaptive fusion."
)]
struct Args {
    /// OpenQASM input files.
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Simulation backend.
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,

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

    /// Time budget per benchmark run in seconds.  Adaptive timing splits this
    /// between warmup and measurement automatically.
    #[arg(long, default_value_t = 5.0)]
    bench_budget: f64,

    /// Force dense kernels: set zero-tolerance to 0 so no matrix entries are
    /// skipped.  Useful for ablation studies comparing sparse vs dense kernels.
    #[arg(long)]
    force_dense: bool,

    /// Print a per-decision fusion log for the hw-adaptive config.
    #[arg(long)]
    fusion_log: bool,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gates_in_order(graph: &CircuitGraph) -> Vec<Arc<QuantumGate>> {
    graph.gates_in_row_order()
}

struct BenchResult {
    label: String,
    n_gates: usize,
    n_rows: usize,
    compile_s: f64,
    timing: TimingStats,
}

// ── CPU runner ───────────────────────────────────────────────────────────────

fn run_cpu(
    graph: &CircuitGraph,
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
    let gates = gates_in_order(graph);

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in &gates {
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
    graph: &CircuitGraph,
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
    let gates = gates_in_order(graph);

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in &gates {
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
    _graph: &CircuitGraph,
    _n_qubits: u32,
    _budget_s: f64,
    _force_dense: bool,
) -> Result<(f64, TimingStats)> {
    anyhow::bail!("CUDA backend requires the `cuda` feature");
}

// ── Benchmark logic ──────────────────────────────────────────────────────────

fn bench_one_config(
    label: &str,
    original: &CircuitGraph,
    config: &FusionConfig,
    n_qubits: u32,
    backend: Backend,
    budget_s: f64,
    force_dense: bool,
) -> Result<BenchResult> {
    let mut graph = original.clone();
    fusion::optimize(&mut graph, config);

    let n_gates = gates_in_order(&graph).len();
    let n_rows = graph.n_rows();

    let (compile_s, timing) = match backend {
        Backend::Cpu => run_cpu(&graph, n_qubits, budget_s, force_dense)?,
        Backend::Cuda => run_cuda(&graph, n_qubits, budget_s, force_dense)?,
    };

    Ok(BenchResult {
        label: label.to_string(),
        n_gates,
        n_rows,
        compile_s,
        timing,
    })
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
        "  Profile: {}  BW={:.1} GiB/s  Compute={:.1} GFLOPs/s  Crossover AI={:.1}\n",
        hw_profile.config.device,
        hw_profile.peak_bw_gib_s,
        hw_profile.peak_gflops_s,
        hw_profile.crossover_ai,
    );

    // ── Configs to compare ──────────────────────────────────────────────────
    let configs: Vec<(&str, FusionConfig)> = vec![
        ("no-fusion", FusionConfig::size_only(1)),
        ("default", FusionConfig::default()),
        ("aggressive", FusionConfig::aggressive()),
        (
            "hw-adaptive",
            FusionConfig::hardware_adaptive(&hw_profile, args.max_size),
        ),
    ];

    // ── Print header ────────────────────────────────────────────────────────
    //
    // "Compile" is the cold-compilation wall time (all kernels generated
    // upfront before any execution).  In practice compilation and GPU
    // execution overlap thanks to the LRU module cache, so the actual
    // overhead is just the startup latency of the first few kernels.
    // "GPU time" is pure device time measured via CUDA events.
    println!(
        "{:<16} {:<14} {:>7} {:>6} {:>10} {:>16}",
        "Circuit", "Config", "Gates", "Depth", "Cold-Start", "GPU time"
    );
    println!("{}", "-".repeat(76));

    // ── Run benchmarks ──────────────────────────────────────────────────────
    for path in &args.files {
        let filename = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        let qasm =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let circuit = parse_qasm(&qasm).with_context(|| format!("parsing {}", path.display()))?;
        let graph = CircuitGraph::from_qasm_circuit(&circuit);
        let n_qubits = graph.n_qubits() as u32;

        for (label, config) in &configs {
            if args.fusion_log && *label == "hw-adaptive" {
                let mut g = graph.clone();
                let log = fusion::optimize_with_log(&mut g, config);
                log.print_summary();
            }
            let r = bench_one_config(
                label,
                &graph,
                config,
                n_qubits,
                args.backend,
                args.bench_budget,
                args.force_dense,
            )?;
            println!(
                "{:<16} {:<14} {:>7} {:>6} {:>10} {:>16}",
                filename,
                r.label,
                r.n_gates,
                r.n_rows,
                fmt_duration(r.compile_s),
                r.timing,
            );
        }
        println!();
    }

    Ok(())
}

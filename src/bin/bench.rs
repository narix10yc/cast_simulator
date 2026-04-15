//! Benchmark circuit simulation under one or more fusion strategies.
//!
//! Loads one or more OpenQASM files and, for each requested fusion mode,
//! builds a fused [`CircuitGraph`], runs [`Simulator::bench`], and prints a
//! row with gate count, depth, cold-compile wall time, and adaptively-measured
//! execution time.
//!
//! Fusion modes are selected with `--fusion` (repeatable or comma-separated).
//! The available modes are:
//!
//! - `none` — no fusion (baseline, `size_only(1)`)
//! - `size-only` — size-gated fusion up to `--max-size` qubits
//! - `hw-adaptive` — roofline-adaptive fusion up to `--max-size` qubits,
//!   using the loaded (or freshly profiled) hardware profile
//!
//! ## Usage
//!
//! ```sh
//! # Default: run all three fusion modes on each circuit, CUDA backend:
//! cargo run --bin bench --features cuda --release -- \
//!       --profile profiles/cuda_f64.json \
//!       examples/journal_examples/*-30*.qasm
//!
//! # CPU backend, only hardware-adaptive fusion, max size 4:
//! cargo run --bin bench --release -- \
//!       --backend cpu --fusion hw-adaptive --max-size 4 \
//!       examples/journal_examples/qft-cx-30.qasm
//!
//! # Sparse-vs-dense ablation: run twice with and without --force-dense,
//! # compare rows. Otherwise identical arguments.
//! cargo run --bin bench --features cuda --release -- \
//!       --profile profiles/cuda_f64.json --fusion hw-adaptive \
//!       examples/journal_examples/mexp-17.qasm
//! cargo run --bin bench --features cuda --release -- \
//!       --profile profiles/cuda_f64.json --fusion hw-adaptive --force-dense \
//!       examples/journal_examples/mexp-17.qasm
//! ```

use anyhow::Context;
use cast::cost_model::{FusionConfig, HardwareProfile};
use cast::fusion;
use cast::openqasm::parse_qasm;
use cast::simulator::{Cpu, Representation, Simulator};
use cast::timing::{fmt_duration, TimingStats};
use cast::CircuitGraph;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use cast::simulator::Cuda;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "lower")]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "lower")]
enum CliPrecision {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CliSimd {
    /// 128-bit (SSE2).
    #[value(name = "128")]
    W128,
    /// 256-bit (AVX2).
    #[value(name = "256")]
    W256,
    /// 512-bit (AVX-512).
    #[value(name = "512")]
    W512,
}

impl CliSimd {
    fn to_simd_width(self) -> cast::cpu::SimdWidth {
        match self {
            CliSimd::W128 => cast::cpu::SimdWidth::W128,
            CliSimd::W256 => cast::cpu::SimdWidth::W256,
            CliSimd::W512 => cast::cpu::SimdWidth::W512,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "kebab-case")]
enum FusionMode {
    /// No fusion (baseline, equivalent to `size_only(1)`).
    None,
    /// Size-gated fusion up to `--max-size` qubits.
    SizeOnly,
    /// Roofline-adaptive fusion up to `--max-size` qubits.
    HwAdaptive,
}

impl FusionMode {
    fn label(self) -> &'static str {
        match self {
            FusionMode::None => "none",
            FusionMode::SizeOnly => "size-only",
            FusionMode::HwAdaptive => "hw-adaptive",
        }
    }

    fn build_config(
        self,
        hw_profile: &HardwareProfile,
        max_size: usize,
        zero_tol: f64,
    ) -> FusionConfig {
        match self {
            FusionMode::None => FusionConfig::size_only(1),
            FusionMode::SizeOnly => FusionConfig::size_only(max_size),
            FusionMode::HwAdaptive => {
                FusionConfig::hardware_adaptive(hw_profile, max_size, zero_tol)
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "bench",
    about = "Benchmark CAST circuit simulation under selectable fusion strategies."
)]
struct Args {
    /// OpenQASM input files.
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Simulation backend.
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,

    /// Fusion modes to benchmark (comma-separated or repeat the flag).
    /// Rows are emitted in the order given.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [FusionMode::None, FusionMode::SizeOnly, FusionMode::HwAdaptive],
    )]
    fusion: Vec<FusionMode>,

    /// Path to a cached HardwareProfile JSON (skips live profiling).
    #[arg(long)]
    profile: Option<PathBuf>,

    /// Number of statevector qubits for profiling (if no --profile is given).
    #[arg(long, default_value_t = 30)]
    profile_qubits: u32,

    /// Profiling time budget in seconds (if no --profile is given).
    #[arg(long, default_value_t = 20.0)]
    profile_budget: f64,

    /// Maximum gate size for size-only and hardware-adaptive fusion.
    #[arg(long, default_value_t = 6)]
    max_size: usize,

    /// Time budget per benchmark run in seconds. Adaptive timing splits this
    /// between warmup and measurement automatically.
    #[arg(long, default_value_t = 5.0)]
    bench_budget: f64,

    /// Force dense kernels: set zero-tolerance to 0 so no matrix entries are
    /// skipped. Useful for sparse-vs-dense ablation (run twice, with and
    /// without this flag, then compare rows).
    #[arg(long)]
    force_dense: bool,

    /// Floating-point precision for simulation kernels.
    #[arg(long, value_enum, default_value_t = CliPrecision::F64)]
    precision: CliPrecision,

    /// Print a per-decision fusion log for each hardware-adaptive run.
    #[arg(long)]
    fusion_log: bool,

    /// Print per-kernel compilation stats (gate qubits, PTX registers, lines)
    /// after each circuit. CUDA only.
    #[arg(long)]
    kernel_stats: bool,

    /// CPU worker threads for kernel dispatch. Defaults to 32 (physical cores
    /// on most workstation CPUs). Overrides the CAST_NUM_THREADS env var.
    #[arg(long, default_value_t = 32)]
    threads: u32,

    /// SIMD register width for CPU kernels (128, 256, 512). Defaults to the
    /// widest width supported by the current CPU.
    #[arg(long, value_enum)]
    simd: Option<CliSimd>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct BenchResult {
    label: String,
    n_gates: usize,
    n_rows: usize,
    compile_s: f64,
    timing: TimingStats,
}

// ---------------------------------------------------------------------------
// Runner dispatch
// ---------------------------------------------------------------------------

//
// Each arm constructs a typed `Simulator<B>` and calls `bench_graph`, which
// compiles the circuit once and adaptively executes it within `budget_s`.
// `RunTiming.exec` is converted to `TimingStats` for the legacy
// table-printing code below.

fn run_bench(
    graph: &CircuitGraph,
    backend: Backend,
    precision: CliPrecision,
    budget_s: f64,
    force_dense: bool,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] kernel_stats: bool,
    simd: Option<CliSimd>,
) -> anyhow::Result<(f64, TimingStats)> {
    let rt = match backend {
        Backend::Cpu => {
            use cast::cpu::CPUKernelGenSpec;
            let mut spec = match precision {
                CliPrecision::F32 => CPUKernelGenSpec::f32(),
                CliPrecision::F64 => CPUKernelGenSpec::f64(),
            };
            if let Some(s) = simd {
                spec.simd_width = s.to_simd_width();
            }
            if force_dense {
                spec.ztol = 0.0;
                spec.otol = 0.0;
            }
            let sim = Simulator::<Cpu>::new(spec);
            sim.bench(graph, Representation::StateVector, budget_s)
                .context("CPU backend bench failed")?
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};
                let (sm_major, sm_minor) =
                    device_sm().context("querying CUDA device SM version")?;
                let cuda_prec = match precision {
                    CliPrecision::F32 => CudaPrecision::F32,
                    CliPrecision::F64 => CudaPrecision::F64,
                };
                let default_tol = match precision {
                    CliPrecision::F32 => 1e-6,
                    CliPrecision::F64 => 1e-12,
                };
                let (ztol, otol) = if force_dense {
                    (0.0, 0.0)
                } else {
                    (default_tol, default_tol)
                };
                let spec = CudaKernelGenSpec {
                    precision: cuda_prec,
                    ztol,
                    otol,
                    sm_major,
                    sm_minor,
                    maxnreg: 128,
                };
                let sim = Simulator::<Cuda>::new(spec);
                let rt = sim
                    .bench(graph, Representation::StateVector, budget_s)
                    .context("CUDA backend bench failed")?;
                if kernel_stats {
                    print_kernel_stats(&sim);
                }
                rt
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA backend requires the `cuda` feature");
        }
    };
    Ok((rt.compile.mean_s(), rt.exec.to_stats()))
}

#[cfg(feature = "cuda")]
fn print_kernel_stats(sim: &Simulator<Cuda>) {
    let stats = sim.kernel_stats();
    if stats.is_empty() {
        return;
    }
    eprintln!(
        "  {:>4}  {:>4}  {:>6}  {:>5}  {:>10}",
        "ID", "Qubs", "VRegs", "Lines", "Compile"
    );
    for (id, nq, _prec, regs, lines, compile_ms) in &stats {
        eprintln!(
            "  {:>4}  {:>4}q  {:>5}  {:>5}  {:>8.1}ms",
            id, nq, regs, lines, compile_ms,
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmark logic
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_one_config(
    label: &str,
    original: &CircuitGraph,
    config: &FusionConfig,
    backend: Backend,
    precision: CliPrecision,
    budget_s: f64,
    force_dense: bool,
    kernel_stats: bool,
    simd: Option<CliSimd>,
) -> anyhow::Result<BenchResult> {
    let mut graph = original.clone();
    fusion::optimize(&mut graph, config);

    let n_gates = graph.gates_in_row_order().len();
    let n_rows = graph.n_rows();

    let (compile_s, timing) = run_bench(
        &graph,
        backend,
        precision,
        budget_s,
        force_dense,
        kernel_stats,
        simd,
    )?;

    Ok(BenchResult {
        label: label.to_string(),
        n_gates,
        n_rows,
        compile_s,
        timing,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Publish thread count so profiling and kernel dispatch both see it.
    std::env::set_var("CAST_NUM_THREADS", args.threads.to_string());

    if args.fusion.is_empty() {
        anyhow::bail!("at least one --fusion mode must be given");
    }

    // ── Obtain hardware profile ─────────────────────────────────────────────
    //
    // Always load or measure a profile, even if hw-adaptive is not selected:
    // it is cheap, every row prints the same profile header, and it keeps the
    // output reproducible.
    let hw_profile = if let Some(ref path) = args.profile {
        eprintln!("Loading cached profile from {}", path.display());
        HardwareProfile::load(path)?
    } else {
        eprintln!("Profiling hardware (budget={:.0}s)...", args.profile_budget);
        match args.backend {
            Backend::Cpu => {
                use cast::cpu::CPUKernelGenSpec;
                let mut spec = match args.precision {
                    CliPrecision::F32 => CPUKernelGenSpec::f32(),
                    CliPrecision::F64 => CPUKernelGenSpec::f64(),
                };
                if let Some(s) = args.simd {
                    spec.simd_width = s.to_simd_width();
                }
                cast::profile::measure_cpu(&spec, args.profile_qubits, args.profile_budget)?
            }
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};
                    let (sm_major, sm_minor) = device_sm()?;
                    let cuda_prec = match args.precision {
                        CliPrecision::F32 => CudaPrecision::F32,
                        CliPrecision::F64 => CudaPrecision::F64,
                    };
                    let default_tol = match args.precision {
                        CliPrecision::F32 => 1e-6,
                        CliPrecision::F64 => 1e-12,
                    };
                    cast::profile::measure_cuda(
                        &CudaKernelGenSpec {
                            precision: cuda_prec,
                            ztol: default_tol,
                            otol: default_tol,
                            sm_major,
                            sm_minor,
                            maxnreg: 128,
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

    // ── Build selected fusion configs ────────────────────────────────────────
    let zero_tol = if args.force_dense {
        0.0
    } else {
        match args.precision {
            CliPrecision::F32 => 1e-6,
            CliPrecision::F64 => 1e-12,
        }
    };
    let configs: Vec<(FusionMode, FusionConfig)> = args
        .fusion
        .iter()
        .map(|&mode| {
            (
                mode,
                mode.build_config(&hw_profile, args.max_size, zero_tol),
            )
        })
        .collect();

    // ── Print header ────────────────────────────────────────────────────────
    //
    // "Compile" is the cold-compilation wall time (all kernels generated
    // upfront before any execution). "Exec time" is adaptively sampled:
    // wall-clock on CPU, summed CUDA event times on GPU.
    println!(
        "{:<16} {:<14} {:>7} {:>6} {:>10} {:>22}",
        "Circuit", "Fusion", "Gates", "Depth", "Compile", "Exec time"
    );
    println!("{}", "-".repeat(80));

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

        for (mode, config) in &configs {
            if args.fusion_log && *mode == FusionMode::HwAdaptive {
                let mut g = graph.clone();
                let log = fusion::optimize_with_log(&mut g, config);
                log.print_summary();
            }
            let r = bench_one_config(
                mode.label(),
                &graph,
                config,
                args.backend,
                args.precision,
                args.bench_budget,
                args.force_dense,
                args.kernel_stats,
                args.simd,
            )
            .with_context(|| format!("benchmarking {} with fusion={}", filename, mode.label()))?;
            println!(
                "{:<16} {:<14} {:>7} {:>6} {:>10} {:>22}",
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

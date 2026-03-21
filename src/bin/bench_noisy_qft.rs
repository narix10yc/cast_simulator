//! Performance comparison for noisy QFT simulation with different fusion strategies.
//!
//! Builds an n-qubit QFT circuit with depolarizing noise, lifts it to a
//! density-matrix (DM) circuit on 2n qubits, then times execution under four
//! fusion configs: no-fusion, size-only(2/3/4), and hardware-adaptive.
//!
//! ## Usage
//!
//! ```sh
//! # CPU only (default n=14, 20 s bench budget):
//! cargo run --bin bench_noisy_qft --release
//!
//! # CPU + CUDA, 10-qubit QFT for a quicker run:
//! cargo run --bin bench_noisy_qft --features cuda --release -- -n 10
//!
//! # Provide cached hardware profiles to skip live profiling:
//! cargo run --bin bench_noisy_qft --release -- --cpu-profile profiles/cpu_f64.json
//!
//! # With CUDA, provide both backend-specific profiles:
//! cargo run --bin bench_noisy_qft --features cuda --release -- \
//!       --cpu-profile profiles/cpu_f64.json --cuda-profile profiles/cuda_f64.json
//!
//! # Override noise probability and per-run time budget:
//! cargo run --bin bench_noisy_qft --release -- --noise-p 0.01 --bench-budget 10
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use cast::{
    cost_model::{FusionConfig, HardwareProfile},
    fusion,
    timing::{fmt_duration, time_adaptive, TimingStats},
    types::QuantumGate,
    CircuitGraph,
};
use clap::Parser;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_noisy_qft",
    about = "Benchmark noisy QFT density-matrix simulation across fusion strategies."
)]
struct Args {
    /// Number of physical qubits in the QFT circuit.
    /// The DM statevector has 2n qubits (2^(2n) amplitudes).
    #[arg(short = 'n', long, default_value_t = 14)]
    n_qubits: u32,

    /// Single-qubit depolarizing error probability after each gate.
    #[arg(long, default_value_t = 0.005)]
    noise_p: f64,

    /// Time budget per benchmark run in seconds.  Adaptive timing splits this
    /// between warmup and measurement automatically.
    #[arg(long, default_value_t = 20.0)]
    bench_budget: f64,

    /// Maximum gate size (in virtual qubits) for hardware-adaptive fusion.
    /// DM gates have even qubit counts (2k for k physical qubits), so this
    /// should be even.  Default 6 = fuse up to 3 physical qubits.
    #[arg(long, default_value_t = 6)]
    max_size: usize,

    /// Cached CPU HardwareProfile JSON; skips live CPU profiling when supplied.
    #[arg(long)]
    cpu_profile: Option<std::path::PathBuf>,

    /// Cached CUDA HardwareProfile JSON; skips live CUDA profiling when supplied.
    #[arg(long)]
    cuda_profile: Option<std::path::PathBuf>,

    /// Time budget for hardware profiling in seconds (ignored when a cached
    /// profile is given for that backend).
    #[arg(long, default_value_t = 20.0)]
    profile_budget: f64,

    /// Qubit count used for hardware profiling (ignored when a cached profile
    /// is given).  Defaults to the DM statevector size (2n) for representative
    /// timings.  Falls back to available-memory auto-detection if 2n would OOM.
    #[arg(long)]
    profile_qubits: Option<u32>,
}

// ── Circuit builders ──────────────────────────────────────────────────────────

/// QFT gate list (H + controlled-phases + bit-reversal SWAPs).
fn qft_gates(n: u32) -> Vec<QuantumGate> {
    let mut gates = Vec::new();
    for q in 0..n {
        gates.push(QuantumGate::h(q));
        for k in 1..(n - q) {
            let theta = std::f64::consts::PI / (1u64 << k) as f64;
            gates.push(QuantumGate::cp(theta, q, q + k));
        }
    }
    for q in 0..(n / 2) {
        gates.push(QuantumGate::swap(q, n - 1 - q));
    }
    gates
}

/// Inserts a single-qubit depolarizing channel after every gate application.
fn add_depolarizing_noise(gates: &[QuantumGate], p: f64) -> Vec<QuantumGate> {
    let mut noisy = Vec::with_capacity(gates.len() * 2);
    for gate in gates {
        noisy.push(gate.clone());
        for &q in gate.qubits() {
            noisy.push(QuantumGate::depolarizing(q, p));
        }
    }
    noisy
}

/// Count unitary and channel gates in a list.
fn gate_stats(gates: &[QuantumGate]) -> (usize, usize) {
    let unitary = gates.iter().filter(|g| g.is_unitary()).count();
    let channels = gates.len() - unitary;
    (unitary, channels)
}

/// Build a CircuitGraph from a gate slice.
fn to_graph(gates: &[QuantumGate]) -> CircuitGraph {
    let mut cg = CircuitGraph::new();
    for g in gates {
        cg.insert_gate(Arc::new(g.clone()));
    }
    cg
}

// ── CPU runner ────────────────────────────────────────────────────────────────

fn run_cpu(graph: &CircuitGraph, n_sv: u32, budget_s: f64) -> Result<(f64, TimingStats)> {
    use cast::cpu::{get_num_threads, CPUKernelGenSpec, CPUStatevector, CpuKernelManager};

    let spec = CPUKernelGenSpec::f64();
    let n_threads = get_num_threads();
    let mgr = CpuKernelManager::new();
    let gates = graph.gates_in_row_order();

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in &gates {
        kernel_ids.push(mgr.generate(&spec, gate)?);
    }
    let compile_s = t0.elapsed().as_secs_f64();

    let mut sv = CPUStatevector::new(n_sv, spec.precision, spec.simd_width);
    let timing = time_adaptive(
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

// ── CUDA runner ───────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn run_cuda(graph: &CircuitGraph, n_sv: u32, budget_s: f64) -> Result<(f64, TimingStats)> {
    use cast::{
        cuda::{device_sm, CudaKernelGenSpec, CudaKernelManager, CudaPrecision, CudaStatevector},
        timing::time_adaptive_with,
    };

    let (sm_major, sm_minor) = device_sm()?;
    let spec = CudaKernelGenSpec {
        precision: CudaPrecision::F64,
        ztol: 1e-12,
        otol: 1e-12,
        sm_major,
        sm_minor,
    };

    let mgr = CudaKernelManager::new();
    let gates = graph.gates_in_row_order();

    let t0 = Instant::now();
    let mut kernel_ids = Vec::with_capacity(gates.len());
    for gate in &gates {
        kernel_ids.push(mgr.generate(gate, spec)?);
    }
    let compile_s = t0.elapsed().as_secs_f64();

    let mut sv = CudaStatevector::new(n_sv, CudaPrecision::F64)?;
    let timing = time_adaptive_with(
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
#[allow(dead_code)]
fn run_cuda(_graph: &CircuitGraph, _n_sv: u32, _budget_s: f64) -> Result<(f64, TimingStats)> {
    anyhow::bail!("CUDA backend requires the `cuda` feature")
}

// ── Hardware profiling ────────────────────────────────────────────────────────

/// Resolve the profiling qubit count: use the explicit flag if given,
/// otherwise default to the DM statevector size (n_dm) clamped to what
/// fits in available memory.
fn resolve_profile_qubits(args: &Args, n_dm: u32) -> u32 {
    if let Some(n) = args.profile_qubits {
        return n;
    }
    // Use n_dm so the profile measures at the actual working-set size.
    // Clamp to available memory if n_dm would OOM.
    let scalar_bytes = 8usize; // f64
    match cast::sysinfo::cpu_free_memory_bytes() {
        Some(free) => {
            let max_n = cast::sysinfo::max_feasible_n_qubits(free, scalar_bytes);
            n_dm.min(max_n)
        }
        None => n_dm.min(28), // conservative fallback
    }
}

fn get_cpu_profile(args: &Args, n_dm: u32) -> Result<HardwareProfile> {
    use cast::cpu::CPUKernelGenSpec;
    if let Some(ref p) = args.cpu_profile {
        eprintln!("  Loading cached CPU profile: {}", p.display());
        return HardwareProfile::load(p);
    }
    let pq = resolve_profile_qubits(args, n_dm);
    eprintln!(
        "  Profiling CPU hardware ({pq}-qubit SV, {:.0}s budget)...",
        args.profile_budget
    );
    cast::profile::measure_cpu(&CPUKernelGenSpec::f64(), pq, args.profile_budget)
}

#[cfg(feature = "cuda")]
fn get_cuda_profile(args: &Args, n_dm: u32) -> Result<HardwareProfile> {
    use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};
    if let Some(ref p) = args.cuda_profile {
        eprintln!("  Loading cached CUDA profile: {}", p.display());
        return HardwareProfile::load(p);
    }
    let pq = resolve_profile_qubits(args, n_dm);
    let (sm_major, sm_minor) = device_sm()?;
    eprintln!(
        "  Profiling CUDA hardware ({pq}-qubit SV, {:.0}s budget)...",
        args.profile_budget
    );
    cast::profile::measure_cuda(
        &CudaKernelGenSpec {
            precision: CudaPrecision::F64,
            ztol: 1e-12,
            otol: 1e-12,
            sm_major,
            sm_minor,
        },
        pq,
        args.profile_budget,
    )
}

// ── Table printing ────────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "{:<14} {:>7} {:>6} {:>10} {:>22} {:>8}",
        "Config", "Gates", "Depth", "Compile", "Exec time", "Speedup"
    );
    println!("{}", "-".repeat(72));
}

fn print_row(
    label: &str,
    n_gates: usize,
    n_rows: usize,
    compile_s: f64,
    t: &TimingStats,
    baseline_mean: f64,
) {
    let speedup = if baseline_mean > 0.0 {
        baseline_mean / t.mean_s
    } else {
        1.0
    };
    println!(
        "{:<14} {:>7} {:>6} {:>10} {:>22} {:>7.2}x",
        label,
        n_gates,
        n_rows,
        fmt_duration(compile_s),
        t,
        speedup
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let n = args.n_qubits;
    let n_dm = 2 * n; // DM statevector qubit count
    let sv_gib = if n_dm < 64 {
        (1u64 << n_dm) as f64 * 16.0 / (1u64 << 30) as f64
    } else {
        f64::INFINITY
    };

    // ── Build circuit ────────────────────────────────────────────────────────
    let unitary = qft_gates(n);
    let noisy = add_depolarizing_noise(&unitary, args.noise_p);
    let dm_gates: Vec<QuantumGate> = noisy
        .iter()
        .map(|g| g.to_density_matrix_gate(n as usize))
        .collect();

    let (uni, ch) = gate_stats(&noisy);
    let n_gates_raw = dm_gates.len();

    eprintln!();
    eprintln!(
        "{n}-qubit noisy QFT — density-matrix circuit (n_sv={n_dm}, SV\u{2248}{sv_gib:.1} GiB F64)"
    );
    eprintln!(
        "  Physical circuit: {uni} unitary + {ch} channel gates  \u{2192}  {n_gates_raw} DM gates"
    );
    eprintln!("  Noise: depolarizing p={}\n", args.noise_p);

    let base_graph = to_graph(&dm_gates);

    // ── Fusion configs ───────────────────────────────────────────────────────
    //
    // DM gates always have even virtual-qubit counts (2k for k physical qubits).
    // fusion::optimize always applies absorb-single-qubit + merge-adjacent-2q
    // as a first pass, so size_only(1) and size_only(2) are identical.
    // Step in increments of 2 for meaningful comparisons.
    //
    // hw-adaptive is evaluated per-backend.

    struct FusedCircuit {
        label: String,
        graph: CircuitGraph,
        n_gates: usize,
        n_rows: usize,
    }

    fn fuse(label: &str, base: &CircuitGraph, config: &FusionConfig) -> FusedCircuit {
        let mut g = base.clone();
        fusion::optimize(&mut g, config);
        let n_gates = g.gates_in_row_order().len();
        let n_rows = g.n_rows();
        FusedCircuit {
            label: label.to_string(),
            graph: g,
            n_gates,
            n_rows,
        }
    }

    let unfused_n_gates = base_graph.gates_in_row_order().len();
    let unfused_n_rows = base_graph.n_rows();

    let mut circuits: Vec<FusedCircuit> = vec![
        // "unfused" = raw circuit, no fusion at all.
        FusedCircuit {
            label: "unfused".into(),
            graph: base_graph.clone(),
            n_gates: unfused_n_gates,
            n_rows: unfused_n_rows,
        },
        fuse("fused(4)", &base_graph, &FusionConfig::size_only(4)),
        fuse("fused(6)", &base_graph, &FusionConfig::size_only(6)),
    ];

    // ── CPU ──────────────────────────────────────────────────────────────────
    {
        let hw = get_cpu_profile(&args, n_dm)?;
        eprintln!(
            "  Profile: {}  BW={:.1} GiB/s  Compute={:.1} GFLOPs/s  Crossover AI={:.1}\n",
            hw.config.device, hw.peak_bw_gib_s, hw.peak_gflops_s, hw.crossover_ai,
        );

        let hw_cfg = FusionConfig::hardware_adaptive(&hw, args.max_size);
        circuits.push(fuse("hw-adaptive", &base_graph, &hw_cfg));

        println!("CPU");
        print_header();

        let mut baseline_mean = 0.0_f64;
        for fc in &circuits {
            let (compile_s, timing) = run_cpu(&fc.graph, n_dm, args.bench_budget)?;
            if fc.label == "unfused" {
                baseline_mean = timing.mean_s;
            }
            print_row(
                &fc.label,
                fc.n_gates,
                fc.n_rows,
                compile_s,
                &timing,
                baseline_mean,
            );
        }
        println!();
    }

    // Remove the hw-adaptive entry so we can add a CUDA-specific one below.
    circuits.pop();

    // ── CUDA ─────────────────────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    {
        let hw = get_cuda_profile(&args, n_dm)?;
        eprintln!(
            "  Profile: {}  BW={:.1} GiB/s  Compute={:.1} GFLOPs/s  Crossover AI={:.1}\n",
            hw.config.device, hw.peak_bw_gib_s, hw.peak_gflops_s, hw.crossover_ai,
        );

        let hw_cfg = FusionConfig::hardware_adaptive(&hw, args.max_size);
        circuits.push(fuse("hw-adaptive", &base_graph, &hw_cfg));

        println!("CUDA");
        print_header();

        let mut baseline_mean = 0.0_f64;
        for fc in &circuits {
            let (compile_s, timing) = run_cuda(&fc.graph, n_dm, args.bench_budget)?;
            if fc.label == "unfused" {
                baseline_mean = timing.mean_s;
            }
            print_row(
                &fc.label,
                fc.n_gates,
                fc.n_rows,
                compile_s,
                &timing,
                baseline_mean,
            );
        }
        println!();
    }

    #[cfg(not(feature = "cuda"))]
    eprintln!("(skipped — rerun with --features cuda to include CUDA results)\n");

    Ok(())
}

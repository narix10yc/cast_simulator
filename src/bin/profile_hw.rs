//! Hardware roofline profiler.
//!
//! Measures the roofline crossover point for one or more backend × precision
//! combinations.  By default profiles all available backends at both F32 and
//! F64; use `--backend` and `--precision` to narrow the scope.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin profile_hw --release                          # CPU only
//! cargo run --bin profile_hw --features cuda --release          # CPU + CUDA
//! cargo run --bin profile_hw --features cuda --release -- \
//!       --backend cuda --precision f32                          # CUDA F32 only
//! CAST_NUM_THREADS=32 cargo run --bin profile_hw --release      # override threads
//! cargo run --bin profile_hw --release -- -n 32 --budget 60     # 32-qubit SV, 60s
//! cargo run --bin profile_hw --release -- --save-profiles ./profiles
//!
//! # Merge several independent runs into one consensus profile:
//! cargo run --bin profile_hw --release -- \
//!       --merge run1.json,run2.json,run3.json \
//!       --merge-out profiles/cuda_f64.json
//! ```

use anyhow::Context;
use cast::cost_model::{HardwareProfile, ProfileConfig, SweepEntry};
use cast::mem_bw::{measure_mem_bw, MemBwResult};
use cast::profile;
use cast::sysinfo::{self, MAX_DEFAULT_QUBITS, MIN_DEFAULT_QUBITS};
use cast::timing::fmt_duration;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "lower")]
enum Backend {
    Cpu,
    Cuda,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
#[value(rename_all = "lower")]
enum CliPrecision {
    F32,
    F64,
    All,
}

#[derive(Parser, Debug)]
#[command(
    name = "profile_hw",
    about = "Adaptive roofline profiler — measures the memory/compute crossover for gate simulation kernels."
)]
struct Args {
    /// Which backend(s) to profile.
    #[arg(long, value_enum, default_value_t = Backend::All)]
    backend: Backend,

    /// Which precision(s) to profile.
    #[arg(long, value_enum, default_value_t = CliPrecision::All)]
    precision: CliPrecision,

    /// CPU worker threads; 0 = CAST_NUM_THREADS or logical CPU count.
    #[arg(long = "threads", default_value_t = 0)]
    threads: u32,

    /// Number of statevector qubits.
    ///
    /// Omit to auto-detect: queries free GPU/CPU memory and picks the largest n ≤ 30
    /// whose statevector fits within 80% of available memory.
    #[arg(long, short)]
    n_qubits: Option<u32>,

    /// Wall-time budget per profile run (seconds).
    #[arg(long, default_value_t = 30.0)]
    budget: f64,

    /// Save each HardwareProfile as a separate JSON file in this directory.
    /// Files are named `<backend>_<precision>.json` (e.g. `cpu_f64.json`).
    #[arg(long)]
    save_profiles: Option<PathBuf>,

    /// Merge mode: skip measurement and instead load several existing
    /// HardwareProfile JSON files, concatenate their raw sweep samples, and
    /// produce one consensus profile via a single piecewise-roofline fit.
    ///
    /// All inputs must share the same backend, precision, and n_qubits.
    /// Pair with `--merge-out <path>` to write the merged profile.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    merge: Vec<PathBuf>,

    /// Output path for `--merge` mode.
    #[arg(long)]
    merge_out: Option<PathBuf>,

    /// Skip the direct memory-bandwidth section (CPU only).
    #[arg(long)]
    skip_mem_bw: bool,

    /// Run only the direct memory-bandwidth section; skip the kernel roofline.
    #[arg(long, conflicts_with = "skip_mem_bw")]
    mem_bw_only: bool,

    /// Per-pattern wall-time budget for the direct memory-bandwidth section.
    /// The section runs four patterns, so total cost is ~4× this value.
    #[arg(long, default_value_t = 3.0)]
    mem_bw_budget: f64,
}

impl Args {
    fn do_cpu(&self) -> bool {
        matches!(self.backend, Backend::Cpu | Backend::All)
    }
    fn do_cuda(&self) -> bool {
        matches!(self.backend, Backend::Cuda | Backend::All)
    }
    fn do_f32(&self) -> bool {
        matches!(self.precision, CliPrecision::F32 | CliPrecision::All)
    }
    fn do_f64(&self) -> bool {
        matches!(self.precision, CliPrecision::F64 | CliPrecision::All)
    }
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

const GIB: f64 = (1u64 << 30) as f64;

/// Statevector size in GiB for `n_qubits` complex scalars of `scalar_bytes` each.
fn sv_gib(n_qubits: u32, scalar_bytes: usize) -> f64 {
    (1u64 << n_qubits) as f64 * 2.0 * scalar_bytes as f64 / GIB
}

/// Resolves the effective n_qubits: returns the user-supplied value if given,
/// otherwise queries free memory on every backend being profiled and picks the
/// largest n ≤ 30 that fits (with an 80% safety margin on the worst-case
/// precision, f64).  Prints one informational line per backend queried.
fn resolve_n_qubits(args: &Args) -> u32 {
    if let Some(n) = args.n_qubits {
        return n;
    }

    // Use f64 scalar_bytes (worst case) so the chosen n works for all precisions.
    const SCALAR_BYTES: usize = 8;
    let mut n = MAX_DEFAULT_QUBITS;

    // ── GPU memory ───────────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    if args.do_cuda() {
        match cast::cuda::cuda_free_memory_bytes() {
            Ok((free, total)) => {
                let max_n = sysinfo::max_feasible_n_qubits(free, SCALAR_BYTES);
                println!(
                    "  [auto] GPU memory: {:.1} GiB free / {:.1} GiB total  →  n_qubits ≤ {}",
                    free as f64 / GIB,
                    total as f64 / GIB,
                    max_n,
                );
                if max_n < n {
                    n = max_n;
                }
            }
            Err(e) => {
                eprintln!(
                    "  Warning: could not query GPU memory ({e}).\n  \
                     Defaulting to n_qubits={n}; use -n to set explicitly."
                );
            }
        }
    }

    // ── CPU / system memory ──────────────────────────────────────────────────
    if args.do_cpu() {
        match sysinfo::cpu_free_memory_bytes() {
            Some(free) => {
                let max_n = sysinfo::max_feasible_n_qubits(free, SCALAR_BYTES);
                println!(
                    "  [auto] System memory: {:.1} GiB free  →  n_qubits ≤ {}",
                    free as f64 / GIB,
                    max_n,
                );
                if max_n < n {
                    n = max_n;
                }
            }
            None => {
                eprintln!(
                    "  Warning: could not query system memory.\n  \
                     Defaulting to n_qubits={n}; use -n to set explicitly."
                );
            }
        }
    }

    n = n.max(MIN_DEFAULT_QUBITS);
    println!(
        "  [auto] Selected n_qubits={n}  ({:.1} GiB statevector, F64)\n",
        sv_gib(n, SCALAR_BYTES),
    );
    n
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_short(p: &HardwareProfile, wall_s: f64) {
    println!(
        "  {:<28}  BW {:>7.1} GiB/s  Compute {:>9.1} GFLOPs/s  Crossover AI {:>5.1}  ({:.1}s)",
        p.config.device, p.peak_bw_gib_s, p.peak_gflops_s, p.crossover_ai, wall_s,
    );
}

fn print_table(profiles: &[(HardwareProfile, f64)]) {
    let bar = "=".repeat(92);
    let line = "-".repeat(92);

    println!("\n  {bar}");
    println!("  Summary");
    println!("  {bar}\n");
    println!(
        "  {:<28} {:>12} {:>14} {:>12} {:>12} {:>8}",
        "Config", "Peak BW", "Peak Compute", "BW Slope", "Crossover", "Time"
    );
    println!(
        "  {:<28} {:>12} {:>14} {:>12} {:>12} {:>8}",
        "", "(GiB/s)", "(GFLOPs/s)", "(GFLOPs/AI)", "(AI)", "(s)"
    );
    println!("  {line}");

    for (p, wall_s) in profiles {
        println!(
            "  {:<28} {:>12.1} {:>14.1} {:>12.2} {:>12.1} {:>8.1}",
            format!(
                "{} {}",
                p.config.device,
                p.config.precision.to_ascii_uppercase()
            ),
            p.peak_bw_gib_s,
            p.peak_gflops_s,
            p.bw_slope,
            p.crossover_ai,
            wall_s,
        );
    }

    println!("  {line}\n");
}

// ---------------------------------------------------------------------------
// Direct memory bandwidth
// ---------------------------------------------------------------------------

fn print_mem_bw(r: &MemBwResult) {
    let buf_gib = r.bytes as f64 / GIB;
    println!(
        "  Direct memory bandwidth  ({:.2} GiB buffer, {} threads)",
        buf_gib, r.n_threads,
    );
    let row = |label: &str, traffic_mul: u64, stats: &cast::timing::TimingStats, gib_s: f64| {
        let cv_pct = stats.cv * 100.0;
        println!(
            "    {label:<6} {traffic_mul}× buffer  {:>10}/iter ± {:<8}  cv={cv_pct:>4.1}%   \
             {gib_s:>6.1} GiB/s",
            fmt_duration(stats.mean_s),
            fmt_duration(stats.stddev_s),
        );
    };
    row("read", 1, &r.read, r.read_gib_s());
    row("write", 1, &r.write, r.write_gib_s());
    row("copy", 2, &r.copy, r.copy_gib_s());
    row("rmw", 2, &r.rmw, r.rmw_gib_s());
    println!("    (rmw = in-place read-modify-write — closest analogue to a gate apply)");
}

/// Runs the direct memory-bandwidth measurement and prints it.  Uses an F64-sized
/// statevector buffer (the worst-case precision) at `n_qubits`.
fn run_mem_bw_section(args: &Args, n_qubits: u32) -> anyhow::Result<()> {
    let n_threads = if args.threads > 0 {
        args.threads
    } else {
        cast::cpu::get_num_threads()
    };
    // F64: 2 scalars × 8 bytes per complex amplitude.
    let buf_bytes = (1usize << n_qubits) * 16;

    let t0 = std::time::Instant::now();
    let r = measure_mem_bw(buf_bytes, n_threads, args.mem_bw_budget)?;
    let wall_s = t0.elapsed().as_secs_f64();
    print_mem_bw(&r);
    println!("  ({:.1}s)\n", wall_s);
    Ok(())
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn oom_hint(backend: &str, precision: &str, n_qubits: u32) -> String {
    let scalar_bytes = if precision == "F64" { 8 } else { 4 };
    format!(
        "Failed to allocate {n_qubits}-qubit statevector ({:.1} GiB) for {backend} {precision} profile.\n  \
         Hint: lower the qubit count with -n <N> (e.g. -n {}).",
        sv_gib(n_qubits, scalar_bytes),
        n_qubits.saturating_sub(1),
    )
}

// ---------------------------------------------------------------------------
// Merge mode
// ---------------------------------------------------------------------------

/// Load several HardwareProfile JSON files, validate that they were all
/// measured under the same `ProfileConfig`, concatenate their raw sweep
/// samples, and re-fit the roofline from the pooled data. Prints the merged
/// profile and optionally writes it to `out_path`.
fn run_merge(inputs: &[PathBuf], out_path: Option<&std::path::Path>) -> anyhow::Result<()> {
    anyhow::ensure!(
        inputs.len() >= 2,
        "--merge needs at least two input profiles (got {})",
        inputs.len()
    );

    let mut all_samples: Vec<SweepEntry> = Vec::new();
    let mut shared_config: Option<ProfileConfig> = None;

    for path in inputs {
        let p = HardwareProfile::load(path)
            .with_context(|| format!("loading profile from {}", path.display()))?;
        match shared_config {
            None => shared_config = Some(p.config.clone()),
            Some(ref cfg) => anyhow::ensure!(
                cfg == &p.config,
                "profile {} has mismatched config (expected {}, got {})",
                path.display(),
                cfg.device,
                p.config.device,
            ),
        }
        eprintln!(
            "  loaded {} ({} samples, peak BW {:.1} GiB/s, crossover AI {:.1})",
            path.display(),
            p.raw.len(),
            p.peak_bw_gib_s,
            p.crossover_ai,
        );
        all_samples.extend(p.raw.into_iter());
    }

    let config = shared_config.expect("at least one input means shared_config is set");
    let n_inputs = inputs.len();
    let n_pooled = all_samples.len();

    let merged = profile::fit_from_samples(config, all_samples)
        .context("fitting roofline from merged samples")?;

    eprintln!(
        "\n  Merged {} profiles  →  {} pooled samples",
        n_inputs, n_pooled,
    );
    print_short(&merged, 0.0);

    if let Some(out) = out_path {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        merged
            .save(out)
            .with_context(|| format!("writing merged profile to {}", out.display()))?;
        eprintln!("  Saved merged profile to {}", out.display());
    } else {
        eprintln!("  (use --merge-out <path> to write the merged profile)");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── Merge mode ───────────────────────────────────────────────────────────
    if !args.merge.is_empty() {
        return run_merge(&args.merge, args.merge_out.as_deref());
    }

    let mut profiles: Vec<(HardwareProfile, f64)> = Vec::new();

    if args.threads > 0 {
        std::env::set_var("CAST_NUM_THREADS", args.threads.to_string());
    }

    println!();

    let n_qubits = resolve_n_qubits(&args);

    // ── Direct memory bandwidth (CPU only) ──────────────────────────────────
    //
    // Runs before the kernel roofline so users see the DRAM ceiling first.
    // This measurement is independent of LLVM codegen — pure streaming Rust.

    if args.do_cpu() && !args.skip_mem_bw {
        run_mem_bw_section(&args, n_qubits)?;
    }

    if args.mem_bw_only {
        return Ok(());
    }

    // ── CPU profiles ─────────────────────────────────────────────────────────

    if args.do_cpu() {
        use cast::cpu::CPUKernelGenSpec;

        if args.do_f64() {
            let t0 = std::time::Instant::now();
            let p = profile::measure_cpu(&CPUKernelGenSpec::f64(), n_qubits, args.budget)
                .with_context(|| oom_hint("CPU", "F64", n_qubits))?;
            let wall_s = t0.elapsed().as_secs_f64();
            print_short(&p, wall_s);
            profiles.push((p, wall_s));
        }
        if args.do_f32() {
            let t0 = std::time::Instant::now();
            let p = profile::measure_cpu(&CPUKernelGenSpec::f32(), n_qubits, args.budget)
                .with_context(|| oom_hint("CPU", "F32", n_qubits))?;
            let wall_s = t0.elapsed().as_secs_f64();
            print_short(&p, wall_s);
            profiles.push((p, wall_s));
        }
    }

    // ── CUDA profiles ────────────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    if args.do_cuda() {
        use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};

        let (sm_major, sm_minor) = device_sm()?;

        if args.do_f64() {
            let spec = CudaKernelGenSpec {
                precision: CudaPrecision::F64,
                ztol: 1e-12,
                otol: 1e-12,
                sm_major,
                sm_minor,
                maxnreg: 128,
            };
            let t0 = std::time::Instant::now();
            let p = profile::measure_cuda(&spec, n_qubits, args.budget)
                .with_context(|| oom_hint("CUDA", "F64", n_qubits))?;
            let wall_s = t0.elapsed().as_secs_f64();
            print_short(&p, wall_s);
            profiles.push((p, wall_s));
        }
        if args.do_f32() {
            let spec = CudaKernelGenSpec {
                precision: CudaPrecision::F32,
                ztol: 1e-6,
                otol: 1e-6,
                sm_major,
                sm_minor,
                maxnreg: 128,
            };
            let t0 = std::time::Instant::now();
            let p = profile::measure_cuda(&spec, n_qubits, args.budget)
                .with_context(|| oom_hint("CUDA", "F32", n_qubits))?;
            let wall_s = t0.elapsed().as_secs_f64();
            print_short(&p, wall_s);
            profiles.push((p, wall_s));
        }
    }

    #[cfg(not(feature = "cuda"))]
    if args.do_cuda() {
        eprintln!(
            "Warning: --backend cuda/all requires the `cuda` feature; skipping CUDA profiles."
        );
    }

    // ── Output ───────────────────────────────────────────────────────────────

    if profiles.is_empty() {
        eprintln!("Nothing to profile. Check --backend and --precision flags.");
    } else if profiles.len() > 1 {
        print_table(&profiles);
    } else {
        println!();
    }

    if let Some(ref dir) = args.save_profiles {
        std::fs::create_dir_all(dir)?;
        for (p, _) in &profiles {
            let backend = match &p.config.device {
                cast::cost_model::Device::Cpu { .. } => "cpu",
                cast::cost_model::Device::Cuda { .. } => "cuda",
            };
            let name = format!("{}_{}.json", backend, p.config.precision);
            let path = dir.join(name);
            p.save(&path)?;
            eprintln!("Saved profile to {}", path.display());
        }
    }

    Ok(())
}

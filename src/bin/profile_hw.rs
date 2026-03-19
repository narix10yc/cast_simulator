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
//! ```

use anyhow::Result;
use cast::cost_model::HardwareProfile;
use cast::profile;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

// ── CLI ──────────────────────────────────────────────────────────────────────

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

    /// Number of statevector qubits for profiling kernels.
    #[arg(long, short, default_value_t = 30)]
    n_qubits: u32,

    /// Wall-time budget per profile run (seconds).
    #[arg(long, default_value_t = 30.0)]
    budget: f64,

    /// Save each HardwareProfile as a separate JSON file in this directory.
    /// Files are named `<backend>_<precision>.json` (e.g. `cpu_f64.json`).
    #[arg(long)]
    save_profiles: Option<PathBuf>,
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

// ── Output formatting ────────────────────────────────────────────────────────

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

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let mut profiles: Vec<(HardwareProfile, f64)> = Vec::new();

    if args.threads > 0 {
        std::env::set_var("CAST_NUM_THREADS", args.threads.to_string());
    }

    println!();

    // ── CPU profiles ─────────────────────────────────────────────────────────

    if args.do_cpu() {
        use cast::cpu::CPUKernelGenSpec;

        if args.do_f64() {
            let t0 = std::time::Instant::now();
            let p = profile::measure_cpu(&CPUKernelGenSpec::f64(), args.n_qubits, args.budget)?;
            let wall_s = t0.elapsed().as_secs_f64();
            print_short(&p, wall_s);
            profiles.push((p, wall_s));
        }
        if args.do_f32() {
            let t0 = std::time::Instant::now();
            let p = profile::measure_cpu(&CPUKernelGenSpec::f32(), args.n_qubits, args.budget)?;
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
            };
            let t0 = std::time::Instant::now();
            let p = profile::measure_cuda(&spec, args.n_qubits, args.budget)?;
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
            };
            let t0 = std::time::Instant::now();
            let p = profile::measure_cuda(&spec, args.n_qubits, args.budget)?;
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

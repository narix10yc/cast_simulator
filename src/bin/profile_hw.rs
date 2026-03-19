//! Hardware roofline profiler using `HardwareProfile::measure_cpu` /
//! `measure_cuda`.
//!
//! Measures the roofline crossover point for one or more backend × precision
//! combinations.  By default profiles all available backends at both F32 and
//! F64; use `--backend` and `--precision` to narrow the scope.
//!
//! A one-line result is printed to stdout after each stage completes, followed
//! by a summary table at the end.  Use `--output <path>` to save the results
//! as JSON (`.json`) or plain text.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --bin profile_hw --release                          # CPU only
//! cargo run --bin profile_hw --features cuda --release          # CPU + CUDA
//! cargo run --bin profile_hw --features cuda --release -- \
//!       --backend cuda --precision f32                          # CUDA F32 only
//! CAST_NUM_THREADS=32 cargo run --bin profile_hw --release      # override threads
//! cargo run --bin profile_hw --release -- --output profile.json # save JSON
//! cargo run --bin profile_hw --release -- --help
//! ```

use anyhow::Result;
use cast::cost_model::HardwareProfile;
use cast::cpu::CPUKernelGenSpec;
use clap::{Parser, ValueEnum};
use std::io::Write;
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

    /// Wall-time budget per profile run (seconds).
    #[arg(long, default_value_t = 30.0)]
    budget: f64,

    /// Save results to a file. Format is inferred from the extension:
    ///   .json  → JSON
    ///   other  → human-readable text table
    #[arg(long, short)]
    output: Option<PathBuf>,
}

impl Args {
    fn n_threads(&self) -> u32 {
        if self.threads == 0 {
            cast::cpu::get_num_threads()
        } else {
            self.threads
        }
    }

    fn do_cpu(&self) -> bool {
        matches!(self.backend, Backend::Cpu | Backend::All)
    }

    #[allow(dead_code)]
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

// ── Profile result ───────────────────────────────────────────────────────────

struct ProfileResult {
    backend: String,
    precision: String,
    detail: String,
    peak_bw_gib_s: f64,
    peak_gflops_s: f64,
    gflops_per_ai: f64,
    crossover_ai: f64,
    wall_s: f64,
}

impl ProfileResult {
    fn new(
        backend: &str,
        precision: &str,
        detail: &str,
        p: &HardwareProfile,
        wall_s: f64,
    ) -> Self {
        Self {
            backend: backend.to_string(),
            precision: precision.to_string(),
            detail: detail.to_string(),
            peak_bw_gib_s: p.peak_bw_gib_s(),
            peak_gflops_s: p.peak_gflops_s(),
            gflops_per_ai: p.gflops_per_ai(),
            crossover_ai: p.crossover_ai(),
            wall_s,
        }
    }

    fn label(&self) -> String {
        format!("{} {} ({})", self.backend, self.precision, self.detail)
    }

    fn print_short(&self) {
        println!(
            "  {:<24}  BW {:>7.1} GiB/s  Compute {:>9.1} GFLOPs/s  Crossover AI {:>5.1}  ({:.1}s)",
            self.label(), self.peak_bw_gib_s, self.peak_gflops_s,
            self.crossover_ai, self.wall_s,
        );
    }
}

// ── Output formatting ────────────────────────────────────────────────────────

fn format_table(results: &[ProfileResult]) -> String {
    let bar = "=".repeat(88);
    let line = "-".repeat(88);
    let mut s = String::new();

    s += &format!("\n  {bar}\n");
    s += "  Summary\n";
    s += &format!("  {bar}\n\n");
    s += &format!(
        "  {:<24} {:>12} {:>14} {:>12} {:>12} {:>8}\n",
        "Config", "Peak BW", "Peak Compute", "GFLOPs/AI", "Crossover", "Time"
    );
    s += &format!(
        "  {:<24} {:>12} {:>14} {:>12} {:>12} {:>8}\n",
        "", "(GiB/s)", "(GFLOPs/s)", "", "(AI)", "(s)"
    );
    s += &format!("  {line}\n");

    for r in results {
        s += &format!(
            "  {:<24} {:>12.1} {:>14.1} {:>12.2} {:>12.1} {:>8.1}\n",
            r.label(), r.peak_bw_gib_s, r.peak_gflops_s, r.gflops_per_ai,
            r.crossover_ai, r.wall_s,
        );
    }

    s += &format!("  {line}\n\n");
    s
}

fn format_json(results: &[ProfileResult]) -> String {
    // Hand-rolled JSON to avoid a serde dependency.
    let mut s = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        s += "  {\n";
        s += &format!("    \"backend\": {:?},\n", r.backend);
        s += &format!("    \"precision\": {:?},\n", r.precision);
        s += &format!("    \"detail\": {:?},\n", r.detail);
        s += &format!("    \"peak_bw_gib_s\": {:.2},\n", r.peak_bw_gib_s);
        s += &format!("    \"peak_gflops_s\": {:.2},\n", r.peak_gflops_s);
        s += &format!("    \"gflops_per_ai\": {:.2},\n", r.gflops_per_ai);
        s += &format!("    \"crossover_ai\": {:.2},\n", r.crossover_ai);
        s += &format!("    \"wall_s\": {:.2}\n", r.wall_s);
        s += "  }";
        if i + 1 < results.len() {
            s += ",";
        }
        s += "\n";
    }
    s += "]\n";
    s
}

fn save_results(path: &PathBuf, results: &[ProfileResult]) -> Result<()> {
    let is_json = path
        .extension()
        .map_or(false, |ext| ext.eq_ignore_ascii_case("json"));

    let content = if is_json {
        format_json(results)
    } else {
        format_table(results)
    };

    let mut f = std::fs::File::create(path)?;
    f.write_all(content.as_bytes())?;
    eprintln!("Saved results to {}", path.display());
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let mut results: Vec<ProfileResult> = Vec::new();

    // If threads were overridden, set the env var so measure_cpu picks it up.
    if args.threads > 0 {
        std::env::set_var("CAST_NUM_THREADS", args.threads.to_string());
    }

    let n_threads = args.n_threads();
    let thread_detail = format!("{n_threads} threads");

    println!();

    // ── CPU profiles ─────────────────────────────────────────────────────────

    if args.do_cpu() {
        if args.do_f64() {
            let t0 = std::time::Instant::now();
            let p = HardwareProfile::measure_cpu(&CPUKernelGenSpec::f64(), args.budget)?;
            let r = ProfileResult::new("CPU", "F64", &thread_detail, &p, t0.elapsed().as_secs_f64());
            r.print_short();
            results.push(r);
        }

        if args.do_f32() {
            let t0 = std::time::Instant::now();
            let p = HardwareProfile::measure_cpu(&CPUKernelGenSpec::f32(), args.budget)?;
            let r = ProfileResult::new("CPU", "F32", &thread_detail, &p, t0.elapsed().as_secs_f64());
            r.print_short();
            results.push(r);
        }
    }

    // ── CUDA profiles ────────────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    if args.do_cuda() {
        use cast::cuda::{device_sm, CudaKernelGenSpec, CudaPrecision};

        let (sm_major, sm_minor) = device_sm()?;
        let sm_detail = format!("sm_{sm_major}{sm_minor}");

        if args.do_f64() {
            let spec = CudaKernelGenSpec {
                precision: CudaPrecision::F64,
                ztol: 1e-12,
                otol: 1e-12,
                sm_major,
                sm_minor,
            };
            let t0 = std::time::Instant::now();
            let p = HardwareProfile::measure_cuda(&spec, args.budget)?;
            let r = ProfileResult::new("CUDA", "F64", &sm_detail, &p, t0.elapsed().as_secs_f64());
            r.print_short();
            results.push(r);
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
            let p = HardwareProfile::measure_cuda(&spec, args.budget)?;
            let r = ProfileResult::new("CUDA", "F32", &sm_detail, &p, t0.elapsed().as_secs_f64());
            r.print_short();
            results.push(r);
        }
    }

    #[cfg(not(feature = "cuda"))]
    if args.do_cuda() {
        eprintln!("Warning: --backend cuda/all requires the `cuda` feature; skipping CUDA profiles.");
    }

    // ── Output ───────────────────────────────────────────────────────────────

    if results.is_empty() {
        eprintln!("Nothing to profile. Check --backend and --precision flags.");
    } else if results.len() > 1 {
        print!("{}", format_table(&results));
    } else {
        println!();
    }

    if let Some(ref path) = args.output {
        save_results(path, &results)?;
    }

    Ok(())
}

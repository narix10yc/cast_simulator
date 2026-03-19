mod types;
pub use types::{MatrixLoadMode, SimdWidth};

mod statevector;
pub(crate) use statevector::get_simd_s;
pub use statevector::CPUStatevector;

mod kernel;
pub use kernel::{CPUKernelGenSpec, CpuKernelManager, KernelId};

mod cost_model;
pub use cost_model::measure_cpu_profile;

#[cfg(test)]
mod tests;

// ── Thread count ──────────────────────────────────────────────────────────────

/// Returns the worker thread count for JIT kernel dispatch.
///
/// Reads `CAST_NUM_THREADS` from the environment. Falls back to the logical
/// CPU count (`num_cpus::get()`) when the variable is unset or invalid.
pub fn get_num_threads() -> u32 {
    std::env::var("CAST_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| num_cpus::get() as u32)
}

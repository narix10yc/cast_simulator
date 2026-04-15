mod types;
pub use types::{MatrixLoadMode, SimdWidth};

mod statevector;
pub(crate) use statevector::get_simd_s;
pub use statevector::CPUStatevector;

mod kernel;
pub use kernel::{CPUKernelGenSpec, CpuKernelManager, KernelId};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Thread count
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SIMD width detection
// ---------------------------------------------------------------------------

/// Detects the widest SIMD register width supported by the current CPU.
///
/// Checks x86 feature flags at runtime:
/// - AVX-512F → [`SimdWidth::W512`]
/// - AVX / AVX2 → [`SimdWidth::W256`]
/// - fallback  → [`SimdWidth::W128`] (SSE2, baseline on x86-64)
#[cfg(target_arch = "x86_64")]
pub fn native_simd_width() -> SimdWidth {
    if is_x86_feature_detected!("avx512f") {
        SimdWidth::W512
    } else if is_x86_feature_detected!("avx2") {
        SimdWidth::W256
    } else {
        SimdWidth::W128
    }
}

mod types;
pub use types::{MatrixLoadMode, SimdWidth};

mod statevector;
pub(crate) use statevector::get_simd_s;
pub use statevector::CPUStatevector;

mod kernel;
pub use kernel::{CPUKernelGenSpec, CpuKernelManager, KernelGenRequest, KernelId};

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
/// On x86-64, checks feature flags at runtime (AVX-512F → W512, AVX2 → W256).
/// On aarch64, detects SVE and queries the hardware vector length via `rdvl`;
/// falls back to NEON (128-bit) when SVE is unavailable.
/// On other architectures, returns [`SimdWidth::W128`] as a safe default.
pub fn native_simd_width() -> SimdWidth {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdWidth::W512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdWidth::W256;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sve") {
            let vl_bytes: u64;
            // SAFETY: SVE is confirmed available by the feature check above.
            // `rdvl` reads the hardware vector length — pure, no side effects.
            unsafe {
                core::arch::asm!(
                    ".arch_extension sve",
                    "rdvl {}, #1",
                    out(reg) vl_bytes,
                    options(pure, nomem, nostack),
                );
            }
            let vl_bits = vl_bytes * 8;
            if vl_bits >= 512 {
                return SimdWidth::W512;
            }
            if vl_bits >= 256 {
                return SimdWidth::W256;
            }
        }
    }
    SimdWidth::W128
}

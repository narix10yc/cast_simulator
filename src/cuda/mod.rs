//! CUDA NVPTX kernel generation and GPU execution.
//!
//! The primary workflow is:
//! 1. Create a [`CudaKernelManager`] — no device required.
//! 2. Call [`CudaKernelManager::generate`] for each gate — runs the full LLVM
//!    IR → O1 → PTX pipeline (device-free) and, with the `cuda` feature,
//!    compiles PTX to device-native cubin via the CUDA JIT linker.
//! 3. Call [`CudaKernelManager::apply`] to enqueue kernel launches (non-blocking).
//! 4. Call [`CudaKernelManager::sync`] to wait for all enqueued work to finish.
//! 5. Inspect PTX via [`CudaKernelManager::emit_ptx`] if needed.

mod kernel;
#[cfg(feature = "cuda")]
mod statevector;
mod types;

#[cfg(test)]
mod tests;

use std::ffi::c_char;

pub use kernel::{CudaKernel, CudaKernelManager};
#[cfg(feature = "cuda")]
pub use kernel::{KernelExecTime, SyncStats};
#[cfg(feature = "cuda")]
pub use statevector::CudaStatevector;
pub use types::{CudaKernelGenSpec, CudaKernelId, CudaPrecision};

/// Raw FFI bindings to `src/cpp/cuda/cuda.h`.  See `ffi.rs` for details.
pub(super) mod ffi;

/// Get the CUDA compute capability.
///
/// Returns `(sm_major, sm_minor)`, e.g. `(8, 6)` for sm_86.
/// If the `cuda` feature is disabled, always returns a default of sm_86 (Ampere generation). Useful
/// for non-CUDA builds to test out PTX generation.
/// With the `cuda` feature, delegates to the CUDA driver to queries from device 0.
pub fn device_sm() -> anyhow::Result<(u32, u32)> {
    #[cfg(not(feature = "cuda"))]
    return Ok((8, 6));

    #[cfg(feature = "cuda")]
    {
        let mut err_buf = [0 as c_char; types::ERR_BUF_LEN];
        let mut major: u32 = 0;
        let mut minor: u32 = 0;
        let status = unsafe {
            ffi::cast_cuda_device_sm(&mut major, &mut minor, err_buf.as_mut_ptr(), err_buf.len())
        };
        if status == 0 {
            Ok((major, minor))
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }
}

/// Query free and total GPU device memory in bytes.
///
/// Returns `(free_bytes, total_bytes)`. Requires the `cuda` feature and an
/// initialized CUDA context (device 0).
#[cfg(feature = "cuda")]
pub fn cuda_free_memory_bytes() -> anyhow::Result<(u64, u64)> {
    let mut err_buf = [0 as c_char; types::ERR_BUF_LEN];
    let mut free: u64 = 0;
    let mut total: u64 = 0;
    let status = unsafe {
        ffi::cast_cuda_free_memory(&mut free, &mut total, err_buf.as_mut_ptr(), err_buf.len())
    };
    if status == 0 {
        Ok((free, total))
    } else {
        Err(anyhow::anyhow!(error_from_buf(&err_buf)))
    }
}

pub(super) fn error_from_buf(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

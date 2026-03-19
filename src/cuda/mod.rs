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

/// Mirrors the "Exported to Rust" section of `src/cpp/cuda/cuda.h`.
///
/// Statevector memory functions (`sv_zero`, `sv_upload`, `sv_download`) are
/// declared separately in `statevector.rs::mod ffi`.
pub(super) mod ffi {
    use super::CudaKernelGenSpec;
    use std::ffi::c_char;

    // ── Types ────────────────────────────────────────────────────────────────

    /// `cast_cuda_complex64_t`
    #[repr(C)]
    pub struct FfiComplex64 {
        pub re: f64,
        pub im: f64,
    }

    // ── Functions ────────────────────────────────────────────────────────────

    unsafe extern "C" {
        // -- Gate PTX compilation --
        pub fn cast_cuda_compile_gate_ptx(
            spec: *const CudaKernelGenSpec,
            matrix: *const FfiComplex64,
            matrix_len: usize,
            qubits: *const u32,
            n_qubits: usize,
            out_ptx: *mut *mut c_char,
            out_func_name: *mut *mut c_char,
            out_n_gate_qubits: *mut u32,
            out_precision: *mut u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_str_free(s: *mut c_char);

        // -- Device capability query --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_device_sm(
            out_major: *mut u32,
            out_minor: *mut u32,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // -- CUDA stream --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_stream_create(
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_stream_destroy(stream: *mut std::ffi::c_void);
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_stream_sync(
            stream: *mut std::ffi::c_void,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // -- PTX → cubin JIT compilation --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_ptx_to_cubin(
            ptx_data: *const c_char,
            out_cubin: *mut *mut u8,
            out_cubin_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_cubin_free(cubin: *mut u8);

        // -- Module loading and kernel launch --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_cubin_load(
            cubin_data: *const u8,
            cubin_len: usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_module_unload(cu_module: *mut std::ffi::c_void);
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_module_get_function(
            cu_module: *mut std::ffi::c_void,
            func_name: *const c_char,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_kernel_launch(
            cu_function: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
            sv_dptr: u64,
            n_gate_qubits: u32,
            sv_n_qubits: u32,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // -- CUDA timing events --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_event_create(
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_event_record(
            event: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_event_destroy(event: *mut std::ffi::c_void);
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_event_elapsed_ms(
            start_event: *mut std::ffi::c_void,
            end_event: *mut std::ffi::c_void,
            out_ms: *mut f32,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // -- Device-to-device async memcpy --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_memcpy_dtod_async(
            dst: u64,
            src: u64,
            n_bytes: usize,
            stream: *mut std::ffi::c_void,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // -- Device memory (statevector) --
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_sv_alloc(
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> u64;
        #[cfg(feature = "cuda")]
        pub fn cast_cuda_sv_free(dptr: u64);
    }
}

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

/// Measure device peak memory bandwidth via repeated D2D async copies.
///
/// Allocates two device buffers of `n_bytes` each, runs `n_warmup` un-timed
/// copies to prime the memory subsystem, then times `n_iters` copies with
/// CUDA events.  Returns bandwidth in GiB/s, counting both the read and the
/// write (i.e. 2 × `n_bytes` per iteration).
///
/// `n_bytes` should match the statevector size that gate kernels will operate
/// on so the bandwidth figure is representative.
#[cfg(feature = "cuda")]
pub fn measure_peak_bw_gib_s(
    n_bytes: usize,
    n_warmup: usize,
    n_iters: usize,
) -> anyhow::Result<f64> {
    use types::ERR_BUF_LEN;

    let mut err_buf = [0 as c_char; ERR_BUF_LEN];

    // Allocate two F64-backed device buffers of the right byte size.
    // cast_cuda_sv_alloc(n_elements, precision=1) allocates n_elements × 8 bytes.
    let n_f64 = n_bytes / 8;
    let src = unsafe { ffi::cast_cuda_sv_alloc(n_f64, 1, err_buf.as_mut_ptr(), err_buf.len()) };
    if src == 0 {
        return Err(anyhow::anyhow!(
            "bw alloc src: {}",
            error_from_buf(&err_buf)
        ));
    }
    let dst = unsafe { ffi::cast_cuda_sv_alloc(n_f64, 1, err_buf.as_mut_ptr(), err_buf.len()) };
    if dst == 0 {
        unsafe { ffi::cast_cuda_sv_free(src) };
        return Err(anyhow::anyhow!(
            "bw alloc dst: {}",
            error_from_buf(&err_buf)
        ));
    }

    // Create a dedicated stream for the BW test.
    let stream = unsafe { ffi::cast_cuda_stream_create(err_buf.as_mut_ptr(), err_buf.len()) };
    if stream.is_null() {
        unsafe {
            ffi::cast_cuda_sv_free(src);
            ffi::cast_cuda_sv_free(dst)
        };
        return Err(anyhow::anyhow!("bw stream: {}", error_from_buf(&err_buf)));
    }

    // Helper: free all resources and return an error.
    let cleanup_err = |msg: String| -> anyhow::Error {
        unsafe {
            ffi::cast_cuda_stream_destroy(stream);
            ffi::cast_cuda_sv_free(dst);
            ffi::cast_cuda_sv_free(src);
        }
        anyhow::anyhow!("{}", msg)
    };

    // Warmup copies.
    for _ in 0..n_warmup {
        let rc = unsafe {
            ffi::cast_cuda_memcpy_dtod_async(
                dst,
                src,
                n_bytes,
                stream,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            return Err(cleanup_err(format!(
                "bw warmup memcpy: {}",
                error_from_buf(&err_buf)
            )));
        }
    }
    let rc = unsafe { ffi::cast_cuda_stream_sync(stream, err_buf.as_mut_ptr(), err_buf.len()) };
    if rc != 0 {
        return Err(cleanup_err(format!(
            "bw warmup sync: {}",
            error_from_buf(&err_buf)
        )));
    }

    // Create timing events.
    let ev_start = unsafe { ffi::cast_cuda_event_create(err_buf.as_mut_ptr(), err_buf.len()) };
    if ev_start.is_null() {
        return Err(cleanup_err(format!(
            "bw ev_start: {}",
            error_from_buf(&err_buf)
        )));
    }
    let ev_end = unsafe { ffi::cast_cuda_event_create(err_buf.as_mut_ptr(), err_buf.len()) };
    if ev_end.is_null() {
        unsafe { ffi::cast_cuda_event_destroy(ev_start) };
        return Err(cleanup_err(format!(
            "bw ev_end: {}",
            error_from_buf(&err_buf)
        )));
    }

    // Record start, issue timed copies, record end, sync.
    let mut rc = unsafe {
        ffi::cast_cuda_event_record(ev_start, stream, err_buf.as_mut_ptr(), err_buf.len())
    };
    if rc != 0 {
        unsafe {
            ffi::cast_cuda_event_destroy(ev_start);
            ffi::cast_cuda_event_destroy(ev_end)
        };
        return Err(cleanup_err(format!(
            "bw record start: {}",
            error_from_buf(&err_buf)
        )));
    }
    for _ in 0..n_iters {
        rc = unsafe {
            ffi::cast_cuda_memcpy_dtod_async(
                dst,
                src,
                n_bytes,
                stream,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if rc != 0 {
            unsafe {
                ffi::cast_cuda_event_destroy(ev_start);
                ffi::cast_cuda_event_destroy(ev_end)
            };
            return Err(cleanup_err(format!(
                "bw timed memcpy: {}",
                error_from_buf(&err_buf)
            )));
        }
    }
    rc =
        unsafe { ffi::cast_cuda_event_record(ev_end, stream, err_buf.as_mut_ptr(), err_buf.len()) };
    if rc != 0 {
        unsafe {
            ffi::cast_cuda_event_destroy(ev_start);
            ffi::cast_cuda_event_destroy(ev_end)
        };
        return Err(cleanup_err(format!(
            "bw record end: {}",
            error_from_buf(&err_buf)
        )));
    }
    rc = unsafe { ffi::cast_cuda_stream_sync(stream, err_buf.as_mut_ptr(), err_buf.len()) };
    if rc != 0 {
        unsafe {
            ffi::cast_cuda_event_destroy(ev_start);
            ffi::cast_cuda_event_destroy(ev_end)
        };
        return Err(cleanup_err(format!(
            "bw sync: {}",
            error_from_buf(&err_buf)
        )));
    }

    let mut ms = 0.0f32;
    rc = unsafe {
        ffi::cast_cuda_event_elapsed_ms(
            ev_start,
            ev_end,
            &mut ms,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    unsafe {
        ffi::cast_cuda_event_destroy(ev_start);
        ffi::cast_cuda_event_destroy(ev_end)
    };
    unsafe {
        ffi::cast_cuda_stream_destroy(stream);
        ffi::cast_cuda_sv_free(dst);
        ffi::cast_cuda_sv_free(src)
    };
    if rc != 0 {
        return Err(anyhow::anyhow!(
            "bw elapsed_ms: {}",
            error_from_buf(&err_buf)
        ));
    }

    // GiB/s = 2 × n_bytes × n_iters / total_time_s / 2^30
    // Factor 2: each copy reads src AND writes dst.
    let total_s = ms as f64 / 1000.0;
    Ok(2.0 * n_bytes as f64 * n_iters as f64 / total_s / (1u64 << 30) as f64)
}

pub(super) fn error_from_buf(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

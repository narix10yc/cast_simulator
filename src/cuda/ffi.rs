//! Raw FFI bindings to `src/cpp/cuda/cuda.h`.
//!
//! Covers gate PTX compilation, device queries, stream management, PTX→cubin
//! JIT linking, module/kernel launch, device-side reductions, async memcpy,
//! and CUDA timing events.
//!
//! Statevector memory functions (`sv_alloc`, `sv_free`, `sv_zero`,
//! `sv_upload`, `sv_download`) are declared privately inside `statevector.rs`
//! because they are only used there.

use super::CudaKernelGenSpec;
use std::ffi::c_char;

// ── Types ────────────────────────────────────────────────────────────────────

/// `cast_cuda_complex64_t` — a complex number passed to the C++ gate compiler.
#[repr(C)]
pub struct FfiComplex64 {
    pub re: f64,
    pub im: f64,
}

// ── Functions ────────────────────────────────────────────────────────────────

unsafe extern "C" {
    // -- Gate PTX compilation -------------------------------------------------

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

    // -- Device capability queries --------------------------------------------

    #[cfg(feature = "cuda")]
    pub fn cast_cuda_device_sm(
        out_major: *mut u32,
        out_minor: *mut u32,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    #[cfg(feature = "cuda")]
    pub fn cast_cuda_free_memory(
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- CUDA stream ----------------------------------------------------------

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

    // -- PTX → cubin JIT compilation ------------------------------------------

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

    // -- Module loading and kernel launch -------------------------------------

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
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- Device-side reductions -----------------------------------------------

    #[cfg(feature = "cuda")]
    pub fn cast_cuda_norm_squared(
        dptr: u64,
        n_elements: usize,
        precision: u8,
        out_norm_sq: *mut f64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    #[cfg(feature = "cuda")]
    pub fn cast_cuda_scale(
        dptr: u64,
        n_elements: usize,
        precision: u8,
        scale: f64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- Device-to-device async memcpy ----------------------------------------

    #[cfg(feature = "cuda")]
    pub fn cast_cuda_memcpy_dtod_async(
        dst: u64,
        src: u64,
        n_bytes: usize,
        stream: *mut std::ffi::c_void,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- CUDA timing events ---------------------------------------------------

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
    pub fn cast_cuda_event_synchronize(
        event: *mut std::ffi::c_void,
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
}

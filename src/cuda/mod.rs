//! CUDA NVPTX kernel generation and GPU execution.
//!
//! The primary workflow is:
//! 1. Create a [`CudaKernelGenerator`] and call [`CudaKernelGenerator::generate`] for each gate.
//! 2. Call [`CudaKernelGenerator::emit_ir`] to retrieve the optimized NVPTX LLVM IR (optional).
//! 3. Call [`CudaKernelGenerator::compile`] to compile all kernels and obtain a
//!    [`CudaCompilationSession`] — a pure-Rust value owning the PTX/cubin data.
//! 4. Build a [`CudaExecSession`] from the compilation session to load kernels into the driver.
//! 5. Allocate a [`CudaStatevector`] and call [`CudaExecSession::apply`].

mod types;
mod kernel;
mod statevector;

#[cfg(test)]
mod tests;

use std::ffi::c_char;

pub use types::{CudaKernelGenSpec, CudaKernelId, CudaPrecision};
pub use kernel::{CompiledKernel, CudaCompilationSession, CudaExecSession, CudaKernelGenerator};
pub use statevector::CudaStatevector;

// ── FFI declarations ──────────────────────────────────────────────────────────

pub(super) mod ffi {
    use super::{CudaKernelGenSpec, CudaKernelId};
    use std::ffi::c_char;

    /// Opaque C++ `cast_cuda_kernel_generator_t`.
    #[repr(C)]
    pub struct CastCudaKernelGenerator {
        _private: [u8; 0],
    }

    /// Opaque C++ `cast_cuda_compilation_session_t`.
    #[repr(C)]
    pub struct CastCudaCompilationSession {
        _private: [u8; 0],
    }

    /// Matches `cast_cuda_complex64_t` in `cuda.h`.
    #[repr(C)]
    pub struct FfiComplex64 {
        pub re: f64,
        pub im: f64,
    }

    unsafe extern "C" {
        // ── Generator ─────────────────────────────────────────────────────────
        pub fn cast_cuda_kernel_generator_new() -> *mut CastCudaKernelGenerator;
        pub fn cast_cuda_kernel_generator_delete(generator: *mut CastCudaKernelGenerator);
        pub fn cast_cuda_kernel_generator_generate(
            generator: *mut CastCudaKernelGenerator,
            spec: *const CudaKernelGenSpec,
            matrix: *const FfiComplex64,
            matrix_len: usize,
            qubits: *const u32,
            n_qubits: usize,
            out_kernel_id: *mut CudaKernelId,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_kernel_generator_emit_ir(
            generator: *mut CastCudaKernelGenerator,
            kernel_id: CudaKernelId,
            out_ir: *mut c_char,
            ir_buf_len: usize,
            out_ir_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_kernel_generator_finish(
            generator: *mut CastCudaKernelGenerator,
            out_session: *mut *mut CastCudaCompilationSession,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // ── Compilation session indexed accessors ─────────────────────────────
        pub fn cast_cuda_compilation_session_n_kernels(
            session: *const CastCudaCompilationSession,
        ) -> u32;
        pub fn cast_cuda_compilation_session_kernel_id_at(
            session: *const CastCudaCompilationSession,
            idx: u32,
        ) -> CudaKernelId;
        pub fn cast_cuda_compilation_session_n_gate_qubits_at(
            session: *const CastCudaCompilationSession,
            idx: u32,
        ) -> u32;
        pub fn cast_cuda_compilation_session_precision_at(
            session: *const CastCudaCompilationSession,
            idx: u32,
        ) -> u8;
        pub fn cast_cuda_compilation_session_func_name_at(
            session: *const CastCudaCompilationSession,
            idx: u32,
        ) -> *const c_char;
        pub fn cast_cuda_compilation_session_emit_ptx(
            session: *mut CastCudaCompilationSession,
            kernel_id: CudaKernelId,
            out_ptx: *mut c_char,
            ptx_buf_len: usize,
            out_ptx_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_compilation_session_emit_cubin(
            session: *mut CastCudaCompilationSession,
            kernel_id: CudaKernelId,
            out_cubin: *mut u8,
            cubin_buf_len: usize,
            out_cubin_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_compilation_session_delete(session: *mut CastCudaCompilationSession);

        // ── Stateless CUDA module loading ──────────────────────────────────────
        /// Load PTX into a CUDA driver module; returns opaque CUmodule as void*.
        pub fn cast_cuda_ptx_load(
            ptx_data: *const c_char,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        pub fn cast_cuda_module_unload(cu_module: *mut std::ffi::c_void);
        /// Resolve an entry-point function from a loaded module.
        pub fn cast_cuda_module_get_function(
            cu_module: *mut std::ffi::c_void,
            func_name: *const c_char,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut std::ffi::c_void;
        /// Launch the kernel and synchronize the device.
        pub fn cast_cuda_kernel_apply(
            cu_function: *mut std::ffi::c_void,
            sv_dptr: u64,
            n_gate_qubits: u32,
            sv_n_qubits: u32,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        // ── Stateless device memory ────────────────────────────────────────────
        /// Allocate device memory; returns CUdeviceptr as u64 (0 on failure).
        pub fn cast_cuda_sv_alloc(
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> u64;
        pub fn cast_cuda_sv_free(dptr: u64);
        pub fn cast_cuda_sv_zero(
            dptr: u64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_sv_upload(
            dptr: u64,
            host_data: *const f64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_sv_download(
            dptr: u64,
            host_data: *mut f64,
            n_elements: usize,
            precision: u8,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
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

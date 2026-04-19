//! Raw FFI bindings to `src/cpp/include/ffi_cpu.h`.

use crate::cpu::{KernelId, MatrixLoadMode, SimdWidth};
use crate::types::Precision;
use std::ffi::{c_char, c_void};

// ── Types ────────────────────────────────────────────────────────────────

/// `cast_complex64_t`
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct c64 {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

/// `cast_cpu_kernel_gen_request_t` — FFI-wire form of a kernel generation
/// request.  Built on the stack at call time from a Rust-side owned
/// `KernelGenRequest`; never stored.  The `qubits` and `matrix` pointers
/// must remain valid for the duration of one FFI call.
#[repr(C)]
pub(crate) struct CastCpuKernelGenRequest {
    pub(crate) precision: Precision,
    pub(crate) simd_width: SimdWidth,
    pub(crate) mode: MatrixLoadMode,
    pub(crate) ztol: f64,
    pub(crate) otol: f64,
    pub(crate) qubits: *const u32,
    pub(crate) n_qubits: usize,
    pub(crate) matrix: *const c64,
    pub(crate) matrix_len: usize,
    pub(crate) capture_ir: bool,
    pub(crate) capture_asm: bool,
}

/// `cast_cpu_kernel_metadata_t`
#[repr(C)]
pub(crate) struct KernelMetadata {
    pub(crate) kernel_id: KernelId,
    pub(crate) precision: Precision,
    pub(crate) simd_width: SimdWidth,
    pub(crate) mode: MatrixLoadMode,
    pub(crate) n_gate_qubits: u32,
}

/// `cast_cpu_jit_kernel_record_t`
#[repr(C)]
pub(crate) struct KernelRecord {
    pub(crate) metadata: KernelMetadata,
    pub(crate) entry: Option<unsafe extern "C" fn(*mut c_void)>,
    pub(crate) matrix: *mut c64,
    pub(crate) matrix_len: usize,
    pub(crate) ir_text: *mut c_char,
    pub(crate) asm_text: *mut c_char,
}

/// `cast_cpu_kernel_generator_t` (opaque)
#[repr(C)]
pub(crate) struct KernelGenerator {
    _private: [u8; 0],
}

/// `cast_cpu_jit_session_t` (opaque)
#[repr(C)]
pub(crate) struct JitSession {
    _private: [u8; 0],
}

// ── Functions ────────────────────────────────────────────────────────────

unsafe extern "C" {
    // -- Generator lifecycle --
    pub(crate) fn cast_cpu_kernel_generator_new() -> *mut KernelGenerator;
    pub(crate) fn cast_cpu_kernel_generator_delete(generator: *mut KernelGenerator);

    // -- Kernel generation --
    // Returns the assigned kernel id (> 0) on success, or 0 on failure
    // (with a message in `err_buf`).
    pub(crate) fn cast_cpu_kernel_generator_generate(
        generator: *mut KernelGenerator,
        request: *const CastCpuKernelGenRequest,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> KernelId;

    // -- JIT compilation --
    pub(crate) fn cast_cpu_kernel_generator_finish(
        generator: *mut KernelGenerator,
        out_session: *mut *mut JitSession,
        out_records: *mut *mut KernelRecord,
        out_n_records: *mut usize,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    pub(crate) fn cast_cpu_jit_kernel_records_free(records: *mut KernelRecord, n: usize);

    // -- Session lifecycle --
    pub(crate) fn cast_cpu_jit_session_delete(session: *mut JitSession);
}

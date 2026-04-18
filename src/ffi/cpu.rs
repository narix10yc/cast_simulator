//! Raw FFI bindings to `src/cpp/include/cast_cpu.h`.

use crate::cpu::{CPUKernelGenSpec, KernelId, MatrixLoadMode, SimdWidth};
use crate::types::Precision;
use std::ffi::{c_char, c_void};

// ── Types ────────────────────────────────────────────────────────────────

/// `cast_cpu_complex64_t`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct c64 {
    pub re: f64,
    pub im: f64,
}

/// `cast_cpu_kernel_metadata_t`
#[repr(C)]
pub struct KernelMetadata {
    pub kernel_id: KernelId,
    pub precision: Precision,
    pub simd_width: SimdWidth,
    pub mode: MatrixLoadMode,
    pub n_gate_qubits: u32,
}

/// `cast_cpu_jit_kernel_record_t`
#[repr(C)]
pub struct KernelRecord {
    pub metadata: KernelMetadata,
    pub entry: Option<unsafe extern "C" fn(*mut c_void)>,
    pub matrix: *mut c64,
    pub matrix_len: usize,
    pub asm_text: *mut c_char,
}

/// `cast_cpu_kernel_generator_t` (opaque)
#[repr(C)]
pub struct KernelGenerator {
    _private: [u8; 0],
}

/// `cast_cpu_jit_session_t` (opaque)
#[repr(C)]
pub struct JitSession {
    _private: [u8; 0],
}

// ── Functions ────────────────────────────────────────────────────────────

unsafe extern "C" {
    // -- Generator lifecycle --
    pub fn cast_cpu_kernel_generator_new() -> *mut KernelGenerator;
    pub fn cast_cpu_kernel_generator_delete(generator: *mut KernelGenerator);

    // -- Kernel generation --
    pub fn cast_cpu_kernel_generator_generate(
        generator: *mut KernelGenerator,
        spec: *const CPUKernelGenSpec,
        matrix: *const c64,
        matrix_len: usize,
        qubits: *const u32,
        n_qubits: usize,
        out_kernel_id: *mut KernelId,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- Diagnostics --
    pub fn cast_cpu_kernel_generator_request_asm(
        generator: *mut KernelGenerator,
        kernel_id: KernelId,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    pub fn cast_cpu_kernel_generator_emit_ir(
        generator: *mut KernelGenerator,
        kernel_id: KernelId,
        out_ir: *mut c_char,
        ir_buf_len: usize,
        out_ir_len: *mut usize,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;

    // -- JIT compilation --
    pub fn cast_cpu_kernel_generator_finish(
        generator: *mut KernelGenerator,
        out_session: *mut *mut JitSession,
        out_records: *mut *mut KernelRecord,
        out_n_records: *mut usize,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    pub fn cast_cpu_jit_kernel_records_free(records: *mut KernelRecord, n: usize);

    // -- Session lifecycle --
    pub fn cast_cpu_jit_session_delete(session: *mut JitSession);
}

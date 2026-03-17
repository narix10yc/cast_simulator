//! CPU statevector simulator with LLVM JIT kernel compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CPUKernelGenerator`] and call [`CPUKernelGenerator::generate`] for each gate
//!    variant you need.
//! 2. Call [`CPUKernelGenerator::init_jit`] to compile all generated kernels and obtain a
//!    [`JitSession`].
//! 3. Allocate a [`CPUStatevector`] and drive simulation by calling [`JitSession::apply`].

use super::*;
use crate::types::{Complex, Precision};

use std::ffi::c_char;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// Configuration passed to the kernel generator for each gate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPUKernelGenSpec {
    pub precision: Precision,
    pub simd_width: SimdWidth,
    pub mode: MatrixLoadMode,
    /// Threshold below which a matrix element is treated as exactly 0.
    pub ztol: f64,
    /// Threshold within which a matrix element is treated as exactly ±1.
    pub otol: f64,
}

impl CPUKernelGenSpec {
    /// Sensible defaults for single-precision (F32, W128, ImmValue).
    pub fn f32() -> Self {
        Self {
            precision: Precision::F32,
            simd_width: SimdWidth::W128,
            mode: MatrixLoadMode::ImmValue,
            ztol: 1e-6,
            otol: 1e-6,
        }
    }

    /// Sensible defaults for double-precision (F64, W128, ImmValue).
    pub fn f64() -> Self {
        Self {
            precision: Precision::F64,
            simd_width: SimdWidth::W128,
            mode: MatrixLoadMode::ImmValue,
            ztol: 1e-12,
            otol: 1e-12,
        }
    }
}

/// Opaque handle returned by [`CPUKernelGenerator::generate`], used to identify a compiled
/// kernel inside a [`JitSession`].
pub type KernelId = u64;

const ERR_BUF_LEN: usize = 1024;

// ── FFI declarations ──────────────────────────────────────────────────────────

/// Raw C FFI bindings to the LLVM JIT kernel generator (`src/cpp/cpu.h`).
mod ffi {
    use super::{CPUKernelGenSpec, KernelId, Precision, SimdWidth};
    use std::ffi::{c_char, c_void};

    /// Opaque C++ `cast_cpu_kernel_generator_t`.
    #[repr(C)]
    pub struct CastCpuKernelGenerator {
        _private: [u8; 0],
    }

    /// Opaque C++ `cast_cpu_jit_session_t`.
    #[repr(C)]
    pub struct CastCpuJitSession {
        _private: [u8; 0],
    }

    /// Matches `cast_cpu_complex64_t` in `cpu.h` (always f64 re/im regardless of precision).
    #[repr(C)]
    pub struct FfiComplex64 {
        pub re: f64,
        pub im: f64,
    }

    unsafe extern "C" {
        pub fn cast_cpu_kernel_generator_new() -> *mut CastCpuKernelGenerator;
        pub fn cast_cpu_kernel_generator_delete(generator: *mut CastCpuKernelGenerator);
        /// Returns 0 on success; writes a human-readable message into `err_buf` on failure.
        pub fn cast_cpu_kernel_generator_generate(
            generator: *mut CastCpuKernelGenerator,
            spec: *const CPUKernelGenSpec,
            matrix: *const FfiComplex64,
            matrix_len: usize,
            qubits: *const u32,
            n_qubits: usize,
            out_kernel_id: *mut KernelId,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Runs O1 on the kernel and returns its optimized IR text (two-call pattern).
        /// Marks `kernel_id` for assembly capture during `init_jit`. Returns 0 on success.
        pub fn cast_cpu_kernel_generator_request_asm(
            generator: *mut CastCpuKernelGenerator,
            kernel_id: KernelId,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Pass `out_ir = null` on the first call to get the required length via
        /// `*out_ir_len`, then allocate and call again with the buffer.
        pub fn cast_cpu_kernel_generator_emit_ir(
            generator: *mut CastCpuKernelGenerator,
            kernel_id: KernelId,
            out_ir: *mut c_char,
            ir_buf_len: usize,
            out_ir_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Consumes the generator and produces a JIT session. Returns 0 on success.
        pub fn cast_cpu_kernel_generator_finish(
            generator: *mut CastCpuKernelGenerator,
            out_session: *mut *mut CastCpuJitSession,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cpu_jit_session_apply(
            session: *mut CastCpuJitSession,
            kernel_id: KernelId,
            sv: *mut c_void,
            n_qubits: u32,
            sv_precision: Precision,
            sv_simd_width: SimdWidth,
            n_threads: i32,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Returns the native assembly text captured during JIT compilation (two-call pattern).
        pub fn cast_cpu_jit_session_emit_asm(
            session: *mut CastCpuJitSession,
            kernel_id: KernelId,
            out_asm: *mut c_char,
            asm_buf_len: usize,
            out_asm_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cpu_jit_session_delete(session: *mut CastCpuJitSession);
    }
}

// ── CPUKernelGenerator ────────────────────────────────────────────────────────

/// Accumulates gate kernels as LLVM IR, then compiles them all at once via
/// [`CPUKernelGenerator::init_jit`].
///
/// Owns the C++ `cast_cpu_kernel_generator_t` object. Calling `init_jit` transfers
/// ownership to a [`JitSession`]; the generator is then consumed and must not be used again.
pub struct CPUKernelGenerator {
    raw: NonNull<ffi::CastCpuKernelGenerator>,
}

impl CPUKernelGenerator {
    pub fn new() -> anyhow::Result<Self> {
        let raw = unsafe { ffi::cast_cpu_kernel_generator_new() };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow::anyhow!("failed to create CPU kernel generator"))?;
        Ok(Self { raw })
    }

    /// Generates LLVM IR for a gate kernel and returns its [`KernelId`].
    ///
    /// `matrix` must be a row-major unitary matrix of size `(2^k)²` where `k = qubits.len()`.
    /// Qubits are given as absolute qubit indices in the statevector (ascending order enforced
    /// by [`QuantumGate`]).
    pub fn generate(
        &mut self,
        spec: &CPUKernelGenSpec,
        matrix: &[Complex],
        qubits: &[u32],
    ) -> anyhow::Result<KernelId> {
        let mut kernel_id: KernelId = 0;
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let ffi_matrix = matrix
            .iter()
            .map(|value| ffi::FfiComplex64 {
                re: value.re,
                im: value.im,
            })
            .collect::<Vec<_>>();
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_generate(
                self.raw.as_ptr(),
                spec as *const CPUKernelGenSpec,
                ffi_matrix.as_ptr(),
                ffi_matrix.len(),
                qubits.as_ptr(),
                qubits.len(),
                &mut kernel_id,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status == 0 {
            Ok(kernel_id)
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }

    /// Returns the optimized LLVM IR for the kernel identified by `kernel_id`.
    ///
    /// Triggers the O1 pass pipeline on the plain `Module` if it hasn't run yet.
    /// Must be called before [`init_jit`](Self::init_jit), which moves the module
    /// into the JIT and makes it inaccessible.
    pub fn emit_ir(&mut self, kernel_id: KernelId) -> anyhow::Result<String> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // First call: query the IR byte length (excluding the null terminator).
        let mut ir_len: usize = 0;
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_emit_ir(
                self.raw.as_ptr(),
                kernel_id,
                std::ptr::null_mut(),
                0,
                &mut ir_len,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // Second call: write IR into a caller-owned buffer.
        let mut ir_buf = vec![0u8; ir_len + 1];
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_emit_ir(
                self.raw.as_ptr(),
                kernel_id,
                ir_buf.as_mut_ptr().cast(),
                ir_buf.len(),
                std::ptr::null_mut(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        ir_buf.truncate(ir_len);
        String::from_utf8(ir_buf).map_err(|e| anyhow::anyhow!("IR is not valid UTF-8: {e}"))
    }

    /// Opts `kernel_id` into assembly capture during [`init_jit`](Self::init_jit).
    ///
    /// Must be called before `init_jit`; has no effect afterwards (the module
    /// has already been consumed). Kernels that were not opted in will return an
    /// error from [`JitSession::emit_asm`].
    pub fn request_asm(&mut self, kernel_id: KernelId) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_request_asm(
                self.raw.as_ptr(),
                kernel_id,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }

    /// Compiles all generated kernels and returns a [`JitSession`], consuming `self`.
    ///
    /// The C++ side takes ownership of the generator's IR modules. On failure the generator
    /// is dropped normally. On success the underlying C++ object is freed by the session.
    pub fn init_jit(self) -> anyhow::Result<JitSession> {
        // Wrap in ManuallyDrop so we decide exactly when the C++ destructor fires.
        let mut this = ManuallyDrop::new(self);
        let mut raw_session = std::ptr::null_mut();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        let status = unsafe {
            ffi::cast_cpu_kernel_generator_finish(
                this.raw.as_ptr(),
                &mut raw_session,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };

        if status != 0 {
            // finish failed; the generator is still valid, so run its destructor.
            unsafe { ManuallyDrop::drop(&mut this) };
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // finish succeeded; the C++ object is now owned by the session.
        let raw = NonNull::new(raw_session)
            .ok_or_else(|| anyhow::anyhow!("C++ side returned a null JIT session"))?;
        Ok(JitSession { raw })
    }
}

impl Default for CPUKernelGenerator {
    fn default() -> Self {
        Self::new().expect("failed to create CPU kernel generator")
    }
}

impl Drop for CPUKernelGenerator {
    fn drop(&mut self) {
        unsafe { ffi::cast_cpu_kernel_generator_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for CPUKernelGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CPUKernelGenerator").finish_non_exhaustive()
    }
}

// ── JitSession ────────────────────────────────────────────────────────────────

/// A compiled JIT session holding ready-to-run native kernel functions.
///
/// Created by [`CPUKernelGenerator::init_jit`]. Individual kernels are applied to
/// statevectors via [`JitSession::apply`].
pub struct JitSession {
    raw: NonNull<ffi::CastCpuJitSession>,
}

impl Drop for JitSession {
    fn drop(&mut self) {
        unsafe { ffi::cast_cpu_jit_session_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for JitSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JitSession").finish_non_exhaustive()
    }
}

impl JitSession {
    /// Returns the native assembly text that was emitted during JIT compilation.
    ///
    /// The assembly is captured once at compile time (before the module is consumed
    /// by the JIT pipeline) and cached for the lifetime of the session. This method
    /// just retrieves the cached text, so it is cheap to call repeatedly.
    pub fn emit_asm(&self, kernel_id: KernelId) -> anyhow::Result<String> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // First call: query the assembly byte length (excluding null terminator).
        let mut asm_len: usize = 0;
        let status = unsafe {
            ffi::cast_cpu_jit_session_emit_asm(
                self.raw.as_ptr(),
                kernel_id,
                std::ptr::null_mut(),
                0,
                &mut asm_len,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // Second call: write the assembly into a caller-owned buffer.
        let mut asm_buf = vec![0u8; asm_len + 1];
        let status = unsafe {
            ffi::cast_cpu_jit_session_emit_asm(
                self.raw.as_ptr(),
                kernel_id,
                asm_buf.as_mut_ptr().cast(),
                asm_buf.len(),
                std::ptr::null_mut(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        asm_buf.truncate(asm_len);
        String::from_utf8(asm_buf).map_err(|e| anyhow::anyhow!("assembly is not valid UTF-8: {e}"))
    }

    /// Applies the kernel identified by `kernel_id` to `statevector` in-place.
    ///
    /// `n_threads`: number of worker threads. `None` lets the C++ side choose based on
    /// hardware concurrency.
    ///
    /// The kernel's precision and SIMD width must match those of the statevector, and the
    /// statevector must have at least `n_gate_qubits + simd_s` qubits.
    pub fn apply(
        &mut self,
        kernel_id: KernelId,
        statevector: &mut CPUStatevector,
        n_threads: u32,
    ) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let n_qubits = statevector.n_qubits();
        let status = unsafe {
            ffi::cast_cpu_jit_session_apply(
                self.raw.as_ptr(),
                kernel_id,
                statevector.raw_mut_ptr(),
                n_qubits,
                statevector.precision(),
                statevector.simd_width(),
                n_threads as i32,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(error_from_buf(&err_buf)))
        }
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Reads a null-terminated C string from `buf` and returns it as a `String`.
fn error_from_buf(buf: &[c_char]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let bytes = buf[..end].iter().map(|&c| c as u8).collect::<Vec<_>>();
    let msg = String::from_utf8_lossy(&bytes).trim().to_owned();
    if msg.is_empty() {
        "unknown CPU FFI error".to_owned()
    } else {
        msg
    }
}

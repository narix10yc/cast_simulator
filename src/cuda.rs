//! CUDA NVPTX kernel generation and PTX/cubin compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CudaKernelGenerator`] and call [`CudaKernelGenerator::generate`] for each gate
//!    variant you need.
//! 2. Call [`CudaKernelGenerator::emit_ir`] to retrieve the optimized NVPTX LLVM IR (optional).
//! 3. Call [`CudaKernelGenerator::compile`] to compile all kernels and obtain a
//!    [`CudaCompilationSession`].
//! 4. Use [`CudaCompilationSession::emit_ptx`] or [`CudaCompilationSession::emit_cubin`] to
//!    retrieve the compiled artifacts for each kernel. GPU execution is out of scope here.

use std::ffi::c_char;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

// ── Enums & config ────────────────────────────────────────────────────────────

/// Floating-point precision used in the CUDA kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CudaPrecision {
    F32 = 0,
    F64 = 1,
}

/// Configuration passed to the kernel generator for each gate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaKernelGenSpec {
    pub precision: CudaPrecision,
    /// Threshold below which a matrix element is treated as exactly 0.
    pub ztol: f64,
    /// Threshold within which a matrix element is treated as exactly ±1.
    pub otol: f64,
    /// CUDA compute capability major version (e.g. 8 for sm_86).
    pub sm_major: u32,
    /// CUDA compute capability minor version (e.g. 6 for sm_86).
    pub sm_minor: u32,
}

impl CudaKernelGenSpec {
    /// Single-precision defaults targeting sm_80.
    pub fn f32_sm80() -> Self {
        Self {
            precision: CudaPrecision::F32,
            ztol: 1e-6,
            otol: 1e-6,
            sm_major: 8,
            sm_minor: 0,
        }
    }

    /// Double-precision defaults targeting sm_80.
    pub fn f64_sm80() -> Self {
        Self {
            precision: CudaPrecision::F64,
            ztol: 1e-12,
            otol: 1e-12,
            sm_major: 8,
            sm_minor: 0,
        }
    }
}

/// Opaque handle returned by [`CudaKernelGenerator::generate`], used to identify a compiled
/// kernel inside a [`CudaCompilationSession`].
pub type CudaKernelId = u64;

const ERR_BUF_LEN: usize = 1024;

// ── FFI declarations ──────────────────────────────────────────────────────────

mod ffi {
    use super::{CudaKernelGenSpec, CudaKernelId, CudaPrecision};
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

    /// Matches `cast_cuda_complex64_t` in `cuda.h` (always f64 re/im regardless of precision).
    #[repr(C)]
    pub struct FfiComplex64 {
        pub re: f64,
        pub im: f64,
    }

    unsafe extern "C" {
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
        /// Pass `out_ir = null` to get the required length via `*out_ir_len`, then allocate
        /// and call again with the buffer.
        pub fn cast_cuda_kernel_generator_emit_ir(
            generator: *mut CastCudaKernelGenerator,
            kernel_id: CudaKernelId,
            out_ir: *mut c_char,
            ir_buf_len: usize,
            out_ir_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Consumes the generator and produces a compilation session. Returns 0 on success.
        pub fn cast_cuda_kernel_generator_finish(
            generator: *mut CastCudaKernelGenerator,
            out_session: *mut *mut CastCudaCompilationSession,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Pass `out_ptx = null` to get required length via `*out_ptx_len`.
        pub fn cast_cuda_compilation_session_emit_ptx(
            session: *mut CastCudaCompilationSession,
            kernel_id: CudaKernelId,
            out_ptx: *mut c_char,
            ptx_buf_len: usize,
            out_ptx_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Pass `out_cubin = null` to get required length via `*out_cubin_len`.
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
    }

    // ── Exec session + statevector ─────────────────────────────────────────────

    /// Opaque C++ `cast_cuda_exec_session_t`.
    #[repr(C)]
    pub struct CastCudaExecSession {
        _private: [u8; 0],
    }

    /// Opaque C++ `cast_cuda_statevector_t`.
    #[repr(C)]
    pub struct CastCudaStatevector {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn cast_cuda_exec_session_new(
            session: *const CastCudaCompilationSession,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut CastCudaExecSession;
        pub fn cast_cuda_exec_session_delete(session: *mut CastCudaExecSession);
        pub fn cast_cuda_exec_session_apply(
            session: *mut CastCudaExecSession,
            kernel_id: CudaKernelId,
            sv: *mut CastCudaStatevector,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;

        pub fn cast_cuda_statevector_new(
            n_qubits: u32,
            precision: CudaPrecision,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> *mut CastCudaStatevector;
        pub fn cast_cuda_statevector_delete(sv: *mut CastCudaStatevector);
        pub fn cast_cuda_statevector_zero(
            sv: *mut CastCudaStatevector,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_statevector_upload(
            sv: *mut CastCudaStatevector,
            host_data: *const f64,
            n_elements: usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        pub fn cast_cuda_statevector_download(
            sv: *const CastCudaStatevector,
            host_data: *mut f64,
            n_elements: usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn error_from_buf(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── CudaKernelGenerator ───────────────────────────────────────────────────────

/// Accumulates gate kernels as NVPTX LLVM IR, then compiles them all at once via
/// [`CudaKernelGenerator::compile`].
pub struct CudaKernelGenerator {
    raw: NonNull<ffi::CastCudaKernelGenerator>,
}

impl CudaKernelGenerator {
    pub fn new() -> anyhow::Result<Self> {
        let raw = unsafe { ffi::cast_cuda_kernel_generator_new() };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow::anyhow!("failed to create CUDA kernel generator"))?;
        Ok(Self { raw })
    }

    /// Generates NVPTX LLVM IR for a gate kernel and returns its [`CudaKernelId`].
    ///
    /// `matrix` must be a row-major flat slice of complex64 values of size `(2^k)²`
    /// where `k = qubits.len()`.
    pub fn generate(
        &mut self,
        spec: &CudaKernelGenSpec,
        matrix: &[(f64, f64)],
        qubits: &[u32],
    ) -> anyhow::Result<CudaKernelId> {
        let mut kernel_id: CudaKernelId = 0;
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let ffi_matrix: Vec<ffi::FfiComplex64> = matrix
            .iter()
            .map(|&(re, im)| ffi::FfiComplex64 { re, im })
            .collect();
        let status = unsafe {
            ffi::cast_cuda_kernel_generator_generate(
                self.raw.as_ptr(),
                spec as *const CudaKernelGenSpec,
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

    /// Returns the optimized NVPTX LLVM IR for the kernel identified by `kernel_id`.
    ///
    /// Triggers the O1 pass pipeline on the NVPTX module if it hasn't run yet.
    /// Must be called before [`compile`](Self::compile), which moves the module into the
    /// compilation pipeline and makes it inaccessible.
    pub fn emit_ir(&mut self, kernel_id: CudaKernelId) -> anyhow::Result<String> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // First call: query the IR byte length (excluding the null terminator).
        let mut ir_len: usize = 0;
        let status = unsafe {
            ffi::cast_cuda_kernel_generator_emit_ir(
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
            ffi::cast_cuda_kernel_generator_emit_ir(
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

    /// Compiles all generated kernels and returns a [`CudaCompilationSession`], consuming `self`.
    ///
    /// On failure the generator is dropped normally. On success the underlying C++ object
    /// is freed by the session.
    pub fn compile(self) -> anyhow::Result<CudaCompilationSession> {
        let mut this = ManuallyDrop::new(self);
        let mut raw_session = std::ptr::null_mut();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        let status = unsafe {
            ffi::cast_cuda_kernel_generator_finish(
                this.raw.as_ptr(),
                &mut raw_session,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };

        if status != 0 {
            unsafe { ManuallyDrop::drop(&mut this) };
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        let raw = NonNull::new(raw_session)
            .ok_or_else(|| anyhow::anyhow!("C++ side returned a null compilation session"))?;
        Ok(CudaCompilationSession { raw })
    }
}

impl Default for CudaKernelGenerator {
    fn default() -> Self {
        Self::new().expect("failed to create CUDA kernel generator")
    }
}

impl Drop for CudaKernelGenerator {
    fn drop(&mut self) {
        unsafe { ffi::cast_cuda_kernel_generator_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for CudaKernelGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaKernelGenerator").finish_non_exhaustive()
    }
}

// ── CudaCompilationSession ────────────────────────────────────────────────────

/// A compiled session holding PTX text and cubin bytes for each kernel.
///
/// Created by [`CudaKernelGenerator::compile`].
pub struct CudaCompilationSession {
    raw: NonNull<ffi::CastCudaCompilationSession>,
}

impl Drop for CudaCompilationSession {
    fn drop(&mut self) {
        unsafe { ffi::cast_cuda_compilation_session_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for CudaCompilationSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaCompilationSession").finish_non_exhaustive()
    }
}

impl CudaCompilationSession {
    /// Returns the PTX assembly text for the kernel identified by `kernel_id`.
    pub fn emit_ptx(&self, kernel_id: CudaKernelId) -> anyhow::Result<String> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // First call: query PTX byte length.
        let mut ptx_len: usize = 0;
        let status = unsafe {
            ffi::cast_cuda_compilation_session_emit_ptx(
                self.raw.as_ptr(),
                kernel_id,
                std::ptr::null_mut(),
                0,
                &mut ptx_len,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // Second call: fill the buffer.
        let mut ptx_buf = vec![0u8; ptx_len + 1];
        let status = unsafe {
            ffi::cast_cuda_compilation_session_emit_ptx(
                self.raw.as_ptr(),
                kernel_id,
                ptx_buf.as_mut_ptr().cast(),
                ptx_buf.len(),
                std::ptr::null_mut(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        ptx_buf.truncate(ptx_len);
        String::from_utf8(ptx_buf).map_err(|e| anyhow::anyhow!("PTX is not valid UTF-8: {e}"))
    }

    /// Returns the cubin binary for the kernel identified by `kernel_id`.
    ///
    /// The returned bytes form an ELF binary (`\x7fELF...`) suitable for loading
    /// via the CUDA driver API.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn emit_cubin(&self, kernel_id: CudaKernelId) -> anyhow::Result<Vec<u8>> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // First call: query cubin byte length.
        let mut cubin_len: usize = 0;
        let status = unsafe {
            ffi::cast_cuda_compilation_session_emit_cubin(
                self.raw.as_ptr(),
                kernel_id,
                std::ptr::null_mut(),
                0,
                &mut cubin_len,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // Second call: fill the buffer.
        let mut cubin_buf = vec![0u8; cubin_len];
        let status = unsafe {
            ffi::cast_cuda_compilation_session_emit_cubin(
                self.raw.as_ptr(),
                kernel_id,
                cubin_buf.as_mut_ptr(),
                cubin_buf.len(),
                std::ptr::null_mut(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        Ok(cubin_buf)
    }
}

// ── CudaStatevector ───────────────────────────────────────────────────────────

/// A statevector allocated in GPU device memory.
///
/// Amplitudes are stored as interleaved `(re, im)` scalars in the precision
/// specified at construction. The host API always uses `f64` regardless of
/// device precision; narrowing to `f32` happens inside the C++ layer.
pub struct CudaStatevector {
    raw: NonNull<ffi::CastCudaStatevector>,
    n_qubits: u32,
    precision: CudaPrecision,
}

impl Drop for CudaStatevector {
    fn drop(&mut self) {
        unsafe { ffi::cast_cuda_statevector_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for CudaStatevector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStatevector")
            .field("n_qubits", &self.n_qubits)
            .field("precision", &self.precision)
            .finish()
    }
}

impl CudaStatevector {
    /// Allocates a device statevector for `2^n_qubits` complex amplitudes.
    pub fn new(n_qubits: u32, precision: CudaPrecision) -> anyhow::Result<Self> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let raw = unsafe {
            ffi::cast_cuda_statevector_new(
                n_qubits,
                precision,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow::anyhow!(error_from_buf(&err_buf)))?;
        Ok(Self { raw, n_qubits, precision })
    }

    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }

    /// Number of complex amplitudes: `2^n_qubits`.
    pub fn len(&self) -> usize {
        1 << self.n_qubits
    }

    /// Sets the device statevector to the `|0⟩` computational basis state.
    pub fn zero(&mut self) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_statevector_zero(
                self.raw.as_ptr(),
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

    /// Uploads amplitudes from a host slice of `(re, im)` pairs.
    ///
    /// The slice length must equal `2^n_qubits`.
    pub fn upload(&mut self, data: &[(f64, f64)]) -> anyhow::Result<()> {
        if data.len() != self.len() {
            anyhow::bail!(
                "upload: expected {} amplitudes, got {}",
                self.len(),
                data.len()
            );
        }
        // Flatten (re, im) pairs into a contiguous f64 slice.
        let flat: Vec<f64> = data.iter().flat_map(|&(re, im)| [re, im]).collect();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_statevector_upload(
                self.raw.as_ptr(),
                flat.as_ptr(),
                flat.len(),
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

    /// Downloads all amplitudes to the host as `(re, im)` pairs.
    pub fn download(&self) -> anyhow::Result<Vec<(f64, f64)>> {
        let n_elements = self.len() * 2;
        let mut flat = vec![0.0f64; n_elements];
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_statevector_download(
                self.raw.as_ptr(),
                flat.as_mut_ptr(),
                flat.len(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }
        Ok(flat.chunks_exact(2).map(|c| (c[0], c[1])).collect())
    }
}

// ── CudaExecSession ───────────────────────────────────────────────────────────

/// Holds CUDA modules loaded from the PTX kernels in a [`CudaCompilationSession`],
/// ready to launch on the current device.
///
/// Created by [`CudaExecSession::new`]; the compilation session does not need
/// to outlive it (the PTX is loaded into driver-owned modules at construction).
pub struct CudaExecSession {
    raw: NonNull<ffi::CastCudaExecSession>,
}

impl Drop for CudaExecSession {
    fn drop(&mut self) {
        unsafe { ffi::cast_cuda_exec_session_delete(self.raw.as_ptr()) };
    }
}

impl fmt::Debug for CudaExecSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaExecSession").finish_non_exhaustive()
    }
}

impl CudaExecSession {
    /// Loads all compiled kernels from `session` into CUDA driver modules.
    ///
    /// Initialises the CUDA driver once per process (device 0). Fails if no
    /// CUDA-capable device is present.
    pub fn new(session: &CudaCompilationSession) -> anyhow::Result<Self> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let raw = unsafe {
            ffi::cast_cuda_exec_session_new(
                session.raw.as_ptr(),
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow::anyhow!(error_from_buf(&err_buf)))?;
        Ok(Self { raw })
    }

    /// Applies the kernel identified by `kernel_id` to `sv` in-place.
    ///
    /// Synchronises the device before returning so the result is immediately
    /// visible on the host via [`CudaStatevector::download`].
    pub fn apply(
        &self,
        kernel_id: CudaKernelId,
        sv: &mut CudaStatevector,
    ) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_exec_session_apply(
                self.raw.as_ptr(),
                kernel_id,
                sv.raw.as_ptr(),
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{CudaKernelGenSpec, CudaKernelGenerator};

    /// Hadamard gate matrix (2x2), row-major, as (re, im) pairs.
    fn hadamard_matrix() -> Vec<(f64, f64)> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        vec![(s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0)]
    }

    /// CNOT gate matrix (4x4), row-major, as (re, im) pairs.
    fn cnot_matrix() -> Vec<(f64, f64)> {
        vec![
            (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
            (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0),
            (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0),
            (0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0),
        ]
    }

    fn default_spec() -> CudaKernelGenSpec {
        CudaKernelGenSpec::f64_sm80()
    }

    #[test]
    fn test_h_gate_emit_ir() {
        let mut gen = CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&default_spec(), &hadamard_matrix(), &[0])
            .expect("generate H kernel");
        let ir = gen.emit_ir(kid).expect("emit IR");
        assert!(
            ir.contains("nvptx64"),
            "IR should mention nvptx64 target; got:\n{ir}"
        );
        assert!(
            ir.contains("define"),
            "IR should contain function definitions; got:\n{ir}"
        );
    }

    #[test]
    fn test_h_gate_emit_ptx() {
        let mut gen = CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&default_spec(), &hadamard_matrix(), &[0])
            .expect("generate H kernel");
        let session = gen.compile().expect("compile");
        let ptx = session.emit_ptx(kid).expect("emit PTX");
        assert!(
            ptx.contains(".visible .entry"),
            "PTX should contain .visible .entry; got:\n{ptx}"
        );
    }

    #[test]
    fn test_cnot_ptx_differs_from_h() {
        let spec = default_spec();

        let mut gen_h = CudaKernelGenerator::new().expect("create H generator");
        let kid_h = gen_h
            .generate(&spec, &hadamard_matrix(), &[0])
            .expect("generate H kernel");
        let session_h = gen_h.compile().expect("compile H");
        let ptx_h = session_h.emit_ptx(kid_h).expect("emit H PTX");

        let mut gen_cnot = CudaKernelGenerator::new().expect("create CNOT generator");
        let kid_cnot = gen_cnot
            .generate(&spec, &cnot_matrix(), &[0, 1])
            .expect("generate CNOT kernel");
        let session_cnot = gen_cnot.compile().expect("compile CNOT");
        let ptx_cnot = session_cnot.emit_ptx(kid_cnot).expect("emit CNOT PTX");

        assert_ne!(ptx_h, ptx_cnot, "H and CNOT should produce different PTX");
    }

    #[test]
    fn test_emit_ir_idempotent() {
        let mut gen = CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&default_spec(), &hadamard_matrix(), &[0])
            .expect("generate kernel");
        let ir1 = gen.emit_ir(kid).expect("first emit_ir");
        let ir2 = gen.emit_ir(kid).expect("second emit_ir");
        assert_eq!(ir1, ir2, "emit_ir should be idempotent");
    }

    #[test]
    fn test_emit_ptx_idempotent() {
        let mut gen = CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&default_spec(), &hadamard_matrix(), &[0])
            .expect("generate kernel");
        let session = gen.compile().expect("compile");
        let ptx1 = session.emit_ptx(kid).expect("first emit_ptx");
        let ptx2 = session.emit_ptx(kid).expect("second emit_ptx");
        assert_eq!(ptx1, ptx2, "emit_ptx should be idempotent");
    }

    #[test]
    #[ignore = "requires CUDA device with nvJitLink support"]
    fn test_h_gate_emit_cubin() {
        let mut gen = CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&default_spec(), &hadamard_matrix(), &[0])
            .expect("generate kernel");
        let session = gen.compile().expect("compile");
        let cubin = session.emit_cubin(kid).expect("emit cubin");
        assert!(cubin.len() >= 4, "cubin should be non-empty");
        assert_eq!(
            &cubin[..4],
            b"\x7fELF",
            "cubin should start with ELF magic"
        );
    }

    // ── GPU execution tests (require a CUDA device) ───────────────────────────

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_statevector_zero_state() {
        let mut sv = super::CudaStatevector::new(3, super::CudaPrecision::F64)
            .expect("alloc statevector");
        sv.zero().expect("zero");
        let amps = sv.download().expect("download");
        assert_eq!(amps.len(), 8);
        assert!((amps[0].0 - 1.0).abs() < 1e-14, "|0> re should be 1");
        for i in 1..8 {
            assert!(amps[i].0.abs() < 1e-14 && amps[i].1.abs() < 1e-14,
                "amp[{i}] should be 0");
        }
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_statevector_upload_download_roundtrip() {
        let data: Vec<(f64, f64)> = (0..4u32)
            .map(|i| (i as f64 * 0.1, i as f64 * -0.05))
            .collect();
        let mut sv = super::CudaStatevector::new(2, super::CudaPrecision::F64)
            .expect("alloc statevector");
        sv.upload(&data).expect("upload");
        let got = sv.download().expect("download");
        for (i, (&want, got)) in data.iter().zip(got.iter()).enumerate() {
            assert!((want.0 - got.0).abs() < 1e-14, "re[{i}] mismatch");
            assert!((want.1 - got.1).abs() < 1e-14, "im[{i}] mismatch");
        }
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_h_gate_apply_to_zero_state() {
        // H|0⟩ = |+⟩ = (|0⟩ + |1⟩) / sqrt(2)
        let spec = default_spec();
        let mut gen = super::CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&spec, &hadamard_matrix(), &[0])
            .expect("generate H kernel");
        let session = gen.compile().expect("compile");
        let exec = super::CudaExecSession::new(&session).expect("create exec session");

        let mut sv = super::CudaStatevector::new(1, super::CudaPrecision::F64)
            .expect("alloc statevector");
        sv.zero().expect("zero");
        exec.apply(kid, &mut sv).expect("apply H");

        let amps = sv.download().expect("download");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].0 - s).abs() < 1e-10, "amp[0].re ≈ 1/√2");
        assert!(amps[0].1.abs() < 1e-10,         "amp[0].im ≈ 0");
        assert!((amps[1].0 - s).abs() < 1e-10, "amp[1].re ≈ 1/√2");
        assert!(amps[1].1.abs() < 1e-10,         "amp[1].im ≈ 0");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_x_gate_apply_to_zero_state() {
        // X|0⟩ = |1⟩
        let x_matrix = vec![
            (0.0, 0.0), (1.0, 0.0),
            (1.0, 0.0), (0.0, 0.0),
        ];
        let spec = default_spec();
        let mut gen = super::CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&spec, &x_matrix, &[0])
            .expect("generate X kernel");
        let session = gen.compile().expect("compile");
        let exec = super::CudaExecSession::new(&session).expect("create exec session");

        let mut sv = super::CudaStatevector::new(1, super::CudaPrecision::F64)
            .expect("alloc statevector");
        sv.zero().expect("zero");
        exec.apply(kid, &mut sv).expect("apply X");

        let amps = sv.download().expect("download");
        assert!(amps[0].0.abs() < 1e-10, "amp[0] ≈ 0 after X");
        assert!((amps[1].0 - 1.0).abs() < 1e-10, "amp[1] ≈ 1 after X");
    }

    #[test]
    #[ignore = "requires CUDA device"]
    fn test_apply_on_larger_statevector() {
        // Apply H to qubit 0 of a 4-qubit |0⟩ statevector.
        // Expected: |0000⟩ + |0001⟩) / sqrt(2)  (qubits ordered LSB = 0)
        let spec = default_spec();
        let mut gen = super::CudaKernelGenerator::new().expect("create generator");
        let kid = gen
            .generate(&spec, &hadamard_matrix(), &[0])
            .expect("generate H kernel");
        let session = gen.compile().expect("compile");
        let exec = super::CudaExecSession::new(&session).expect("create exec session");

        let mut sv = super::CudaStatevector::new(4, super::CudaPrecision::F64)
            .expect("alloc statevector");
        sv.zero().expect("zero");
        exec.apply(kid, &mut sv).expect("apply H on 4-qubit SV");

        let amps = sv.download().expect("download");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].0 - s).abs() < 1e-10, "amp[0] ≈ 1/√2");
        assert!((amps[1].0 - s).abs() < 1e-10, "amp[1] ≈ 1/√2");
        for i in 2..16 {
            assert!(amps[i].0.abs() < 1e-10 && amps[i].1.abs() < 1e-10,
                "amp[{i}] should be 0");
        }
    }
}

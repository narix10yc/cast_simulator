use std::ffi::{CStr, CString};
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use super::ffi;
use super::types::{CudaKernelId, CudaPrecision, ERR_BUF_LEN};
use super::{error_from_buf, CudaStatevector};

// ── CudaKernelGenerator ───────────────────────────────────────────────────────

/// Accumulates gate kernels as NVPTX LLVM IR, then compiles them via
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
        spec: &super::CudaKernelGenSpec,
        matrix: &[(f64, f64)],
        qubits: &[u32],
    ) -> anyhow::Result<CudaKernelId> {
        let mut kernel_id: CudaKernelId = 0;
        let mut err_buf = [0 as std::ffi::c_char; ERR_BUF_LEN];
        let ffi_matrix: Vec<ffi::FfiComplex64> = matrix
            .iter()
            .map(|&(re, im)| ffi::FfiComplex64 { re, im })
            .collect();
        let status = unsafe {
            ffi::cast_cuda_kernel_generator_generate(
                self.raw.as_ptr(),
                spec as *const super::CudaKernelGenSpec,
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
    /// Must be called before [`compile`](Self::compile).
    pub fn emit_ir(&mut self, kernel_id: CudaKernelId) -> anyhow::Result<String> {
        let mut err_buf = [0 as std::ffi::c_char; ERR_BUF_LEN];

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
    /// The compilation session is a pure-Rust value owning the PTX/cubin data; the underlying
    /// C++ compilation object is deleted before this function returns.
    pub fn compile(self) -> anyhow::Result<CudaCompilationSession> {
        let mut this = ManuallyDrop::new(self);
        let mut raw_session = std::ptr::null_mut::<ffi::CastCudaCompilationSession>();
        let mut err_buf = [0 as std::ffi::c_char; ERR_BUF_LEN];

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

        // Drain compiled kernel data from the C++ session into a Rust Vec, then delete it.
        let result = drain_compilation_session(raw_session, &mut err_buf);
        unsafe { ffi::cast_cuda_compilation_session_delete(raw_session) };
        result
    }
}

/// Extracts all compiled kernel data from `raw` into a [`CudaCompilationSession`].
fn drain_compilation_session(
    raw: *mut ffi::CastCudaCompilationSession,
    err_buf: &mut [std::ffi::c_char; ERR_BUF_LEN],
) -> anyhow::Result<CudaCompilationSession> {
    let n = unsafe { ffi::cast_cuda_compilation_session_n_kernels(raw) };
    let mut kernels = Vec::with_capacity(n as usize);

    for idx in 0..n {
        let kernel_id =
            unsafe { ffi::cast_cuda_compilation_session_kernel_id_at(raw, idx) };
        let n_gate_qubits =
            unsafe { ffi::cast_cuda_compilation_session_n_gate_qubits_at(raw, idx) };
        let precision_byte =
            unsafe { ffi::cast_cuda_compilation_session_precision_at(raw, idx) };
        let precision = if precision_byte == 0 {
            CudaPrecision::F32
        } else {
            CudaPrecision::F64
        };
        let func_name_ptr =
            unsafe { ffi::cast_cuda_compilation_session_func_name_at(raw, idx) };
        let func_name = unsafe { CStr::from_ptr(func_name_ptr) }
            .to_string_lossy()
            .into_owned();

        let ptx = extract_ptx(raw, kernel_id, err_buf)?;
        let cubin = extract_cubin(raw, kernel_id, err_buf)?;

        kernels.push(CompiledKernel {
            kernel_id,
            n_gate_qubits,
            precision,
            func_name,
            ptx,
            cubin,
        });
    }
    Ok(CudaCompilationSession { kernels })
}

fn extract_ptx(
    raw: *mut ffi::CastCudaCompilationSession,
    kernel_id: CudaKernelId,
    err_buf: &mut [std::ffi::c_char; ERR_BUF_LEN],
) -> anyhow::Result<String> {
    let mut ptx_len: usize = 0;
    let status = unsafe {
        ffi::cast_cuda_compilation_session_emit_ptx(
            raw,
            kernel_id,
            std::ptr::null_mut(),
            0,
            &mut ptx_len,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        return Err(anyhow::anyhow!(error_from_buf(err_buf)));
    }
    let mut buf = vec![0u8; ptx_len + 1];
    let status = unsafe {
        ffi::cast_cuda_compilation_session_emit_ptx(
            raw,
            kernel_id,
            buf.as_mut_ptr().cast(),
            buf.len(),
            std::ptr::null_mut(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        return Err(anyhow::anyhow!(error_from_buf(err_buf)));
    }
    buf.truncate(ptx_len);
    String::from_utf8(buf).map_err(|e| anyhow::anyhow!("PTX is not valid UTF-8: {e}"))
}

fn extract_cubin(
    raw: *mut ffi::CastCudaCompilationSession,
    kernel_id: CudaKernelId,
    err_buf: &mut [std::ffi::c_char; ERR_BUF_LEN],
) -> anyhow::Result<Vec<u8>> {
    let mut cubin_len: usize = 0;
    let status = unsafe {
        ffi::cast_cuda_compilation_session_emit_cubin(
            raw,
            kernel_id,
            std::ptr::null_mut(),
            0,
            &mut cubin_len,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        return Err(anyhow::anyhow!(error_from_buf(err_buf)));
    }
    let mut buf = vec![0u8; cubin_len];
    let status = unsafe {
        ffi::cast_cuda_compilation_session_emit_cubin(
            raw,
            kernel_id,
            buf.as_mut_ptr(),
            buf.len(),
            std::ptr::null_mut(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        return Err(anyhow::anyhow!(error_from_buf(err_buf)));
    }
    Ok(buf)
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

/// Metadata and artifacts for a single compiled kernel.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub kernel_id: CudaKernelId,
    pub n_gate_qubits: u32,
    pub precision: CudaPrecision,
    pub func_name: String,
    /// PTX assembly text (null-terminated, ASCII).
    pub ptx: String,
    /// cubin ELF binary (`\x7fELF…`).
    pub cubin: Vec<u8>,
}

/// A pure-Rust compilation session owning PTX/cubin for each compiled kernel.
///
/// Created by [`CudaKernelGenerator::compile`]. The underlying C++ compilation object
/// is fully drained and deleted before this value is returned.
#[derive(Debug)]
pub struct CudaCompilationSession {
    pub kernels: Vec<CompiledKernel>,
}

impl CudaCompilationSession {
    /// Returns the PTX assembly text for the kernel identified by `kernel_id`.
    pub fn emit_ptx(&self, kernel_id: CudaKernelId) -> anyhow::Result<String> {
        self.kernels
            .iter()
            .find(|k| k.kernel_id == kernel_id)
            .map(|k| k.ptx.clone())
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found in compilation session", kernel_id))
    }

    /// Returns the cubin binary for the kernel identified by `kernel_id`.
    pub fn emit_cubin(&self, kernel_id: CudaKernelId) -> anyhow::Result<Vec<u8>> {
        self.kernels
            .iter()
            .find(|k| k.kernel_id == kernel_id)
            .map(|k| k.cubin.clone())
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found in compilation session", kernel_id))
    }
}

// ── CudaExecSession ───────────────────────────────────────────────────────────

struct ExecEntry {
    kernel_id: CudaKernelId,
    n_gate_qubits: u32,
    precision: CudaPrecision,
    cu_module: *mut std::ffi::c_void,
    cu_function: *mut std::ffi::c_void,
}

/// Holds CUDA modules loaded from a [`CudaCompilationSession`], ready to launch.
///
/// Owns the `CUmodule`/`CUfunction` handles; modules are unloaded on drop.
pub struct CudaExecSession {
    entries: Vec<ExecEntry>,
}

impl Drop for CudaExecSession {
    fn drop(&mut self) {
        for e in &self.entries {
            unsafe { ffi::cast_cuda_module_unload(e.cu_module) };
        }
    }
}

impl fmt::Debug for CudaExecSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaExecSession")
            .field("n_kernels", &self.entries.len())
            .finish_non_exhaustive()
    }
}

impl CudaExecSession {
    /// Loads all compiled kernels from `session` into CUDA driver modules.
    ///
    /// Initialises the CUDA driver once per process (device 0).
    pub fn new(session: &CudaCompilationSession) -> anyhow::Result<Self> {
        let mut this = Self { entries: Vec::with_capacity(session.kernels.len()) };

        for kernel in &session.kernels {
            let mut err_buf = [0 as std::ffi::c_char; ERR_BUF_LEN];

            let ptx_cstr = CString::new(kernel.ptx.as_bytes())
                .map_err(|e| anyhow::anyhow!("PTX contains null byte: {e}"))?;
            let cu_module = unsafe {
                ffi::cast_cuda_ptx_load(ptx_cstr.as_ptr(), err_buf.as_mut_ptr(), err_buf.len())
            };
            if cu_module.is_null() {
                return Err(anyhow::anyhow!(
                    "failed to load PTX for '{}': {}",
                    kernel.func_name,
                    error_from_buf(&err_buf)
                ));
            }

            let func_cstr = CString::new(kernel.func_name.as_bytes())
                .map_err(|e| anyhow::anyhow!("func_name contains null byte: {e}"))?;
            let cu_function = unsafe {
                ffi::cast_cuda_module_get_function(
                    cu_module,
                    func_cstr.as_ptr(),
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if cu_function.is_null() {
                // Unload the module we just loaded (not yet in entries, so drop won't touch it).
                unsafe { ffi::cast_cuda_module_unload(cu_module) };
                return Err(anyhow::anyhow!(
                    "failed to get function '{}': {}",
                    kernel.func_name,
                    error_from_buf(&err_buf)
                ));
            }

            this.entries.push(ExecEntry {
                kernel_id: kernel.kernel_id,
                n_gate_qubits: kernel.n_gate_qubits,
                precision: kernel.precision,
                cu_module,
                cu_function,
            });
        }
        Ok(this)
    }

    /// Applies the kernel identified by `kernel_id` to `sv` in-place.
    ///
    /// Synchronises the device before returning.
    pub fn apply(&self, kernel_id: CudaKernelId, sv: &mut CudaStatevector) -> anyhow::Result<()> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.kernel_id == kernel_id)
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found in exec session", kernel_id))?;

        if sv.n_qubits() < entry.n_gate_qubits {
            anyhow::bail!("statevector has fewer qubits than the gate kernel requires");
        }
        if sv.precision() as u8 != entry.precision as u8 {
            anyhow::bail!("statevector precision does not match kernel precision");
        }

        let mut err_buf = [0 as std::ffi::c_char; ERR_BUF_LEN];
        let status = unsafe {
            ffi::cast_cuda_kernel_apply(
                entry.cu_function,
                sv.dptr(),
                entry.n_gate_qubits,
                sv.n_qubits(),
                entry.precision as u8,
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

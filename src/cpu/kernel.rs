//! CPU statevector simulator with LLVM JIT kernel compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CpuKernelManager`].
//! 2. Call [`CpuKernelManager::generate`] for each gate variant you need.
//! 3. Allocate a [`CPUStatevector`] and drive simulation by calling
//!    [`CpuKernelManager::apply`].

use super::*;
use crate::types::{Precision, QuantumGate};

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

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
    /// Sensible defaults for single-precision (native SIMD, ImmValue).
    pub fn f32() -> Self {
        Self {
            precision: Precision::F32,
            simd_width: super::native_simd_width(),
            mode: MatrixLoadMode::ImmValue,
            ztol: 1e-6,
            otol: 1e-6,
        }
    }

    /// Sensible defaults for double-precision (native SIMD, ImmValue).
    pub fn f64() -> Self {
        Self {
            precision: Precision::F64,
            simd_width: super::native_simd_width(),
            mode: MatrixLoadMode::ImmValue,
            ztol: 1e-12,
            otol: 1e-12,
        }
    }
}

/// Opaque handle returned by [`CpuKernelManager::generate`], used to identify a compiled
/// kernel inside a [`CpuKernelManager`].
pub type KernelId = u64;

const ERR_BUF_LEN: usize = 1024;

// ── FFI declarations ──────────────────────────────────────────────────────────

/// Mirrors the "Exported to Rust" section of `src/cpp/cpu/cpu.h`.
///
/// Types not listed here (`cast_cpu_launch_args_t`, `cast_cpu_kernel_entry_t`)
/// are internal to C++; their Rust mirrors live outside this module (see
/// `CastCpuLaunchArgs`).
mod ffi {
    use super::{CPUKernelGenSpec, KernelId, Precision, SimdWidth};
    use crate::cpu::MatrixLoadMode;
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
}

// ── CastCpuLaunchArgs ─────────────────────────────────────────────────────────

/// Matches `cast_cpu_launch_args_t` in `cpu.h`.
/// Each JIT-compiled kernel reads its work range and matrix pointer from this struct.
#[repr(C)]
struct CpuLaunchArgs {
    sv: *mut c_void,
    ctr_begin: u64,
    ctr_end: u64,
    p_mat: *mut c_void,
}

// ── MatrixBuffer ─────────────────────────────────────────────────────────────

/// Typed matrix buffer for StackLoad dispatch; ensures correct scalar alignment.
/// Built once at kernel construction time and reused across all `apply` calls.
enum MatrixBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    Empty,
}

impl MatrixBuffer {
    fn from_ffi(mode: MatrixLoadMode, precision: Precision, ffi_data: &[ffi::c64]) -> Self {
        if !matches!(mode, MatrixLoadMode::StackLoad) || ffi_data.is_empty() {
            return MatrixBuffer::Empty;
        }
        match precision {
            Precision::F32 => {
                let v: Vec<f32> = ffi_data
                    .iter()
                    .flat_map(|c| [c.re as f32, c.im as f32])
                    .collect();
                MatrixBuffer::F32(v)
            }
            Precision::F64 => {
                let v: Vec<f64> = ffi_data.iter().flat_map(|c| [c.re, c.im]).collect();
                MatrixBuffer::F64(v)
            }
        }
    }

    fn as_ptr(&self) -> *const c_void {
        match self {
            MatrixBuffer::F32(v) => v.as_ptr().cast(),
            MatrixBuffer::F64(v) => v.as_ptr().cast(),
            MatrixBuffer::Empty => std::ptr::null(),
        }
    }
}

// ── FFI helpers ───────────────────────────────────────────────────────────

/// Runs O1 optimisation on the kernel and returns the LLVM IR text (two-call
/// pattern: first call queries length, second fills the buffer).
fn ffi_emit_ir(gen: *mut ffi::KernelGenerator, kernel_id: KernelId) -> anyhow::Result<String> {
    let mut err_buf = [0 as c_char; ERR_BUF_LEN];

    let mut ir_len: usize = 0;
    let status = unsafe {
        ffi::cast_cpu_kernel_generator_emit_ir(
            gen,
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
        ffi::cast_cpu_kernel_generator_emit_ir(
            gen,
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

// ── JIT session / generator RAII handles ─────────────────────────────────

/// RAII wrapper for a C++ LLJIT session.  Shared via `Arc` across all
/// kernels compiled in the same batch — the session (and thus the compiled
/// code pages) stays alive as long as any kernel from the batch is alive.
struct JitSessionHandle(NonNull<ffi::JitSession>);

impl Drop for JitSessionHandle {
    fn drop(&mut self) {
        unsafe { ffi::cast_cpu_jit_session_delete(self.0.as_ptr()) };
    }
}

// The session is never mutated after construction.
unsafe impl Send for JitSessionHandle {}
unsafe impl Sync for JitSessionHandle {}

/// RAII wrapper for the C++ kernel generator.  On drop, the generator is
/// deleted via `cast_cpu_kernel_generator_delete`.
struct GeneratorHandle(NonNull<ffi::KernelGenerator>);

impl GeneratorHandle {
    fn new() -> anyhow::Result<Self> {
        let raw = unsafe { ffi::cast_cpu_kernel_generator_new() };
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| anyhow::anyhow!("failed to create CPU kernel generator"))
    }

    fn as_ptr(&self) -> *mut ffi::KernelGenerator {
        self.0.as_ptr()
    }

    /// Consumes the handle and calls `finish`, returning the JIT session and
    /// kernel records.  On success the C++ generator is deleted by `finish`;
    /// on failure it is deleted here.
    fn finish(self) -> anyhow::Result<(NonNull<ffi::JitSession>, *mut ffi::KernelRecord, usize)> {
        let mut raw_session: *mut ffi::JitSession = std::ptr::null_mut();
        let mut raw_records: *mut ffi::KernelRecord = std::ptr::null_mut();
        let mut n_records: usize = 0;
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        let status = unsafe {
            ffi::cast_cpu_kernel_generator_finish(
                self.as_ptr(),
                &mut raw_session,
                &mut raw_records,
                &mut n_records,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };

        // `finish` deletes the C++ generator on success.  On failure the
        // generator is left intact — our Drop will clean it up.
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // Ownership transferred to the C++ side on success — prevent our
        // Drop from double-freeing.
        std::mem::forget(self);

        let jit_session = NonNull::new(raw_session)
            .ok_or_else(|| anyhow::anyhow!("finish returned null session"))?;
        Ok((jit_session, raw_records, n_records))
    }
}

impl Drop for GeneratorHandle {
    fn drop(&mut self) {
        unsafe { ffi::cast_cpu_kernel_generator_delete(self.as_ptr()) };
    }
}

// ── CpuKernelManager ──────────────────────────────────────────────────────

/// A single compiled kernel owned by the manager.
///
/// Holds a shared reference to the LLJIT session (to keep the compiled code
/// pages alive), the JIT-compiled function pointer, the source gate, and
/// optional diagnostics (LLVM IR / native assembly text).
struct KernelEntry {
    /// Shared LLJIT session; multiple kernels from the same batch share this.
    /// Not read directly — kept alive to pin the JIT-compiled code pages.
    _jit_session: Arc<JitSessionHandle>,
    /// The source gate this kernel was compiled from.  Also used to query
    /// `n_gate_qubits` at apply time (avoids storing a redundant copy).
    gate: Arc<QuantumGate>,
    /// JIT-compiled entry point.
    func: unsafe extern "C" fn(*mut c_void),
    precision: Precision,
    simd_width: SimdWidth,
    /// Pre-built matrix buffer for StackLoad dispatch (Empty for ImmValue).
    matrix_buf: MatrixBuffer,
    /// Cached LLVM IR text (only if generated with diagnostics).
    ir_text: Option<String>,
    /// Cached native assembly text (only if generated with diagnostics).
    asm_text: Option<String>,
}

impl KernelEntry {
    /// Applies this kernel to `statevector` in-place using a thread pool.
    fn apply_kernel(&self, statevector: &mut CPUStatevector, n_threads: u32) -> anyhow::Result<()> {
        if statevector.precision() != self.precision {
            return Err(anyhow::anyhow!(
                "statevector precision does not match kernel"
            ));
        }
        if statevector.simd_width() != self.simd_width {
            return Err(anyhow::anyhow!(
                "statevector SIMD width does not match kernel"
            ));
        }

        let n_gate_qubits = self.gate.n_qubits() as u32;
        let n_qubits = statevector.n_qubits();
        let simd_s = get_simd_s(self.simd_width, self.precision);
        let n_task_bits = n_qubits
            .checked_sub(n_gate_qubits + simd_s)
            .ok_or_else(|| anyhow::anyhow!("statevector has too few qubits for this kernel"))?;

        let n_tasks: u64 = 1 << n_task_bits;
        let n_threads = if n_threads == 0 {
            super::get_num_threads()
        } else {
            n_threads
        };
        let n_threads = (n_threads as u64).min(n_tasks).max(1) as u32;
        let n_tasks_per_thread = n_tasks / n_threads as u64;

        log::debug!(
            "cpu: apply kernel ({} gate qubits, {} tasks, {} thread(s))",
            n_gate_qubits,
            n_tasks,
            n_threads,
        );

        let entry = self.func;
        let sv_addr = statevector.raw_mut_ptr() as usize;
        let p_mat_addr = self.matrix_buf.as_ptr() as usize;

        // SAFETY:
        // - `entry` is a valid JIT-compiled function pointer backed by the LLJIT session.
        // - `sv_addr` points to a valid, aligned statevector buffer; the `&mut` borrow
        //   ensures exclusive access.
        // - Each thread receives a disjoint `[ctr_begin, ctr_end)` counter range.
        // - `p_mat_addr` is either null (ImmValue) or points to `self.matrix_buf`.
        std::thread::scope(|s| {
            for i in 0..n_threads {
                let ctr_begin = n_tasks_per_thread * i as u64;
                let ctr_end = if i + 1 == n_threads {
                    n_tasks
                } else {
                    n_tasks_per_thread * (i as u64 + 1)
                };
                s.spawn(move || {
                    let mut args = CpuLaunchArgs {
                        sv: sv_addr as *mut c_void,
                        ctr_begin,
                        ctr_end,
                        p_mat: p_mat_addr as *mut c_void,
                    };
                    unsafe { entry(&mut args as *mut CpuLaunchArgs as *mut c_void) };
                });
            }
        });

        Ok(())
    }
}

// All fields are immutable after construction.  The JitSessionHandle is
// shared via Arc, function pointers are Copy, matrix_buf is read-only.
unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

/// Metadata for a kernel that has been added to the C++ generator but not
/// yet JIT-compiled.  Promoted to a [`KernelEntry`] by [`CpuKernelManager::finalize`].
struct PendingKernel {
    gate: Arc<QuantumGate>,
    /// C++ kernel id within the current generator batch.  Not read on the
    /// Rust side — present for debugging and to document the mapping.
    #[allow(dead_code)]
    raw_kid: KernelId,
    /// Captured LLVM IR text (only if `request_llvm_ir` was set).
    ir_text: Option<String>,
}

// ── Kernel deduplication ─────────────────────────────────────────────────────

/// Key capturing every input that affects the compiled kernel output.
///
/// The spec is fixed per manager, so only the gate-specific fields vary.
///
/// - **ImmValue** mode bakes matrix values as immediates → key uses raw bytes.
/// - **StackLoad** mode loads values at runtime → key uses the element
///   classification (zero / +1 / −1 / other) since only the sparsity pattern
///   is baked into the generated code.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DedupKey {
    /// Raw matrix bytes (ImmValue) or per-scalar classification (StackLoad).
    matrix_fingerprint: Vec<u8>,
    qubits: Vec<u32>,
}

impl DedupKey {
    fn new(spec: &CPUKernelGenSpec, gate: &QuantumGate) -> Self {
        let matrix_fingerprint = match spec.mode {
            MatrixLoadMode::ImmValue => matrix_bytes(gate),
            MatrixLoadMode::StackLoad => matrix_signature(gate, spec.ztol, spec.otol),
        };
        Self {
            matrix_fingerprint,
            qubits: gate.qubits().to_vec(),
        }
    }
}

fn matrix_bytes(gate: &QuantumGate) -> Vec<u8> {
    gate.matrix().as_bytes().to_vec()
}

/// Classify each real/imaginary scalar: 0 = zero, 1 = +1, 2 = −1, 3 = other.
fn classify_scalar(val: f64, ztol: f64, otol: f64) -> u8 {
    if val.abs() < ztol {
        0
    } else if (val - 1.0).abs() < otol {
        1
    } else if (val + 1.0).abs() < otol {
        2
    } else {
        3
    }
}

/// Per-scalar classification of the gate matrix, capturing the sparsity pattern
/// that determines generated code in StackLoad mode.
fn matrix_signature(gate: &QuantumGate, ztol: f64, otol: f64) -> Vec<u8> {
    gate.matrix()
        .data()
        .iter()
        .flat_map(|z| {
            [
                classify_scalar(z.re, ztol, otol),
                classify_scalar(z.im, ztol, otol),
            ]
        })
        .collect()
}

struct CpuManagerInner {
    next_id: KernelId,
    /// Compiled, ready-to-apply kernel entries.
    entries: HashMap<KernelId, Arc<KernelEntry>>,
    dedup: HashMap<DedupKey, KernelId>,
    /// Kernels added to the generator but not yet JIT-compiled.
    /// Order matches the C++ generator's internal kernel vector.
    pending: Vec<(KernelId, PendingKernel)>,
}

/// CPU kernel manager: generate → JIT-compile → apply.
///
/// ```ignore
/// let mgr = CpuKernelManager::new(spec);
/// let k1 = mgr.generate(&gate_a)?;         // LLVM IR (deferred JIT)
/// let k2 = mgr.generate(&gate_b)?;         // batched with k1
/// mgr.apply(k1, &mut statevector, 0)?;     // auto-finalizes both
/// mgr.apply(k2, &mut statevector, 0)?;     // already compiled
/// ```
///
/// Kernels are accumulated in a shared C++ generator during [`generate`]
/// calls and batch-compiled into a single LLJIT session on the first
/// [`apply`], [`finalize`], or diagnostic query.  This amortises LLJIT
/// setup overhead and shares compiled code pages across kernels.
///
/// Identical gates are deduplicated via content-based keys.
/// `apply` dispatches work across threads synchronously; the manager lock
/// is released before kernel execution.
pub struct CpuKernelManager {
    spec: CPUKernelGenSpec,
    inner: std::sync::Mutex<CpuManagerInner>,
    /// Shared C++ generator accumulating un-compiled kernels.
    /// Lock ordering: always acquire `generator` before `inner`.
    generator: std::sync::Mutex<Option<GeneratorHandle>>,
}

impl CpuKernelManager {
    /// Creates a new kernel manager bound to the given spec.
    pub fn new(spec: CPUKernelGenSpec) -> Self {
        Self {
            spec,
            inner: std::sync::Mutex::new(CpuManagerInner {
                next_id: 0,
                entries: HashMap::new(),
                dedup: HashMap::new(),
                pending: Vec::new(),
            }),
            generator: std::sync::Mutex::new(None),
        }
    }

    /// Returns the spec this manager was created with.
    pub fn spec(&self) -> &CPUKernelGenSpec {
        &self.spec
    }

    /// Generates, optimises, and JIT-compiles a gate kernel.
    ///
    /// Runs the full LLVM pipeline on the calling thread, then briefly locks
    /// to store the result.  The `Arc<QuantumGate>` is cloned and stored
    /// alongside the compiled code so the source gate remains accessible via
    /// [`gate`](Self::gate).
    ///
    /// Multiple threads may call `generate` concurrently.
    pub fn generate(&self, gate: &Arc<QuantumGate>) -> anyhow::Result<KernelId> {
        self.generate_inner(gate.clone(), false, false)
    }

    /// Like [`generate`](Self::generate), but also captures diagnostics.
    ///
    /// - `request_llvm_ir`: capture optimised LLVM IR (retrieve via [`emit_ir`](Self::emit_ir)).
    /// - `request_asm`: capture native assembly (retrieve via [`emit_asm`](Self::emit_asm)).
    pub fn generate_with_diagnostics(
        &self,
        gate: &Arc<QuantumGate>,
        request_llvm_ir: bool,
        request_asm: bool,
    ) -> anyhow::Result<KernelId> {
        self.generate_inner(gate.clone(), request_llvm_ir, request_asm)
    }

    fn generate_inner(
        &self,
        gate: Arc<QuantumGate>,
        request_llvm_ir: bool,
        request_asm: bool,
    ) -> anyhow::Result<KernelId> {
        // ── First dedup check (fast path, no generator lock) ─────────────
        let spec = &self.spec;
        let want_dedup = !request_llvm_ir && !request_asm;
        let key = if want_dedup {
            let k = DedupKey::new(spec, &gate);
            let guard = self.inner.lock().unwrap();
            if let Some(&id) = guard.dedup.get(&k) {
                return Ok(id);
            }
            Some(k)
        } else {
            None
        };

        // Extract FFI data upfront so the borrow on `gate` is released
        // before we move the Arc into the pending entry.
        let ffi_matrix: Vec<ffi::c64> = gate
            .matrix()
            .data()
            .iter()
            .map(|c| ffi::c64 { re: c.re, im: c.im })
            .collect();
        let qubits = gate.qubits().to_vec();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // ── Acquire generator lock for the remainder of the function ────
        // Lock order: generator → inner.  Held through the C++ codegen and
        // the pending push so the generator state and pending list stay
        // consistent — no concurrent generate_inner can race with us.
        let mut gen_guard = self.generator.lock().unwrap();

        // ── Second dedup re-check, BEFORE enqueueing into the C++ batch ──
        // Another generate_inner may have completed between our first
        // dedup miss and our acquisition of the generator lock.  We must
        // catch the duplicate here so we never add an orphan kernel to the
        // C++ generator's batch (which would cause a record/pending size
        // mismatch in finalize).
        if let Some(ref key) = key {
            let inner = self.inner.lock().unwrap();
            if let Some(&existing_id) = inner.dedup.get(key) {
                return Ok(existing_id);
            }
        }

        // ── Create generator if this is the first kernel of the batch ───
        let gen = match gen_guard.as_ref() {
            Some(g) => g,
            None => {
                *gen_guard = Some(GeneratorHandle::new()?);
                gen_guard.as_ref().unwrap()
            }
        };

        let mut raw_kid: KernelId = 0;
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_generate(
                gen.as_ptr(),
                spec as *const CPUKernelGenSpec,
                ffi_matrix.as_ptr(),
                ffi_matrix.len(),
                qubits.as_ptr(),
                qubits.len(),
                &mut raw_kid,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // ── Diagnostics (IR + asm capture) ────────────────────────────────
        // emit_ir triggers O1 optimisation for this kernel (idempotent —
        // finish will skip re-optimising it).  request_asm only sets a flag;
        // assembly is emitted during finish.
        let ir_text = if request_llvm_ir {
            Some(ffi_emit_ir(gen.as_ptr(), raw_kid)?)
        } else {
            None
        };
        if request_asm {
            let status = unsafe {
                ffi::cast_cpu_kernel_generator_request_asm(
                    gen.as_ptr(),
                    raw_kid,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if status != 0 {
                return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
            }
        }

        // ── Register pending entry ───────────────────────────────────────
        // We still hold the generator lock, so no concurrent generate_inner
        // can have inserted into dedup since our second re-check above.
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        if let Some(key) = key {
            inner.dedup.insert(key, id);
        }
        inner.pending.push((
            id,
            PendingKernel {
                gate,
                raw_kid,
                ir_text,
            },
        ));
        log::info!(
            "cpu: queued kernel {} ({} qubits, {:?} precision, batch pos {})",
            id,
            qubits.len(),
            spec.precision,
            inner.pending.len() - 1,
        );
        Ok(id)
    }

    /// Batch-compiles all pending kernels into a single LLJIT session.
    ///
    /// Called automatically by [`apply`](Self::apply) and diagnostic
    /// accessors when the requested kernel is still pending.  Explicit
    /// calls allow the caller to control when the compilation cost is paid.
    ///
    /// No-op if there are no pending kernels.
    pub fn finalize(&self) -> anyhow::Result<()> {
        // ── Take the generator and pending list atomically ───────────────
        // Hold the generator lock while taking pending so concurrent
        // generate_inner cannot push new entries between the two takes
        // (which would leave them stranded — their kernels are in a
        // different generator than the one we're about to finish).
        // Lock order: generator → inner.
        let mut gen_guard = self.generator.lock().unwrap();
        let gen_handle = match gen_guard.take() {
            Some(g) => g,
            None => return Ok(()), // no generator → nothing pending
        };
        let pending = {
            let mut inner = self.inner.lock().unwrap();
            std::mem::take(&mut inner.pending)
        };
        // Release the generator lock — concurrent generate_inner calls
        // can now create a fresh GeneratorHandle for the next batch.
        drop(gen_guard);

        if pending.is_empty() {
            // Generator existed but had no pending kernels (should not happen
            // in normal use, but handle gracefully — drop the generator).
            return Ok(());
        }

        // ── JIT-compile the entire batch (no locks held) ─────────────────
        let (jit_session, raw_records, n_records) = gen_handle.finish()?;
        let session = Arc::new(JitSessionHandle(jit_session));

        debug_assert_eq!(
            n_records,
            pending.len(),
            "finish returned {} records but {} pending kernels",
            n_records,
            pending.len(),
        );

        let records = unsafe { std::slice::from_raw_parts(raw_records, n_records) };

        // ── Promote pending entries to compiled entries ───────────────────
        let mut inner = self.inner.lock().unwrap();
        for ((kid, pend), record) in pending.iter().zip(records.iter()) {
            let entry_fn = record
                .entry
                .ok_or_else(|| anyhow::anyhow!("finish returned null entry pointer"))?;

            let matrix_data = if record.matrix.is_null() {
                &[] as &[ffi::c64]
            } else {
                unsafe { std::slice::from_raw_parts(record.matrix, record.matrix_len) }
            };

            let asm_text = if record.asm_text.is_null() {
                None
            } else {
                Some(
                    unsafe { std::ffi::CStr::from_ptr(record.asm_text) }
                        .to_string_lossy()
                        .into_owned(),
                )
            };

            let precision = record.metadata.precision;
            let simd_width = record.metadata.simd_width;
            let matrix_buf = MatrixBuffer::from_ffi(record.metadata.mode, precision, matrix_data);

            inner.entries.insert(
                *kid,
                Arc::new(KernelEntry {
                    _jit_session: session.clone(),
                    gate: pend.gate.clone(),
                    func: entry_fn,
                    precision,
                    simd_width,
                    matrix_buf,
                    ir_text: pend.ir_text.clone(),
                    asm_text,
                }),
            );
        }
        unsafe { ffi::cast_cpu_jit_kernel_records_free(raw_records, n_records) };

        log::info!(
            "cpu: finalized batch of {} kernels into shared JIT session",
            n_records,
        );
        Ok(())
    }

    /// Ensures kernel `id` is compiled, calling [`finalize`](Self::finalize)
    /// if it is still pending.  Returns the compiled entry.
    fn ensure_compiled(&self, id: KernelId) -> anyhow::Result<Arc<KernelEntry>> {
        // Fast path: already compiled.
        {
            let inner = self.inner.lock().unwrap();
            if let Some(entry) = inner.entries.get(&id) {
                return Ok(entry.clone());
            }
            if !inner.pending.iter().any(|(kid, _)| *kid == id) {
                return Err(anyhow::anyhow!("kernel id {} not found", id));
            }
        }
        // Kernel is pending — finalize the batch.
        self.finalize()?;
        let inner = self.inner.lock().unwrap();
        inner
            .entries
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found after finalize", id))
    }

    /// Returns the optimised LLVM IR for a kernel, if it was generated with
    /// diagnostics.  Triggers [`finalize`](Self::finalize) if the kernel is
    /// still pending.
    pub fn emit_ir(&self, id: KernelId) -> Option<String> {
        self.ensure_compiled(id).ok()?.ir_text.clone()
    }

    /// Returns the native assembly for a kernel, if it was generated with
    /// diagnostics.  Triggers [`finalize`](Self::finalize) if the kernel is
    /// still pending.
    pub fn emit_asm(&self, id: KernelId) -> Option<String> {
        self.ensure_compiled(id).ok()?.asm_text.clone()
    }

    /// Returns the source gate for the kernel identified by `id`.
    pub fn gate(&self, id: KernelId) -> Option<Arc<QuantumGate>> {
        let inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.entries.get(&id) {
            return Some(entry.gate.clone());
        }
        inner
            .pending
            .iter()
            .find(|(kid, _)| *kid == id)
            .map(|(_, p)| p.gate.clone())
    }

    /// Applies the kernel identified by `id` to `statevector` in-place.
    ///
    /// `n_threads`: number of worker threads. Pass `0` to use the hardware
    /// thread count.
    ///
    /// If the kernel is still pending, the entire batch is finalized first.
    /// The manager lock is released before kernel execution begins, so
    /// concurrent `generate` and `apply` calls on other kernels are not blocked.
    pub fn apply(
        &self,
        id: KernelId,
        statevector: &mut CPUStatevector,
        n_threads: u32,
    ) -> anyhow::Result<()> {
        let entry = self.ensure_compiled(id)?;
        // Lock released — kernel execution does not block the manager.
        entry.apply_kernel(statevector, n_threads)
    }

    /// Times a kernel adaptively within `budget_s` seconds.
    ///
    /// Delegates to [`crate::timing::time_adaptive`]; see that function for
    /// details on the warmup / measurement budget split.
    pub fn time_adaptive(
        &self,
        id: KernelId,
        statevector: &mut CPUStatevector,
        n_threads: u32,
        budget_s: f64,
    ) -> anyhow::Result<crate::timing::TimingStats> {
        crate::timing::time_adaptive(|| self.apply(id, statevector, n_threads), budget_s)
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

//! CPU statevector simulator with LLVM JIT kernel compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CpuKernelManager`].
//! 2. Build a [`KernelGenRequest`] from a spec + gate (or construct it
//!    directly) and submit it via [`CpuKernelManager::generate`] for every
//!    gate variant you need.
//! 3. Allocate a [`CPUStatevector`] and drive simulation by calling
//!    [`CpuKernelManager::apply`].

use super::*;
use crate::types::{Complex, Precision, QuantumGate};

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CPUKernelGenSpec
// ---------------------------------------------------------------------------

/// Codegen-invariant configuration: precision, SIMD width, matrix load mode,
/// and the tolerances used to classify matrix elements in StackLoad mode.
///
/// Not stored on the manager — each [`KernelGenRequest`] carries its own.
/// Provided as a bundle so callers can reuse "their usual" defaults across
/// many requests.
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

// ---------------------------------------------------------------------------
// KernelGenRequest
// ---------------------------------------------------------------------------

/// Complete description of a single kernel generation: spec + gate identity
/// + diagnostic flags.  This is the canonical Rust-side source of truth and
/// the key for kernel deduplication inside [`CpuKernelManager`].
///
/// Construction via [`KernelGenRequest::from_gate`] is the recommended path.
/// It computes a canonical *dedup fingerprint* once:
///
/// - `ImmValue` — raw matrix bytes (values are baked into the IR).
/// - `StackLoad` — per-scalar classification (0 = zero, 1 = +1, 2 = −1,
///   3 = other) of every real/imaginary component, since only the sparsity
///   pattern is baked into the generated code.
///
/// The fingerprint drives `Hash` and `Eq`, so requests that produce
/// identical compiled output collide in the dedup map even when the live
/// matrix values differ (StackLoad case).
///
/// The `matrix` field preserves the original complex values for the FFI
/// handoff; the fingerprint is only used for hashing/equality.
#[derive(Debug, Clone)]
pub struct KernelGenRequest {
    pub spec: CPUKernelGenSpec,
    pub qubits: Vec<u32>,
    matrix: Vec<Complex>,
    dedup_fingerprint: Vec<u8>,
    pub capture_ir: bool,
    pub capture_asm: bool,
}

impl KernelGenRequest {
    /// Builds a request from a spec and gate, computing the canonical
    /// dedup fingerprint up front.  Diagnostic flags default to `false`;
    /// use [`with_ir`](Self::with_ir) / [`with_asm`](Self::with_asm) to
    /// opt in.
    pub fn from_gate(spec: CPUKernelGenSpec, gate: &QuantumGate) -> Self {
        let matrix: Vec<Complex> = gate.matrix().data().to_vec();
        let qubits = gate.qubits().to_vec();
        let dedup_fingerprint = match spec.mode {
            MatrixLoadMode::ImmValue => gate.matrix().as_bytes().to_vec(),
            MatrixLoadMode::StackLoad => matrix_signature(&matrix, spec.ztol, spec.otol),
        };
        Self {
            spec,
            qubits,
            matrix,
            dedup_fingerprint,
            capture_ir: false,
            capture_asm: false,
        }
    }

    /// Requests capture of the optimized LLVM IR.  The text is returned
    /// from the C++ side alongside the JIT-compiled entry at finalize time
    /// and can be fetched via [`CpuKernelManager::emit_ir`].
    pub fn with_ir(mut self) -> Self {
        self.capture_ir = true;
        self
    }

    /// Requests capture of the native assembly.  The assembly text is
    /// emitted during JIT finalization and can be fetched via
    /// [`CpuKernelManager::emit_asm`].
    pub fn with_asm(mut self) -> Self {
        self.capture_asm = true;
        self
    }

    /// Raw gate matrix (row-major, length 2^n × 2^n for an n-qubit gate).
    pub fn matrix(&self) -> &[Complex] {
        &self.matrix
    }

    /// Number of gate qubits.
    pub fn n_qubits(&self) -> u32 {
        self.qubits.len() as u32
    }
}

impl PartialEq for KernelGenRequest {
    fn eq(&self, other: &Self) -> bool {
        self.spec.precision == other.spec.precision
            && self.spec.simd_width == other.spec.simd_width
            && self.spec.mode == other.spec.mode
            && self.spec.ztol.to_bits() == other.spec.ztol.to_bits()
            && self.spec.otol.to_bits() == other.spec.otol.to_bits()
            && self.qubits == other.qubits
            && self.dedup_fingerprint == other.dedup_fingerprint
            && self.capture_ir == other.capture_ir
            && self.capture_asm == other.capture_asm
    }
}

impl Eq for KernelGenRequest {}

impl Hash for KernelGenRequest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.spec.precision.hash(state);
        self.spec.simd_width.hash(state);
        self.spec.mode.hash(state);
        self.spec.ztol.to_bits().hash(state);
        self.spec.otol.to_bits().hash(state);
        self.qubits.hash(state);
        self.dedup_fingerprint.hash(state);
        self.capture_ir.hash(state);
        self.capture_asm.hash(state);
    }
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

/// Per-scalar classification of the gate matrix, capturing the sparsity
/// pattern baked into the generated code in StackLoad mode.
fn matrix_signature(matrix: &[Complex], ztol: f64, otol: f64) -> Vec<u8> {
    matrix
        .iter()
        .flat_map(|z| {
            [
                classify_scalar(z.re, ztol, otol),
                classify_scalar(z.im, ztol, otol),
            ]
        })
        .collect()
}

/// Opaque handle returned by [`CpuKernelManager::generate`], used to identify
/// a compiled kernel inside a [`CpuKernelManager`].  Nonzero; 0 is reserved
/// as a "no kernel" sentinel.
pub type KernelId = u64;

const ERR_BUF_LEN: usize = 1024;

use crate::ffi::cpu as ffi;

// ---------------------------------------------------------------------------
// CastCpuLaunchArgs
// ---------------------------------------------------------------------------

/// Matches `cast_cpu_launch_args_t` in `ffi_cpu.h`.
/// Each JIT-compiled kernel reads its work range and matrix pointer from this struct.
#[repr(C)]
struct CpuLaunchArgs {
    sv: *mut c_void,
    ctr_begin: u64,
    ctr_end: u64,
    p_mat: *mut c_void,
}

// ---------------------------------------------------------------------------
// MatrixBuffer
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// JIT session / generator RAII handles
// ---------------------------------------------------------------------------

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
            anyhow::bail!(error_from_buf(&err_buf));
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

// ---------------------------------------------------------------------------
// CpuKernelManager
// ---------------------------------------------------------------------------

/// A single compiled kernel owned by the manager.
///
/// Holds a shared reference to the LLJIT session (to keep the compiled code
/// pages alive), the JIT-compiled function pointer, and optional diagnostics.
struct KernelEntry {
    /// Shared LLJIT session; multiple kernels from the same batch share this.
    /// Not read directly — kept alive to pin the JIT-compiled code pages.
    _jit_session: Arc<JitSessionHandle>,
    /// JIT-compiled entry point.
    func: unsafe extern "C" fn(*mut c_void),
    precision: Precision,
    simd_width: SimdWidth,
    n_gate_qubits: u32,
    /// Pre-built matrix buffer for StackLoad dispatch (Empty for ImmValue).
    matrix_buf: MatrixBuffer,
    /// Cached LLVM IR text (only if generated with `capture_ir`).
    ir_text: Option<String>,
    /// Cached native assembly text (only if generated with `capture_asm`).
    asm_text: Option<String>,
}

impl KernelEntry {
    /// Applies this kernel to `sv` in-place using a thread pool.
    fn apply_kernel(&self, sv: &mut CPUStatevector, n_threads: u32) -> anyhow::Result<()> {
        if sv.precision() != self.precision {
            anyhow::bail!("statevector precision does not match kernel");
        }
        if sv.simd_width() != self.simd_width {
            anyhow::bail!("statevector SIMD width does not match kernel");
        }

        let n_qubits = sv.n_qubits();
        let simd_s = get_simd_s(self.simd_width, self.precision);
        let n_task_bits = n_qubits
            .checked_sub(self.n_gate_qubits + simd_s)
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
            self.n_gate_qubits,
            n_tasks,
            n_threads,
        );

        let func = self.func;
        let sv_addr = sv.raw_mut_ptr() as usize;
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
                    unsafe { func(&mut args as *mut CpuLaunchArgs as *mut c_void) };
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
    /// C++-side kernel id within the current generator batch.  Kept for
    /// logging/debug only; after finalize, the Rust-side `KernelId`
    /// identifies the compiled entry.
    #[allow(dead_code)]
    raw_kid: KernelId,
}

struct CpuManagerInner {
    next_id: KernelId,
    /// Compiled, ready-to-apply kernel entries.
    entries: HashMap<KernelId, Arc<KernelEntry>>,
    /// Dedup map: identical requests → same kernel id.
    dedup: HashMap<KernelGenRequest, KernelId>,
    /// Kernels added to the generator but not yet JIT-compiled.
    /// Order matches the C++ generator's internal kernel vector.
    pending: Vec<(KernelId, PendingKernel)>,
}

/// CPU kernel manager: generate → JIT-compile → apply.
///
/// ```ignore
/// let mgr = CpuKernelManager::new();
/// let k1 = mgr.generate(KernelGenRequest::from_gate(spec, &gate_a))?;
/// let k2 = mgr.generate(KernelGenRequest::from_gate(spec, &gate_b))?;
/// mgr.apply(k1, &mut sv, 0)?;     // auto-finalizes both
/// mgr.apply(k2, &mut sv, 0)?;     // already compiled
/// ```
///
/// Kernels are accumulated in a shared C++ generator during [`generate`]
/// calls and batch-compiled into a single LLJIT session on the first
/// [`apply`], [`finalize`], or diagnostic query.  This amortizes LLJIT
/// setup overhead and shares compiled code pages across kernels.
///
/// Identical requests are deduplicated via `Hash + Eq` on [`KernelGenRequest`].
/// `apply` dispatches work across threads synchronously; the manager lock
/// is released before kernel execution.
///
/// **Concurrency:** `generate` may be called from multiple threads, but
/// LLVM IR codegen serializes on the shared C++ generator's lock — only
/// one thread runs the codegen at a time.  `apply` for *different*
/// kernels can run in parallel because each `apply` clones its
/// `KernelEntry` (via `Arc`) before releasing the manager lock.
pub struct CpuKernelManager {
    inner: std::sync::Mutex<CpuManagerInner>,
    /// Shared C++ generator accumulating un-compiled kernels.
    /// Lock ordering: always acquire `generator` before `inner`.
    generator: std::sync::Mutex<Option<GeneratorHandle>>,
}

impl Default for CpuKernelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuKernelManager {
    /// Creates a new kernel manager.  The manager holds no default spec —
    /// each kernel carries its own in the [`KernelGenRequest`].
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(CpuManagerInner {
                next_id: 1,
                entries: HashMap::new(),
                dedup: HashMap::new(),
                pending: Vec::new(),
            }),
            generator: std::sync::Mutex::new(None),
        }
    }

    /// Adds `request` to the shared compile queue and returns its kernel ID.
    ///
    /// LLVM IR generation runs on the calling thread; O1 optimization and
    /// native code generation are deferred until [`apply`](Self::apply),
    /// [`finalize`](Self::finalize), or a diagnostic accessor is called.
    ///
    /// Multiple threads may call `generate` concurrently.
    pub fn generate(&self, request: KernelGenRequest) -> anyhow::Result<KernelId> {
        // ── First dedup check (fast path, no generator lock) ─────────────
        {
            let guard = self.inner.lock().unwrap();
            if let Some(&id) = guard.dedup.get(&request) {
                return Ok(id);
            }
        }

        // Build the FFI payload upfront so the borrow on `request` is held
        // only across the single FFI call.
        let ffi_matrix: Vec<ffi::c64> = request
            .matrix
            .iter()
            .map(|c| ffi::c64 { re: c.re, im: c.im })
            .collect();
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // ── Acquire generator lock for the remainder of the function ────
        // Lock order: generator → inner.  Held through the C++ codegen and
        // the pending push so the generator state and pending list stay
        // consistent — no concurrent generate can race with us.
        let mut gen_guard = self.generator.lock().unwrap();

        // ── Second dedup re-check, BEFORE enqueueing into the C++ batch ──
        // Another generate may have completed between our first dedup miss
        // and our acquisition of the generator lock.  Catch the duplicate
        // here so we never add an orphan kernel to the C++ generator's
        // batch (which would cause a record/pending size mismatch in
        // finalize).
        {
            let inner = self.inner.lock().unwrap();
            if let Some(&existing_id) = inner.dedup.get(&request) {
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

        let ffi_request = ffi::CastCpuKernelGenRequest {
            precision: request.spec.precision,
            simd_width: request.spec.simd_width,
            mode: request.spec.mode,
            ztol: request.spec.ztol,
            otol: request.spec.otol,
            qubits: request.qubits.as_ptr(),
            n_qubits: request.qubits.len(),
            matrix: ffi_matrix.as_ptr(),
            matrix_len: ffi_matrix.len(),
            capture_ir: request.capture_ir,
            capture_asm: request.capture_asm,
        };

        let raw_kid = unsafe {
            ffi::cast_cpu_kernel_generator_generate(
                gen.as_ptr(),
                &ffi_request as *const ffi::CastCpuKernelGenRequest,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if raw_kid == 0 {
            anyhow::bail!(error_from_buf(&err_buf));
        }

        // ── Register pending entry ───────────────────────────────────────
        // We still hold the generator lock, so no concurrent generate can
        // have inserted into dedup since our second re-check above.
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        let n_qubits = request.qubits.len();
        let precision = request.spec.precision;
        inner.dedup.insert(request, id);
        inner.pending.push((id, PendingKernel { raw_kid }));
        log::info!(
            "cpu: queued kernel {} ({} qubits, {:?} precision, batch pos {})",
            id,
            n_qubits,
            precision,
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
    ///
    /// **Failure semantics:** if `finish` returns an error mid-batch, the
    /// kernel IDs in the failed batch are already in `dedup` but will not
    /// appear in `entries`.  Subsequent `apply` / `ensure_compiled` calls
    /// for those IDs return `Err("kernel id N not found")`.  There is no
    /// per-kernel recovery; the manager must be discarded and rebuilt.
    pub(crate) fn finalize(&self) -> anyhow::Result<()> {
        // ── Take the generator and pending list atomically ───────────────
        // Hold the generator lock while taking pending so concurrent
        // generate cannot push new entries between the two takes (which
        // would leave them stranded — their kernels are in a different
        // generator than the one we're about to finish).
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
        // Release the generator lock — concurrent generate calls can now
        // create a fresh GeneratorHandle for the next batch.
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
        for ((kid, _pend), record) in pending.iter().zip(records.iter()) {
            let entry_fn = record
                .entry
                .ok_or_else(|| anyhow::anyhow!("finish returned null entry pointer"))?;

            let matrix_data = if record.matrix.is_null() {
                &[] as &[ffi::c64]
            } else {
                unsafe { std::slice::from_raw_parts(record.matrix, record.matrix_len) }
            };

            let c_string_to_owned = |p: *mut c_char| -> Option<String> {
                if p.is_null() {
                    None
                } else {
                    Some(
                        unsafe { std::ffi::CStr::from_ptr(p) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
            };
            let ir_text = c_string_to_owned(record.ir_text);
            let asm_text = c_string_to_owned(record.asm_text);

            let precision = record.metadata.precision;
            let simd_width = record.metadata.simd_width;
            let matrix_buf = MatrixBuffer::from_ffi(record.metadata.mode, precision, matrix_data);

            inner.entries.insert(
                *kid,
                Arc::new(KernelEntry {
                    _jit_session: session.clone(),
                    func: entry_fn,
                    precision,
                    simd_width,
                    n_gate_qubits: record.metadata.n_gate_qubits,
                    matrix_buf,
                    ir_text,
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
                anyhow::bail!("kernel id {} not found", id);
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

    /// Returns the optimized LLVM IR for a kernel, if it was generated with
    /// `capture_ir`.  Triggers [`finalize`](Self::finalize) if the kernel is
    /// still pending.
    pub fn emit_ir(&self, id: KernelId) -> Option<String> {
        self.ensure_compiled(id).ok()?.ir_text.clone()
    }

    /// Returns the native assembly for a kernel, if it was generated with
    /// `capture_asm`.  Triggers [`finalize`](Self::finalize) if the kernel
    /// is still pending.
    pub fn emit_asm(&self, id: KernelId) -> Option<String> {
        self.ensure_compiled(id).ok()?.asm_text.clone()
    }

    /// Applies the kernel identified by `id` to `sv` in-place.
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
        sv: &mut CPUStatevector,
        n_threads: u32,
    ) -> anyhow::Result<()> {
        let entry = self.ensure_compiled(id)?;
        // Lock released — kernel execution does not block the manager.
        entry.apply_kernel(sv, n_threads)
    }

    /// Times a kernel adaptively within `budget_s` seconds.
    ///
    /// Delegates to [`crate::timing::time_adaptive`]; see that function for
    /// details on the warmup / measurement budget split.
    pub fn time_adaptive(
        &self,
        id: KernelId,
        sv: &mut CPUStatevector,
        n_threads: u32,
        budget_s: f64,
    ) -> anyhow::Result<crate::timing::TimingStats> {
        crate::timing::time_adaptive(|| self.apply(id, sv, n_threads), budget_s)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors / callers
// ---------------------------------------------------------------------------

impl CpuKernelManager {
    /// Convenience: build a request from `(spec, gate)` and call
    /// [`generate`](Self::generate).  Equivalent to
    /// `self.generate(KernelGenRequest::from_gate(spec, gate))`.
    ///
    /// `&Arc<QuantumGate>` coerces to `&QuantumGate` via `Deref`.
    pub fn generate_gate(
        &self,
        spec: CPUKernelGenSpec,
        gate: &QuantumGate,
    ) -> anyhow::Result<KernelId> {
        self.generate(KernelGenRequest::from_gate(spec, gate))
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

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

//! CPU statevector simulator with LLVM JIT kernel compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CpuKernelManager`].
//! 2. Call [`CpuKernelManager::generate`] for each gate variant you need.
//! 3. Allocate a [`CPUStatevector`] and drive simulation by calling
//!    [`CpuKernelManager::apply`].

use super::*;
use crate::types::{Complex, Precision};

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::ptr::NonNull;
use std::time::Instant;

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

/// Opaque handle returned by [`CpuKernelManager::generate`], used to identify a compiled
/// kernel inside a [`CpuKernelManager`].
pub type KernelId = u64;

const ERR_BUF_LEN: usize = 1024;

// ── TimingStats ───────────────────────────────────────────────────────────────

/// Statistics returned by [`CpuKernelManager::time_adaptive`].
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub n_iters: usize,
    pub mean_s: f64,
    pub stddev_s: f64,
    /// Coefficient of variation: `stddev / mean` (scale-free noise metric).
    pub cv: f64,
    pub min_s: f64,
    pub max_s: f64,
}

// ── FFI declarations ──────────────────────────────────────────────────────────

/// Raw C FFI bindings to the LLVM JIT kernel generator (`src/cpp/cpu.h`).
mod ffi {
    use super::{CPUKernelGenSpec, KernelId, Precision, SimdWidth};
    use std::ffi::{c_char, c_void};

    use crate::cpu::MatrixLoadMode;

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
    #[derive(Clone, Copy)]
    pub struct FfiComplex64 {
        pub re: f64,
        pub im: f64,
    }

    /// Matches `cast_cpu_kernel_metadata_t` in `cpu.h`.
    #[repr(C)]
    pub struct KernelMetadata {
        pub kernel_id: KernelId,
        pub precision: Precision,
        pub simd_width: SimdWidth,
        pub mode: MatrixLoadMode,
        pub n_gate_qubits: u32,
    }

    /// Matches `cast_cpu_jit_kernel_record_t` in `cpu.h`.
    #[repr(C)]
    pub struct FfiKernelRecord {
        pub metadata: KernelMetadata,
        /// JIT-compiled function pointer; always non-null on a successful finish.
        pub entry: Option<unsafe extern "C" fn(*mut c_void)>,
        /// NULL for ImmValue mode; malloc'd otherwise.
        pub matrix: *mut FfiComplex64,
        pub matrix_len: usize,
        /// NULL if request_asm was not called; malloc'd otherwise.
        pub asm_text: *mut c_char,
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
        /// Marks `kernel_id` for assembly capture during `finish`. Returns 0 on success.
        pub fn cast_cpu_kernel_generator_request_asm(
            generator: *mut CastCpuKernelGenerator,
            kernel_id: KernelId,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Runs O1 on the kernel and returns its optimized IR text (two-call pattern).
        /// Returns 0 on success.
        pub fn cast_cpu_kernel_generator_emit_ir(
            generator: *mut CastCpuKernelGenerator,
            kernel_id: KernelId,
            out_ir: *mut c_char,
            ir_buf_len: usize,
            out_ir_len: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Compiles all kernels into a JIT session and returns per-kernel records.
        /// On success, deletes the generator and returns 0.
        /// Caller must call `cast_cpu_jit_kernel_records_free` after use.
        pub fn cast_cpu_kernel_generator_finish(
            generator: *mut CastCpuKernelGenerator,
            out_session: *mut *mut CastCpuJitSession,
            out_records: *mut *mut FfiKernelRecord,
            out_n_records: *mut usize,
            err_buf: *mut c_char,
            err_buf_len: usize,
        ) -> i32;
        /// Frees the records array returned by `cast_cpu_kernel_generator_finish`.
        pub fn cast_cpu_jit_kernel_records_free(records: *mut FfiKernelRecord, n: usize);
        pub fn cast_cpu_jit_session_delete(session: *mut CastCpuJitSession);
    }
}

// ── CastCpuLaunchArgs ─────────────────────────────────────────────────────────

/// Matches `cast_cpu_launch_args_t` in `cpu.h`.
/// Each JIT-compiled kernel reads its work range and matrix pointer from this struct.
#[repr(C)]
struct CastCpuLaunchArgs {
    sv: *mut c_void,
    ctr_begin: u64,
    ctr_end: u64,
    p_mat: *mut c_void,
}

// ── RustJittedKernel ──────────────────────────────────────────────────────────

/// Rust-owned data for a single compiled kernel.
/// The `entry` pointer is valid for the lifetime of the owning `KernelEntry`
/// (the C++ LLJIT session inside the entry keeps the code pages alive).
struct RustJittedKernel {
    entry: unsafe extern "C" fn(*mut c_void),
    precision: Precision,
    simd_width: SimdWidth,
    mode: MatrixLoadMode,
    n_gate_qubits: u32,
    /// Non-empty only for StackLoad mode.
    matrix: Vec<ffi::FfiComplex64>,
    /// Some only if `request_asm` was called before `init_jit`.
    asm_text: Option<String>,
}

// ── MatrixBuffer ──────────────────────────────────────────────────────────────

/// Typed matrix buffer for StackLoad dispatch; ensures correct scalar alignment.
enum MatrixBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    Empty,
}

impl MatrixBuffer {
    fn as_ptr(&self) -> *const c_void {
        match self {
            MatrixBuffer::F32(v) => v.as_ptr().cast(),
            MatrixBuffer::F64(v) => v.as_ptr().cast(),
            MatrixBuffer::Empty => std::ptr::null(),
        }
    }
}

fn build_matrix_buffer(kernel: &RustJittedKernel) -> MatrixBuffer {
    if !matches!(kernel.mode, MatrixLoadMode::StackLoad) || kernel.matrix.is_empty() {
        return MatrixBuffer::Empty;
    }
    match kernel.precision {
        Precision::F32 => {
            let v: Vec<f32> = kernel
                .matrix
                .iter()
                .flat_map(|c| [c.re as f32, c.im as f32])
                .collect();
            MatrixBuffer::F32(v)
        }
        Precision::F64 => {
            let v: Vec<f64> = kernel.matrix.iter().flat_map(|c| [c.re, c.im]).collect();
            MatrixBuffer::F64(v)
        }
    }
}

/// Applies a single JIT-compiled kernel to `statevector` in-place using a thread pool.
///
/// `n_threads`: number of worker threads. Pass `0` to use the hardware thread count.
///
/// The kernel's precision and SIMD width must match those of the statevector, and the
/// statevector must have at least `n_gate_qubits + simd_s` qubits.
fn apply_kernel(
    kernel: &RustJittedKernel,
    statevector: &mut CPUStatevector,
    n_threads: u32,
) -> anyhow::Result<()> {
    if statevector.precision() as u32 != kernel.precision as u32 {
        return Err(anyhow::anyhow!(
            "statevector precision does not match kernel"
        ));
    }
    if statevector.simd_width() as u32 != kernel.simd_width as u32 {
        return Err(anyhow::anyhow!(
            "statevector SIMD width does not match kernel"
        ));
    }

    let n_qubits = statevector.n_qubits();
    let simd_s = get_simd_s(kernel.simd_width, kernel.precision);
    let n_task_bits = n_qubits
        .checked_sub(kernel.n_gate_qubits + simd_s)
        .ok_or_else(|| anyhow::anyhow!("statevector has too few qubits for this kernel"))?;

    let n_tasks: u64 = 1 << n_task_bits;
    let n_threads = if n_threads == 0 {
        super::get_num_threads()
    } else {
        n_threads
    };
    // Clamp so every thread gets at least one task.
    let n_threads = (n_threads as u64).min(n_tasks).max(1) as u32;
    let n_tasks_per_thread = n_tasks / n_threads as u64;

    log::debug!(
        "cpu: apply kernel ({} gate qubits, {} tasks, {} thread(s))",
        kernel.n_gate_qubits,
        n_tasks,
        n_threads,
    );

    let matrix_buf = build_matrix_buffer(kernel);

    let entry = kernel.entry;
    let sv_addr = statevector.raw_mut_ptr() as usize;
    let p_mat_addr = matrix_buf.as_ptr() as usize;

    // SAFETY:
    // - `entry` is a valid JIT-compiled function pointer backed by the LLJIT that owns
    //   the code pages (the caller must ensure the owning session/manager is alive).
    // - `sv_addr` points to a valid, aligned statevector buffer with the correct precision
    //   and SIMD width; the `&mut statevector` borrow ensures exclusive access.
    // - Each thread receives a disjoint `[ctr_begin, ctr_end)` counter range, so there are
    //   no data races on the statevector buffer.
    // - `p_mat_addr` is either null (ImmValue) or points to `matrix_buf` which lives for
    //   the duration of the `thread::scope` call.
    std::thread::scope(|s| {
        for i in 0..n_threads {
            let ctr_begin = n_tasks_per_thread * i as u64;
            let ctr_end = if i + 1 == n_threads {
                n_tasks
            } else {
                n_tasks_per_thread * (i as u64 + 1)
            };
            s.spawn(move || {
                let mut args = CastCpuLaunchArgs {
                    sv: sv_addr as *mut c_void,
                    ctr_begin,
                    ctr_end,
                    p_mat: p_mat_addr as *mut c_void,
                };
                unsafe { entry(&mut args as *mut CastCpuLaunchArgs as *mut c_void) };
            });
        }
    });

    Ok(())
}

// ── FFI helpers ───────────────────────────────────────────────────────────

/// Runs O1 optimisation on the kernel and returns the LLVM IR text (two-call
/// pattern: first call queries length, second fills the buffer).
fn ffi_emit_ir(
    gen: *mut ffi::CastCpuKernelGenerator,
    kernel_id: KernelId,
) -> anyhow::Result<String> {
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

// ── CpuKernelManager ──────────────────────────────────────────────────────

/// A single kernel entry owned by the manager.
///
/// Holds the LLJIT session pointer (to keep the compiled code pages alive)
/// alongside the Rust-owned kernel data needed for dispatch.
struct KernelEntry {
    /// C++ LLJIT session; keeps the JIT-compiled code pages alive.
    /// Destroyed via `cast_cpu_jit_session_delete` on drop.
    jit_session: NonNull<ffi::CastCpuJitSession>,
    /// Compiled kernel data (entry pointer, matrix, metadata).
    kernel: RustJittedKernel,
    /// Cached LLVM IR text (only if generated with diagnostics).
    ir_text: Option<String>,
}

impl Drop for KernelEntry {
    fn drop(&mut self) {
        unsafe { ffi::cast_cpu_jit_session_delete(self.jit_session.as_ptr()) };
    }
}

// jit_session is not Send by default (raw pointer), but access is
// serialised through the manager's Mutex.
unsafe impl Send for KernelEntry {}

struct CpuManagerInner {
    next_id: KernelId,
    entries: HashMap<KernelId, KernelEntry>,
}

/// Unified manager for CPU kernel generation, JIT compilation, and execution.
///
/// # Workflow
///
/// ```ignore
/// let mgr = CpuKernelManager::new();
/// let kid = mgr.generate(&spec, gate.matrix().data(), gate.qubits())?;
/// mgr.apply(kid, &mut statevector, n_threads)?;
/// ```
///
/// Each [`generate`](Self::generate) call runs the full LLVM pipeline
/// (IR generation → O1 optimisation → native JIT compilation) independently
/// before briefly locking to insert the result. Multiple threads may call
/// `generate` concurrently.
///
/// [`apply`](Self::apply) dispatches work across a thread pool synchronously.
pub struct CpuKernelManager {
    inner: std::sync::Mutex<CpuManagerInner>,
}

impl fmt::Debug for CpuKernelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self
            .inner
            .lock()
            .map(|g| g.entries.len())
            .unwrap_or(0);
        f.debug_struct("CpuKernelManager")
            .field("n_kernels", &n)
            .finish_non_exhaustive()
    }
}

impl Default for CpuKernelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuKernelManager {
    /// Creates a new kernel manager.
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(CpuManagerInner {
                next_id: 0,
                entries: HashMap::new(),
            }),
        }
    }

    /// Generates, optimises, and JIT-compiles a gate kernel.
    ///
    /// Runs the full LLVM pipeline on the calling thread, then briefly locks
    /// to store the result. Multiple threads may call `generate` concurrently.
    pub fn generate(
        &self,
        spec: &CPUKernelGenSpec,
        matrix: &[Complex],
        qubits: &[u32],
    ) -> anyhow::Result<KernelId> {
        self.generate_inner(spec, matrix, qubits, false)
    }

    /// Like [`generate`](Self::generate), but also captures LLVM IR and
    /// native assembly for later retrieval via [`emit_ir`](Self::emit_ir)
    /// and [`emit_asm`](Self::emit_asm).
    pub fn generate_with_diagnostics(
        &self,
        spec: &CPUKernelGenSpec,
        matrix: &[Complex],
        qubits: &[u32],
    ) -> anyhow::Result<KernelId> {
        self.generate_inner(spec, matrix, qubits, true)
    }

    fn generate_inner(
        &self,
        spec: &CPUKernelGenSpec,
        matrix: &[Complex],
        qubits: &[u32],
        diagnostics: bool,
    ) -> anyhow::Result<KernelId> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];

        // ── Create generator ──────────────────────────────────────────────
        let raw_gen = unsafe { ffi::cast_cpu_kernel_generator_new() };
        let raw_gen = NonNull::new(raw_gen)
            .ok_or_else(|| anyhow::anyhow!("failed to create CPU kernel generator"))?;

        // ── Generate LLVM IR for the gate ─────────────────────────────────
        let ffi_matrix: Vec<ffi::FfiComplex64> = matrix
            .iter()
            .map(|c| ffi::FfiComplex64 { re: c.re, im: c.im })
            .collect();

        let mut raw_kid: KernelId = 0;
        let status = unsafe {
            ffi::cast_cpu_kernel_generator_generate(
                raw_gen.as_ptr(),
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
            unsafe { ffi::cast_cpu_kernel_generator_delete(raw_gen.as_ptr()) };
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        // ── Diagnostics (IR + asm capture) ────────────────────────────────
        let ir_text = if diagnostics {
            Some(ffi_emit_ir(raw_gen.as_ptr(), raw_kid)?)
        } else {
            None
        };
        if diagnostics {
            let status = unsafe {
                ffi::cast_cpu_kernel_generator_request_asm(
                    raw_gen.as_ptr(),
                    raw_kid,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if status != 0 {
                unsafe { ffi::cast_cpu_kernel_generator_delete(raw_gen.as_ptr()) };
                return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
            }
        }

        // ── JIT-compile ───────────────────────────────────────────────────
        // `finish` deletes the C++ generator on success; on failure the
        // generator is left intact — but we give up ownership either way
        // to avoid a double-free, so we do NOT call generator_delete after
        // this point regardless of the outcome.
        let mut raw_session: *mut ffi::CastCpuJitSession = std::ptr::null_mut();
        let mut raw_records: *mut ffi::FfiKernelRecord = std::ptr::null_mut();
        let mut n_records: usize = 0;

        let status = unsafe {
            ffi::cast_cpu_kernel_generator_finish(
                raw_gen.as_ptr(),
                &mut raw_session,
                &mut raw_records,
                &mut n_records,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        let jit_session = NonNull::new(raw_session)
            .ok_or_else(|| anyhow::anyhow!("finish returned null session"))?;

        // ── Extract the single kernel record ──────────────────────────────
        // Each generate_inner produces exactly one kernel.
        debug_assert!(n_records == 1);
        let records = unsafe { std::slice::from_raw_parts(raw_records, n_records) };
        let record = &records[0];

        let entry_fn = record
            .entry
            .ok_or_else(|| anyhow::anyhow!("finish returned null entry pointer"))?;

        let matrix_data = if record.matrix.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(record.matrix, record.matrix_len) }.to_vec()
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

        let kernel = RustJittedKernel {
            entry: entry_fn,
            precision: record.metadata.precision,
            simd_width: record.metadata.simd_width,
            mode: record.metadata.mode,
            n_gate_qubits: record.metadata.n_gate_qubits,
            matrix: matrix_data,
            asm_text,
        };

        unsafe { ffi::cast_cpu_jit_kernel_records_free(raw_records, n_records) };

        // ── Insert under lock (brief) ─────────────────────────────────────
        let mut guard = self.inner.lock().unwrap();
        let id = guard.next_id;
        guard.next_id += 1;
        guard.entries.insert(
            id,
            KernelEntry {
                jit_session,
                kernel,
                ir_text,
            },
        );
        log::info!(
            "cpu: manager registered kernel {} ({} qubits, {:?} precision)",
            id,
            qubits.len(),
            spec.precision,
        );
        Ok(id)
    }

    /// Returns the optimised LLVM IR for a kernel, if it was generated with
    /// diagnostics.
    pub fn emit_ir(&self, id: KernelId) -> Option<String> {
        let guard = self.inner.lock().unwrap();
        guard.entries.get(&id)?.ir_text.clone()
    }

    /// Returns the native assembly for a kernel, if it was generated with
    /// diagnostics.
    pub fn emit_asm(&self, id: KernelId) -> Option<String> {
        let guard = self.inner.lock().unwrap();
        guard.entries.get(&id)?.kernel.asm_text.clone()
    }

    /// Applies the kernel identified by `id` to `statevector` in-place.
    ///
    /// `n_threads`: number of worker threads. Pass `0` to use the hardware
    /// thread count.
    pub fn apply(
        &self,
        id: KernelId,
        statevector: &mut CPUStatevector,
        n_threads: u32,
    ) -> anyhow::Result<()> {
        let guard = self.inner.lock().unwrap();
        let entry = guard
            .entries
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found", id))?;
        apply_kernel(&entry.kernel, statevector, n_threads)
    }

    /// Times a kernel adaptively within `budget_s` seconds.
    ///
    /// A single probe run starts the clock and estimates per-iteration cost.
    /// If that one run already exceeds `1.5 × budget_s`, it is returned as the
    /// sole sample. Otherwise the budget is split: ~1/4 for warmup, ~3/4 for
    /// timed measurements.
    pub fn time_adaptive(
        &self,
        id: KernelId,
        statevector: &mut CPUStatevector,
        n_threads: u32,
        budget_s: f64,
    ) -> anyhow::Result<TimingStats> {
        const WARMUP_FRACTION: f64 = 0.25;
        const OVER_BUDGET_FACTOR: f64 = 1.5;

        // Phase 1: single probe — warmup and cost estimate.
        let t_start = Instant::now();
        let t = Instant::now();
        self.apply(id, statevector, n_threads)?;
        let probe_time = t.elapsed().as_secs_f64();
        let est_per_iter = probe_time.max(1e-9);

        if probe_time > budget_s * OVER_BUDGET_FACTOR {
            return Ok(TimingStats {
                n_iters: 1,
                mean_s: probe_time,
                stddev_s: 0.0,
                cv: 0.0,
                min_s: probe_time,
                max_s: probe_time,
            });
        }

        // Phase 2: fill remaining warmup budget with un-timed iterations.
        let warmup_budget = budget_s * WARMUP_FRACTION;
        let remaining_warmup = (warmup_budget - probe_time).max(0.0);
        let n_extra_warmup = (remaining_warmup / est_per_iter) as u32;
        for _ in 0..n_extra_warmup {
            self.apply(id, statevector, n_threads)?;
        }

        // Phase 3: timed measurements over the remaining ~3/4 of budget.
        let elapsed = t_start.elapsed().as_secs_f64();
        let remaining = (budget_s - elapsed).max(0.0);
        let n_timed = ((remaining / est_per_iter) as u32).max(1);
        let mut samples: Vec<f64> = Vec::with_capacity(n_timed as usize);
        for _ in 0..n_timed {
            let t = Instant::now();
            self.apply(id, statevector, n_threads)?;
            samples.push(t.elapsed().as_secs_f64());
        }

        let n = samples.len() as f64;
        let mean = samples.iter().copied().sum::<f64>() / n;
        let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(TimingStats {
            n_iters: samples.len(),
            mean_s: mean,
            stddev_s: stddev,
            cv: stddev / mean,
            min_s: min,
            max_s: max,
        })
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

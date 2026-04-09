use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::error_from_buf;
use super::ffi;
use super::types::{CudaKernelGenSpec, CudaKernelId, CudaPrecision, ERR_BUF_LEN};
#[cfg(feature = "cuda")]
use super::CudaStatevector;
use crate::types::QuantumGate;

/// Shorthand for opaque CUDA driver handles (CUmodule, CUfunction, CUstream,
/// CUevent) passed through the FFI layer.
type CuPtr = *mut std::ffi::c_void;

// ── CudaKernel ───────────────────────────────────────────────────────────────

/// A compiled CUDA kernel, ready to be loaded and launched.
///
/// Owned by [`CudaKernelManager`]; obtained via [`CudaKernelManager::generate`].
/// CUDA module handles are not stored here — the [`sync`](CudaKernelManager::sync)
/// path manages them via an LRU cache, while the
/// [`execute_pipelined`](CudaKernelManager::execute_pipelined) path uses a
/// per-call local module cache.
pub struct CudaKernel {
    /// The source gate this kernel was compiled from.
    gate: Arc<QuantumGate>,
    precision: CudaPrecision,
    /// PTX assembly text produced by the LLVM NVPTX backend.
    ptx: String,
    /// Entry-point name in the PTX / cubin (e.g. `"k_gate"`).
    func_name: String,
    /// Wall time for the LLVM IR → PTX stage.
    ptx_compile_time: Duration,

    /// Device-native binary produced by the CUDA JIT linker.
    /// Module loading is deferred to `sync`; the manager's LRU cache owns
    /// the CUmodule handles.
    #[cfg(feature = "cuda")]
    cubin: Option<Vec<u8>>,
    /// Wall time for the PTX → cubin JIT stage.
    #[cfg(feature = "cuda")]
    jit_compile_time: Duration,
}

impl CudaKernel {
    /// Returns the source gate this kernel was compiled from.
    pub fn gate(&self) -> &Arc<QuantumGate> {
        &self.gate
    }

    /// Returns the number of qubits in the gate this kernel implements.
    pub fn n_gate_qubits(&self) -> u32 {
        self.gate.n_qubits() as u32
    }

    /// Returns the floating-point precision this kernel operates on.
    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }

    /// Returns the PTX assembly text for this kernel.
    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    /// Returns the kernel entry-point name.
    pub fn func_name(&self) -> &str {
        &self.func_name
    }

    /// Wall time spent in the LLVM IR → PTX stage of `generate`.
    pub fn ptx_compile_time(&self) -> Duration {
        self.ptx_compile_time
    }

    /// Wall time spent in the PTX → cubin JIT stage of `generate`.
    #[cfg(feature = "cuda")]
    pub fn jit_compile_time(&self) -> Duration {
        self.jit_compile_time
    }
}

impl fmt::Debug for CudaKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("CudaKernel");
        s.field("n_gate_qubits", &self.n_gate_qubits())
            .field("precision", &self.precision)
            .field("func_name", &self.func_name)
            .field("ptx_bytes", &self.ptx.len())
            .field("ptx_compile_time", &self.ptx_compile_time);
        #[cfg(feature = "cuda")]
        s.field("jit_compile_time", &self.jit_compile_time);
        s.finish_non_exhaustive()
    }
}

// ── LRU module cache ──────────────────────────────────────────────────────────

/// Default number of CUmodule slots in the manager's LRU cache.
#[cfg(feature = "cuda")]
const DEFAULT_LRU_SIZE: usize = 4;

/// A CUmodule / CUfunction pair held in the manager's LRU cache.
#[cfg(feature = "cuda")]
struct LoadedModule {
    kernel_id: CudaKernelId,
    cu_module: CuPtr,
    cu_function: CuPtr,
    /// Monotonically increasing tick set on each cache access; smaller → older.
    lru_tick: u64,
}

/// One enqueued but not-yet-launched apply request.
#[cfg(feature = "cuda")]
struct PendingApply {
    kernel_id: CudaKernelId,
    sv_dptr: u64,
    sv_n_qubits: u32,
}

/// CUDA event pair bracketing a single kernel launch, plus the metadata needed
/// to build a [`KernelExecTime`] after the stream is synchronized.
///
/// Events are destroyed automatically on drop, so error paths never need
/// explicit cleanup loops — they just let the `Vec<EventPair>` go out of scope.
///
/// Used by both the [`CudaKernelManager::sync`] and
/// [`CudaKernelManager::execute_pipelined`] paths.
#[cfg(feature = "cuda")]
struct EventPair {
    kernel_id: CudaKernelId,
    n_gate_qubits: u32,
    precision: CudaPrecision,
    ptx_compile_time: Duration,
    jit_compile_time: Duration,
    start_ev: CuPtr,
    end_ev: CuPtr,
}

#[cfg(feature = "cuda")]
impl Drop for EventPair {
    fn drop(&mut self) {
        unsafe {
            ffi::cast_cuda_event_destroy(self.start_ev);
            ffi::cast_cuda_event_destroy(self.end_ev);
        }
    }
}

#[cfg(feature = "cuda")]
impl EventPair {
    /// Wait for this kernel's end event to complete, then query GPU time.
    fn wait_and_collect(
        &self,
        err_buf: &mut [c_char; ERR_BUF_LEN],
    ) -> anyhow::Result<KernelExecTime> {
        let status = unsafe {
            ffi::cast_cuda_event_synchronize(self.end_ev, err_buf.as_mut_ptr(), err_buf.len())
        };
        if status != 0 {
            return Err(anyhow::anyhow!(
                "event_synchronize: {}",
                error_from_buf(err_buf)
            ));
        }
        self.collect_timing(err_buf)
    }

    /// Query GPU execution time from already-completed events.
    fn collect_timing(
        &self,
        err_buf: &mut [c_char; ERR_BUF_LEN],
    ) -> anyhow::Result<KernelExecTime> {
        let mut ms = 0.0f32;
        let status = unsafe {
            ffi::cast_cuda_event_elapsed_ms(
                self.start_ev,
                self.end_ev,
                &mut ms,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(
                "event_elapsed_ms: {}",
                error_from_buf(err_buf)
            ));
        }
        Ok(KernelExecTime {
            kernel_id: self.kernel_id,
            n_gate_qubits: self.n_gate_qubits,
            precision: self.precision,
            ptx_compile_time: self.ptx_compile_time,
            jit_compile_time: self.jit_compile_time,
            gpu_time: Duration::from_secs_f64(ms as f64 / 1000.0),
        })
    }
}

// ── Kernel deduplication ─────────────────────────────────────────────────────

/// Key capturing every input that affects the compiled CUDA kernel output.
/// The spec is fixed per manager, so only the gate-specific fields vary.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DedupKey {
    matrix_bytes: Vec<u8>,
    qubits: Vec<u32>,
}

impl DedupKey {
    fn new(gate: &QuantumGate) -> Self {
        Self {
            matrix_bytes: gate.matrix().as_bytes().to_vec(),
            qubits: gate.qubits().to_vec(),
        }
    }
}

// ── Shared FFI helpers ──────────────────────────────────────────────────────

/// GPU-side parameters for a single kernel launch.
#[cfg(feature = "cuda")]
struct LaunchParams {
    cu_function: CuPtr,
    stream: CuPtr,
    sv_dptr: u64,
    sv_n_qubits: u32,
}

/// Record a start event, launch the kernel, then record an end event.
/// Returns a fully armed [`EventPair`] on success.  On failure, any
/// partially-created events are destroyed before returning.
#[cfg(feature = "cuda")]
fn record_launch(
    params: &LaunchParams,
    kernel_id: CudaKernelId,
    kernel: &CudaKernel,
    err_buf: &mut [c_char; ERR_BUF_LEN],
) -> anyhow::Result<EventPair> {
    let start_ev = create_and_record_event(params.stream, err_buf)?;

    let status = unsafe {
        ffi::cast_cuda_kernel_launch(
            params.cu_function,
            params.stream,
            params.sv_dptr,
            kernel.n_gate_qubits(),
            params.sv_n_qubits,
            kernel.precision() as u8,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        unsafe { ffi::cast_cuda_event_destroy(start_ev) };
        return Err(anyhow::anyhow!(
            "kernel {} launch failed: {}",
            kernel_id,
            error_from_buf(err_buf)
        ));
    }

    let end_ev = match create_and_record_event(params.stream, err_buf) {
        Ok(ev) => ev,
        Err(e) => {
            unsafe { ffi::cast_cuda_event_destroy(start_ev) };
            return Err(e);
        }
    };

    log::debug!(
        "cuda: launched kernel {} ({} gate qubits, {} sv qubits, {:?})",
        kernel_id,
        kernel.n_gate_qubits(),
        params.sv_n_qubits,
        kernel.precision(),
    );

    Ok(EventPair {
        kernel_id,
        n_gate_qubits: kernel.n_gate_qubits(),
        precision: kernel.precision(),
        ptx_compile_time: kernel.ptx_compile_time(),
        jit_compile_time: kernel.jit_compile_time(),
        start_ev,
        end_ev,
    })
}

/// Load a cubin into a CUmodule and look up the entry-point function.
/// On failure, any partially-loaded module is unloaded before returning.
#[cfg(feature = "cuda")]
fn load_cubin_module(
    cubin: &[u8],
    func_name: &str,
    err_buf: &mut [c_char; ERR_BUF_LEN],
) -> anyhow::Result<(CuPtr, CuPtr)> {
    let cu_module = unsafe {
        ffi::cast_cuda_cubin_load(
            cubin.as_ptr(),
            cubin.len(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if cu_module.is_null() {
        return Err(anyhow::anyhow!(
            "cubin load failed: {}",
            error_from_buf(err_buf)
        ));
    }
    let func_cstr = CString::new(func_name.as_bytes())
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
        unsafe { ffi::cast_cuda_module_unload(cu_module) };
        return Err(anyhow::anyhow!(
            "module_get_function failed: {}",
            error_from_buf(err_buf)
        ));
    }
    Ok((cu_module, cu_function))
}

/// Create a CUDA event and record it on `stream`.
/// On failure, the event is destroyed before returning.
#[cfg(feature = "cuda")]
fn create_and_record_event(
    stream: CuPtr,
    err_buf: &mut [c_char; ERR_BUF_LEN],
) -> anyhow::Result<CuPtr> {
    let ev = unsafe { ffi::cast_cuda_event_create(err_buf.as_mut_ptr(), err_buf.len()) };
    if ev.is_null() {
        return Err(anyhow::anyhow!(
            "event_create failed: {}",
            error_from_buf(err_buf)
        ));
    }
    let status =
        unsafe { ffi::cast_cuda_event_record(ev, stream, err_buf.as_mut_ptr(), err_buf.len()) };
    if status != 0 {
        unsafe { ffi::cast_cuda_event_destroy(ev) };
        return Err(anyhow::anyhow!(
            "event_record failed: {}",
            error_from_buf(err_buf)
        ));
    }
    Ok(ev)
}

/// Run the LLVM IR → PTX (→ cubin) pipeline for `gate`.
///
/// This is a pure function of `spec` and `gate` — no manager state is accessed.
/// Returns the compiled kernel, its gate qubit count, and the precision used.
fn compile_gate(
    spec: CudaKernelGenSpec,
    gate: &Arc<QuantumGate>,
) -> anyhow::Result<(Arc<CudaKernel>, u32, CudaPrecision)> {
    let gate = gate.clone();
    let ffi_matrix: Vec<ffi::FfiComplex64> = gate
        .matrix()
        .data()
        .iter()
        .map(|c| ffi::FfiComplex64 { re: c.re, im: c.im })
        .collect();
    let qubits = gate.qubits();

    let mut err_buf = [0 as c_char; ERR_BUF_LEN];
    let mut out_ptx: *mut c_char = std::ptr::null_mut();
    let mut out_func_name: *mut c_char = std::ptr::null_mut();
    let mut n_gate_qubits: u32 = 0;
    let mut precision_byte: u8 = 0;

    let ptx_t0 = Instant::now();
    let status = unsafe {
        ffi::cast_cuda_compile_gate_ptx(
            &spec as *const CudaKernelGenSpec,
            ffi_matrix.as_ptr(),
            ffi_matrix.len(),
            qubits.as_ptr(),
            qubits.len(),
            &mut out_ptx,
            &mut out_func_name,
            &mut n_gate_qubits,
            &mut precision_byte,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    let ptx_compile_time = ptx_t0.elapsed();
    if status != 0 {
        return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
    }

    // Safety: C++ guarantees non-null, null-terminated strings on success.
    let ptx = unsafe { CStr::from_ptr(out_ptx) }
        .to_string_lossy()
        .into_owned();
    let func_name = unsafe { CStr::from_ptr(out_func_name) }
        .to_string_lossy()
        .into_owned();
    unsafe { ffi::cast_cuda_str_free(out_ptx) };
    unsafe { ffi::cast_cuda_str_free(out_func_name) };

    let precision = if precision_byte == 0 {
        CudaPrecision::F32
    } else {
        CudaPrecision::F64
    };

    log::debug!(
        "cuda: compiled kernel ({} gate qubits, {:?} precision, {} B PTX, {:?} PTX stage)",
        n_gate_qubits,
        precision,
        ptx.len(),
        ptx_compile_time,
    );

    // ── PTX → cubin (cuda feature only) ─────────────────────────────────
    #[cfg(feature = "cuda")]
    let (cubin, jit_compile_time) = {
        let ptx_cstr = CString::new(ptx.as_bytes())
            .map_err(|e| anyhow::anyhow!("PTX contains null byte: {e}"))?;
        let mut raw_cubin: *mut u8 = std::ptr::null_mut();
        let mut cubin_len: usize = 0;
        let jit_t0 = Instant::now();
        let status = unsafe {
            ffi::cast_cuda_ptx_to_cubin(
                ptx_cstr.as_ptr(),
                &mut raw_cubin,
                &mut cubin_len,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        let jit_compile_time = jit_t0.elapsed();
        if status != 0 {
            return Err(anyhow::anyhow!(
                "PTX → cubin failed: {}",
                error_from_buf(&err_buf)
            ));
        }
        // Safety: raw_cubin points to cubin_len bytes allocated by C++.
        let bytes = unsafe { std::slice::from_raw_parts(raw_cubin, cubin_len) }.to_vec();
        unsafe { ffi::cast_cuda_cubin_free(raw_cubin) };
        log::debug!("cuda: JIT PTX → cubin took {:?}", jit_compile_time);
        (Some(bytes), jit_compile_time)
    };

    let kernel = Arc::new(CudaKernel {
        gate,
        precision,
        ptx,
        func_name,
        ptx_compile_time,
        #[cfg(feature = "cuda")]
        cubin,
        #[cfg(feature = "cuda")]
        jit_compile_time,
    });

    Ok((kernel, n_gate_qubits, precision))
}

// ── ManagerInner ─────────────────────────────────────────────────────────────

struct ManagerInner {
    kernels: HashMap<CudaKernelId, Arc<CudaKernel>>,
    next_id: u64,
    dedup: HashMap<DedupKey, CudaKernelId>,
    /// Keys currently being compiled by a worker thread.  Other threads
    /// finding their key here wait on `CudaKernelManager::compile_done`
    /// instead of redundantly compiling the same gate.
    compiling: HashSet<DedupKey>,
    /// Ordered queue of apply requests not yet launched onto the stream.
    #[cfg(feature = "cuda")]
    pending: VecDeque<PendingApply>,
    /// LRU cache of loaded CUmodule handles.
    #[cfg(feature = "cuda")]
    loaded: Vec<Option<LoadedModule>>,
    #[cfg(feature = "cuda")]
    lru_tick: u64,
    /// Lazily initialized on first `sync`; `None` until then.
    #[cfg(feature = "cuda")]
    stream: Option<CuPtr>,
}

// loaded / stream are accessed only through the Mutex.
unsafe impl Send for ManagerInner {}

#[cfg(feature = "cuda")]
impl Drop for ManagerInner {
    fn drop(&mut self) {
        // Destroy the stream first so in-flight GPU work completes before we
        // unload modules that those launches may still reference.
        if let Some(stream) = self.stream.take() {
            unsafe { ffi::cast_cuda_stream_destroy(stream) };
        }
        for slot in &mut self.loaded {
            if let Some(m) = slot.take() {
                unsafe { ffi::cast_cuda_module_unload(m.cu_module) };
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl ManagerInner {
    /// Returns the CUDA stream, creating it on the first call.
    fn ensure_stream(&mut self) -> anyhow::Result<CuPtr> {
        if let Some(stream) = self.stream {
            return Ok(stream);
        }
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let stream = unsafe { ffi::cast_cuda_stream_create(err_buf.as_mut_ptr(), err_buf.len()) };
        if stream.is_null() {
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }
        self.stream = Some(stream);
        Ok(stream)
    }

    /// Ensures the module for `kernel` is resident in one of the LRU slots,
    /// loading (and possibly evicting the least-recently-used entry) as needed.
    /// Returns `cu_function` for immediate use.
    fn ensure_module(
        &mut self,
        kernel: &CudaKernel,
        id: CudaKernelId,
    ) -> anyhow::Result<CuPtr> {
        // ── Cache hit ────────────────────────────────────────────────────────
        let hit_idx = self
            .loaded
            .iter()
            .position(|s| s.as_ref().is_some_and(|m| m.kernel_id == id));
        if let Some(idx) = hit_idx {
            self.lru_tick += 1;
            let tick = self.lru_tick;
            let m = self.loaded[idx].as_mut().unwrap();
            m.lru_tick = tick;
            return Ok(m.cu_function);
        }

        // ── Select a slot: prefer empty, else evict LRU ──────────────────────
        let slot_idx = if let Some(i) = self.loaded.iter().position(|s| s.is_none()) {
            i
        } else {
            let idx = self
                .loaded
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.as_ref().unwrap().lru_tick)
                .unwrap()
                .0;
            let evicted = self.loaded[idx].take().unwrap();
            log::debug!("cuda: evicting module for kernel {}", evicted.kernel_id);
            unsafe { ffi::cast_cuda_module_unload(evicted.cu_module) };
            idx
        };

        // ── Load module from cubin ────────────────────────────────────────────
        let cubin = kernel
            .cubin
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kernel {} has no cubin", id))?;
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let (cu_module, cu_function) = load_cubin_module(cubin, kernel.func_name(), &mut err_buf)?;

        self.lru_tick += 1;
        self.loaded[slot_idx] = Some(LoadedModule {
            kernel_id: id,
            cu_module,
            cu_function,
            lru_tick: self.lru_tick,
        });
        log::debug!(
            "cuda: loaded module for kernel {} into slot {}",
            id,
            slot_idx
        );
        Ok(cu_function)
    }

    /// Launches every pending apply request in order, bracketing each launch
    /// with a start/end CUDA event for later GPU time measurement.
    /// Returns one [`EventPair`] per launch; events are destroyed via [`Drop`]
    /// when the returned `Vec` goes out of scope.
    fn drain_pending(&mut self, stream: CuPtr) -> anyhow::Result<Vec<EventPair>> {
        // Collect into a Vec first: launch_all needs &mut self (for
        // ensure_module), which conflicts with borrowing self.pending
        // through a drain iterator.
        let pending: Vec<PendingApply> = self.pending.drain(..).collect();
        self.launch_all(stream, &pending)
    }

    /// Enqueues one kernel per item, bracketed by CUDA events.
    /// On failure, already-accumulated [`EventPair`]s are destroyed via [`Drop`];
    /// the current item's partially-created events are destroyed before returning.
    fn launch_all(
        &mut self,
        stream: CuPtr,
        pending: &[PendingApply],
    ) -> anyhow::Result<Vec<EventPair>> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let mut pairs: Vec<EventPair> = Vec::with_capacity(pending.len());
        for item in pending {
            // Clone the Arc so the immutable borrow on self.kernels ends before
            // ensure_module takes &mut self.
            let kernel = self
                .kernels
                .get(&item.kernel_id)
                .ok_or_else(|| anyhow::anyhow!("kernel {} not found", item.kernel_id))?
                .clone();

            let cu_function = self.ensure_module(&kernel, item.kernel_id)?;

            let params = LaunchParams {
                cu_function,
                stream,
                sv_dptr: item.sv_dptr,
                sv_n_qubits: item.sv_n_qubits,
            };
            pairs.push(record_launch(&params, item.kernel_id, &kernel, &mut err_buf)?);
        }
        Ok(pairs)
    }
}

// ── Public timing types ───────────────────────────────────────────────────────

/// Timing metadata for a single kernel launch within a [`CudaKernelManager::sync`] call.
#[cfg(feature = "cuda")]
pub struct KernelExecTime {
    pub kernel_id: CudaKernelId,
    pub n_gate_qubits: u32,
    pub precision: CudaPrecision,
    /// Wall time for the LLVM IR → PTX stage, recorded at `generate` time.
    pub ptx_compile_time: Duration,
    /// Wall time for the PTX → cubin JIT stage, recorded at `generate` time.
    pub jit_compile_time: Duration,
    /// GPU execution time for this launch, measured by CUDA events.
    pub gpu_time: Duration,
}

/// Returned by [`CudaKernelManager::sync`].
#[cfg(feature = "cuda")]
pub struct SyncStats {
    /// One entry per [`apply`](CudaKernelManager::apply) call flushed in this sync, in order.
    /// Empty if nothing was queued.
    pub kernels: Vec<KernelExecTime>,
    /// CPU wall time for the entire `sync()` call (dispatch + GPU wait).
    pub wall_time: Duration,
}

// ── CudaKernelManager ────────────────────────────────────────────────────────

/// CUDA kernel manager: generate → enqueue → sync.
///
/// ```ignore
/// let mgr = CudaKernelManager::new(spec);
/// let kid = mgr.generate(&gate)?;      // LLVM IR → PTX → cubin
/// mgr.apply(kid, &mut statevector)?;   // enqueue (non-blocking)
/// mgr.sync()?;                         // flush queue, wait for GPU
/// ```
///
/// `generate` is thread-safe with content-based deduplication.
/// `apply` enqueues without blocking. `sync` drains the queue via an
/// LRU module cache (default 4 slots), then blocks until GPU completion.
///
/// # Safety
///
/// A `CudaStatevector` passed to `apply` must outlive the next `sync` call.
pub struct CudaKernelManager {
    spec: CudaKernelGenSpec,
    inner: Mutex<ManagerInner>,
    /// Notified whenever a compilation finishes, so threads waiting on a
    /// duplicate key can wake up and find the result in `dedup`.
    compile_done: Condvar,
}

impl fmt::Debug for CudaKernelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.inner.lock().map(|g| g.kernels.len()).unwrap_or(0);
        f.debug_struct("CudaKernelManager")
            .field("n_kernels", &n)
            .finish_non_exhaustive()
    }
}

impl CudaKernelManager {
    /// Creates a new manager with the default LRU cache size.
    /// Does not initialize the CUDA driver.
    pub fn new(spec: CudaKernelGenSpec) -> Self {
        Self::with_lru_size(spec, DEFAULT_LRU_SIZE)
    }

    /// Creates a new manager with a custom LRU cache size.
    ///
    /// `lru_cache_size` controls the number of CUmodule slots kept resident
    /// on the GPU.  Circuits with many distinct kernel shapes benefit from a
    /// larger cache to avoid module load/unload thrashing.  Minimum 1 slot.
    ///
    /// **Note:** the LRU cache is only used by the [`apply`](Self::apply) /
    /// [`sync`](Self::sync) path (currently exercised by trajectory mode).
    /// The simulator's non-trajectory path goes through
    /// [`execute_pipelined`](Self::execute_pipelined), which maintains its
    /// own per-call module cache keyed by kernel ID and bypasses this LRU.
    pub fn with_lru_size(spec: CudaKernelGenSpec, lru_cache_size: usize) -> Self {
        let lru_cache_size = lru_cache_size.max(1);
        Self {
            spec,
            inner: Mutex::new(ManagerInner {
                kernels: HashMap::new(),
                next_id: 0,
                dedup: HashMap::new(),
                compiling: HashSet::new(),
                #[cfg(feature = "cuda")]
                pending: VecDeque::new(),
                #[cfg(feature = "cuda")]
                loaded: (0..lru_cache_size).map(|_| None).collect(),
                #[cfg(feature = "cuda")]
                lru_tick: 0,
                #[cfg(feature = "cuda")]
                stream: None,
            }),
            compile_done: Condvar::new(),
        }
    }

    /// Returns the spec this manager was created with.
    pub fn spec(&self) -> &CudaKernelGenSpec {
        &self.spec
    }

    /// Generates a CUDA kernel for `gate`.
    ///
    /// Runs LLVM IR generation → O1 optimisation → NVPTX PTX emission on the
    /// calling thread (no CUDA device required).  When the `cuda` feature is
    /// enabled the PTX is further compiled to device-native cubin via the
    /// CUDA JIT linker.  Multiple threads may call `generate` concurrently;
    /// the manager lock is only held briefly at the start and end.
    ///
    /// If another thread is already compiling the same gate (same matrix bytes
    /// and qubit indices), this thread waits on `compile_done` instead of
    /// redundantly running the LLVM pipeline.
    pub fn generate(&self, gate: &Arc<QuantumGate>) -> anyhow::Result<CudaKernelId> {
        let spec = self.spec;
        let key = DedupKey::new(gate);

        // ── Dedup / in-flight check ─────────────────────────────────────
        {
            let mut guard = self.inner.lock().unwrap();
            loop {
                if let Some(&id) = guard.dedup.get(&key) {
                    return Ok(id);
                }
                if !guard.compiling.contains(&key) {
                    guard.compiling.insert(key.clone());
                    break; // this thread owns the compilation
                }
                // Another thread is compiling this key — wait.
                guard = self.compile_done.wait(guard).unwrap();
            }
        }

        // ── Compile (no lock held) ──────────────────────────────────────
        let result = compile_gate(spec, gate);

        let (kernel, n_gate_qubits, precision) = match result {
            Ok(v) => v,
            Err(e) => {
                // Remove key from `compiling` and wake waiters so they can
                // either retry or surface the same error via their own attempt.
                let mut guard = self.inner.lock().unwrap();
                guard.compiling.remove(&key);
                self.compile_done.notify_all();
                return Err(e);
            }
        };

        // ── Insert under lock (brief) ────────────────────────────────────
        let mut guard = self.inner.lock().unwrap();
        guard.compiling.remove(&key);

        // Re-check dedup: another path (unlikely) may have inserted first.
        if let Some(&existing_id) = guard.dedup.get(&key) {
            self.compile_done.notify_all();
            return Ok(existing_id);
        }

        let id = guard.next_id;
        guard.next_id += 1;
        guard.dedup.insert(key, id);
        guard.kernels.insert(id, kernel);
        self.compile_done.notify_all();
        log::info!(
            "cuda: manager registered kernel {} ({} gate qubits, {:?})",
            id,
            n_gate_qubits,
            precision,
        );
        Ok(id)
    }

    /// Returns the PTX assembly text for the given kernel. We have to return a copy because
    /// `CudaKernel` is shared via `Arc`.
    pub fn emit_ptx(&self, id: CudaKernelId) -> Option<String> {
        let guard = self.inner.lock().unwrap();
        let kernel = guard.kernels.get(&id)?;
        Some(kernel.ptx().into())
    }

    /// Queues a launch of kernel `id` on `sv`, without touching the GPU immediately.
    ///
    /// Validates that `id` exists and that the statevector's qubit count and
    /// precision match the kernel, then pushes the request onto an internal queue.
    /// Call [`sync`](Self::sync) to flush the queue and wait for completion.
    ///
    /// # Safety contract
    ///
    /// `sv` must remain valid (not dropped) until the next call to `sync`.
    #[cfg(feature = "cuda")]
    pub fn apply(&self, id: CudaKernelId, sv: &mut CudaStatevector) -> anyhow::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        let kernel = guard
            .kernels
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("kernel {} not found", id))?;

        if sv.n_qubits() < kernel.n_gate_qubits() {
            anyhow::bail!("statevector has fewer qubits than the gate kernel requires");
        }
        if sv.precision() as u8 != kernel.precision() as u8 {
            anyhow::bail!("statevector precision does not match kernel precision");
        }

        guard.pending.push_back(PendingApply {
            kernel_id: id,
            sv_dptr: sv.dptr(),
            sv_n_qubits: sv.n_qubits(),
        });

        log::debug!(
            "cuda: queued kernel {} ({} gate qubits, {} sv qubits, {:?})",
            id,
            kernel.n_gate_qubits(),
            sv.n_qubits(),
            kernel.precision(),
        );

        Ok(())
    }

    /// Flushes the apply queue — loading and evicting CUmodules via the LRU
    /// cache — enqueues all pending kernel launches onto the internal CUDA stream,
    /// then blocks until the stream is idle.
    ///
    /// Returns a [`SyncStats`] with per-kernel GPU execution times (measured via
    /// CUDA events) and the total CPU wall time for the call.
    /// Returns immediately with empty stats if nothing has been queued and no
    /// stream was ever created.
    #[cfg(feature = "cuda")]
    pub fn sync(&self) -> anyhow::Result<SyncStats> {
        let wall_t0 = Instant::now();

        let (stream, pairs) = {
            let mut guard = self.inner.lock().unwrap();

            if guard.pending.is_empty() && guard.stream.is_none() {
                return Ok(SyncStats {
                    kernels: vec![],
                    wall_time: Duration::ZERO,
                });
            }

            let stream = guard.ensure_stream()?;
            let pairs = guard.drain_pending(stream)?;
            (stream, pairs)
        }; // lock released here — stream_sync runs without holding the mutex

        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let status =
            unsafe { ffi::cast_cuda_stream_sync(stream, err_buf.as_mut_ptr(), err_buf.len()) };
        if status != 0 {
            // pairs drops here, destroying all events via Drop.
            return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
        }

        let wall_time = wall_t0.elapsed();

        // Query GPU elapsed time for each launch.  Events are destroyed when
        // `pairs` drops at the end of this function.
        let mut kernels: Vec<KernelExecTime> = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            kernels.push(pair.collect_timing(&mut err_buf)?);
        }

        log::debug!(
            "cuda: sync complete — {} kernels, wall {:?}",
            kernels.len(),
            wall_time,
        );

        Ok(SyncStats { kernels, wall_time })
    }

    /// Times a single-kernel apply+sync cycle adaptively within `budget_s`.
    ///
    /// Delegates to [`crate::timing::time_adaptive_with`].  Each iteration
    /// queues one kernel launch, syncs the stream, and reports the **GPU event
    /// time** (not wall-clock) as the sample duration.  This excludes launch
    /// overhead and gives the most accurate measure of kernel execution cost.
    #[cfg(feature = "cuda")]
    pub fn time_adaptive(
        &self,
        id: CudaKernelId,
        sv: &mut CudaStatevector,
        budget_s: f64,
    ) -> anyhow::Result<crate::timing::TimingStats> {
        crate::timing::time_adaptive_with(
            || {
                self.apply(id, sv)?;
                let stats = self.sync()?;
                Ok(stats.kernels[0].gpu_time)
            },
            budget_s,
        )
    }

    /// Pipelined compile-and-execute: a thread pool compiles gate kernels while
    /// the main thread loads CUmodules and launches kernels on the GPU stream.
    ///
    /// Compilation of gate N+1 overlaps with GPU execution of gate N.
    /// `window_size` limits the number of in-flight (launched but not yet
    /// waited-on) kernels.  Loaded CUmodules are cached for the duration of
    /// the call so that repeated gates avoid redundant load/unload cycles.
    ///
    /// Gates are launched in order (each modifies the statevector in place).
    /// Out-of-order compilation results are buffered until needed.
    ///
    /// # Safety contract
    ///
    /// `sv` must remain valid for the duration of this call.
    #[cfg(feature = "cuda")]
    pub fn execute_pipelined(
        &self,
        gates: &[Arc<QuantumGate>],
        sv: &mut CudaStatevector,
        window_size: usize,
        n_compile_threads: usize,
    ) -> anyhow::Result<SyncStats> {
        let wall_t0 = Instant::now();
        let window_size = window_size.max(1);
        let n_compile_threads = n_compile_threads.max(1);

        if gates.is_empty() {
            return Ok(SyncStats {
                kernels: vec![],
                wall_time: Duration::ZERO,
            });
        }

        // Obtain the CUDA stream (created lazily on first use).
        let stream = {
            let mut guard = self.inner.lock().unwrap();
            guard.ensure_stream()?
        };

        let sv_dptr = sv.dptr();
        let sv_n_qubits = sv.n_qubits();
        let sv_precision = sv.precision();

        let mut timings: Vec<KernelExecTime> = Vec::with_capacity(gates.len());
        let mut window: VecDeque<EventPair> = VecDeque::with_capacity(window_size);

        // Local module cache: CudaKernelId → (CUmodule, CUfunction).
        // Modules stay loaded for the duration of this call so that
        // repeated kernel IDs do not trigger redundant load/unload cycles.
        let mut module_cache: HashMap<CudaKernelId, (CuPtr, CuPtr)> =
            HashMap::new();

        // Bounded work channel with capacity proportional to the thread pool,
        // so large circuits don't buffer the entire gate list upfront.
        let (work_tx, work_rx) =
            std::sync::mpsc::sync_channel::<(usize, Arc<QuantumGate>)>(2 * n_compile_threads);
        let work_rx = std::sync::Mutex::new(work_rx);

        let scope_result = std::thread::scope(|s| {
            // Result channel (unbounded — results are small).
            let (result_tx, result_rx) =
                std::sync::mpsc::channel::<anyhow::Result<(usize, CudaKernelId)>>();

            // ── Feed thread ─────────────────────────────────────────────────
            // Runs in its own thread so the bounded channel provides
            // backpressure without blocking the compile workers.
            let feed_gates = &*gates;
            s.spawn(move || {
                for (i, gate) in feed_gates.iter().enumerate() {
                    if work_tx.send((i, gate.clone())).is_err() {
                        break;
                    }
                }
                // work_tx drops here, closing the channel.
            });

            // ── Compile workers ─────────────────────────────────────────────
            for _ in 0..n_compile_threads {
                let tx = result_tx.clone();
                let wrx = &work_rx;
                s.spawn(move || loop {
                    let item = {
                        let rx = wrx.lock().unwrap();
                        rx.recv().ok()
                    };
                    match item {
                        Some((idx, gate)) => {
                            let res = self.generate(&gate).map(|kid| (idx, kid));
                            if tx.send(res).is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                });
            }
            drop(result_tx);

            // ── Main dispatch loop ──────────────────────────────────────────
            // Reorder buffer: dense Vec indexed by gate position.
            let mut compiled: Vec<Option<CudaKernelId>> = vec![None; gates.len()];
            let mut err_buf = [0 as c_char; ERR_BUF_LEN];

            for next in 0..gates.len() {
                // 1. Free a window slot if full.
                if window.len() >= window_size {
                    let pair = window.pop_front().unwrap();
                    timings.push(pair.wait_and_collect(&mut err_buf)?);
                }

                // 2. Wait for gate `next` to finish compiling (in order).
                let kid = Self::await_compiled(
                    next,
                    &mut compiled,
                    &result_rx,
                )?;

                // 3. Look up and validate the kernel, then load/launch.
                let pair = self.pipelined_launch(
                    kid,
                    sv_dptr,
                    sv_n_qubits,
                    sv_precision,
                    stream,
                    &mut module_cache,
                    &mut err_buf,
                )?;

                window.push_back(pair);
            }

            // ── Drain remaining window ──────────────────────────────────────
            if unsafe { ffi::cast_cuda_stream_sync(stream, err_buf.as_mut_ptr(), err_buf.len()) }
                != 0
            {
                return Err(anyhow::anyhow!(error_from_buf(&err_buf)));
            }

            while let Some(pair) = window.pop_front() {
                timings.push(pair.collect_timing(&mut err_buf)?);
            }

            Ok(())
        });

        // On the error path, in-flight kernels may still reference cached
        // CUmodules.  Sync the stream before unloading them.
        if scope_result.is_err() && !module_cache.is_empty() {
            let mut err_buf = [0 as c_char; ERR_BUF_LEN];
            let sync_status =
                unsafe { ffi::cast_cuda_stream_sync(stream, err_buf.as_mut_ptr(), err_buf.len()) };
            if sync_status != 0 {
                log::warn!(
                    "cuda: stream_sync failed during pipelined cleanup: {}",
                    error_from_buf(&err_buf),
                );
            }
        }

        // Unload all cached modules (events already destroyed via EventPair Drop).
        for (_, (cu_module, _)) in module_cache.drain() {
            unsafe { ffi::cast_cuda_module_unload(cu_module) };
        }

        scope_result?;

        log::debug!(
            "cuda: pipelined execution complete — {} kernels, wall {:?}",
            timings.len(),
            wall_t0.elapsed(),
        );

        Ok(SyncStats {
            kernels: timings,
            wall_time: wall_t0.elapsed(),
        })
    }

    /// Block until compilation result for gate `idx` is available in the
    /// reorder buffer, draining the result channel as needed.
    #[cfg(feature = "cuda")]
    fn await_compiled(
        idx: usize,
        compiled: &mut [Option<CudaKernelId>],
        result_rx: &std::sync::mpsc::Receiver<anyhow::Result<(usize, CudaKernelId)>>,
    ) -> anyhow::Result<CudaKernelId> {
        while compiled[idx].is_none() {
            match result_rx.recv() {
                Ok(Ok((i, kid))) => {
                    compiled[i] = Some(kid);
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "compile workers exited before gate {} was compiled",
                        idx
                    ));
                }
            }
        }
        Ok(compiled[idx].take().unwrap())
    }

    /// Look up a compiled kernel, load its module (from cache or cubin),
    /// and launch it on `stream` via [`record_launch`].
    #[cfg(feature = "cuda")]
    fn pipelined_launch(
        &self,
        kid: CudaKernelId,
        sv_dptr: u64,
        sv_n_qubits: u32,
        sv_precision: CudaPrecision,
        stream: CuPtr,
        module_cache: &mut HashMap<CudaKernelId, (CuPtr, CuPtr)>,
        err_buf: &mut [c_char; ERR_BUF_LEN],
    ) -> anyhow::Result<EventPair> {
        let kernel = {
            let guard = self.inner.lock().unwrap();
            guard
                .kernels
                .get(&kid)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("kernel {} not found", kid))?
        };

        if sv_n_qubits < kernel.n_gate_qubits() {
            anyhow::bail!(
                "statevector has fewer qubits ({}) than kernel requires ({})",
                sv_n_qubits,
                kernel.n_gate_qubits()
            );
        }
        if sv_precision as u8 != kernel.precision() as u8 {
            anyhow::bail!("statevector precision does not match kernel");
        }

        // Load or reuse cached module.
        let cu_function = match module_cache.get(&kid) {
            Some(&(_, f)) => f,
            None => {
                let cubin = kernel
                    .cubin
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("kernel {} has no cubin", kid))?;
                let (m, f) = load_cubin_module(cubin, kernel.func_name(), err_buf)?;
                module_cache.insert(kid, (m, f));
                f
            }
        };

        let params = LaunchParams {
            cu_function,
            stream,
            sv_dptr,
            sv_n_qubits,
        };
        record_launch(&params, kid, &kernel, err_buf)
    }
}

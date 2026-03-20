use std::collections::{HashMap, VecDeque};
use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::error_from_buf;
use super::ffi;
use super::types::{CudaKernelGenSpec, CudaKernelId, CudaPrecision, ERR_BUF_LEN};
#[cfg(feature = "cuda")]
use super::CudaStatevector;
use crate::types::QuantumGate;

// ── CudaKernel ───────────────────────────────────────────────────────────────

/// A compiled CUDA kernel, ready to be loaded and launched.
///
/// Owned by [`CudaKernelManager`]; obtained via [`CudaKernelManager::generate`].
/// CUDA module handles are not stored here — they live in the manager's LRU cache.
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

/// Number of CUmodule slots in the manager's LRU cache.
#[cfg(feature = "cuda")]
const LRU_SIZE: usize = 2;

/// A CUmodule / CUfunction pair held in the manager's LRU cache.
#[cfg(feature = "cuda")]
struct LoadedModule {
    kernel_id: CudaKernelId,
    cu_module: *mut std::ffi::c_void,
    cu_function: *mut std::ffi::c_void,
    /// Monotonically increasing tick set on each cache access; smaller → older.
    lru_tick: u64,
}

/// One enqueued but not-yet-launched apply request.
#[cfg(feature = "cuda")]
struct PendingApply {
    kernel_id: CudaKernelId,
    sv_dptr: u64,
    sv_n_qubits: u32,
    n_gate_qubits: u32,
    precision: CudaPrecision,
}

/// CUDA event pair bracketing a single kernel launch, plus the metadata needed
/// to build a [`KernelExecTime`] after the stream is synchronised.
///
/// Events are destroyed automatically on drop, so error paths never need
/// explicit cleanup loops — they just let the `Vec<EventPair>` go out of scope.
#[cfg(feature = "cuda")]
struct EventPair {
    kernel_id: CudaKernelId,
    n_gate_qubits: u32,
    precision: CudaPrecision,
    ptx_compile_time: Duration,
    jit_compile_time: Duration,
    start_ev: *mut std::ffi::c_void,
    end_ev: *mut std::ffi::c_void,
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

// ── ManagerInner ─────────────────────────────────────────────────────────────

struct ManagerInner {
    kernels: HashMap<CudaKernelId, Arc<CudaKernel>>,
    next_id: u64,
    /// Ordered queue of apply requests not yet launched onto the stream.
    #[cfg(feature = "cuda")]
    pending: VecDeque<PendingApply>,
    /// LRU cache of loaded CUmodule handles; capacity is [`LRU_SIZE`].
    #[cfg(feature = "cuda")]
    loaded: [Option<LoadedModule>; LRU_SIZE],
    #[cfg(feature = "cuda")]
    lru_tick: u64,
    /// Lazily initialised on first `sync`; `None` until then.
    #[cfg(feature = "cuda")]
    stream: Option<*mut std::ffi::c_void>,
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
    fn ensure_stream(&mut self) -> anyhow::Result<*mut std::ffi::c_void> {
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

    /// Ensures the module for `kernel` is resident in one of the two LRU slots,
    /// loading (and possibly evicting the LRU entry) as needed.
    /// Returns `cu_function` for immediate use.
    fn ensure_module(
        &mut self,
        kernel: &CudaKernel,
        id: CudaKernelId,
    ) -> anyhow::Result<*mut std::ffi::c_void> {
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
                error_from_buf(&err_buf)
            ));
        }

        let func_cstr = CString::new(kernel.func_name().as_bytes())
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
                error_from_buf(&err_buf)
            ));
        }

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
    fn drain_pending(&mut self, stream: *mut std::ffi::c_void) -> anyhow::Result<Vec<EventPair>> {
        let pending: Vec<PendingApply> = self.pending.drain(..).collect();
        self.launch_all(stream, &pending)
    }

    /// Enqueues one kernel per item, bracketed by CUDA events.
    /// On failure, already-accumulated [`EventPair`]s are destroyed via [`Drop`];
    /// the current item's partially-created events are destroyed before returning.
    fn launch_all(
        &mut self,
        stream: *mut std::ffi::c_void,
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
                .ok_or_else(|| anyhow::anyhow!("kernel {} missing during drain", item.kernel_id))?
                .clone();

            let cu_function = self.ensure_module(&kernel, item.kernel_id)?;

            // ── start event ──────────────────────────────────────────────────
            let start_ev =
                unsafe { ffi::cast_cuda_event_create(err_buf.as_mut_ptr(), err_buf.len()) };
            if start_ev.is_null() {
                return Err(anyhow::anyhow!(
                    "event_create (start) failed: {}",
                    error_from_buf(&err_buf)
                ));
            }
            let status = unsafe {
                ffi::cast_cuda_event_record(start_ev, stream, err_buf.as_mut_ptr(), err_buf.len())
            };
            if status != 0 {
                unsafe { ffi::cast_cuda_event_destroy(start_ev) };
                return Err(anyhow::anyhow!(
                    "event_record (start) failed: {}",
                    error_from_buf(&err_buf)
                ));
            }

            // ── kernel launch ─────────────────────────────────────────────────
            let status = unsafe {
                ffi::cast_cuda_kernel_launch(
                    cu_function,
                    stream,
                    item.sv_dptr,
                    item.n_gate_qubits,
                    item.sv_n_qubits,
                    item.precision as u8,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if status != 0 {
                unsafe { ffi::cast_cuda_event_destroy(start_ev) };
                return Err(anyhow::anyhow!(
                    "kernel {} launch failed: {}",
                    item.kernel_id,
                    error_from_buf(&err_buf)
                ));
            }

            // ── end event ─────────────────────────────────────────────────────
            let end_ev =
                unsafe { ffi::cast_cuda_event_create(err_buf.as_mut_ptr(), err_buf.len()) };
            if end_ev.is_null() {
                unsafe { ffi::cast_cuda_event_destroy(start_ev) };
                return Err(anyhow::anyhow!(
                    "event_create (end) failed: {}",
                    error_from_buf(&err_buf)
                ));
            }
            let status = unsafe {
                ffi::cast_cuda_event_record(end_ev, stream, err_buf.as_mut_ptr(), err_buf.len())
            };
            if status != 0 {
                unsafe { ffi::cast_cuda_event_destroy(start_ev) };
                unsafe { ffi::cast_cuda_event_destroy(end_ev) };
                return Err(anyhow::anyhow!(
                    "event_record (end) failed: {}",
                    error_from_buf(&err_buf)
                ));
            }

            log::debug!(
                "cuda: launched kernel {} ({} gate qubits, {} sv qubits, {:?})",
                item.kernel_id,
                item.n_gate_qubits,
                item.sv_n_qubits,
                item.precision,
            );

            pairs.push(EventPair {
                kernel_id: item.kernel_id,
                n_gate_qubits: kernel.n_gate_qubits(),
                precision: kernel.precision(),
                ptx_compile_time: kernel.ptx_compile_time(),
                jit_compile_time: kernel.jit_compile_time(),
                start_ev,
                end_ev,
            });
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

/// Unified manager for CUDA kernel generation, PTX storage, and GPU execution.
///
/// # Workflow
///
/// ```ignore
/// use std::sync::Arc;
///
/// let mgr = CudaKernelManager::new();
/// let gate = Arc::new(QuantumGate::h(0));
/// let kid = mgr.generate(&gate, spec)?;   // LLVM IR → PTX (→ cubin with cuda feature)
/// mgr.apply(kid, &mut statevector)?;      // queue for launch (non-blocking)
/// mgr.sync()?;                            // flush queue, then wait for GPU
/// let amps = statevector.download()?;
/// ```
///
/// Each [`CudaKernel`] stores the source `Arc<QuantumGate>` alongside the
/// compiled PTX/cubin, accessible via [`CudaKernel::gate`].
///
/// `generate` can be called from multiple threads concurrently; each call runs
/// the full LLVM pipeline independently before briefly locking to insert the result.
/// `apply` is non-blocking — it validates and enqueues the request.  `sync` drains
/// the queue in order, loading and evicting CUmodules via a 2-slot LRU policy
/// (keeping at most two modules resident at once), then blocks until all GPU work
/// is complete.
///
/// # Safety contract
///
/// A `CudaStatevector` passed to `apply` must remain valid (not dropped) until
/// the next call to `sync`.
pub struct CudaKernelManager {
    inner: Mutex<ManagerInner>,
}

impl fmt::Debug for CudaKernelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.inner.lock().map(|g| g.kernels.len()).unwrap_or(0);
        f.debug_struct("CudaKernelManager")
            .field("n_kernels", &n)
            .finish_non_exhaustive()
    }
}

impl Default for CudaKernelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaKernelManager {
    /// Creates a new manager. Does not initialise the CUDA driver.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(ManagerInner {
                kernels: HashMap::new(),
                next_id: 0,
                #[cfg(feature = "cuda")]
                pending: VecDeque::new(),
                #[cfg(feature = "cuda")]
                loaded: std::array::from_fn(|_| None),
                #[cfg(feature = "cuda")]
                lru_tick: 0,
                #[cfg(feature = "cuda")]
                stream: None,
            }),
        }
    }

    /// Generates a CUDA kernel for `gate`.
    ///
    /// Runs LLVM IR generation → O1 optimisation → NVPTX PTX emission on the
    /// calling thread (no CUDA device required).  When the `cuda` feature is
    /// enabled the PTX is further compiled to device-native cubin via the
    /// CUDA JIT linker.  Multiple threads may call `generate` concurrently;
    /// the manager lock is only held briefly at the end to insert the result.
    pub fn generate(
        &self,
        gate: &Arc<QuantumGate>,
        spec: CudaKernelGenSpec,
    ) -> anyhow::Result<CudaKernelId> {
        let gate = gate.clone();
        let ffi_matrix: Vec<ffi::FfiComplex64> = gate
            .matrix()
            .data()
            .iter()
            .map(|c| ffi::FfiComplex64 { re: c.re, im: c.im })
            .collect();
        let qubits = gate.qubits();

        // ── C++ LLVM pipeline (blocking, no lock held) ────────────────────
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

        // ── PTX → cubin (cuda feature only) ──────────────────────────────
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

        // ── Insert under lock (brief) ─────────────────────────────────────
        let mut guard = self.inner.lock().unwrap();
        let id = guard.next_id;
        guard.next_id += 1;
        guard.kernels.insert(id, kernel);
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
            .ok_or_else(|| anyhow::anyhow!("kernel id {} not found", id))?;

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
            n_gate_qubits: kernel.n_gate_qubits(),
            precision: kernel.precision(),
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

    /// Flushes the apply queue — loading and evicting CUmodules via a 2-slot LRU
    /// policy — enqueues all pending kernel launches onto the internal CUDA stream,
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

        // Query GPU elapsed time for each launch. Events are destroyed when
        // `pairs` drops at the end of this function.
        let mut query_err: Option<anyhow::Error> = None;
        let mut kernels = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let mut ms = 0.0f32;
            if query_err.is_none() {
                let status = unsafe {
                    ffi::cast_cuda_event_elapsed_ms(
                        pair.start_ev,
                        pair.end_ev,
                        &mut ms,
                        err_buf.as_mut_ptr(),
                        err_buf.len(),
                    )
                };
                if status != 0 {
                    query_err = Some(anyhow::anyhow!(
                        "event_elapsed_ms: {}",
                        error_from_buf(&err_buf)
                    ));
                } else {
                    kernels.push(KernelExecTime {
                        kernel_id: pair.kernel_id,
                        n_gate_qubits: pair.n_gate_qubits,
                        precision: pair.precision,
                        ptx_compile_time: pair.ptx_compile_time,
                        jit_compile_time: pair.jit_compile_time,
                        gpu_time: Duration::from_secs_f64(ms as f64 / 1000.0),
                    });
                }
            }
        }
        // pairs drops here, destroying all events via Drop.

        if let Some(e) = query_err {
            return Err(e);
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
}

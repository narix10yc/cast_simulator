//! CPU statevector simulator with LLVM JIT kernel compilation.
//!
//! The primary workflow is:
//! 1. Create a [`CPUKernelGenerator`] and call [`CPUKernelGenerator::generate`] for each gate
//!    variant you need.
//! 2. Call [`CPUKernelGenerator::init_jit`] to compile all generated kernels and obtain a
//!    [`JitSession`].
//! 3. Allocate a [`CPUStatevector`] and drive simulation by calling [`JitSession::apply`].

use crate::types::{Complex, QuantumGate};
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use std::alloc::{self, Layout};
use std::ffi::c_char;
use std::ffi::c_void;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

// ── Enums & config ────────────────────────────────────────────────────────────

/// Floating-point precision used for the statevector and generated kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F64,
}

/// SIMD register width, in bits.
///
/// This controls how many amplitudes are processed together per SIMD lane.
/// The corresponding `simd_s` exponent (register holds `2^simd_s` scalars of
/// the given precision) determines the statevector memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum SimdWidth {
    W128 = 128,
    W256 = 256,
    W512 = 512,
}

/// Controls how the gate matrix is embedded in the JIT-compiled kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MatrixLoadMode {
    /// Matrix elements are baked in as immediate constants (faster for fixed gates).
    ImmValue,
    /// Matrix is loaded from a runtime pointer (allows reusing the kernel for different
    /// matrices of the same shape).
    StackLoad,
}

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

// ── Scalar trait ──────────────────────────────────────────────────────────────

/// Abstraction over f32/f64 for generic statevector storage.
trait CpuScalar: Copy + Default {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl CpuScalar for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl CpuScalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

// ── AlignedVec ────────────────────────────────────────────────────────────────

/// A heap-allocated buffer with a guaranteed minimum alignment.
///
/// `Vec<T>` only guarantees `align_of::<T>()` alignment. `AlignedVec<T>` lets
/// callers request SIMD-register-width alignment (e.g. 64 bytes for AVX-512)
/// so JIT-compiled kernels can use aligned load/store instructions.
struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

// SAFETY: same reasoning as Vec<T> — the buffer is uniquely owned.
unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    /// Allocates `len` zero-initialized `T` values with `align`-byte alignment.
    /// `align` must be a power of two and ≥ `align_of::<T>()`.
    fn zeroed(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two());
        assert!(align >= std::mem::align_of::<T>());
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: Layout::from_size_align(0, align).unwrap(),
            };
        }
        let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), align)
            .expect("AlignedVec: invalid layout");
        let ptr = unsafe { alloc::alloc_zeroed(layout) }.cast::<T>();
        let ptr = NonNull::new(ptr).expect("AlignedVec: allocation failed");
        Self { ptr, len, layout }
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let align = self.layout.align();
        if self.len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: Layout::from_size_align(0, align).unwrap(),
            };
        }
        let layout = Layout::from_size_align(self.len * std::mem::size_of::<T>(), align)
            .expect("AlignedVec: invalid layout");
        let ptr = unsafe { alloc::alloc(layout) }.cast::<T>();
        let ptr = NonNull::new(ptr).expect("AlignedVec: allocation failed");
        for (i, item) in self.iter().enumerate() {
            unsafe { ptr.as_ptr().add(i).write(item.clone()) };
        }
        Self { ptr, len: self.len, layout }
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T> IntoIterator for &'a AlignedVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlignedVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        unsafe {
            for i in 0..self.len {
                self.ptr.as_ptr().add(i).drop_in_place();
            }
            alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout);
        }
    }
}

// ── Statevector ───────────────────────────────────────────────────────────────

/// Inner typed storage for a statevector.
///
/// ## Memory layout
///
/// Amplitudes are stored split into real and imaginary scalars in a SIMD-friendly layout.
/// For amplitude index `i`, the real part lives at `insert_zero_to_bit(i, simd_s)` and
/// the imaginary part at that offset OR'd with `1 << simd_s`. This groups `2^simd_s`
/// reals followed by `2^simd_s` imaginaries so a single SIMD load captures a full register.
#[derive(Clone, Debug)]
struct CPUStatevectorData<T> {
    /// Buffer aligned to `simd_width / 8` bytes so JIT kernels can use aligned SIMD moves.
    data: AlignedVec<T>,
    n_qubits: usize,
    simd_width: SimdWidth,
    /// `simd_s = log2(simd_register_size / scalar_size)`. Determines the memory layout.
    simd_s: usize,
}

#[derive(Clone, Debug)]
enum CPUStatevectorInner {
    F32(CPUStatevectorData<f32>),
    F64(CPUStatevectorData<f64>),
}

/// A CPU quantum statevector supporting both a scalar reference implementation and
/// LLVM-JIT-compiled gate kernels.
///
/// The internal buffer layout is SIMD-aware (see [`CPUStatevectorData`]); both the
/// scalar methods and JIT kernels must agree on it.
#[derive(Clone, Debug)]
pub struct CPUStatevector {
    inner: CPUStatevectorInner,
}

impl CPUStatevector {
    /// Allocates a zeroed statevector for `n_qubits` qubits.
    ///
    /// `simd_width` selects the SIMD memory layout; it must be compatible with the
    /// `simd_width` used when generating kernels that will be applied to this statevector.
    pub fn new(n_qubits: usize, precision: Precision, simd_width: SimdWidth) -> Self {
        assert!(n_qubits > 0, "statevector must have at least one qubit");
        let simd_s = get_simd_s(simd_width, precision);
        // scalar_len accounts for simd_s so imaginary parts never go out of bounds.
        let scalar_len = scalar_len_for(n_qubits, simd_s);
        // Align to the SIMD register width so JIT kernels can use aligned moves.
        let align = simd_width as usize / 8;
        let inner = match precision {
            Precision::F32 => CPUStatevectorInner::F32(CPUStatevectorData {
                data: AlignedVec::zeroed(scalar_len, align),
                n_qubits,
                simd_width,
                simd_s,
            }),
            Precision::F64 => CPUStatevectorInner::F64(CPUStatevectorData {
                data: AlignedVec::zeroed(scalar_len, align),
                n_qubits,
                simd_width,
                simd_s,
            }),
        };
        Self { inner }
    }

    pub fn precision(&self) -> Precision {
        match &self.inner {
            CPUStatevectorInner::F32(_) => Precision::F32,
            CPUStatevectorInner::F64(_) => Precision::F64,
        }
    }

    pub fn simd_width(&self) -> SimdWidth {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.simd_width,
            CPUStatevectorInner::F64(data) => data.simd_width,
        }
    }

    /// The `simd_s` exponent that governs the internal memory layout.
    pub fn simd_s(&self) -> usize {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.simd_s,
            CPUStatevectorInner::F64(data) => data.simd_s,
        }
    }

    pub fn n_qubits(&self) -> usize {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.n_qubits,
            CPUStatevectorInner::F64(data) => data.n_qubits,
        }
    }

    /// Number of amplitudes: `2^n_qubits`.
    pub fn len(&self) -> usize {
        1usize << self.n_qubits()
    }

    /// Number of scalar values in the internal buffer (≥ `2 * len()`).
    pub fn scalar_len(&self) -> usize {
        scalar_len_for(self.n_qubits(), self.simd_s())
    }

    /// Size of the internal buffer in bytes.
    pub fn byte_len(&self) -> usize {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.data.len() * std::mem::size_of::<f32>(),
            CPUStatevectorInner::F64(data) => data.data.len() * std::mem::size_of::<f64>(),
        }
    }

    /// Raw pointer to the internal buffer, for passing to JIT kernels via FFI.
    pub fn raw_ptr(&self) -> *const c_void {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.data.as_ptr().cast(),
            CPUStatevectorInner::F64(data) => data.data.as_ptr().cast(),
        }
    }

    /// Mutable raw pointer to the internal buffer, for passing to JIT kernels via FFI.
    pub fn raw_mut_ptr(&mut self) -> *mut c_void {
        match &mut self.inner {
            CPUStatevectorInner::F32(data) => data.data.as_mut_ptr().cast(),
            CPUStatevectorInner::F64(data) => data.data.as_mut_ptr().cast(),
        }
    }

    /// Sets the state to |0⟩ (amplitude 1 for basis state 0, 0 elsewhere).
    pub fn initialize(&mut self) {
        self.with_data_mut(|data| data.initialize());
    }

    /// Fills with random amplitudes from a standard normal distribution, then normalizes.
    pub fn randomize(&mut self) {
        self.with_data_mut(|data| data.randomize());
    }

    pub fn normalize(&mut self) {
        self.with_data_mut(|data| data.normalize());
    }

    pub fn norm_squared(&self) -> f64 {
        self.with_data(|data| data.norm_squared())
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Returns the complex amplitude for basis state `idx`.
    pub fn amp(&self, idx: usize) -> Complex {
        self.with_data(|data| data.amp(idx))
    }

    /// Sets the complex amplitude for basis state `idx`.
    pub fn set_amp(&mut self, idx: usize, value: Complex) {
        self.with_data_mut(|data| data.set_amp(idx, value));
    }

    /// Returns all amplitudes in computational-basis order.
    pub fn amplitudes(&self) -> Vec<Complex> {
        (0..self.len()).map(|idx| self.amp(idx)).collect()
    }

    /// Returns `|⟨self|other⟩|²`.
    pub fn fidelity(&self, other: &Self) -> f64 {
        assert_eq!(
            self.n_qubits(),
            other.n_qubits(),
            "statevector size mismatch"
        );
        let mut inner = Complex::default();
        for idx in 0..self.len() {
            inner += self.amp(idx).conj() * other.amp(idx);
        }
        inner.norm_sqr()
    }

    /// Applies `gate` using the scalar (non-JIT) path. Use [`JitSession::apply`] for
    /// performance-critical paths.
    pub fn apply_gate(&mut self, gate: &QuantumGate) {
        self.with_data_mut(|data| data.apply_gate(gate));
    }

    fn with_data<R>(&self, f: impl FnOnce(&dyn CPUStatevectorView) -> R) -> R {
        match &self.inner {
            CPUStatevectorInner::F32(data) => f(data),
            CPUStatevectorInner::F64(data) => f(data),
        }
    }

    fn with_data_mut<R>(&mut self, f: impl FnOnce(&mut dyn CPUStatevectorViewMut) -> R) -> R {
        match &mut self.inner {
            CPUStatevectorInner::F32(data) => f(data),
            CPUStatevectorInner::F64(data) => f(data),
        }
    }
}

trait CPUStatevectorView {
    fn amp(&self, idx: usize) -> Complex;
    fn norm_squared(&self) -> f64;
}

trait CPUStatevectorViewMut: CPUStatevectorView {
    fn set_amp(&mut self, idx: usize, value: Complex);
    fn initialize(&mut self);
    fn randomize(&mut self);
    fn normalize(&mut self);
    fn apply_gate(&mut self, gate: &QuantumGate);
}

impl<T: CpuScalar> CPUStatevectorData<T> {
    fn amp(&self, idx: usize) -> Complex {
        assert!(
            idx < (1usize << self.n_qubits),
            "amplitude index out of bounds"
        );
        let base = insert_zero_to_bit(idx, self.simd_s);
        Complex::new(
            self.data[base].to_f64(),
            self.data[base | (1 << self.simd_s)].to_f64(),
        )
    }

    fn set_amp(&mut self, idx: usize, value: Complex) {
        assert!(
            idx < (1usize << self.n_qubits),
            "amplitude index out of bounds"
        );
        let base = insert_zero_to_bit(idx, self.simd_s);
        self.data[base] = T::from_f64(value.re);
        self.data[base | (1 << self.simd_s)] = T::from_f64(value.im);
    }

    fn initialize(&mut self) {
        self.data.fill(T::default());
        self.data[0] = T::from_f64(1.0);
    }

    fn norm_squared(&self) -> f64 {
        // The buffer holds all real and imaginary scalars; summing their squares
        // gives Σ |aᵢ|² = ‖ψ‖².
        self.data
            .iter()
            .map(|value| {
                let value = value.to_f64();
                value * value
            })
            .sum()
    }

    fn normalize(&mut self) {
        let norm = self.norm_squared().sqrt();
        assert!(norm > 0.0, "cannot normalize the zero statevector");
        let factor = 1.0 / norm;
        for value in &mut self.data {
            *value = T::from_f64(value.to_f64() * factor);
        }
    }

    fn randomize(&mut self) {
        let mut rng = thread_rng();
        for value in &mut self.data {
            let sample: f64 = StandardNormal.sample(&mut rng);
            *value = T::from_f64(sample);
        }
        self.normalize();
    }

    /// Scalar gate application using bit-deposit addressing (non-JIT reference path).
    fn apply_gate(&mut self, gate: &QuantumGate) {
        assert!(
            gate.qubits()
                .last()
                .is_none_or(|&qubit| (qubit as usize) < self.n_qubits),
            "gate acts on a qubit outside the statevector"
        );

        let k = gate.n_qubits();
        let gate_dim = 1usize << k;
        assert_eq!(
            gate.matrix().edge_size(),
            gate_dim,
            "gate matrix size does not match qubits"
        );

        // Mask of target qubit positions; the complement is iterated as task IDs.
        let mut target_mask = 0usize;
        for &qubit in gate.qubits() {
            target_mask |= 1usize << qubit;
        }
        let full_mask = (1usize << self.n_qubits) - 1;
        let task_mask = full_mask ^ target_mask;
        let n_tasks = 1usize << (self.n_qubits - k);

        let mut amp_indices = vec![0usize; gate_dim];
        let mut updated = vec![Complex::default(); gate_dim];

        for task_id in 0..n_tasks {
            let deposited_task = pdep_usize(task_id, task_mask);
            for (amp_id, amp_index) in amp_indices.iter_mut().enumerate() {
                *amp_index = deposited_task | pdep_usize(amp_id, target_mask);
            }

            for row in 0..gate_dim {
                let mut acc = Complex::default();
                for col in 0..gate_dim {
                    acc += gate.matrix().get(row, col) * self.amp(amp_indices[col]);
                }
                updated[row] = acc;
            }

            for (amp_index, value) in amp_indices.iter().zip(updated.iter()) {
                self.set_amp(*amp_index, *value);
            }
        }
    }
}

impl<T: CpuScalar> CPUStatevectorView for CPUStatevectorData<T> {
    fn amp(&self, idx: usize) -> Complex {
        CPUStatevectorData::amp(self, idx)
    }
    fn norm_squared(&self) -> f64 {
        CPUStatevectorData::norm_squared(self)
    }
}

impl<T: CpuScalar> CPUStatevectorViewMut for CPUStatevectorData<T> {
    fn set_amp(&mut self, idx: usize, value: Complex) {
        CPUStatevectorData::set_amp(self, idx, value);
    }
    fn initialize(&mut self) {
        CPUStatevectorData::initialize(self);
    }
    fn randomize(&mut self) {
        CPUStatevectorData::randomize(self);
    }
    fn normalize(&mut self) {
        CPUStatevectorData::normalize(self);
    }
    fn apply_gate(&mut self, gate: &QuantumGate) {
        CPUStatevectorData::apply_gate(self, gate);
    }
}

// ── Layout helpers ────────────────────────────────────────────────────────────

/// Total scalar slots needed for the SIMD-split statevector layout.
///
/// Each amplitude is stored as two scalars (real + imaginary). The imaginary part of
/// amplitude `i` lives at bit-offset `insert_zero_to_bit(i, simd_s) | (1 << simd_s)`,
/// so the buffer must span `2^(max(n_qubits, simd_s) + 1)` slots — not merely
/// `2^(n_qubits + 1)` — to avoid out-of-bounds access when `simd_s > n_qubits`.
fn scalar_len_for(n_qubits: usize, simd_s: usize) -> usize {
    2usize
        .checked_shl(n_qubits.max(simd_s) as u32)
        .expect("too many qubits for CPU statevector")
}

/// Returns `simd_s = log2(simd_register_scalars)` for the given width and precision.
fn get_simd_s(simd_width: SimdWidth, precision: Precision) -> usize {
    match precision {
        Precision::F32 => match simd_width {
            SimdWidth::W128 => 2,
            SimdWidth::W256 => 3,
            SimdWidth::W512 => 4,
        },
        Precision::F64 => match simd_width {
            SimdWidth::W128 => 1,
            SimdWidth::W256 => 2,
            SimdWidth::W512 => 3,
        },
    }
}

/// Inserts a zero bit at position `bit`, shifting higher bits left by one.
///
/// Used to map an amplitude index to its real-part offset in the SIMD layout.
fn insert_zero_to_bit(x: usize, bit: usize) -> usize {
    let mask_lo = (1usize << bit) - 1;
    let mask_hi = !mask_lo;
    (x & mask_lo) + ((x & mask_hi) << 1)
}

/// Bit-deposit: scatter the bits of `src` into the set positions of `mask`.
///
/// Equivalent to the x86 PDEP instruction but written in portable Rust.
fn pdep_usize(src: usize, mask: usize) -> usize {
    let mut out = 0usize;
    let mut src_bit = 0usize;
    let nbits = usize::BITS as usize;
    for dst_bit in 0..nbits {
        if (mask & (1usize << dst_bit)) == 0 {
            continue;
        }
        if (src & (1usize << src_bit)) != 0 {
            out |= 1usize << dst_bit;
        }
        src_bit += 1;
    }
    out
}

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
            .ok_or_else(|| anyhow::anyhow!("C++ finish returned a null JIT session"))?;
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
        n_threads: Option<usize>,
    ) -> anyhow::Result<()> {
        let mut err_buf = [0 as c_char; ERR_BUF_LEN];
        let n_threads = n_threads
            .map(|threads| i32::try_from(threads).expect("thread count exceeds i32"))
            .unwrap_or(0); // 0 → C++ picks based on hardware_concurrency
        let n_qubits = u32::try_from(statevector.n_qubits()).expect("qubit count exceeds u32");
        let status = unsafe {
            ffi::cast_cpu_jit_session_apply(
                self.raw.as_ptr(),
                kernel_id,
                statevector.raw_mut_ptr(),
                n_qubits,
                statevector.precision(),
                statevector.simd_width(),
                n_threads,
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{CPUKernelGenSpec, CPUKernelGenerator, CPUStatevector, Precision, SimdWidth};
    use crate::types::{Complex, QuantumGate};

    /// Creates a normalized statevector with deterministic amplitudes.
    fn seeded_statevector(
        n_qubits: usize,
        precision: Precision,
        simd_width: SimdWidth,
    ) -> CPUStatevector {
        let mut sv = CPUStatevector::new(n_qubits, precision, simd_width);
        for idx in 0..sv.len() {
            let re = (idx as f64) + 1.0;
            let im = (idx as f64) * 0.5 - 0.25;
            sv.set_amp(idx, Complex::new(re, im));
        }
        sv.normalize();
        sv
    }

    fn assert_statevectors_close(lhs: &CPUStatevector, rhs: &CPUStatevector, tol: f64) {
        assert_eq!(lhs.n_qubits(), rhs.n_qubits(), "statevector size mismatch");
        for idx in 0..lhs.len() {
            let diff = lhs.amp(idx) - rhs.amp(idx);
            assert!(
                diff.norm() < tol,
                "statevectors differ at index {}: lhs={:?}, rhs={:?}, diff={:?}",
                idx,
                lhs.amp(idx),
                rhs.amp(idx),
                diff
            );
        }
    }

    /// Generates a JIT kernel for `gate` and checks the result against the scalar path.
    /// `n_threads`: number of worker threads passed to `apply`.
    fn run_jit_and_compare_full(
        gate: &QuantumGate,
        n_qubits_sv: usize,
        spec: CPUKernelGenSpec,
        n_threads: usize,
        tol: f64,
    ) {
        let mut generator = CPUKernelGenerator::new().expect("create generator");
        let kernel_id = generator
            .generate(&spec, gate.matrix().data(), gate.qubits())
            .expect("generate kernel");
        let mut jit = generator.init_jit().expect("init jit");

        let mut sv_jit = seeded_statevector(n_qubits_sv, spec.precision, spec.simd_width);
        let mut sv_ref = sv_jit.clone();

        sv_ref.apply_gate(gate);
        jit.apply(kernel_id, &mut sv_jit, Some(n_threads))
            .expect("apply kernel");

        assert_statevectors_close(&sv_jit, &sv_ref, tol);
        assert!((sv_jit.norm() - 1.0).abs() < tol);
    }

    fn default_spec(precision: Precision, simd_width: SimdWidth) -> CPUKernelGenSpec {
        CPUKernelGenSpec {
            precision,
            simd_width,
            mode: super::MatrixLoadMode::ImmValue,
            ztol: match precision {
                Precision::F32 => 1e-6,
                Precision::F64 => 1e-12,
            },
            otol: match precision {
                Precision::F32 => 1e-6,
                Precision::F64 => 1e-12,
            },
        }
    }

    fn run_jit_and_compare(
        gate: QuantumGate,
        n_qubits_sv: usize,
        precision: Precision,
        simd_width: SimdWidth,
        tol: f64,
    ) {
        run_jit_and_compare_full(
            &gate,
            n_qubits_sv,
            default_spec(precision, simd_width),
            1,
            tol,
        );
    }

    #[test]
    fn initializes_to_zero_state() {
        let mut sv = CPUStatevector::new(3, Precision::F64, SimdWidth::W128);
        sv.initialize();

        assert_eq!(sv.amp(0), Complex::new(1.0, 0.0));
        for idx in 1..sv.len() {
            assert_eq!(sv.amp(idx), Complex::new(0.0, 0.0));
        }
        assert!((sv.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn applies_single_qubit_gate() {
        let mut sv = CPUStatevector::new(1, Precision::F64, SimdWidth::W128);
        sv.initialize();
        sv.apply_gate(&QuantumGate::h(0));

        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv.amp(0) - Complex::new(expected, 0.0)).norm() < 1e-12);
        assert!((sv.amp(1) - Complex::new(expected, 0.0)).norm() < 1e-12);
        assert!((sv.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jit_applies_single_qubit_gate() {
        let gate = QuantumGate::h(0);
        let mut generator = CPUKernelGenerator::new().expect("create generator");
        let kernel_id = generator
            .generate(
                &CPUKernelGenSpec::f64(),
                gate.matrix().data(),
                gate.qubits(),
            )
            .expect("generate kernel");
        let mut jit = generator.init_jit().expect("init jit");

        let mut sv = CPUStatevector::new(2, Precision::F64, SimdWidth::W128);
        sv.initialize();
        jit.apply(kernel_id, &mut sv, Some(1))
            .expect("apply kernel");

        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv.amp(0) - Complex::new(expected, 0.0)).norm() < 1e-10);
        assert!((sv.amp(1) - Complex::new(expected, 0.0)).norm() < 1e-10);
        assert!((sv.amp(2) - Complex::new(0.0, 0.0)).norm() < 1e-10);
        assert!((sv.amp(3) - Complex::new(0.0, 0.0)).norm() < 1e-10);
        assert!((sv.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jit_matches_scalar_for_x_gate_imm_mode() {
        run_jit_and_compare(QuantumGate::x(1), 3, Precision::F64, SimdWidth::W128, 1e-10);
    }

    #[test]
    fn jit_matches_scalar_for_cx_gate_imm_mode() {
        run_jit_and_compare(
            QuantumGate::cx(0, 1),
            3,
            Precision::F64,
            SimdWidth::W128,
            1e-10,
        );
    }

    #[test]
    fn jit_matches_scalar_for_nonadjacent_gate_imm_mode() {
        run_jit_and_compare(
            QuantumGate::cx(0, 2),
            4,
            Precision::F64,
            SimdWidth::W128,
            1e-10,
        );
    }

    #[test]
    fn jit_matches_scalar_for_fp32_imm_mode() {
        run_jit_and_compare(QuantumGate::h(2), 4, Precision::F32, SimdWidth::W128, 5e-5);
    }

    // ── SIMD width coverage ────────────────────────────────────────────────────

    // F64/W256: simd_s=2, needs ≥3 qubits for a 1-qubit gate.
    #[test]
    fn jit_f64_w256() {
        run_jit_and_compare(QuantumGate::h(1), 4, Precision::F64, SimdWidth::W256, 1e-10);
    }

    // F64/W512: simd_s=3, needs ≥4 qubits for a 1-qubit gate.
    #[test]
    fn jit_f64_w512() {
        run_jit_and_compare(QuantumGate::h(1), 5, Precision::F64, SimdWidth::W512, 1e-10);
    }

    // F32/W256: simd_s=3, needs ≥4 qubits for a 1-qubit gate.
    #[test]
    fn jit_f32_w256() {
        run_jit_and_compare(QuantumGate::h(2), 5, Precision::F32, SimdWidth::W256, 5e-5);
    }

    // F32/W512: simd_s=4, needs ≥5 qubits for a 1-qubit gate.
    #[test]
    fn jit_f32_w512() {
        run_jit_and_compare(QuantumGate::h(2), 6, Precision::F32, SimdWidth::W512, 5e-5);
    }

    // ── StackLoad mode ─────────────────────────────────────────────────────────

    // StackLoad embeds a runtime matrix pointer rather than immediate constants;
    // the numerical result must be identical to ImmValue for the same gate.
    #[test]
    fn jit_stack_load_matches_imm_value() {
        let gate = QuantumGate::cx(0, 2);
        let n_qubits_sv = 4;
        let precision = Precision::F64;
        let simd_width = SimdWidth::W128;
        let tol = 1e-10;

        let mut sv_imm = seeded_statevector(n_qubits_sv, precision, simd_width);
        let mut sv_stack = sv_imm.clone();

        let spec_imm = default_spec(precision, simd_width);
        let spec_stack = CPUKernelGenSpec {
            mode: super::MatrixLoadMode::StackLoad,
            ..spec_imm
        };

        let mut gen_imm = CPUKernelGenerator::new().expect("create generator");
        let kid_imm = gen_imm
            .generate(&spec_imm, gate.matrix().data(), gate.qubits())
            .expect("generate imm kernel");
        let mut jit_imm = gen_imm.init_jit().expect("init jit");
        jit_imm.apply(kid_imm, &mut sv_imm, Some(1)).expect("apply imm");

        let mut gen_stack = CPUKernelGenerator::new().expect("create generator");
        let kid_stack = gen_stack
            .generate(&spec_stack, gate.matrix().data(), gate.qubits())
            .expect("generate stack kernel");
        let mut jit_stack = gen_stack.init_jit().expect("init jit");
        jit_stack.apply(kid_stack, &mut sv_stack, Some(1)).expect("apply stack");

        assert_statevectors_close(&sv_imm, &sv_stack, tol);
    }

    // ── Gate variety ───────────────────────────────────────────────────────────

    #[test]
    fn jit_swap_nonadjacent() {
        // SWAP(0,2): non-adjacent, exercises hi_bits path same as CX(0,2).
        run_jit_and_compare(QuantumGate::swap(0, 2), 4, Precision::F64, SimdWidth::W128, 1e-10);
    }

    #[test]
    fn jit_cz_nonadjacent() {
        run_jit_and_compare(QuantumGate::cz(1, 3), 5, Precision::F64, SimdWidth::W128, 1e-10);
    }

    #[test]
    fn jit_ccx_gate() {
        // 3-qubit Toffoli: exercises the multi-qubit hi_bits partitioning.
        run_jit_and_compare(QuantumGate::ccx(0, 1, 2), 5, Precision::F64, SimdWidth::W128, 1e-10);
    }

    // Rx has a dense, fully complex matrix — no zero or ±1 entries — so the
    // kernel exercises FMA paths that sparse gates like CX skip.
    #[test]
    fn jit_rx_gate() {
        run_jit_and_compare(
            QuantumGate::rx(std::f64::consts::PI / 3.0, 1),
            3,
            Precision::F64,
            SimdWidth::W128,
            1e-10,
        );
    }

    #[test]
    fn jit_rz_gate() {
        run_jit_and_compare(
            QuantumGate::rz(std::f64::consts::PI / 5.0, 0),
            3,
            Precision::F64,
            SimdWidth::W128,
            1e-10,
        );
    }

    // ── Multi-kernel session ────────────────────────────────────────────────────

    // Generates H and CX kernels in one session and applies them sequentially;
    // verifies that the session correctly dispatches by kernel id.
    #[test]
    fn jit_multiple_kernels_in_one_session() {
        let spec = default_spec(Precision::F64, SimdWidth::W128);
        let h_gate = QuantumGate::h(0);
        let cx_gate = QuantumGate::cx(0, 1);

        let mut generator = CPUKernelGenerator::new().expect("create generator");
        let kid_h = generator
            .generate(&spec, h_gate.matrix().data(), h_gate.qubits())
            .expect("generate H kernel");
        let kid_cx = generator
            .generate(&spec, cx_gate.matrix().data(), cx_gate.qubits())
            .expect("generate CX kernel");
        let mut jit = generator.init_jit().expect("init jit");

        let mut sv_jit = seeded_statevector(3, Precision::F64, SimdWidth::W128);
        let mut sv_ref = sv_jit.clone();

        sv_ref.apply_gate(&h_gate);
        sv_ref.apply_gate(&cx_gate);

        jit.apply(kid_h, &mut sv_jit, Some(1)).expect("apply H");
        jit.apply(kid_cx, &mut sv_jit, Some(1)).expect("apply CX");

        assert_statevectors_close(&sv_jit, &sv_ref, 1e-10);
        assert!((sv_jit.norm() - 1.0).abs() < 1e-10);
    }

    // ── Multithreaded apply ────────────────────────────────────────────────────

    // Result must be identical to single-threaded regardless of how work is split.
    #[test]
    fn jit_multithreaded_matches_single_thread() {
        let gate = QuantumGate::rx(std::f64::consts::PI / 7.0, 2);
        let spec = default_spec(Precision::F64, SimdWidth::W128);

        run_jit_and_compare_full(&gate, 6, spec, 4, 1e-10);
    }
}

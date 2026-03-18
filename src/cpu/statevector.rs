use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

use rand_distr::Distribution as _;

use super::SimdWidth;
use crate::types::{Complex, Precision, QuantumGate};

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

    unsafe fn uninit(len: usize, align: usize) -> Self {
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
        let ptr = unsafe { alloc::alloc(layout) }.cast::<T>();
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
        Self {
            ptr,
            len: self.len,
            layout,
        }
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
    n_qubits: u32,
    simd_width: super::SimdWidth,
    /// `simd_s = log2(simd_register_size / scalar_size)`. Determines the memory layout.
    simd_s: u32,
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
    pub fn new(n_qubits: u32, precision: Precision, simd_width: SimdWidth) -> Self {
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

    pub unsafe fn uninit(n_qubits: u32, precision: Precision, simd_width: SimdWidth) -> Self {
        let simd_s = get_simd_s(simd_width, precision);
        let scalar_len = scalar_len_for(n_qubits, simd_s);
        let align = simd_width as usize / 8;
        let inner = match precision {
            Precision::F32 => CPUStatevectorInner::F32(CPUStatevectorData {
                data: unsafe { AlignedVec::uninit(scalar_len, align) },
                n_qubits,
                simd_width,
                simd_s,
            }),
            Precision::F64 => CPUStatevectorInner::F64(CPUStatevectorData {
                data: unsafe { AlignedVec::uninit(scalar_len, align) },
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
    pub fn simd_s(&self) -> u32 {
        match &self.inner {
            CPUStatevectorInner::F32(data) => data.simd_s,
            CPUStatevectorInner::F64(data) => data.simd_s,
        }
    }

    pub fn n_qubits(&self) -> u32 {
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

    /// Applies `gate` using the scalar (non-JIT) path. Use [`CpuJitSession::apply`] for
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
        let mut rng = rand::thread_rng();
        for value in &mut self.data {
            let sample: f64 = rand_distr::StandardNormal.sample(&mut rng);
            *value = T::from_f64(sample);
        }
        self.normalize();
    }

    /// Scalar gate application using bit-deposit addressing (non-JIT reference path).
    fn apply_gate(&mut self, gate: &QuantumGate) {
        assert!(
            gate.qubits()
                .last()
                .is_none_or(|&qubit| qubit < self.n_qubits),
            "gate acts on a qubit outside the statevector"
        );

        let k = gate.n_qubits() as u32;
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
fn scalar_len_for(n_qubits: u32, simd_s: u32) -> usize {
    2usize
        .checked_shl(n_qubits.max(simd_s))
        .expect("too many qubits for CPU statevector")
}

/// Returns `simd_s = log2(simd_register_scalars)` for the given width and precision.
pub(crate) fn get_simd_s(simd_width: SimdWidth, precision: Precision) -> u32 {
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
fn insert_zero_to_bit(x: usize, bit: u32) -> usize {
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

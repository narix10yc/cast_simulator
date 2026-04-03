// ── Enums & config ────────────────────────────────────────────────────────────

/// SIMD register width, in bits.
///
/// This controls how many amplitudes are processed together per SIMD lane.
/// The corresponding `simd_s` exponent (register holds `2^simd_s` scalars of
/// the given precision) determines the statevector memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

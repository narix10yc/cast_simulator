/// Floating-point precision used in the CUDA kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Opaque handle returned by [`super::CudaKernelGenerator::generate`], used to identify a
/// compiled kernel inside a [`super::CudaKernelArtifacts`].
pub type CudaKernelId = u64;

pub(super) const ERR_BUF_LEN: usize = 1024;

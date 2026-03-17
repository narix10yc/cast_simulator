/// Floating-point precision used for the statevector and generated kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F64,
}

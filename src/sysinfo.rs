//! Helpers for querying available system memory.

/// Maximum n_qubits the auto-selector will ever return.
pub const MAX_DEFAULT_QUBITS: u32 = 30;

/// Minimum n_qubits the auto-selector will return (prevents a degenerate run).
pub const MIN_DEFAULT_QUBITS: u32 = 10;

/// Fraction of free memory considered safe to use (leaves headroom for fragmentation).
const SAFETY_FRACTION: f64 = 0.80;

/// Returns the amount of free system memory in bytes by reading `/proc/meminfo`.
///
/// Returns `None` if the file cannot be read or parsed (e.g. non-Linux systems).
pub fn cpu_free_memory_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Returns the largest n such that a statevector of `2^n` complex scalars fits
/// within `free_bytes` (after applying an 80% safety margin).
///
/// `scalar_bytes` is 8 for f64, 4 for f32. A statevector element is a complex
/// number, so each element occupies `2 * scalar_bytes` bytes.
///
/// The result is clamped to `[MIN_DEFAULT_QUBITS, MAX_DEFAULT_QUBITS]`.
pub fn max_feasible_n_qubits(free_bytes: u64, scalar_bytes: usize) -> u32 {
    let usable = (free_bytes as f64 * SAFETY_FRACTION) as u64;
    let bytes_per_element = (2 * scalar_bytes) as u64; // complex scalar
    if usable < bytes_per_element {
        return MIN_DEFAULT_QUBITS;
    }
    let max_elements = usable / bytes_per_element;
    let n = (max_elements as f64).log2().floor() as u32;
    n.clamp(MIN_DEFAULT_QUBITS, MAX_DEFAULT_QUBITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_feasible_f64() {
        // 16 GiB free → n=30 exactly (2^30 * 16 = 16 GiB, but 80% of 16 GiB = 12.8 GiB → n=29)
        let gib_16: u64 = 16 * (1 << 30);
        assert_eq!(max_feasible_n_qubits(gib_16, 8), 29);

        // 20 GiB free → 80% = 16 GiB → n=30
        let gib_20: u64 = 20 * (1u64 << 30);
        assert_eq!(max_feasible_n_qubits(gib_20, 8), 30);

        // 8 GiB free → 80% = 6.4 GiB → n=28
        let gib_8: u64 = 8 * (1u64 << 30);
        assert_eq!(max_feasible_n_qubits(gib_8, 8), 28);
    }

    #[test]
    fn max_feasible_f32() {
        // 8 GiB free, f32 (4 bytes/scalar) → 80% = 6.4 GiB → n=29
        let gib_8: u64 = 8 * (1u64 << 30);
        assert_eq!(max_feasible_n_qubits(gib_8, 4), 29);
    }

    #[test]
    fn clamps_to_min() {
        assert_eq!(max_feasible_n_qubits(0, 8), MIN_DEFAULT_QUBITS);
        assert_eq!(max_feasible_n_qubits(1024, 8), MIN_DEFAULT_QUBITS);
    }

    #[test]
    fn clamps_to_max() {
        assert_eq!(max_feasible_n_qubits(u64::MAX, 8), MAX_DEFAULT_QUBITS);
    }
}

//! Direct DRAM bandwidth measurement — no JIT codegen.
//!
//! This module hits DRAM through straight-line Rust streaming loops, giving
//! an independent roofline that is *not* entangled with LLVM codegen quality.
//!
//! The [`crate::profile`] roofline reports a "peak BW" value, but it derives
//! it from JIT-compiled gate kernels — so a slow kernel yields a low "peak BW"
//! even on a fast DRAM controller.  Use [`measure_mem_bw`] to get the raw
//! hardware ceiling, then compare gate-kernel bandwidth against it to decide
//! whether a kernel is DRAM-bound or codegen-bound.
//!
//! Four patterns are reported:
//!
//! | Pattern | Logical DRAM traffic per iter | Notes |
//! |---------|------------------------------|-------|
//! | `read`  | 1 × buffer | streaming reduction (XOR) |
//! | `write` | 1 × buffer | streaming fill; on x86 this pays RFO unless the compiler emits non-temporal stores |
//! | `copy`  | 2 × buffer | `src → dst`; RFO on dst |
//! | `rmw`   | 2 × buffer | in-place update; RFO on the same lines just read. **Best match for a quantum gate apply.** |
//!
//! The buffer is partitioned into disjoint chunks across `n_threads`; each
//! thread drives its own chunk, so there is no false sharing at chunk
//! boundaries and no synchronization inside the timed region.

use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

use crate::timing::{time_adaptive_with, TimingStats};

const GIB: f64 = (1u64 << 30) as f64;

/// Result of one memory-bandwidth sweep across the four streaming patterns.
#[derive(Debug, Clone)]
pub struct MemBwResult {
    /// Buffer size in bytes.
    pub bytes: u64,
    /// Thread count used for all four patterns.
    pub n_threads: u32,
    /// Streaming read (XOR reduction). Traffic = `bytes` per iter.
    pub read: TimingStats,
    /// Streaming fill. Traffic = `bytes` per iter (logical; hardware may pay RFO).
    pub write: TimingStats,
    /// Streaming copy `src → dst`. Traffic = `2 * bytes` per iter.
    pub copy: TimingStats,
    /// In-place read-modify-write. Traffic = `2 * bytes` per iter.
    /// Closest analogue to a quantum gate apply.
    pub rmw: TimingStats,
}

impl MemBwResult {
    pub fn read_gib_s(&self) -> f64 {
        self.bytes as f64 / self.read.mean_s / GIB
    }
    pub fn write_gib_s(&self) -> f64 {
        self.bytes as f64 / self.write.mean_s / GIB
    }
    pub fn copy_gib_s(&self) -> f64 {
        2.0 * self.bytes as f64 / self.copy.mean_s / GIB
    }
    pub fn rmw_gib_s(&self) -> f64 {
        2.0 * self.bytes as f64 / self.rmw.mean_s / GIB
    }
}

// ---------------------------------------------------------------------------
// Streaming kernels
// ---------------------------------------------------------------------------

#[inline(never)]
fn read_kernel(chunk: &[u64]) -> u64 {
    // 8-way unrolled XOR: keeps 8 independent accumulators so the CPU can
    // issue them in parallel without a reduction dependency chain.  LLVM
    // auto-vectorises this to ZMM XORs on AVX-512.
    let mut acc = [0u64; 8];
    let n = chunk.len();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: bounds guaranteed by the while condition.
        unsafe {
            acc[0] ^= *chunk.get_unchecked(i);
            acc[1] ^= *chunk.get_unchecked(i + 1);
            acc[2] ^= *chunk.get_unchecked(i + 2);
            acc[3] ^= *chunk.get_unchecked(i + 3);
            acc[4] ^= *chunk.get_unchecked(i + 4);
            acc[5] ^= *chunk.get_unchecked(i + 5);
            acc[6] ^= *chunk.get_unchecked(i + 6);
            acc[7] ^= *chunk.get_unchecked(i + 7);
        }
        i += 8;
    }
    while i < n {
        acc[0] ^= chunk[i];
        i += 1;
    }
    acc.iter().fold(0u64, |a, &b| a ^ b)
}

#[inline(never)]
fn write_kernel(chunk: &mut [u64]) {
    let pat = black_box(0xDEADBEEF_CAFEBABEu64);
    for x in chunk.iter_mut() {
        *x = pat;
    }
}

#[inline(never)]
fn rmw_kernel(chunk: &mut [u64]) {
    let k = black_box(1u64);
    for x in chunk.iter_mut() {
        *x = x.wrapping_add(k);
    }
}

// ---------------------------------------------------------------------------
// Threaded drivers
// ---------------------------------------------------------------------------

fn run_read_parallel(buf: &[u64], n_threads: usize) -> Duration {
    let chunk = buf.len().div_ceil(n_threads);
    let t0 = Instant::now();
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for piece in buf.chunks(chunk) {
            handles.push(s.spawn(move || read_kernel(piece)));
        }
        let mut acc = 0u64;
        for h in handles {
            acc ^= h.join().unwrap();
        }
        black_box(acc);
    });
    t0.elapsed()
}

fn run_write_parallel(buf: &mut [u64], n_threads: usize) -> Duration {
    let chunk = buf.len().div_ceil(n_threads);
    let t0 = Instant::now();
    thread::scope(|s| {
        for piece in buf.chunks_mut(chunk) {
            s.spawn(move || write_kernel(piece));
        }
    });
    t0.elapsed()
}

fn run_rmw_parallel(buf: &mut [u64], n_threads: usize) -> Duration {
    let chunk = buf.len().div_ceil(n_threads);
    let t0 = Instant::now();
    thread::scope(|s| {
        for piece in buf.chunks_mut(chunk) {
            s.spawn(move || rmw_kernel(piece));
        }
    });
    t0.elapsed()
}

fn run_copy_parallel(src: &[u64], dst: &mut [u64], n_threads: usize) -> Duration {
    debug_assert_eq!(src.len(), dst.len());
    let chunk = src.len().div_ceil(n_threads);
    let t0 = Instant::now();
    thread::scope(|s| {
        for (s_piece, d_piece) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            s.spawn(move || d_piece.copy_from_slice(s_piece));
        }
    });
    t0.elapsed()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Measures raw DRAM bandwidth through streaming read, write, copy, and RMW
/// patterns.
///
/// `buf_bytes` should exceed last-level cache by several× so the result
/// reflects DRAM and not cache.  Matching it to your target statevector size
/// is recommended.
///
/// `n_threads` partitions the buffer into disjoint chunks, one per thread.
/// One thread per physical core typically gives the most stable numbers.
///
/// `budget_s` is the wall-time budget **per pattern** (total run time is
/// roughly `4 * budget_s` plus buffer allocation).
pub fn measure_mem_bw(
    buf_bytes: usize,
    n_threads: u32,
    budget_s: f64,
) -> anyhow::Result<MemBwResult> {
    anyhow::ensure!(n_threads > 0, "need at least one thread");
    anyhow::ensure!(buf_bytes >= 64, "buffer too small (< 1 cache line)");

    let n_u64 = buf_bytes / 8;
    let nt = n_threads as usize;

    // Allocate and fault every page in.  `(0..N).collect()` writes every
    // element, so physical pages are backed before timing starts; `vec![0; N]`
    // may return CoW zero pages, so we explicitly touch it.
    let src: Vec<u64> = (0..n_u64 as u64).collect();
    let mut dst: Vec<u64> = vec![0u64; n_u64];
    for x in dst.iter_mut() {
        *x = 0;
    }

    let read = time_adaptive_with(|| Ok(run_read_parallel(&src, nt)), budget_s)?;
    let write = time_adaptive_with(|| Ok(run_write_parallel(&mut dst, nt)), budget_s)?;
    let copy = time_adaptive_with(|| Ok(run_copy_parallel(&src, &mut dst, nt)), budget_s)?;
    let rmw = time_adaptive_with(|| Ok(run_rmw_parallel(&mut dst, nt)), budget_s)?;

    Ok(MemBwResult {
        bytes: (n_u64 * 8) as u64,
        n_threads,
        read,
        write,
        copy,
        rmw,
    })
}

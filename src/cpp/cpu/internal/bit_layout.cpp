#include "bit_layout.h"

namespace cast_cpu_detail {

BitLayout compute_bit_layout(const uint32_t *qubits, size_t n_qubits, unsigned simd_s) {
  BitLayout layout;
  unsigned q = 0;
  size_t qi = 0;

  while (layout.simd_bits.size() < simd_s) {
    if (qi < n_qubits && qubits[qi] == q) {
      layout.lo_bits.push_back(q);
      ++qi;
    } else {
      layout.simd_bits.push_back(q);
    }
    ++q;
  }
  while (qi < n_qubits) {
    layout.hi_bits.push_back(qubits[qi++]);
  }

  for (auto *vec : {&layout.lo_bits, &layout.simd_bits, &layout.hi_bits}) {
    for (auto &bit : *vec) {
      if (bit >= simd_s)
        ++bit;
    }
  }

  layout.sep_bit = layout.simd_bits.empty() ? 0u : layout.simd_bits.back() + 1u;
  // When `simd_bits` fills positions [0, simd_s), sep_bit lands exactly on
  // the implicit-zero bit the statevector inserts at `simd_s`.  Step over it.
  if (layout.sep_bit == simd_s)
    ++layout.sep_bit;

  return layout;
}

std::vector<PtrSegment> compute_hi_ptr_segments(const std::vector<unsigned> &hi_bits,
                                                unsigned sep_bit) {
  std::vector<PtrSegment> segs;
  unsigned src_bit = 0;
  unsigned prev_end = 0;

  for (const unsigned hb : hi_bits) {
    const unsigned hi_pos = hb - sep_bit;
    if (hi_pos > prev_end) {
      const unsigned width = hi_pos - prev_end;
      const uint64_t mask = ((uint64_t(1) << width) - 1) << src_bit;
      segs.push_back({mask, prev_end - src_bit});
      src_bit += width;
    }
    prev_end = hi_pos + 1;
  }

  // Catch-all tail segment: all src bits above `src_bit` have no hi_bit
  // punching through them, so they shift by the final (prev_end - src_bit).
  segs.push_back({~((uint64_t(1) << src_bit) - 1), prev_end - src_bit});
  return segs;
}

} // namespace cast_cpu_detail

#include "bit_layout.hpp"

namespace cast::cpu {

BitLayout computeBitLayout(const uint32_t *qubits, size_t nQubits, unsigned simdS) {
  BitLayout layout;
  unsigned q = 0;
  size_t qi = 0;

  while (layout.simdBits.size() < simdS) {
    if (qi < nQubits && qubits[qi] == q) {
      layout.loBits.push_back(q);
      ++qi;
    } else {
      layout.simdBits.push_back(q);
    }
    ++q;
  }
  while (qi < nQubits) {
    layout.hiBits.push_back(qubits[qi++]);
  }

  for (auto *vec : {&layout.loBits, &layout.simdBits, &layout.hiBits}) {
    for (auto &bit : *vec) {
      if (bit >= simdS)
        ++bit;
    }
  }

  layout.sepBit = layout.simdBits.empty() ? 0u : layout.simdBits.back() + 1u;
  // When `simdBits` fills positions [0, simdS), sepBit lands exactly on
  // the implicit-zero bit the statevector inserts at `simdS`.  Step over it.
  if (layout.sepBit == simdS)
    ++layout.sepBit;

  return layout;
}

std::vector<PtrSegment> computeHiPtrSegments(const std::vector<unsigned> &hiBits, unsigned sepBit) {
  std::vector<PtrSegment> segs;
  unsigned srcBit = 0;
  unsigned prevEnd = 0;

  for (const unsigned hb : hiBits) {
    const unsigned hiPos = hb - sepBit;
    if (hiPos > prevEnd) {
      const unsigned width = hiPos - prevEnd;
      const uint64_t mask = ((uint64_t(1) << width) - 1) << srcBit;
      segs.push_back({mask, prevEnd - srcBit});
      srcBit += width;
    }
    prevEnd = hiPos + 1;
  }

  // Catch-all tail segment: all src bits above `srcBit` have no hi_bit
  // punching through them, so they shift by the final (prevEnd - srcBit).
  segs.push_back({~((uint64_t(1) << srcBit) - 1), prevEnd - srcBit});
  return segs;
}

} // namespace cast::cpu

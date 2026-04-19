#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP

#include "bit_layout.hpp"

#include <vector>

namespace cast::cpu {

// Precomputed shufflevector indices used by Phase 1 (load → split) and
// Phase 3 (merge → store).  `reSplit`/`imSplit` flatten as (li, si) with
// stride `s`; `merge` holds one mask per loBit for the reassembly round.
struct ShuffleMasks {
  std::vector<int> reSplit;
  std::vector<int> imSplit;
  std::vector<std::vector<int>> merge;
  std::vector<int> reimMerge;
};

ShuffleMasks computeShuffleMasks(const BitLayout &layout, unsigned s, unsigned simdS,
                                 unsigned vecSize);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP

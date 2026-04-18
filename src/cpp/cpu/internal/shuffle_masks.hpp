#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP

#include "bit_layout.hpp"

#include <vector>

namespace cast::cpu {

// Precomputed shufflevector indices used by Phase 1 (load → split) and
// Phase 3 (merge → store).  `re_split`/`im_split` flatten as (li, si) with
// stride `s`; `merge` holds one mask per lo_bit for the reassembly round.
struct ShuffleMasks {
  std::vector<int> re_split;
  std::vector<int> im_split;
  std::vector<std::vector<int>> merge;
  std::vector<int> reim_merge;
};

ShuffleMasks compute_shuffle_masks(const BitLayout &layout, unsigned s, unsigned simd_s,
                                   unsigned vec_size);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_SHUFFLE_MASKS_HPP

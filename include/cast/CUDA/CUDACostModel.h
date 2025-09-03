#ifndef CAST_CUDA_CUDACOSTMODEL_H
#define CAST_CUDA_CUDACOSTMODEL_H

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/Core/CostModel.h"
#include "utils/CSVParsable.h"
#include <span>

namespace cast {

class CUDAPerformanceCache {
  struct Item : utils::CSVParsable<Item> {
    int k;
    double opCount;
    Precision precision;
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;

    Item(int k, double opCount, Precision precision, double memUpdateSpeed)
        : k(k), opCount(opCount), precision(precision),
          memUpdateSpeed(memUpdateSpeed) {}

    CSV_DATA_FIELD(k, opCount, precision, memUpdateSpeed);
  }; // struct Item

  std::vector<Item> items_;

public:
  static constexpr auto CSVTitle() { return Item::CSV_TITLE; }

  std::span<const Item> items() const { return items_; }

  llvm::Error runExperiments(const CUDAKernelGenConfig& kernelConfig,
                             int nQubits,
                             int nWorkerThreads,
                             int nRuns,
                             int verbose = 1);

  void writeResults(std::ostream& os) const;

}; // class CUDAPerformanceCache

class CUDACostModel : public CostModel {
public:
  CUDACostModel() : CostModel(CM_CUDA) {}
};

} // namespace cast

#endif // CAST_CUDA_CUDACOSTMODEL_H
#ifndef CAST_CUDA_CUDACOSTMODEL_H
#define CAST_CUDA_CUDACOSTMODEL_H

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/Core/CostModel.h"
#include "utils/CSVParsable.h"
#include <span>

namespace cast {

class CUDAPerformanceCache {

  struct Item : utils::CSVParsable<Item> {
    int nQubits;
    double opCount;
    Precision precision;
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;

    CSV_DATA_FIELD(nQubits, opCount, precision, memUpdateSpeed);
  }; // struct Item

  std::vector<Item> items_;

public:
  std::span<const Item> items() const { return items_; }

  void runExperiments(const CUDAKernelGenConfig& kernelConfig,
                      int nQubits,
                      int nWorkerThreads,
                      int nRuns,
                      int verbose = 1);

}; // class CUDAPerformanceCache

class CUDACostModel : public CostModel {
public:
  CUDACostModel() : CostModel(CM_CUDA) {}
};

} // namespace cast

#endif // CAST_CUDA_CUDACOSTMODEL_H
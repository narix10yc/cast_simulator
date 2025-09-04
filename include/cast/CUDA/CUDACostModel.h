#ifndef CAST_CUDA_CUDACOSTMODEL_H
#define CAST_CUDA_CUDACOSTMODEL_H

#include "cast/ADT/SortedVectorMap.h"
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
                             int nQubitsSV,
                             int nWorkerThreads,
                             int nRuns,
                             int verbose = 1);

  void writeResults(std::ostream& os) const;

  llvm::Error loadFrom(std::istream& is) {
    items_.clear();
    std::string line;
    // Skip the title line
    if (!std::getline(is, line))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to read CSV title line");
    while (std::getline(is, line)) {
      Item item(0, 0.0, Precision::Unknown, 0.0);
      item.parse(line);
      items_.push_back(item);
    }
    return llvm::Error::success();
  }

  static llvm::Expected<CUDAPerformanceCache> LoadFrom(std::istream& is) {
    CUDAPerformanceCache cache;
    if (auto err = cache.loadFrom(is))
      return std::move(err);
    return cache;
  }

}; // class CUDAPerformanceCache

class CUDACostModel : public CostModel {
  struct BucketKey {
    int k; // number of gate qubits
    Precision precision;

    bool operator<(const BucketKey& other) const {
      if (precision != other.precision)
        return precision < other.precision;
      return k < other.k;
    }
  };
  // Value: GiB time per op count
  SortedVectorMap<BucketKey, float> bucket_;
  Precision queryPrecision_ = Precision::Unknown;

  // The minimum time it will take to update 1GiB memory. Calculated by
  // 1.0 / bandwidth. In the cost model initialization, this is set by the
  // maximum memory update speed across buckets.
  float minGiBTimeCap{};

public:
  CUDACostModel() : CostModel(CM_CUDA) {}

  CUDACostModel(const CUDAPerformanceCache& cache);

  void setQueryPrecision(Precision p) { queryPrecision_ = p; }

  double computeGiBTime(const QuantumGate* gate) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;
};

} // namespace cast

#endif // CAST_CUDA_CUDACOSTMODEL_H
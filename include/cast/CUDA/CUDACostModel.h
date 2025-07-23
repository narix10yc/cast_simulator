#ifndef CAST_CUDA_CUDACOSTMODEL_H
#define CAST_CUDA_CUDACOSTMODEL_H

#include "cast/Core/CostModel.h"
#include "utils/CSVParsable.h"

namespace cast {

class CUDAPerformanceCache {
public:
  struct Item : public utils::CSVParsable<Item> {
    int nQubits;
    int opCount;
    int precision;
    int blockSize;
    double occupancy;
    double coalescingScore;
    double memUpdateSpeed;

    CSV_DATA_FIELD(nQubits, opCount, precision, blockSize,
                   occupancy, coalescingScore, memUpdateSpeed)

    double getAvgGibTimePerOpCount() const {
        return (1.0 / memUpdateSpeed) / opCount;
    }
  };

  std::vector<Item> items;
  int defaultBlockSize = 256;

  void runExperiments(
    const CUDAKernelGenConfig& gpuConfig,
    int nQubits, int blockSize, int nRuns, int nWorkerThreads);
  void writeResults(std::ostream& os) const;
  void writeResults(const std::string& filename) const;
  static CUDAPerformanceCache LoadFromCSV(const std::string& filename);
  const Item* findClosestMatch(
      const QuantumGate& gate, int precision, int blockSize) const;
};

class CUDACostModel : public CostModel {
    const CUDAPerformanceCache* cache;
    double zeroTol;
    int currentBlockSize;
    double minGibTimeCap;
    
public:
    explicit CUDACostModel(const CUDAPerformanceCache* c, double zt = 1e-8)
      : CostModel(CM_CUDA)
      , cache(c), zeroTol(zt), currentBlockSize(256), minGibTimeCap(1e-9) {}
    
    double computeGiBTime(const QuantumGate& gate, int precision, int) const override;

    void setBlockSize(int blockSize) { 
      if (blockSize < 32 || blockSize > 1024 || 
          (blockSize & (blockSize-1)) != 0) {
          throw std::invalid_argument(
              "Block size must be power of 2 between 32-1024");
      }
      currentBlockSize = blockSize;
    }
    
private:
    double calculateOccupancyPenalty(const CUDAPerformanceCache::Item& item) const;
    double calculateCoalescingPenalty(const CUDAPerformanceCache::Item& item) const;
};

} // namespace cast

#endif // CAST_CUDA_CUDACOSTMODEL_H
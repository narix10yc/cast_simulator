#ifndef CAST_CORE_COSTMODEL_H
#define CAST_CORE_COSTMODEL_H

#include "cast/Core/QuantumGate.h"
#include <string>

namespace cast {
  
// An abstract class for all cost models.
class CostModel {
public:
  enum CostModelKind {
    CM_Base,
    CM_SizeOnly, // Size only cost model
    CM_Constant, // Constant cost model. Every gate takes the same time.
    CM_CPU, // CPU cost model
    CM_CUDA,
    CM_End
  };
protected:
  CostModelKind _kind;
public:
  explicit CostModel(CostModelKind kind) : _kind(kind) {}

  virtual ~CostModel() = default;

  CostModelKind getKind() const { return _kind; }

  // The time it takes to update 1 GiB of memory, in seconds.
  virtual double computeGiBTime(const QuantumGate* gate) const = 0;

  virtual std::ostream& displayInfo(std::ostream& os, int verbose) const {
    return os << "CostModel::displayInfo() not implemented";
  }
};

/// @brief \c NaiveCostModel is based on the size and operation count of fused
/// gates.
class SizeOnlyCostModel : public CostModel {
  int maxSize;
  int maxOp;
  double zeroTol;
public:
  SizeOnlyCostModel(int maxSize, int maxOp, double zeroTol)
    : CostModel(CM_SizeOnly)
    , maxSize(maxSize), maxOp(maxOp), zeroTol(zeroTol) {}

  double computeGiBTime(const QuantumGate* gate) const override;
  
  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_SizeOnly;
  }
};

class ConstantCostModel : public CostModel {
public:
  ConstantCostModel() : CostModel(CM_Constant) {}

  double computeGiBTime(const QuantumGate* gate) const override {
    return 1.0;
  }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_Constant;
  }
};

#ifdef CAST_USE_CUDA

class CUDAPerformanceCache {
public:
  struct Item {
    int nQubits;
    int opCount;
    Precision precision;
    int blockSize;
    double occupancy;
    double coalescingScore;
    double memUpdateSpeed;
    int nData = 1;
    
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
      const legacy::QuantumGate& gate, Precision precision, int blockSize) const;
  
  constexpr static const char* CSV_HEADER = 
      "nQubits,opCount,precision,blockSize,occupancy,coalescing,memSpd";
};

class CUDACostModel : public CostModel {
    const CUDAPerformanceCache* cache;
    double zeroTol;
    int currentBlockSize;
    double minGibTimeCap;
    
public:
    explicit CUDACostModel(const CUDAPerformanceCache* c, double zt = 1e-8)
      : cache(c), zeroTol(zt), currentBlockSize(256), minGibTimeCap(1e-9) {}
    
    double computeGiBTime(const legacy::QuantumGate& gate, Precision precision, int) const override;

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

#endif // CAST_USE_CUDA


} // namespace cast

#endif // CAST_CORE_COSTMODEL_H
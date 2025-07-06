#ifndef CAST_COSTMODEL_H
#define CAST_COSTMODEL_H

#include "cast/Core/QuantumGate.h"
#include "cast/Core/Precision.h"
#include <string>

namespace cast {
  
class CPUKernelGenConfig;
class PerformanceCache;

class CostModel {
public:
  enum CostModelKind {
    CM_Base,
    CM_SizeOnly, // Size only cost model
    CM_Standard, // Standard cost model based on performance cache
    CM_Constant, // Constant cost model
    CM_End
  };
protected:
  CostModelKind _kind;
public:
  explicit CostModel(CostModelKind kind) : _kind(kind) {}

  virtual ~CostModel() = default;

  CostModelKind getKind() const { return _kind; }

  // The time it takes to update 1 GiB of memory, in seconds.
  virtual double computeGiBTime(
      QuantumGatePtr gate, Precision precision, int nThreads) const = 0;
};

/// @brief \c NaiveCostModel is based on the size and operation count of fused
/// gates.
class SizeOnlyCostModel : public CostModel {
  int maxNQubits;
  int maxOp;
  double zeroTol;

public:
  SizeOnlyCostModel(int maxNQubits, int maxOp, double zeroTol)
    : CostModel(CM_SizeOnly)
    , maxNQubits(maxNQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  double computeGiBTime(
      QuantumGatePtr gate, Precision precision, int nThreads) const override;
  
  static bool classof(const CostModel* model) {
    return model->getKind() == CM_SizeOnly;
  }
};

/// \c StandardCostModel assumes simulation time is proportional to opCount and
/// independent to target qubits.
class StandardCostModel : public CostModel {
  PerformanceCache* cache;
  double zeroTol;
  // Minimum time it will take to update 1GiB memory. Calculated by
  // 1.0 / bandwidth
  double minGibTimeCap;

  struct Item {
    int nQubits;
    Precision precision;
    int nThreads;
    int nData; // number of data points;
    double totalGibTimePerOpCount;

    double getAvgGibTimePerOpCount() const {
      return totalGibTimePerOpCount / nData;
    }
  };
  std::vector<Item> items;
public:
  StandardCostModel(PerformanceCache* cache, double zeroTol = 1e-8);

  std::ostream& display(std::ostream& os, int nLines = 0) const;

  double computeGiBTime(
      QuantumGatePtr gate, Precision precision, int nThreads) const override;
  
  static bool classof(const CostModel* model) {
    return model->getKind() == CM_Standard;
  }
};

class ConstantCostModel : public CostModel {
public:
  ConstantCostModel() : CostModel(CM_Constant) {}

  double computeGiBTime(
      QuantumGatePtr gate, Precision precision, int nThreads) const override {
    return 1.0;
  }

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_Constant;
  }
};

class PerformanceCache {
public:
  struct Item {
    int nQubits;
    double opCount;
    Precision precision;
    int nThreads;
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;
  };

  std::vector<Item> items;
  PerformanceCache() : items() {}

  void runExperiments(
      const CPUKernelGenConfig& cpuConfig,
      int nQubits, int nThreads, int nRuns);

  void writeResults(std::ostream& os) const;
  
  static PerformanceCache LoadFromCSV(const std::string& fileName);
  
  constexpr static const char*
  CSV_Title = "nQubits,opCount,precision,nThreads,memSpd";
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

#endif // CAST_COSTMODEL_H
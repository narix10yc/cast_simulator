#ifndef CAST_COSTMODEL_H
#define CAST_COSTMODEL_H

#include "cast/Core/QuantumGate.h"
#include <cassert>
#include <string>
#include <vector>

namespace cast {
  
class CPUKernelGenConfig;
class PerformanceCache;

class CostModel {
public:
  virtual ~CostModel() = default;

  // The time it takes to update 1 GiB of memory, in seconds.
  virtual double computeGiBTime(
      QuantumGatePtr gate, int precision, int nThreads) const = 0;
};

/// @brief \c NaiveCostModel is based on the size and operation count of fused
/// gates.
class NaiveCostModel : public CostModel {
  int maxNQubits;
  int maxOp;
  double zeroTol;

public:
  NaiveCostModel(int maxNQubits, int maxOp, double zeroTol)
    : maxNQubits(maxNQubits), maxOp(maxOp), zeroTol(zeroTol) {}

  double computeGiBTime(
      QuantumGatePtr gate, int precision, int nThreads) const override;
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
    int precision;
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
      QuantumGatePtr gate, int precision, int nThreads) const override;
};

class PerformanceCache {
public:
  struct Item {
    int nQubits;
    double opCount;
    int precision;
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
    int precision;
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
      const legacy::QuantumGate& gate, int precision, int blockSize) const;
  
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
    
    double computeGiBTime(const legacy::QuantumGate& gate, int precision, int) const override;

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
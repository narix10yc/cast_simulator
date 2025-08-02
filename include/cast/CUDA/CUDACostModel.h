// #ifndef CAST_CUDA_CUDACOSTMODEL_H
// #define CAST_CUDA_CUDACOSTMODEL_H

// #include "cast/Core/CostModel.h"
// #include "utils/CSVParsable.h"

// namespace cast {

// class CUDAPerformanceCache {
// public:
//   struct Item : public utils::CSVParsable<Item> {
//     int nQubits;
//     int opCount;
//     int precision;
//     int blockSize;
//     double occupancy;
//     double coalescingScore;
//     double memUpdateSpeed;

//     CSV_DATA_FIELD(nQubits,
//                    opCount,
//                    precision,
//                    blockSize,
//                    occupancy,
//                    coalescingScore,
//                    memUpdateSpeed)

//     double getAvgGibTimePerOpCount() const {
//       return (1.0 / memUpdateSpeed) / opCount;
//     }
//   };

//   std::vector<Item> items;
//   int defaultBlockSize = 256;

//   void runExperiments(const CUDAKernelGenConfig& gpuConfig,
//                       int nQubits,
//                       int blockSize,
//                       int nRuns,
//                       int nWorkerThreads);
//   void writeResults(std::ostream& os) const;
//   void writeResults(const std::string& filename) const;
//   static CUDAPerformanceCache LoadFromCSV(const std::string& filename);
//   const Item*
//   findClosestMatch(const QuantumGate& gate, int precision, int blockSize) const;
// };

// class CUDACostModel : public CostModel {
//   const CUDAPerformanceCache* cache;
//   double zeroTol;
//   int currentBlockSize;
//   double minGibTimeCap;

// public:
//   explicit CUDACostModel(const CUDAPerformanceCache* c, double zt = 1e-8)
//       : CostModel(CM_CUDA), cache(c), zeroTol(zt), currentBlockSize(256),
//         minGibTimeCap(1e-9) {}

//   double
//   computeGiBTime(const QuantumGate& gate, int precision, int) const override;

//   void setBlockSize(int blockSize) {
//     if (blockSize < 32 || blockSize > 1024 ||
//         (blockSize & (blockSize - 1)) != 0) {
//       throw std::invalid_argument(
//           "Block size must be power of 2 between 32-1024");
//     }
//     currentBlockSize = blockSize;
//   }

// private:
//   double
//   calculateOccupancyPenalty(const CUDAPerformanceCache::Item& item) const;
//   double
//   calculateCoalescingPenalty(const CUDAPerformanceCache::Item& item) const;
// };

// } // namespace cast

// #endif // CAST_CUDA_CUDACOSTMODEL_H



#ifndef CAST_CUDA_CUDACOSTMODEL_H
#define CAST_CUDA_CUDACOSTMODEL_H

#ifdef CAST_USE_CUDA

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/Precision.h"
#include "utils/CSVParsable.h"

#include <array>
#include <fstream>
#include <memory>

namespace cast {

class CUDAPerformanceCache {
public:
  struct Item : utils::CSVParsable<Item> {
    int nQubits = 0;
    double opCount = 0.0;
    Precision precision = Precision::Unknown;
    int blockSize = 256;
    double occupancy = 1.0;
    double coalescingScore = 1.0;
    double memUpdateSpeed = 0.0;

    Item() = default;
    Item(int nQ, double ops, Precision p, int blk,
         double occ, double coal, double bw)
        : nQubits(nQ), opCount(ops), precision(p), blockSize(blk),
          occupancy(occ), coalescingScore(coal), memUpdateSpeed(bw) {}

    CSV_DATA_FIELD(nQubits, opCount, precision,
                   blockSize, occupancy, coalescingScore, memUpdateSpeed);
  };

  std::vector<Item> items;
  int defaultBlockSize = 256;

  CUDAPerformanceCache() = default;
  explicit CUDAPerformanceCache(const std::string &fileName);

  void runExperiments(const CUDAKernelGenConfig &cfg,
                      int nQubits,
                      int blockSize,
                      int nRuns,
                      int nWorkerThreads,
                      int verbose = 1);

  void writeResults(std::ostream &os) const;

  const Item *findClosestMatch(const QuantumGate *gate,
                               Precision precision,
                               int blockSize) const;
};

class CUDACostModel : public CostModel {
  // One row per (nQ, precision, blockSize)
  struct Item {
    int nQubits;
    Precision precision;
    int blockSize;
    int nData = 0;
    double totalGiBTimePerOpCnt = 0.0;
    double occupancy = 1.0;
    double coalescingScore = 1.0;

    double getAvgGiBTimePerOpCnt() const {
      return totalGiBTimePerOpCnt / nData;
    }
  };

  std::unique_ptr<CUDAPerformanceCache> cache;
  std::vector<Item>                     items;
  double                                minGiBTimeCap = 0.0;
  double                                zeroTol       = 1e-8;

  int queryBlockSize = -1;
  Precision queryPrecision = Precision::Unknown;

public:
  explicit CUDACostModel(std::unique_ptr<CUDAPerformanceCache> cache,
                         double zeroTol = 1e-8);

  void setQueryBlockSize(int blk)     { queryBlockSize = blk; }
  void setQueryPrecision(Precision p) { queryPrecision = p;   }

  double computeGiBTime(const QuantumGate *gate) const override;
  std::ostream &displayInfo(std::ostream &os,
                            int verbose = 1) const override;

  static bool classof(const CostModel *m) { return m->getKind() == CM_CUDA; }
};

} // namespace cast
#endif   // CAST_USE_CUDA
#endif   // CAST_CUDA_CUDACOSTMODEL_H
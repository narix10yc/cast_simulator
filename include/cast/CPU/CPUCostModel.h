#ifndef CAST_CPU_CPUCOSTMODEL_H
#define CAST_CPU_CPUCOSTMODEL_H

#include "cast/Core/CostModel.h"
#include "cast/Core/Precision.h"
#include "utils/CSVParsable.h"
#include "cast/CPU/Config.h"

#include <fstream>

namespace cast {

class CPUKernelGenConfig;

class CPUPerformanceCache {
public:
  struct Item : utils::CSVParsable<Item> {
    int nQubits;
    double opCount;
    Precision precision;
    int nThreads;
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;

    Item() = default;

    Item(int nQubits, double opCount, Precision precision,
         int nThreads, double memUpdateSpeed)
      : nQubits(nQubits), opCount(opCount), precision(precision),
        nThreads(nThreads), memUpdateSpeed(memUpdateSpeed) {}

    CSV_DATA_FIELD(nQubits, opCount, precision, nThreads, memUpdateSpeed);
  };

  std::vector<Item> items;
// private:
  using WeightType = std::array<int, CPU_GLOBAL_MAX_SIZE>;
  void runPreliminaryExperiments(const CPUKernelGenConfig& cpuConfig,
                                 int nQubits,
                                 int nThreads,
                                 WeightType& weights,
                                 int verbose = 1);
public:
  CPUPerformanceCache() = default;

  CPUPerformanceCache(const std::string& fileName) {
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      assert(false && "Failed to open CPUPerformanceCache file");
      return;
    }
    std::string line;
    std::getline(ifs, line); // Read header
    assert(line == Item::CSV_TITLE);

    while (std::getline(ifs, line)) {
      Item item;
      item.parse(line);
      items.push_back(std::move(item));
    }
  }

  void runExperiments(const CPUKernelGenConfig& cpuConfig,
                      int nQubits, int nThreads, int nRuns, int verbose = 1);

  void writeResults(std::ostream& os) const;
};

/// \c CPUCostModel assumes simulation time is proportional to opCount and
/// independent to target qubits.
class CPUCostModel : public CostModel {
  std::unique_ptr<CPUPerformanceCache> cache;
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

  int queryNThreads = -1; // -1 means not set
  Precision queryPrecision = Precision::Unknown;
public:
  CPUCostModel(std::unique_ptr<CPUPerformanceCache> cache,
               double zeroTol = 1e-8);

  void setQueryNThreads(int nThreads) {
    queryNThreads = nThreads;
  }
  
  void setQueryPrecision(Precision precision) {
    queryPrecision = precision;
  }

  double computeGiBTime(const QuantumGate* gate) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;
  
  static bool classof(const CostModel* model) {
    return model->getKind() == CM_CPU;
  }
};

} // end namespace cast

#endif // CAST_CPU_CPUCOSTMODEL_H
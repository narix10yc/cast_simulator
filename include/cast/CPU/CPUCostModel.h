#ifndef CAST_CPU_CPUCOSTMODEL_H
#define CAST_CPU_CPUCOSTMODEL_H

#include "cast/CPU/Config.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/Precision.h"
#include "utils/CSVParsable.h"

#include <fstream>
#include <span>

namespace cast {

struct CPUKernelGenConfig;

class CPUPerformanceCache {
  struct Item : utils::CSVParsable<Item> {
    int nQubits;
    double opCount;
    Precision precision;
    int nThreads;
    /// memory update speed in Gigabytes per second (GiBps)
    double memUpdateSpeed;

    Item() = default;

    Item(int nQubits,
         double opCount,
         Precision precision,
         int nThreads,
         double memUpdateSpeed)
        : nQubits(nQubits), opCount(opCount), precision(precision),
          nThreads(nThreads), memUpdateSpeed(memUpdateSpeed) {}

    CSV_DATA_FIELD(nQubits, opCount, precision, nThreads, memUpdateSpeed);
  };

  std::vector<Item> items_;
  // private:
  using WeightType = std::array<int, CPU_GLOBAL_MAX_SIZE>;
  void runPreliminaryExperiments(const CPUKernelGenConfig& cpuConfig,
                                 int nQubits,
                                 int nThreads,
                                 WeightType& weights,
                                 int verbose = 1);

public:
  CPUPerformanceCache() = default;

  static constexpr auto CSVTitle() { return Item::CSV_TITLE; }

  // Returns null if we cannot find performance cache or the cache is empty
  static std::unique_ptr<CPUPerformanceCache>
  LoadFromFile(const std::string& filename) {
    auto pc = std::make_unique<CPUPerformanceCache>();
    if (!pc->loadFromFile(filename))
      return nullptr;
    if (pc->items().empty())
      return nullptr;
    return pc;
  }

  std::span<const Item> items() const { return items_; }

  // return true on success
  bool loadFromFile(const std::string& fileName) {
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      return false;
    }
    std::string line;
    std::getline(ifs, line); // Read header
    if (line != Item::CSV_TITLE) {
      return false;
    }

    while (std::getline(ifs, line)) {
      items_.emplace_back();
      items_.back().parse(line);
    }
    return true;
  }

  void runExperiments(const CPUKernelGenConfig& cpuConfig,
                      int nQubits,
                      int nThreads,
                      int nRuns,
                      int verbose = 1);

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

  static std::unique_ptr<CPUCostModel>
  LoadFromFile(const std::string& filename) {
    auto pc = CPUPerformanceCache::LoadFromFile(filename);
    if (!pc)
      return nullptr;
    return std::make_unique<CPUCostModel>(std::move(pc));
  }

  void setQueryNThreads(int nThreads) { queryNThreads = nThreads; }

  void setQueryPrecision(Precision precision) { queryPrecision = precision; }

  double computeGiBTime(const QuantumGate* gate) const override;

  void showEntries(std::ostream& os, int nLines) const;

  void displayInfo(utils::InfoLogger logger) const override {}

  static bool classof(const CostModel* model) {
    return model->getKind() == CM_CPU;
  }
};

} // end namespace cast

#endif // CAST_CPU_CPUCOSTMODEL_H
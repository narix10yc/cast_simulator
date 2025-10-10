#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/Internal/PerfCacheHelper.h"

#include "timeit/timeit.h"
#include "utils/Formats.h"
#include "utils/PrintSpan.h"
#include "utils/utils.h"

using namespace cast;

CPUCostModel::CPUCostModel(std::unique_ptr<CPUPerformanceCache> cache,
                           double zeroTol)
    : CostModel(CM_CPU), cache(std::move(cache)), zeroTol(zeroTol), items() {
  items.reserve(32);

  // loop through cache and initialize this->items
  for (const auto& cacheItem : this->cache->items()) {
    auto it =
        std::ranges::find_if(this->items, [&cacheItem](const Item& thisItem) {
          return thisItem.nQubits == cacheItem.nQubits &&
                 thisItem.precision == cacheItem.precision &&
                 thisItem.nThreads == cacheItem.nThreads;
        });
    // time it takes to update 1 GiB memory per opCount
    auto t = 1.0 / (cacheItem.memUpdateSpeed * cacheItem.opCount);
    if (it == this->items.end()) {
      items.emplace_back(
          cacheItem.nQubits, cacheItem.precision, cacheItem.nThreads, 1, t);
    } else {
      it->nData++;
      it->totalGibTimePerOpCount += t;
    }
  }

  // initialize minGibTimeCap
  assert(!items.empty());
  this->minGibTimeCap = 1e6; // a large number
  for (const auto& item : items) {
    // The time it takes to update 1 GiB memory on dense gates
    double opCount = static_cast<double>(1ULL << (item.nQubits + 2));
    auto t = item.totalGibTimePerOpCount * opCount / item.nData;
    if (t < this->minGibTimeCap)
      this->minGibTimeCap = t;
  }
}

double CPUCostModel::computeGiBTime(const QuantumGate* gate) const {
  assert(gate != nullptr);
  assert(!items.empty());

  assert(queryNThreads > 0 &&
         "CPUCostModel: Must set queryNThreads before calling computeGiBTime");
  assert(queryPrecision != Precision::Unknown &&
         "CPUCostModel: Must set queryPrecision before calling computeGiBTime");

  auto gateNQubits = gate->nQubits();
  auto gateOpCount = gate->opCount(zeroTol);
  // Try to find an exact match
  for (const auto& item : items) {
    if (item.nQubits == gateNQubits && item.precision == queryPrecision &&
        item.nThreads == queryNThreads) {
      auto t = item.getAvgGibTimePerOpCount() * gateOpCount;
      return std::max(t, this->minGibTimeCap);
    }
  }

  // No exact match. Estimate it
  auto bestMatchIt = items.begin();

  auto it = items.cbegin();
  const auto end = items.cend();
  while (++it != end) {
    // priority: nThreads > nQubits > precision
    const int bestNThreadsDiff =
        std::abs(queryNThreads - bestMatchIt->nThreads);
    const int thisNThreadsDiff = std::abs(queryNThreads - it->nThreads);
    if (thisNThreadsDiff > bestNThreadsDiff)
      continue;
    if (thisNThreadsDiff < bestNThreadsDiff) {
      bestMatchIt = it;
      continue;
    }

    const int bestNQubitsDiff = std::abs(gateNQubits - bestMatchIt->nQubits);
    const int thisNQubitsDiff = std::abs(gateNQubits - it->nQubits);
    if (thisNQubitsDiff > bestNQubitsDiff)
      continue;
    if (thisNQubitsDiff < bestNQubitsDiff) {
      bestMatchIt = it;
      continue;
    }

    if (queryPrecision == bestMatchIt->precision)
      continue;
    if (queryPrecision == it->precision) {
      bestMatchIt = it;
      continue;
    }
  }

  // best match GiBTime per opCount
  auto bestMatchT0 = bestMatchIt->getAvgGibTimePerOpCount();
  // estimated GibTime per opCount
  auto estT0 = bestMatchT0 * bestMatchIt->nThreads / queryNThreads;
  // estimated GibTime for this gate
  auto estTime = std::max(estT0 * gateOpCount, this->minGibTimeCap);

  // std::cerr << YELLOW("Warning: ") << "No exact match to "
  //              "[nQubits, precision, nThreads] = ["
  //           << gateNQubits << ", " << gateOpCount << ", "
  //           << static_cast<int>(queryPrecision) << ", " << queryNThreads
  //           << "] found. We estimate it by ["
  //           << bestMatchIt->nQubits << ", "
  //           << static_cast<int>(bestMatchIt->precision)
  //           << ", " << bestMatchIt->nThreads
  //           << "] @ " << bestMatchT0 << " s/GiB/op => "
  //              "Est. " << estT0 << " s/GiB/op.\n";

  return estTime;
}

void CPUCostModel::showEntries(std::ostream& os, int nLines) const {
  const int nLinesToDisplay = std::min<int>(nLines, items.size());

  os << "Memory Bandwidth: " << utils::fmt_1_to_1e3(1.0 / this->minGibTimeCap)
     << " GiBps\n";
  os << "  nQubits | Precision | nThreads | Dense MemSpd \n";
  for (int i = 0; i < nLinesToDisplay; ++i) {
    double opCount = static_cast<double>(1ULL << (items[i].nQubits + 2));
    double timePerGiB = items[i].getAvgGibTimePerOpCount() * opCount;
    os << "    " << std::fixed << std::setw(2) << items[i].nQubits
       << "    |    f" << static_cast<int>(items[i].precision) << "    |    "
       << items[i].nThreads << "    |    "
       << utils::fmt_1_to_1e3(1.0 / timePerGiB) << "\n";
  }
}

void CPUPerformanceCache::runPreliminaryExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits,
    int nThreads,
    WeightType& weights,
    int verbose) {
  CPUKernelManager km;
  std::vector<int> qubits;

  const auto generateGatesAndInitJit = [&]() {
    for (int k = 1; k <= 5; ++k) {
      utils::sampleNoReplacement(nQubits, k, qubits);
      auto gate = StandardQuantumGate::RandomUnitary(qubits);
      llvm::cantFail(
          km.genGate(cpuConfig, gate, "gate_k" + std::to_string(k)));
    }
    llvm::cantFail(km.compileAll(llvm::OptimizationLevel::O1, false));
  };

  if (verbose > 0) {
    utils::timedExecute(generateGatesAndInitJit,
                        "Code Generation and JIT Initialization");
  } else {
    generateGatesAndInitJit();
  }

  CPUStatevectorWrapper sv(cpuConfig.precision, nQubits, cpuConfig.simdWidth);

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;
  std::array<double, 5> memSpds;
  for (int k = 1; k <= 5; ++k) {
    tr = timer.timeit([&]() {
      llvm::cantFail(km.applyCPUKernel(
          sv.data(), nQubits, "gate_k" + std::to_string(k), nThreads));
    });
    memSpds[k - 1] =
        internal::calculateMemUpdateSpeed(nQubits, cpuConfig.precision, tr.min);
    if (verbose > 0) {
      std::cerr << "Dense " << k << "-qubit gate @ " << memSpds[k - 1]
                << " GiBps\n";
    }
  }

  assert(weights.size() >= 5);
  weights[0] = 100; // 1-qubit gates

  // The ratio decides the scaling of weights when mem speed halves.
  // Set to a larger value to focus more on the transition region.
  constexpr double ratio = 1.1;

  // Decay decides the decaying rate for >=5 qubit gates
  constexpr double decay = 0.45;
  for (int k = 2; k <= 5; ++k) {
    // weights[k] is the weight for k-qubit gates
    weights[k - 1] = static_cast<int>(
        static_cast<double>(weights[k - 2]) *
        ((ratio - 1.0) * (memSpds[k - 2] / memSpds[k - 1]) + 2 - ratio));
  }
  for (int k = 6; k < weights.size() + 1; ++k) {
    weights[k - 1] =
        static_cast<int>(static_cast<double>(weights[k - 2]) * decay);
  }

  int maxIdx = 0;
  double maxWeight = 0.0;
  for (int k = 0; k < weights.size(); ++k) {
    if (weights[k] > maxWeight) {
      maxWeight = static_cast<double>(weights[k]);
      maxIdx = k;
    }
  }
  // An extra round of weight decay. More decay for distant weights
  for (int k = 0; k < weights.size(); ++k) {
    if (k == maxIdx)
      continue;
    constexpr double ratio = 1.5;
    double newWeight =
        static_cast<double>(weights[k]) / std::pow(ratio, std::abs(maxIdx - k));
    weights[k] = static_cast<int>(newWeight);
  }

  if (verbose > 1) {
    double sum = std::reduce(weights.begin(), weights.end(), 0.0);
    std::cerr << "Relative weights:\n";
    for (int k = 1; k <= weights.size(); ++k) {
      std::cerr << "  " << k << "-qubit: " << "weight = " << weights[k - 1]
                << "; percentage = "
                << (100.0 * static_cast<double>(weights[k - 1]) / sum) << "\n";
    }
  }
}

void CPUPerformanceCache::runExperiments(const CPUKernelGenConfig& cpuConfig,
                                         int nQubits,
                                         int nThreads,
                                         int nRuns,
                                         int verbose) {
  std::vector<StandardQuantumGatePtr> gates;
  gates.reserve(nRuns);

  // nQubitsWeights[k-1] denotes the weight for k-qubit gates
  WeightType nQubitsWeights;
  runPreliminaryExperiments(
      cpuConfig, nQubits, nThreads, nQubitsWeights, verbose);

  // Add a random quantum gate whose size follows distribution of nQubitsWeights
  const auto addRandomQuGate = [&](float erasureProb) {
    int sum = 0;
    for (const auto& weight : nQubitsWeights)
      sum += weight;
    assert(sum > 0 && "nQubitsWeight is empty");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, sum - 1);
    int r = dist(gen);
    int acc = 0;
    for (int i = 0; i < nQubitsWeights.size(); ++i) {
      acc += nQubitsWeights[i];
      if (r <= acc) {
        std::vector<int> targetQubits;
        utils::sampleNoReplacement(nQubits, i + 1, targetQubits);
        gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));
        internal::randRemoveQuantumGate(gates.back().get(), erasureProb);
        return;
      }
    }
    assert(false && "Unreachable: addRandomQuGate failed to add a gate");
  };

  // For the initial run, we add some random dense 1 to 5-qubit gates
  for (int q = 1; q <= 5; ++q) {
    std::vector<int> targetQubits;
    utils::sampleNoReplacement(nQubits, q, targetQubits);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));

    utils::sampleNoReplacement(nQubits, q, targetQubits);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));
  }

  int initialRuns = gates.size();
  for (int run = 0; run < nRuns - initialRuns; ++run) {
    float ratio = static_cast<float>(run) / (nRuns - initialRuns);
    float prob = ratio * 1.0f + (1.0f - ratio) * 0.25f;
    addRandomQuGate(prob);
  }

  CPUKernelManager km;
  utils::timedExecute(
      [&]() {
        int i = 0;
        for (const auto& gate : gates) {
          if (auto e = km.genGate(
                  cpuConfig, gate, "gate_" + std::to_string(i++))) {
            std::cerr << RED("Error: ") << "Failed to generate kernel for gate "
                      << i - 1 << ": " << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        }
      },
      "Code Generation");

  utils::timedExecute(
      [&]() {
        if (auto e = km.compileAll(llvm::OptimizationLevel::O1, false)) {
          std::cerr << RED("Error: ") << "Failed to initialize JIT engine: "
                    << llvm::toString(std::move(e)) << "\n";
          std::exit(1);
        }
      },
      "Initialize JIT Engine");

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  CPUStatevectorWrapper sv(cpuConfig.precision, nQubits, cpuConfig.simdWidth);
  utils::timedExecute([&]() { sv.randomize(nThreads); },
                      "Initialize statevector");

  for (auto& kernel : km.all_kernels()) {
    tr = timer.timeit([&]() {
      llvm::cantFail(
          km.applyCPUKernel(sv.data(), sv.nQubits(), *kernel, nThreads));
    });
    auto memSpd =
        internal::calculateMemUpdateSpeed(nQubits, kernel->precision, tr.min);
    items_.emplace_back(kernel->gate->nQubits(),
                        kernel->opCount,
                        kernel->precision,
                        nThreads,
                        memSpd);
    if (verbose > 0) {
      utils::printSpan(std::cerr << "Gate @ ",
                       std::span(kernel->gate->qubits()));
      std::cerr << ": " << memSpd << " GiBps\n";
    }
  }
}

void CPUPerformanceCache::writeResults(std::ostream& os) const {
  for (const auto& item : items_) {
    item.write(os);
    os << "\n";
  }
}
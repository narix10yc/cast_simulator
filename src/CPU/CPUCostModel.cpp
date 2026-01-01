#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/Internal/PerfCacheHelper.h"

#include "timeit/timeit.h"
#include "utils/Formats.h"
#include "utils/PrintSpan.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <random>

using namespace cast;

void CPUCostModel::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUCostModel")
      .put("Query nThreads   ", queryNThreads)
      .put("Query precision  ", static_cast<int>(queryPrecision))
      .put("Number of buckets", buckets.size())
      .put("Memory Bandwidth ", utils::fmt_mem(1e9 / this->minGibTimeCap));
}

llvm::Error CPUCostModel::loadCache(const CPUPerformanceCache& cache) {
  // loop through cache and initialize this->items
  for (const auto& item : cache.items()) {
    auto it =
        std::ranges::find_if(this->buckets, [&item](const Bucket& thisItem) {
          return thisItem.nQubits == item.nQubits &&
                 thisItem.precision == item.precision &&
                 thisItem.nThreads == item.nThreads;
        });
    // time it takes to update 1 GiB memory per opCount
    auto t = 1.0 / (item.memUpdateSpeed * item.opCount);
    if (it == this->buckets.end()) {
      buckets.emplace_back(item.nQubits, item.precision, item.nThreads, 1, t);
    } else {
      it->nData++;
      it->totalGibTimePerOpCount += t;
    }
  }

  // initialize minGibTimeCap

  if (buckets.empty()) {
    return llvm::createStringError(
        "CPUCostModel: empty buckets after loading cache");
  }

  this->minGibTimeCap = 1e6; // a large number
  for (const auto& item : buckets) {
    // The time it takes to update 1 GiB memory on dense gates
    double opCount = static_cast<double>(1ULL << (item.nQubits + 2));
    auto t = item.totalGibTimePerOpCount * opCount / item.nData;
    if (t < this->minGibTimeCap)
      this->minGibTimeCap = t;
  }

  return llvm::Error::success();
}

double CPUCostModel::computeGiBTime(const QuantumGate* gate) const {
  assert(gate != nullptr);
  assert(!buckets.empty());

  assert(queryNThreads > 0 &&
         "CPUCostModel: Must set queryNThreads before calling computeGiBTime");
  assert(queryPrecision != Precision::Unknown &&
         "CPUCostModel: Must set queryPrecision before calling computeGiBTime");

  auto gateNQubits = gate->nQubits();
  auto gateOpCount = gate->opCount(zTol);
  // Try to find an exact match
  for (const auto& item : buckets) {
    if (item.nQubits == gateNQubits && item.precision == queryPrecision &&
        item.nThreads == queryNThreads) {
      auto t = item.getAvgGibTimePerOpCount() * gateOpCount;
      return std::max(t, this->minGibTimeCap);
    }
  }

  // No exact match. Estimate it
  auto bestMatchIt = buckets.begin();

  auto it = buckets.cbegin();
  const auto end = buckets.cend();
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
  const int nLinesToDisplay = std::min<int>(nLines, buckets.size());

  os << "Memory Bandwidth: " << utils::fmt_1_to_1e3(1.0 / this->minGibTimeCap)
     << " GiBps\n";
  os << "  nQubits | Precision | nThreads | Dense MemSpd \n";
  for (int i = 0; i < nLinesToDisplay; ++i) {
    double opCount = static_cast<double>(1ULL << (buckets[i].nQubits + 2));
    double timePerGiB = buckets[i].getAvgGibTimePerOpCount() * opCount;
    os << "    " << std::fixed << std::setw(2) << buckets[i].nQubits
       << "    |    f" << static_cast<int>(buckets[i].precision) << "    |    "
       << buckets[i].nThreads << "    |    "
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

  double t;
  t = timeit::once([&]() {
    for (int k = 1; k <= 5; ++k) {
      utils::sampleNoReplacement(nQubits, k, qubits);
      auto gate = StandardQuantumGate::RandomUnitary(qubits);
      llvm::cantFail(km.genGate(cpuConfig, gate, "gate_k" + std::to_string(k)));
    }
    llvm::cantFail(km.compileAllPools(llvm::OptimizationLevel::O1, false));
  });

  if (verbose > 0) {
    std::cerr << "Preliminary code generation time: " << utils::fmt_time(t)
              << "\n";
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
  for (unsigned k = 6; k < weights.size() + 1; ++k) {
    weights[k - 1] =
        static_cast<int>(static_cast<double>(weights[k - 2]) * decay);
  }

  unsigned maxIdx = 0;
  double maxWeight = 0.0;
  for (unsigned k = 0; k < weights.size(); ++k) {
    if (weights[k] > maxWeight) {
      maxWeight = static_cast<double>(weights[k]);
      maxIdx = k;
    }
  }
  // An extra round of weight decay. More decay for distant weights
  for (unsigned k = 0; k < weights.size(); ++k) {
    if (k == maxIdx)
      continue;
    constexpr double ratio = 1.5;
    double newWeight = static_cast<double>(weights[k]) /
                       std::pow(ratio, std::abs<int>(maxIdx - k));
    weights[k] = static_cast<int>(newWeight);
  }

  if (verbose > 1) {
    double sum = std::reduce(weights.begin(), weights.end(), 0.0);
    std::cerr << "Relative weights:\n";
    for (unsigned k = 1; k <= weights.size(); ++k) {
      std::cerr << "  " << k << "-qubit: " << "weight = " << weights[k - 1]
                << "; percentage = "
                << (100.0 * static_cast<double>(weights[k - 1]) / sum) << "\n";
    }
  }
}

llvm::Error
CPUPerformanceCache::runExperiments(const CPUKernelGenConfig& cpuConfig,
                                    int nQubits,
                                    int nThreads,
                                    int nRuns,
                                    int logLevel) {

  if (logLevel > 0) {
    std::cerr << "Running CPU performance experiments with config:\n"
              << "  nQubits :   " << nQubits << "\n"
              << "  nThreads :  " << nThreads << "\n"
              << "  precision : " << cpuConfig.precision << "\n";
  }

  // timing logs
  double t;

  std::vector<StandardQuantumGatePtr> gates;
  gates.reserve(nRuns);

  // nQubitsWeights[k-1] denotes the weight for k-qubit gates
  WeightType nQubitsWeights;
  runPreliminaryExperiments(
      cpuConfig, nQubits, nThreads, nQubitsWeights, logLevel);

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
    for (unsigned i = 0; i < nQubitsWeights.size(); ++i) {
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
  // code gen
  t = timeit::once([&]() {
    int i = 0;
    for (const auto& gate : gates) {
      if (auto e = km.genGate(cpuConfig, gate, "gate_" + std::to_string(i++))) {
        std::cerr << RED("Error: ") << "Failed to generate kernel for gate "
                  << i - 1 << ": " << llvm::toString(std::move(e)) << "\n";
        std::exit(1);
      }
    }
  });
  if (logLevel > 0) {
    std::cerr << "Code generation time for " << gates.size()
              << " gates: " << utils::fmt_time(t) << "\n";
  }

  // JIT compile
  t = timeit::once([&]() {
    if (auto e = km.compileAllPools(llvm::OptimizationLevel::O1, false)) {
      std::cerr << RED("Error: ") << "Failed to initialize JIT engine: "
                << llvm::toString(std::move(e)) << "\n";
      std::exit(1);
    }
  });
  if (logLevel > 0) {
    std::cerr << "JIT initialization time: " << utils::fmt_time(t) << "\n";
  }

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  CPUStatevectorWrapper sv(cpuConfig.precision, nQubits, cpuConfig.simdWidth);
  t = timeit::once([&]() { sv.randomize(nThreads); });
  if (logLevel > 0) {
    std::cerr << "Randomize statevector: " << utils::fmt_time(t) << "\n";
  }

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
    if (logLevel > 0) {
      utils::printSpan(std::cerr << "Gate @ ",
                       std::span(kernel->gate->qubits()));
      std::cerr << ": " << memSpd << " GiBps\n";
    }
  }

  return llvm::Error::success();
}

void CPUPerformanceCache::writeCache(std::ostream& os) const {
  for (const auto& item : items_) {
    item.write(os);
    os << "\n";
  }
}

llvm::Error CPUPerformanceCache::save(const std::string& filename,
                                      bool overwrite) const {
  namespace fs = std::filesystem;

  bool needTitle = true;
  if (!overwrite && fs::exists(filename) && fs::file_size(filename) > 0) {
    // non-overwrite mode: non-empty file => no title needed
    needTitle = false;
  }

  std::ofstream ofs;
  if (overwrite)
    ofs.open(filename, std::ios::out | std::ios::trunc);
  else
    ofs.open(filename, std::ios::out | std::ios::app);

  if (!ofs.is_open()) {
    return llvm::createStringError("Failed to open file for writing: " +
                                   filename);
  }

  if (needTitle) {
    ofs << cast::CPUPerformanceCache::CSVTitle() << "\n";
  }

  this->writeCache(ofs);
  return llvm::Error::success();
}
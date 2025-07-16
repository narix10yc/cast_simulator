#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "utils/Formats.h"
#include "utils/utils.h"
#include "utils/PrintSpan.h"
#include "timeit/timeit.h"

using namespace cast;

CPUCostModel::CPUCostModel(std::unique_ptr<CPUPerformanceCache> cache,
                           double zeroTol)
  : CostModel(CM_CPU)
  , cache(std::move(cache))
  , zeroTol(zeroTol)
  , items() {
  items.reserve(32);

  // loop through cache and initialize this->items
  for (const auto& cacheItem : cache->items) {
    auto it = std::ranges::find_if(this->items,
      [&cacheItem](const Item& thisItem) {
      return thisItem.nQubits == cacheItem.nQubits &&
             thisItem.precision == cacheItem.precision &&
             thisItem.nThreads == cacheItem.nThreads;
    });
    if (it == this->items.end())
      // a new item
      items.emplace_back(
        cacheItem.nQubits, cacheItem.precision, cacheItem.nThreads,
        1,  // nData
        1.0 / (cacheItem.memUpdateSpeed * cacheItem.opCount) // totalGiBTimePerOpCount
      ); 
    else {
      it->nData++;
      it->totalGibTimePerOpCount += 1.0 / (cacheItem.opCount * cacheItem.memUpdateSpeed);
    }
  }

  // initialize maxMemUpdateSpd
  assert(!items.empty());
  auto it = items.cbegin();
  auto end = items.cend();
  this->minGibTimeCap = 0.0;
}

double CPUCostModel::computeGiBTime(const QuantumGate* gate) const {
  assert(gate != nullptr);
  assert(!items.empty());

  assert(queryNThreads > 0 &&
         "CPUCostModel: Must set queryNThreads before calling computeGiBTime");
  assert(queryPrecision != Precision::Unknown &&
         "CPUCostModel: Must set queryPrecision before calling computeGiBTime");

  const auto gateNQubits = gate->nQubits();
  auto gateOpCount = gate->opCount(zeroTol);

  // Try to find an exact match
  for (const auto& item : items) {
    if (item.nQubits == gateNQubits &&
        item.precision == queryPrecision &&
        item.nThreads == queryNThreads) {
      auto avg = std::max(item.getAvgGibTimePerOpCount(), this->minGibTimeCap);
      return avg * gateOpCount;
    }
  }

  // No exact match. Estimate it
  auto bestMatchIt = items.begin();

  auto it = items.cbegin();
  const auto end = items.cend();
  while (++it != end) {
    // priority: nThreads > nQubits > precision
    const int bestNThreadsDiff = std::abs(queryNThreads - bestMatchIt->nThreads);
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

  // best match avg GiB time per opCount
  auto bestMatchT0 = bestMatchIt->getAvgGibTimePerOpCount();
  // estimated avg Gib time per opCount
  auto estT0 = bestMatchT0 * bestMatchIt->nThreads / queryNThreads;
  auto estimateTime = std::max<double>(estT0, this->minGibTimeCap) * gateOpCount;

  // std::cerr << YELLOW("Warning: ") << "No exact match to "
  //              "[nQubits, precision, nThreads] = ["
  //           << gateNQubits << ", " << gateOpCount << ", "
  //           << precision << ", " << nThreads
  //           << "] found. We estimate it by ["
  //           << bestMatchIt->nQubits << ", " << bestMatchIt->precision
  //           << ", " << bestMatchIt->nThreads
  //           << "] @ " << bestMatchT0 << " s/GiB/op => "
  //              "Est. " << estT0 << " s/GiB/op.\n";

  return estimateTime;
}

std::ostream& CPUCostModel::displayInfo(std::ostream& os, int verbose) const {
  const int nLinesToDisplay = std::min<int>(5 * verbose, items.size());

  os << "Gib Time Cap: " << this->minGibTimeCap << " per op\n";
  os << "  nQubits | Precision | nThreads | Dense MemSpd \n";
  for (int i = 0; i < nLinesToDisplay; ++i) {
    int denseOpCount = 1ULL << (items[i].nQubits + 1);
    double GibTimePerOpCount = items[i].getAvgGibTimePerOpCount();
    double denseMemSpd = 1.0 / (GibTimePerOpCount * denseOpCount);
    os << "    " << std::fixed << std::setw(2) << items[i].nQubits
       << "    |    f" << static_cast<int>(items[i].precision)
       << "    |    " << items[i].nThreads
       << "    |    " << utils::fmt_1_to_1e3(denseMemSpd, 5)
       << "\n";
  }
  return os;
}

namespace {
  /// @return Speed in gigabytes per second (GiBps)
  double calculateMemUpdateSpeed(int nQubits, Precision precision, double t) {
    assert(nQubits >= 0);
    assert(precision != Precision::Unknown);
    assert(t >= 0.0);

    return static_cast<double>(
      (precision == Precision::F32 ? 8ULL : 16ULL) << nQubits) * 1e-9 / t;
  }

  // Take the scalar gate matrix representation of the gate and randomly zero
  // out some of the elements with probability p. This methods is not versatile
  // and should only be used for testing purposes.
  // For \c StandardQuantumGate, it only applies to the gate matrix.
  // For \c SuperopQuantumGate, it applies to the superoperator matrix (not implemented yet).
  // This method does not apply direct removal. It keeps the matrix to be valid,
  // meaning non of the rows or columns will be completely zeroed out.
  void randRemoveQuantumGate(QuantumGatePtr quGate, float p) {
    assert(0.0f <= p && p <= 1.0f);
    if (p == 0.0f)
      return; // nothing to do

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distF(0.0f, 1.0f);

    auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(quGate.get());
    assert(stdQuGate != nullptr);
    auto scalarGM = stdQuGate->getScalarGM();
    assert(scalarGM != nullptr);
    
    auto& mat = scalarGM->matrix();
    auto edgeSize = mat.edgeSize();

    // randomly zero out some elements
    for (unsigned r = 0; r < edgeSize; ++r) {
      for (unsigned c = 0; c < edgeSize; ++c) {
        if (distF(gen) < p) {
          // zero out the element
          mat.reData()[r * edgeSize + c] = 0.0;
          mat.imData()[r * edgeSize + c] = 0.0;
        }
      }
    }

    std::uniform_int_distribution<size_t> distI(0, edgeSize - 1);
    // check if some row is completely zeroed out
    for (unsigned r = 0; r < edgeSize; ++r) {
      bool isRowZeroed = true;
      for (unsigned c = 0; c < edgeSize; ++c) {
        if (mat.reData()[r * edgeSize + c] != 0.0 ||
            mat.imData()[r * edgeSize + c] != 0.0) {
          isRowZeroed = false;
          break;
        }
      }
      if (isRowZeroed) {
        // randomly choose a non-zero element to keep
        auto keepCol = distI(gen);
        mat.reData()[r * edgeSize + keepCol] = 0.5;
        mat.imData()[r * edgeSize + keepCol] = 0.5;
      }
    }
    
    // check if some column is completely zeroed out
    for (unsigned c = 0; c < edgeSize; ++c) {
      bool isColZeroed = true;
      for (unsigned r = 0; r < edgeSize; ++r) {
        if (mat.reData()[r * edgeSize + c] != 0.0 ||
            mat.imData()[r * edgeSize + c] != 0.0) {
          isColZeroed = false;
          break;
        }
      }
      if (isColZeroed) {
        // randomly choose a non-zero element to keep
        auto keepRow = distI(gen);
        mat.reData()[keepRow * edgeSize + c] = 0.5;
        mat.imData()[keepRow * edgeSize + c] = 0.5;
      }
    }
  } // randRemoveQuantumGate

} // anonymous namespace

void CPUPerformanceCache::runPreliminaryExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits, int nThreads,
    WeightType& weights,
    int verbose) {
  CPUKernelManager kernelMgr;
  std::vector<int> qubits;
  utils::timedExecute([&]() {
    for (int k = 1; k <= 5; ++k) {
      utils::sampleNoReplacement(nQubits, k, qubits);
      kernelMgr.genStandaloneGate(
        cpuConfig,
        StandardQuantumGate::RandomUnitary(qubits),
        "gate_k" + std::to_string(k)
      ).consumeError(); // ignore possible errors
    }
    kernelMgr.initJIT(nThreads,
                      llvm::OptimizationLevel::O1,
                      false, // useLazyJIT
                      0 // verbose
    ).consumeError(); // ignore possible errors
  }, "Code Generation and JIT Initialization");
  
  void* sv;
  void* svData;
  utils::timedExecute([&]() {
    if (cpuConfig.precision == Precision::F32) {
      auto* p = new CPUStatevectorF32(nQubits, cpuConfig.simdWidth);
      p->randomize(nThreads);
      sv = p;
      svData = p->data();
    } else {
      assert(cpuConfig.precision == Precision::F64);
      auto* p = new CPUStatevectorF64(nQubits, cpuConfig.simdWidth);
      p->randomize(nThreads);
      sv = p;
      svData = p->data();
    }
  }, "Initialize statevector");


  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;
  std::array<double, 5> memSpds { 120.0, 120.0, 77.6, 32.1, 12.9};
  for (int k = 1; k <= 5; ++k) {
    tr = timer.timeit([&]() {
      kernelMgr.applyCPUKernel(
        svData, nQubits, "gate_k" + std::to_string(k), nThreads
      ).consumeError(); // ignore possible errors
    });
    memSpds[k-1] = 
      calculateMemUpdateSpeed(nQubits, cpuConfig.precision, tr.min);
    if (verbose > 0) {
      std::cerr << "Dense " << k << "-qubit gate @ "
                << memSpds[k-1] << " GiBps\n";
    }
  }

  assert(weights.size() >= 5);
  weights[0] = 100; // 1-qubit gates
  for (int k = 2; k <= 5; ++k) {
    // The ratio decides the scaling of weights when mem speeds halves.
    // Set to a larger value to focus more on the transition region.
    constexpr double ratio = 1.1;
    // weights[k] is the weight for k-qubit gates
    weights[k-1] = static_cast<int>(
      static_cast<double>(weights[k-2]) *
      ((ratio - 1.0) * (memSpds[k-2] / memSpds[k-1]) + 2 - ratio)
    );
  }
  for (int k = 6; k < weights.size() + 1; ++k) {
    weights[k-1] = static_cast<int>(static_cast<double>(weights[k-2]) * 0.45);
  }

  int maxIdx = 0;
  double maxWeight = 0.0;
  for (int k = 0; k < weights.size(); ++k) {
    if (weights[k] > maxWeight) {
      maxWeight = static_cast<double>(weights[k]);
      maxIdx = k;
    }
  }
  for (int k = 0; k < weights.size(); ++k) {
    if (k == maxIdx)
      continue;
    constexpr double ratio = 1.5;
    double newWeight = static_cast<double>(weights[k]) /
                       std::pow(ratio, std::abs(maxIdx - k));
    weights[k] = static_cast<int>(newWeight);
  }

  
  if (verbose > 0) {
    double sum = 0.0;
    for (const auto& w : weights)
      sum += static_cast<double>(w);
    std::cerr << "Relative weights:\n";
    for (int k = 1; k <= CPU_GLOBAL_MAX_SIZE; ++k) {
      std::cerr << "  " << k << "-qubit: "
                << "weight = " << weights[k-1]
                << "; percentage = "
                << (100.0 * static_cast<double>(weights[k-1]) / sum)
                << "\n";
    }
  }

  if (cpuConfig.precision == Precision::F32)
    delete static_cast<cast::CPUStatevectorF32*>(sv);
  else {
    assert(cpuConfig.precision == Precision::F64);
    delete static_cast<cast::CPUStatevectorF64*>(sv);
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
  runPreliminaryExperiments(cpuConfig, nQubits, nThreads,
                            nQubitsWeights, verbose);
  
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
        randRemoveQuantumGate(gates.back(), erasureProb);
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

  CPUKernelManager kernelMgr;
  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates) {
      // ignore possible errors
      kernelMgr.genStandaloneGate(
        cpuConfig, gate, "gate_" + std::to_string(i++)
      ).consumeError();
    }
  }, "Code Generation");

  utils::timedExecute([&]() {
    kernelMgr.initJIT(nThreads, llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false, /* verbose */ 1
    ).consumeError(); // ignore possible errors
  }, "Initialize JIT Engine");

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  cast::CPUStatevector<double> sv(nQubits, cpuConfig.simdWidth);
  utils::timedExecute([&]() {
    sv.randomize(nThreads);
  }, "Initialize statevector");

  for (const auto& kernel : kernelMgr.getAllStandaloneKernels()) {
    tr = timer.timeit([&]() {
      // ignore possible errors
      kernelMgr.applyCPUKernel(
        sv.data(), sv.nQubits(), *kernel, nThreads).consumeError();
    });
    auto memSpd = calculateMemUpdateSpeed(nQubits, kernel->precision, tr.min);
    items.emplace_back(kernel->gate->nQubits(),
                       kernel->opCount,
                       Precision::F64,
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
  for (const auto& item : items) {
    item.write(os);
    os << "\n";
  }
}
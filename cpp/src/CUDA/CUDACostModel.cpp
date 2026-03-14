#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/Detail/PerfCacheHelper.h"

#include "utils/Formats.h"
#include "utils/utils.h"

#include <numeric> // for std::reduce
#include <random>

using namespace cast;

static llvm::Error
runPreliminaryExperiments(const CUDAKernelGenConfig& kernelConfig,
                          int nQubits,
                          int nWorkerThreads,
                          int verbose,
                          impl::CostModelWeightType& weights) {
  CUDAKernelManager km(nWorkerThreads);
  auto sv = std::make_unique<CUDAStatevectorFP64>(nQubits);
  sv->initialize();
  km.attachStatevector(std::move(sv));
  km.enableTiming();

  // generate {1,2,3,4,5}-qubit gates acting on MSB qubits
  for (int k = 1; k <= 5; ++k) {
    QuantumGate::TargetQubitsType qubits;
    qubits.reserve(k);
    for (int q = nQubits - k; q < nQubits; ++q)
      qubits.push_back(q);
    auto eKernel = km.genGate(kernelConfig,
                              StandardQuantumGate::RandomUnitary(qubits),
                              "gate_k" + std::to_string(k));
    if (!eKernel) {
      return llvm::joinErrors(
          llvm::createStringError("Preliminary gate gen failed"),
          eKernel.takeError());
    }
  }

  if (auto e = km.syncCompilation()) {
    return llvm::joinErrors(
        llvm::createStringError("Preliminary compilation failed"),
        std::move(e));
  }

  // Launch every kernel 5 times (warm-up + timed)
  std::array<float, 5> tarr;
  auto& pool = km.getDefaultPool();
  int i = 0;
  for (auto& kernelPtr : pool) {
    auto* kernel = kernelPtr.get();

    if (auto eTask = km.enqueueKernelExecution(kernel); !eTask) {
      return llvm::joinErrors(
          llvm::createStringError("Warm-up kernel launch failed"),
          eTask.takeError());
    }
    if (auto e = km.syncKernelExecution()) {
      return llvm::joinErrors(
          llvm::createStringError("Warm-up kernel execution failed"),
          std::move(e));
    }

    auto eTask = km.enqueueKernelExecution(kernel);
    if (!eTask) {
      return llvm::joinErrors(llvm::createStringError("Kernel launch failed"),
                              eTask.takeError());
    }
    if (auto e = km.syncKernelExecution()) {
      return llvm::joinErrors(
          llvm::createStringError("Kernel execution failed"), std::move(e));
    }
    tarr[i++] = eTask->getKernelTimeMs() * 1e-3f;
  }

  for (int k = 1; k <= 5; ++k) {
    std::cerr << k << "-qubit dense gate takes " << utils::fmt_time(tarr[k - 1])
              << "\n";
  }

  impl::computeGateWeights(tarr, weights);
  if (verbose >= 2) {
    auto sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    std::cerr << "Frequencies of gates in tests:\n"
              << std::fixed << std::setprecision(2);
    for (unsigned k = 1; k <= weights.size(); ++k) {
      std::cerr << k << "-qubit gate: " << 100.f * weights[k - 1] / sum
                << "%\n";
    }
  }

  return llvm::Error::success();
}

static QuantumGatePtr
createRandomSizedGate(int nQubitsSV,
                      const impl::CostModelWeightType& weights,
                      float erasureProb) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng) * std::accumulate(weights.begin(), weights.end(), 0.0f);
  float acc = 0;
  for (unsigned i = 0; i < weights.size(); ++i) {
    acc += weights[i];
    if (r <= acc) {
      QuantumGate::TargetQubitsType qubits;
      utils::sampleNoReplacement(nQubitsSV, i + 1, qubits);
      auto gate = StandardQuantumGate::RandomUnitary(qubits);
      internal::randRemoveQuantumGate(gate.get(), erasureProb);
      return gate;
    }
  }
  return nullptr;
}

llvm::Error
CUDAPerformanceCache::runExperiments(const CUDAKernelGenConfig& kernelConfig,
                                     int nQubitsSV,
                                     int nWorkerThreads,
                                     int nRuns,
                                     int verbose) {
  impl::CostModelWeightType weights;
  if (auto e = runPreliminaryExperiments(
          kernelConfig, nQubitsSV, nWorkerThreads, verbose, weights)) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to run preliminary experiments"),
        std::move(e));
  }

  CUDAKernelManager km(nWorkerThreads);
  auto sv = std::make_unique<CUDAStatevectorFP64>(nQubitsSV);
  sv->initialize();
  km.attachStatevector(std::move(sv));
  km.enableTiming();

  int count = 0;
  for (int k = 1; k < 5; ++k) {
    QuantumGate::TargetQubitsType qubits;
    utils::sampleNoReplacement(nQubitsSV, k, qubits);
    auto gate1 = StandardQuantumGate::RandomUnitary(qubits);
    utils::sampleNoReplacement(nQubitsSV, k, qubits);
    auto gate2 = StandardQuantumGate::RandomUnitary(qubits);

    auto eKernel1 =
        km.genGate(kernelConfig, gate1, "gate_" + std::to_string(count++));
    if (!eKernel1)
      return eKernel1.takeError();

    auto eKernel2 =
        km.genGate(kernelConfig, gate2, "gate_" + std::to_string(count++));
    if (!eKernel2)
      return eKernel2.takeError();
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 0.8f);
  for (; count < nRuns; ++count) {
    auto erasureProb = dist(rng);
    auto gate = createRandomSizedGate(nQubitsSV, weights, erasureProb);
    assert(gate);
    auto eKernel =
        km.genGate(kernelConfig, gate, "gate_" + std::to_string(count));
    if (!eKernel)
      return eKernel.takeError();
  }

  if (auto e = km.syncCompilation()) {
    return llvm::joinErrors(
        llvm::createStringError("Compilation failed for cost model runs"),
        std::move(e));
  }

  constexpr int nReplications = 3;
  for (auto& kernelPtr : km.getDefaultPool()) {
    auto* kernel = kernelPtr.get();
    float t = 1e6f;
    for (int rep = 0; rep < nReplications; ++rep) {
      auto eTask = km.enqueueKernelExecution(kernel);
      if (!eTask)
        return eTask.takeError();
      if (auto e = km.syncKernelExecution())
        return e;
      t = std::min(t, eTask->getKernelTimeMs() * 1e-3f);
      if (t > 0.5f)
        break; // no replicate if the time of each run > 0.5 second
    }

    items_.emplace_back(
        kernel->gate->nQubits(),
        kernel->gate->opCount(kernelConfig.zeroTol),
        kernel->precision,
        internal::calculateMemUpdateSpeed(nQubitsSV, kernel->precision, t));
    if (verbose >= 1) {
      std::cerr << std::fixed << std::setprecision(2);
      std::cerr << "Gate on " << kernel->gate->nQubits()
                << " qubits with opCount=" << items_.back().opCount << ": "
                << items_.back().memUpdateSpeed << " GiB/s\n";
    }
  }

  return llvm::Error::success();
}

void CUDAPerformanceCache::writeResults(std::ostream& os) const {
  for (const auto& item : items_) {
    item.write(os);
    os << "\n";
  }
}

/* CUDA Cost Model */

CUDACostModel::CUDACostModel(const CUDAPerformanceCache& cache)
    : CostModel(CM_CUDA) {
  struct TmpBucketValue {
    float totalGiBTimePerOpCount;
    int count;
  };
  SortedVectorMap<BucketKey, TmpBucketValue> tmpBucket;

  for (const auto& item : cache.items()) {
    BucketKey key{item.k, item.precision};
    if (bucket_.find(key) == bucket_.end()) {
      tmpBucket[key] = TmpBucketValue{
          1.0f / static_cast<float>(item.memUpdateSpeed * item.opCount), 1};
    } else {
      tmpBucket[key].totalGiBTimePerOpCount +=
          1.0f / static_cast<float>(item.memUpdateSpeed * item.opCount);
      tmpBucket[key].count += 1;
    }
  }

  for (const auto& [key, value] : tmpBucket)
    bucket_[key] = value.totalGiBTimePerOpCount / value.count;

  // setup minGiBTimeCap
  if (bucket_.empty())
    minGiBTimeCap = 0.0f;
  else {
    minGiBTimeCap = std::numeric_limits<float>::max();
    for (const auto& [key, value] : bucket_) {
      float gbt = value * (1U << (key.k + 2));
      minGiBTimeCap = std::min(minGiBTimeCap, gbt);
    }
  }
}

void CUDACostModel::showEntries(std::ostream& os, int nLines) const {
  int nToDisplay = std::min<int>(bucket_.size(), nLines);
  os << "Two-way Bandwidth Cap: " << utils::fmt_mem(1.0f / minGiBTimeCap * 1e9)
     << "\n";
  os << "k | Precision | Dense (GiB/s) | Per Op Per Sec\n";
  for (const auto& [key, value] : bucket_) {
    auto denseGateGiBTime = 1.0f / (value * (1U << (key.k + 2)));
    os << key.k << " | " << (key.precision == Precision::FP32 ? "FP32" : "FP64")
       << " | " << utils::fmt_1_to_1e3(denseGateGiBTime) << " | "
       << utils::fmt_mem(value * 1e6) << "\n";
    if (--nToDisplay <= 0)
      break;
  }
}

double CUDACostModel::computeGiBTime(const QuantumGate* gate) const {
  assert(queryPrecision_ != Precision::Unknown);
  if (bucket_.empty())
    return minGiBTimeCap;

  // exact match
  if (auto it = bucket_.find(BucketKey{gate->nQubits(), queryPrecision_});
      it != bucket_.end()) {
    return std::max(static_cast<double>(it->second) * gate->opCount(1e-8),
                    static_cast<double>(this->minGiBTimeCap));
  }

  // no exact match, make an estimation
  // priority: precision > k
  auto gateNQubits = gate->nQubits();
  auto gateOpCount = gate->opCount(1e-8);
  auto bestMatchIt = bucket_.begin();

  auto it = bucket_.begin();
  const auto end = bucket_.end();
  while (++it != end) {
    // priority: precision > k
    if (queryPrecision_ == bestMatchIt->first.precision)
      continue;
    if (queryPrecision_ == it->first.precision) {
      bestMatchIt = it;
      continue;
    }

    const int bestKDiff = std::abs(gateNQubits - bestMatchIt->first.k);
    const int thisKDiff = std::abs(gateNQubits - it->first.k);
    if (thisKDiff > bestKDiff)
      continue;
    if (thisKDiff < bestKDiff) {
      bestMatchIt = it;
      continue;
    }
  }

  return std::max(static_cast<double>(bestMatchIt->second) * gateOpCount,
                  static_cast<double>(this->minGiBTimeCap));
}

#include "cast/CostModel.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/CPU/CPUKernelManager.h"
#include "utils/Formats.h"
#include "utils/PrintSpan.h"
#include "timeit/timeit.h"

#include "llvm/Support/Casting.h"

#include <fstream>
#include <iomanip>
#include <unordered_set>

using namespace cast;
using namespace llvm;

double NaiveCostModel::computeGiBTime(
    QuantumGatePtr gate, int precision, int nThreads) const {
  assert(gate != nullptr);
  if (gate->nQubits() > maxNQubits)
    return 1.0;
  if (maxOp > 0 && gate->opCount(zeroTol) > maxOp)
    return 1.0;
  return 0.0;
}

StandardCostModel::StandardCostModel(PerformanceCache* cache, double zeroTol)
  : cache(cache), zeroTol(zeroTol), items() {
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

  std::cerr << "StandardCostModel: "
               "A total of " << items.size() << " items found!\n";
}

double StandardCostModel::computeGiBTime(
    QuantumGatePtr gate, int precision, int nThreads) const {
  assert(gate != nullptr);
  assert(!items.empty());
  const auto gateNQubits = gate->nQubits();
  auto gateOpCount = gate->opCount(zeroTol);

  // Try to find an exact match
  for (const auto& item : items) {
    if (item.nQubits == gateNQubits &&
        item.precision == precision &&
        item.nThreads == nThreads) {
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
    const int bestNThreadsDiff = std::abs(nThreads - bestMatchIt->nThreads);
    const int thisNThreadsDiff = std::abs(nThreads - it->nThreads);
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

    if (precision == bestMatchIt->precision)
      continue;
    if (precision == it->precision) {
      bestMatchIt = it;
      continue;
    }
  }

  // best match avg GiB time per opCount
  auto bestMatchT0 = bestMatchIt->getAvgGibTimePerOpCount();
  // estimated avg Gib time per opCount
  auto estT0 = bestMatchT0 * bestMatchIt->nThreads / nThreads;
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

std::ostream& StandardCostModel::display(std::ostream& os, int nLines) const {
  const int nLinesToDisplay = nLines > 0 ?
    std::min<int>(nLines, items.size()) :
    static_cast<int>(items.size());

  os << "Gib Time Cap: " << this->minGibTimeCap << " per op\n";
  os << "  nQubits | Precision | nThreads | Dense MemSpd \n";
  for (int i = 0; i < nLinesToDisplay; ++i) {
    int denseOpCount = 1ULL << (items[i].nQubits + 1);
    double GibTimePerOpCount = items[i].getAvgGibTimePerOpCount();
    double denseMemSpd = 1.0 / (GibTimePerOpCount * denseOpCount);
    os << "    " << std::fixed << std::setw(2) << items[i].nQubits
       << "    |    f" << items[i].precision
       << "    |    " << items[i].nThreads
       << "    |    " << utils::fmt_1_to_1e3(denseMemSpd, 5)
       << "\n";
  }

  return os;
}

void PerformanceCache::writeResults(std::ostream& os) const {
  for (const auto&
      [nQubits, opCount, precision, nThreads, memUpdateSpeed] : items) {
    os << nQubits << "," << opCount << ","
       << precision << "," << nThreads << ","
       << std::scientific << std::setw(6) << memUpdateSpeed << "\n";
  }
}

namespace {
  /// @return Speed in gigabytes per second (GiBps)
  double calculateMemUpdateSpeed(int nQubits, int precision, double t) {
    assert(nQubits >= 0);
    assert(precision == 32 || precision == 64);
    assert(t >= 0.0);

    return static_cast<double>(
      (precision == 32 ? 8ULL : 16ULL) << nQubits) * 1e-9 / t;
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
    if (stdQuGate != nullptr) {
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
    }
  }

} // anonymous namespace


// PerformanceCache::LoadFromCSV helper functions
namespace {
int parseInt(const char*& curPtr, const char* bufferEnd) {
  const auto* beginPtr = curPtr;
  while (curPtr < bufferEnd && *curPtr >= '0' && *curPtr <= '9')
    ++curPtr;
  assert(curPtr == bufferEnd || *curPtr == ',' || *curPtr == '\n');
  return std::stoi(std::string(beginPtr, curPtr));
}

double parseDouble(const char*& curPtr, const char* bufferEnd) {
  const auto* beginPtr = curPtr;
  while (curPtr < bufferEnd &&
         ((*curPtr >= '0' && *curPtr <= '9') ||
           *curPtr == 'e' || *curPtr == 'E' ||
           *curPtr == '.' || *curPtr == '-' || *curPtr == '+'))
    ++curPtr;
  assert(curPtr == bufferEnd || *curPtr == ',' || *curPtr == '\n');
  return std::stod(std::string(beginPtr, curPtr));
}

PerformanceCache::Item parseLine(const char*& curPtr, const char* bufferEnd) {
  PerformanceCache::Item item;

  item.nQubits = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.opCount = parseDouble(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.precision = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.nThreads = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.memUpdateSpeed = parseDouble(curPtr, bufferEnd);
  assert(*curPtr == '\n' || curPtr == bufferEnd);
  return item;
}

} // anonymous namespace

PerformanceCache PerformanceCache::LoadFromCSV(const std::string& fileName) {
  PerformanceCache cache;

  std::ifstream file(fileName, std::ifstream::binary);
  assert(file.is_open());
  file.seekg(0, file.end);
  const auto bufferLength = file.tellg();
  file.seekg(0, file.beg);
  auto* bufferBegin = new char[bufferLength];
  const auto* bufferEnd = bufferBegin + bufferLength;
  file.read(bufferBegin, bufferLength);
  file.close();
  const auto* curPtr = bufferBegin;

  // parse the header
  while (*curPtr != '\n')
    ++curPtr;
  assert(std::string(bufferBegin, curPtr - bufferBegin) ==
         PerformanceCache::CSV_Title && "Tile does not match");
  ++curPtr;

  while (curPtr < bufferEnd) {
    cache.items.push_back(parseLine(curPtr, bufferEnd));
    assert(*curPtr == '\n' || curPtr == bufferEnd);
    if (*curPtr == '\n')
      ++curPtr;
  }

  delete[] bufferBegin;
  return cache;
}

template<std::size_t K>
static inline void sampleNoReplacement(int n, std::vector<int>& holder) {
  assert(n > 0);
  assert(K <= n);
  std::vector<int> indices(n);
  for (int i = 0; i < n; ++i)
    indices[i] = i;

  std::random_device rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < K; ++i) {
    std::uniform_int_distribution<int> dist(i, n - 1);
    int j = dist(gen);
    std::swap(indices[i], indices[j]);
  }
  for (int i = 0; i < K; ++i)
    holder[i] = indices[i];
}

void PerformanceCache::runExperiments(
    const CPUKernelGenConfig& cpuConfig,
    int nQubits, int nThreads, int nRuns) {
  std::vector<StandardQuantumGatePtr> gates;
  gates.reserve(nRuns);

  // nQubitsWeights[k-1] denotes the weight for k-qubit gates
  std::array<int, 7> nQubitsWeights;
  
  // Add a random quantum gate whose size follows distribution of nQubitsWeights
  const auto addRandomQuGate = [&](float erasureProb) {
    int sum = 0;
    for (auto weight : nQubitsWeights)
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
    assert(false && "Unreachable: randAdd failed to add a gate");
  };

  // For the initial run, we add some random 1 to 4-qubit gates
  for (int q = 1; q <= 4; ++q) {
    std::vector<int> targetQubits;
    utils::sampleNoReplacement(nQubits, q, targetQubits);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));
    randRemoveQuantumGate(gates.back(), 0.0f);

    utils::sampleNoReplacement(nQubits, q, targetQubits);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));
    randRemoveQuantumGate(gates.back(), 0.0f);
  }

  nQubitsWeights = { 3, 5, 5, 3, 2, 2, 1 };
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
    kernelMgr.initJIT(nThreads, OptimizationLevel::O1,
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
    items.emplace_back(
      kernel->gate->nQubits(), kernel->opCount, 64, nThreads, memSpd);
    std::cerr << "Gate @ ";
    utils::printSpan(std::cerr, std::span(kernel->gate->qubits()));
    std::cerr << ": " << memSpd << " GiBps\n";
  }
}

#ifdef CAST_USE_CUDA

#include "cast/CUDA/StatevectorCUDA.h"
#include "utils/cuda_api_call.h"

namespace {

inline void driverGetFuncAttributes(CUfunction func, int& numRegs, int& sharedSizeBytes)
{
  // Number of registers used
  CUresult rc = cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
  if (rc != CUDA_SUCCESS) {
    std::cerr << "cuFuncGetAttribute(NUM_REGS) failed\n";
    numRegs = 0;
  }
  // Static shared memory usage
  rc = cuFuncGetAttribute(&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
  if (rc != CUDA_SUCCESS) {
    std::cerr << "cuFuncGetAttribute(SHARED_SIZE_BYTES) failed\n";
    sharedSizeBytes = 0;
  }
}

double calculateMemUpdateSpeedCuda(int nQubits, int precision, double t,
                                double coalescingScore,
                                double occupancy) {
  assert(nQubits >= 0);
  assert(precision == 32 || precision == 64);
  assert(t > 0.0);
  assert(coalescingScore > 0.0 && coalescingScore <= 1.0);
  assert(occupancy > 0.0 && occupancy <= 1.0);

  const double bytesTouched =
       (precision == 32 ? 8.0 : 16.0) // 8 or 16 bytes per amplitude
       * std::pow(2.0, nQubits);

  return bytesTouched /
      (static_cast<double>(1u << 30) // bytes GiB
      * t                            // divide by time
      * coalescingScore              // coalescing penalty
      * std::sqrt(occupancy));       // occupancy penalty
}


double estimateOccupancy(const CUDAKernelInfo& k, int blockSize) {
  // Device properties (unchanged for the whole run, so probably should cache)
  static cudaDeviceProp props;
  static bool propsInit = false;
  if (!propsInit) {
      CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
      propsInit = true;
  }

  // Fast path: kernel has been JITed -> ask the driver
  if (k.kernelFunction()) {
      int maxBlocksPerSM = 0;
      CUresult res = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                          &maxBlocksPerSM,
                          k.kernelFunction(),
                          blockSize,
                          k.sharedMemUsage());          // dynamic SMEM
      if (res != CUDA_SUCCESS) {
          std::cerr << "cuOccupancyMaxActiveBlocksPerMultiprocessor failed\n";
          return 1.0;
      }

      double occ = double(maxBlocksPerSM * blockSize) /
                    double(props.maxThreadsPerMultiProcessor);
      return std::clamp(occ, 0.05, 1.0);
  }

  // Metadata‑only path : have an estimate of regs & SMEM
  unsigned regsPerThr = k.registerUsage();
  size_t dynSmem = k.sharedMemUsage();

  unsigned maxBlocksThreads =
      props.maxThreadsPerMultiProcessor / blockSize;

  unsigned maxBlocksRegs =
      (regsPerThr == 0)
          ? maxBlocksThreads
          : props.regsPerBlock / (regsPerThr * blockSize);

  unsigned maxBlocksSmem =
      (dynSmem == 0)
          ? maxBlocksThreads
          : props.sharedMemPerBlock / dynSmem;

  unsigned activeBlocks = std::max(1u,
      std::min({maxBlocksThreads, maxBlocksRegs, maxBlocksSmem}));

  double occ = double(activeBlocks * blockSize) /
                double(props.maxThreadsPerMultiProcessor);

  return std::clamp(occ, 0.05, 1.0);
}

double estimateCoalescingScore(const LegacyQuantumGate& gate, int nQubits) {
  const auto& targets = gate.qubits;
  const std::vector<int> controls = {}; // Assuming no controls for now
    
    // Check for perfect coalescing (all strides = 1)
    bool perfect_coalescing = true;
    int max_stride = 1;
    std::unordered_set<int> unique_strides;
    
    for (int t : targets) {
        int stride = 1 << t;
        unique_strides.insert(stride);
        if (stride != 1) perfect_coalescing = false;
        if (stride > max_stride) max_stride = stride;
    }
    
    if (perfect_coalescing) {
        return 1.0;  // Ideal case
    }
    
    double score = 1.0;
    // Stride-based penalties
    if (max_stride >= 32) {
        score *= 0.2;  // Worst case for 32-byte transactions
    } else if (max_stride >= 8) {
        score *= 0.4;
    } else if (max_stride >= 2) {
        score *= 0.7;
    }
    
    // Control qubits cause broadcast patterns
    if (!controls.empty()) {
        score *= 0.6 * pow(0.9, controls.size());  // Base penalty + diminishing returns
    }
    // Multiple strides hurt coalescing
    if (unique_strides.size() > 1) {
        score *= 0.8 / log2(unique_strides.size() + 1);
    }
    // Warp divergence penalty for conditional gates
    // if (gate.hasConditionalOperations()) {
    //     score *= 0.7;
    // }
    
    // Ensure score stays in valid range
    return std::clamp(score, 0.05, 1.0);
}

} // anonymous namespace

// Not included: Warp Efficiency Metrics, Shared Memory Bank Conflict Detection, L1/L2 Cache Modeling
CUDAPerformanceCache CUDAPerformanceCache::LoadFromCSV(const std::string& filename) {
  CUDAPerformanceCache cache;
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);

  std::string line;
  std::getline(file, line); // Header
  if (line != CUDAPerformanceCache::CSV_HEADER) throw std::runtime_error("Invalid CSV header");

  while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::istringstream iss(line);
      CUDAPerformanceCache::Item item;
      char comma;
      iss >> item.nQubits >> comma
          >> item.opCount >> comma
          >> item.precision >> comma
          >> item.blockSize >> comma
          >> item.occupancy >> comma
          >> item.coalescingScore >> comma
          >> item.memUpdateSpeed;
      cache.items.push_back(item);
  }
  return cache;
}

void CUDAPerformanceCache::writeResults(std::ostream& os) const {
    for (const auto& item : items) {
        os << item.nQubits << ","
           << item.opCount << ","
           << item.precision << ","
           << item.blockSize << ","
           << item.occupancy << ","
           << item.coalescingScore << ","
           << std::scientific << std::setw(6) << item.memUpdateSpeed << "\n";
    }
}

void CUDAPerformanceCache::writeResults(const std::string& filename) const {
    std::ofstream out(filename);
    out << "nQubits,opCount,precision,blockSize,occupancy,coalescing,memSpd\n";
    writeResults(out);
}

const CUDAPerformanceCache::Item* CUDAPerformanceCache::findClosestMatch(
    const legacy::QuantumGate& gate, int precision, int blockSize) const {
    
    const Item* bestMatch = nullptr;
    double bestScore = -1.0;
    
    for (const auto& item : items) {
        if (item.precision != precision) continue;
        
        double score = 0.0;
        // Prefer same block size
        score += (item.blockSize == blockSize) ? 1.0 : 0.3;
        // Prefer similar gate size
        score += 1.0 / (1.0 + abs(item.nQubits - gate.nQubits()));
        // Prefer similar op count
        score += 1.0 / (1.0 + abs(item.opCount - gate.opCount(1e-8)));
        
        if (score > bestScore) {
            bestScore = score;
            bestMatch = &item;
        }
    }
    
    return bestMatch;
}

inline double regPenalty(int regsPerThr, int blk, int regsPerSM = 65536)
{
  if (regsPerThr == 0) return 1.0; // unknown -> no penalty
  double maxWarps = double(regsPerSM) /
                    (regsPerThr * (blk / 32.0));
  double occLimit = std::min(1.0, maxWarps / 64.0); // 64 warps = 100 %
  return 1.0 / std::max(0.05, occLimit);
}

double CUDACostModel::computeGiBTime(const legacy::QuantumGate& gate,
                                     int precision, int) const
{
  const double gpuPeakTFLOPs = (precision == 32 ? 35.6 : 0.556); // RTX‑3090
  const double launchOverhead = 3.0e-6;
  const double opCnt = gate.opCount(zeroTol);
  const int nQ = gate.nQubits();

  // pick the best anchor (same precision, closest nQ & blockSize)
  const CUDAPerformanceCache::Item* anchor = nullptr;
  double bestScore = -1.0;
  for (const auto& it : cache->items) {
      if (it.precision != precision) continue;
      double s  = (it.blockSize == currentBlockSize ? 1.0 : 0.25);
      s        += 1.0 / (1.0 + std::abs(it.nQubits - nQ));
      if (s > bestScore) { bestScore = s; anchor = &it; }
  }
  if (!anchor) return 1e8; // no data → discourage fuse

  // live resource estimates for the fused candidate gate
  CUDAKernelInfo tmp;
  tmp.setGate(&gate);
  tmp.setKernelFunction(nullptr);   // metadata only

  auto estSmemBytes = [&](const legacy::QuantumGate& g) {
      int localQ = g.nQubits();
      const int bytesPerAmp = (precision == 32 ? 8 : 16);
      return (localQ <= 5 ? 0
                          : (std::size_t{1} << localQ) * bytesPerAmp);
  };
  auto estRegisters = [&](const legacy::QuantumGate& g) {
      return 32 + 4 * g.nQubits();
  };

  tmp.setSharedMemUsage(estSmemBytes(gate));
  tmp.setRegisterUsage(estRegisters(gate));

  double occLive  = std::clamp(estimateOccupancy(tmp, currentBlockSize), 0.05, 1.0);
  double coalLive = std::clamp(estimateCoalescingScore(gate, nQ), 0.05, 1.0);

  // scale the anchor bandwidth by live penalties
  double blkScale = std::pow(double(anchor->blockSize) / currentBlockSize, 0.5);
  double occScale = occLive/std::max(0.05, anchor->occupancy);
  double coalScale = std::pow(coalLive/std::max(0.05, anchor->coalescingScore), 1.5);

  double effGiBps = anchor->memUpdateSpeed * occScale * blkScale * coalScale;
  effGiBps = std::max(effGiBps, 1.0 / minGibTimeCap); // clamp

  // turn bandwidth into time (memory + FLOP + launch const)
  const double bytesPerAmp = (precision == 32 ? 8.0 : 16.0);
  double memGiB = bytesPerAmp * opCnt / (1ULL << 30);
  double memTime = memGiB / effGiBps;
  double flopTime = opCnt * 2.0 / (gpuPeakTFLOPs * 1e12);

  constexpr double gpuBW_Bytes = 840.0 * (1ULL << 30);
  double ridgeAI = (gpuPeakTFLOPs * 1e12) / gpuBW_Bytes;
  double ai = 2.0 / bytesPerAmp;
  double execCore = (ai < ridgeAI) ? memTime : flopTime;

  return execCore + launchOverhead;
}


double CUDACostModel::calculateOccupancyPenalty(const CUDAPerformanceCache::Item& item) const {
    // Quadratic penalty
    return 1.0 / (item.occupancy * item.occupancy); 
}

double CUDACostModel::calculateCoalescingPenalty(const CUDAPerformanceCache::Item& item) const {
    // Exponential penalty for memory access patterns
    const double exponent = 1.5;
    return pow(1.0 / item.coalescingScore, exponent);
}


void CUDAPerformanceCache::runExperiments(
    const CUDAKernelGenConfig& gpuConfig,
    int nQubits, int blockSize, int nRuns, int nWorkerThreads) {
  std::vector<std::shared_ptr<legacy::QuantumGate>> gates;
  // Rethink this
  // gates.reserve(nRuns);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disFloat(0.0, 1.0);
  std::uniform_int_distribution<int> disInt(0, nQubits - 1);
  float prob = 1.0f;

  const auto randFloat = [&]() { return disFloat(gen); };
  const auto randRemove = [&](legacy::QuantumGate& gate) {
    if (prob >= 1.0)
      return;
    auto* cMat = gate.gateMatrix.getConstantMatrix();
    assert(cMat != nullptr);
    for (size_t i = 0; i < cMat->size(); ++i) {
      if (randFloat() > prob)
        cMat->data()[i].real(0.0);
      if (randFloat() > prob)
        cMat->data()[i].imag(0.0);
    }
  };

  // nQubitsWeights[q] denotes the weight for n-qubit gates
  std::array<int, 8> nQubitsWeights;

  const auto addRandU1q = [&]() {
    auto a = disInt(gen);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a)));
    randRemove(*gates.back());
  };

  const auto addRandU2q = [&]() {
    int a,b;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b)));
    randRemove(*gates.back());
  };

  const auto addRandU3q = [&]() {
    int a,b,c;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b, c)));
    randRemove(*gates.back());
  };

  const auto addRandU4q = [&]() {
    int a,b,c,d;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b, c, d)));
    randRemove(*gates.back());
  };

  const auto addRandU5q = [&]() {
    int a,b,c,d,e;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b, c, d, e)));
    randRemove(*gates.back());
  };

  const auto addRandU6q = [&]() {
    int a,b,c,d,e,f;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    do { f = disInt(gen); }
    while (f == a || f == b || f == c || f == d || f == e);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b, c, d, e, f)));
    randRemove(*gates.back());
  };

  const auto addRandU7q = [&]() {
    int a,b,c,d,e,f,g;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    do { f = disInt(gen); }
    while (f == a || f == b || f == c || f == d || f == e);
    do { g = disInt(gen); }
    while (g == a || g == b || g == c || g == d || g == e || g == f);
    gates.emplace_back(std::make_shared<legacy::QuantumGate>(
      legacy::QuantumGate::RandomUnitary(a, b, c, d, e, f, g)));
    randRemove(*gates.back());
  };

  const auto addRandU = [&](int nQubits) {
    assert(nQubits != 0);
    switch (nQubits) {
      case 1: addRandU1q(); break;
      case 2: addRandU2q(); break;
      case 3: addRandU3q(); break;
      case 4: addRandU4q(); break;
      case 5: addRandU5q(); break;
      case 6: addRandU6q(); break;
      case 7: addRandU7q(); break;
      default: assert(false && "Unknown nQubits");
    }
  };

  const auto randAdd = [&]() {
    int sum = 0;
    for (auto weight : nQubitsWeights)
      sum += weight;
    assert(sum > 0 && "nQubitsWeight is empty");
    std::uniform_int_distribution<int> dist(0, sum - 1);
    int r = dist(gen);
    int acc = 0;
    for (int i = 1; i < nQubitsWeights.size(); ++i) {
      acc += nQubitsWeights[i];
      if (r <= acc)
        return addRandU(i);
    }
  };

  // Initial deterministic gates (1-5 qubits)
  prob = 1.0f;
  for (int n = 1; n <= 5; ++n) {
    addRandU(n);
    addRandU(n);
  }

  // Add randomized gates with varying sparsity
  nQubitsWeights = {0, 1, 2, 3, 5, 5, 3, 2};
  int initialRuns = gates.size();
  for (int run = 0; run < nRuns - initialRuns; ++run) {
    float ratio = static_cast<float>(run) / (nRuns - initialRuns);
    prob = ratio * 1.0f + (1.0f - ratio) * 0.25f;
    randAdd();
  }

  CUDAKernelManager kernelMgr;
  
  // Generate kernels
  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelMgr.genCUDAGate(gpuConfig, gate, "gate_" + std::to_string(i++));
  }, "CUDA Kernel Generation");

  // Initialize JIT with GPU-specific parameters
  utils::timedExecute([&]() {
    kernelMgr.emitPTX(nWorkerThreads, llvm::OptimizationLevel::O3);
    kernelMgr.initCUJIT(nWorkerThreads, 1); // 1 stream
  }, "JIT Compilation");

  // Prepare statevector
  utils::StatevectorCUDA<double> sv(nQubits);
  utils::timedExecute([&]() {
    sv.randomize();
  }, "Initialize statevector");

  for (auto& kernel : kernelMgr.kernels()) {
    // Warmup
    for (int i = 0; i < 3; ++i) {
      kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), kernel, blockSize);
    }
    cudaDeviceSynchronize();

    // Timed measurement using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Time multiple runs for better accuracy
    const int measurementRuns = 15;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < measurementRuns; ++i) {
      kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), kernel, blockSize);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double avgTimeSec = milliseconds * 1e-3 / measurementRuns;

    // Calculate GPU-specific metrics
    double occupancy = estimateOccupancy(kernel, blockSize);
    double coalescing = estimateCoalescingScore(*kernel.gate, nQubits);
    double memSpeed = calculateMemUpdateSpeedCuda(
      nQubits, 
      kernel.precision, 
      avgTimeSec,
      coalescing,
      occupancy
    );

    items.emplace_back(Item{
        kernel.gate->nQubits(),
        kernel.opCount,
        kernel.precision,
        blockSize,
        occupancy,
        coalescing,
        memSpeed
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cerr << "Benchmarked gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel.gate->qubits.begin(), kernel.gate->qubits.size()));
    std::cerr << ": " << memSpeed << " GiB/s, "
              << "occupancy=" << occupancy << ", "
              << "coalescing=" << coalescing << "\n";
  }
}

#endif
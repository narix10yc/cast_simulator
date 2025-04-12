#include "cast/CostModel.h"
#include "cast/QuantumGate.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"
#include "utils/Formats.h"
#include "timeit/timeit.h"
#include "utils/cuda_api_call.h"

#include <fstream>
#include <iomanip>
#include <unordered_set>

using namespace cast;
using namespace llvm;

double NaiveCostModel::computeGiBTime(
    const QuantumGate& gate, int precision, int nThreads) const {
  if (gate.nQubits() > maxNQubits)
    return 1.0;
  if (maxOp > 0 && gate.opCount(zeroTol) > maxOp)
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
    const QuantumGate& gate, int precision, int nThreads) const {
  assert(!items.empty());
  const auto gateNQubits = gate.nQubits();
  auto gateOpCount = gate.opCount(zeroTol);

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

  std::cerr << YELLOW("Warning: ") << "No exact match to "
               "[nQubits, precision, nThreads] = ["
            << gateNQubits << ", " << gateOpCount << ", "
            << precision << ", " << nThreads
            << "] found. We estimate it by ["
            << bestMatchIt->nQubits << ", " << bestMatchIt->precision
            << ", " << bestMatchIt->nThreads
            << "] @ " << bestMatchT0 << " s/GiB/op => "
               "Est. " << estT0 << " s/GiB/op.\n";

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
      [nQubits, opCount, precision,
       irregularity, nThreads, memUpdateSpeed] : items) {
    os << nQubits << "," << opCount << ","
       << precision << "," << irregularity << ","
       << nThreads << ","
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
  item.opCount = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.precision = parseInt(curPtr, bufferEnd);
  assert(*curPtr == ',');
  ++curPtr;
  item.irregularity = parseInt(curPtr, bufferEnd);
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
        "nQubits,opCount,precision,irregularity,nThreads,memSpd");
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

static inline void randomRemove(QuantumGate& gate, float p) {
  auto* cMat = gate.gateMatrix.getConstantMatrix();
  assert(cMat != nullptr);
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
  std::vector<std::shared_ptr<QuantumGate>> gates;
  gates.reserve(nRuns);
//  constexpr int maxAllowedK = 7;
//  std::vector<int> holder(maxAllowedK);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disFloat(0.0, 1.0);
  std::uniform_int_distribution<int> disInt(0, nQubits - 1);
  float prob = 1.0f;

  const auto randFloat = [&]() { return disFloat(gen); };
  const auto randRemove = [&](QuantumGate& gate) {
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
  // so length-8 array means we allow up to 7-qubit gates
  std::array<int, 8> nQubitsWeights;

  const auto addRandU1q = [&]() {
    auto a = disInt(gen);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a)));
    randRemove(*gates.back());
  };

  const auto addRandU2q = [&]() {
    int a,b;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b)));
    randRemove(*gates.back());
  };

  const auto addRandU3q = [&]() {
    int a,b,c;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c)));
    randRemove(*gates.back());
  };


  const auto addRandU4q = [&]() {
    int a,b,c,d;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d)));
    randRemove(*gates.back());
  };

  const auto addRandU5q = [&]() {
    int a,b,c,d,e;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e)));
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

    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e, f)));
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

    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e, f, g)));
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

  prob = 1.0f;
  for (int n = 1; n <= 5; ++n) {
    addRandU(n);
    addRandU(n);
  }

  nQubitsWeights = {0, 1, 2, 3, 5, 5, 3, 2};
  int initialRuns = gates.size();
  for (int run = 0; run < nRuns - initialRuns; ++run) {
    float ratio = static_cast<float>(run) / (nRuns - initialRuns);
    prob = ratio * 1.0f + (1.0f - ratio) * 0.25f;
    randAdd();
  }

  CPUKernelManager kernelMgr;
  utils::timedExecute([&]() {
    int i = 0;
    for (const auto& gate : gates)
      kernelMgr.genCPUGate(cpuConfig, gate, "gate_" + std::to_string(i++));
  }, "Code Generation");

  utils::timedExecute([&]() {
    kernelMgr.initJIT(nThreads, OptimizationLevel::O1,
      /* useLazyJIT */ false, /* verbose */ 1);
  }, "Initialize JIT Engine");

  timeit::Timer timer(3, /* verbose */ 0);
  timeit::TimingResult tr;

  utils::StatevectorCPU<double> sv(nQubits, cpuConfig.simd_s);
  utils::timedExecute([&]() {
    sv.randomize(nThreads);
  }, "Initialize statevector");

  for (auto& kernel : kernelMgr.kernels()) {
    if (nThreads == 1)
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), kernel);
      });
    else
      tr = timer.timeit([&]() {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), kernel, nThreads);
      });
    auto memSpd = calculateMemUpdateSpeed(nQubits, kernel.precision, tr.min);
    items.emplace_back(
      kernel.gate->nQubits(), kernel.opCount, 64,
      kernel.nLoBits, nThreads, memSpd);
    std::cerr << "Gate @ ";
    utils::printArray(
      std::cerr, ArrayRef(kernel.gate->qubits.begin(), kernel.gate->qubits.size()));
    std::cerr << ": " << memSpd << " GiBps\n";
  }
}

#ifdef CAST_USE_CUDA

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

  if (nQubits > 6) return 1e9;

  // Theoretical memory accessed (bytes)
  const size_t theoreticalBytes = (precision == 32 ? 8ULL : 16ULL) << nQubits;
  
  // Apply penalties for imperfect access patterns
  const double effectiveBytes = theoreticalBytes / 
                              (coalescingScore * sqrt(occupancy));
  return effectiveBytes * 1e-9 / t;
}

double estimateOccupancy(const CUDAKernelInfo& kernel, int blockSize) {
  CUcontext ctx;
  CUresult res = cuCtxGetCurrent(&ctx);
  if (res != CUDA_SUCCESS || !ctx) {
    std::cerr << "ERROR: No active CUDA context!\n";
    return 1.0;
  }
  if (!kernel.kernelFunction()) {
    std::cerr << "ERROR: Kernel function is NULL!\n";
    return 1.0;
  }

  int numRegs = 0, staticShmem = 0;
  driverGetFuncAttributes(kernel.kernelFunction(), numRegs, staticShmem);

  // query the maximum blocks-per-SM from the driver API
  int maxBlocksPerSM = 0;
  res = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel.kernelFunction(),
        blockSize,
        kernel.sharedMemUsage() // dynamic shared mem usage
      );

  if (res != CUDA_SUCCESS) {
    std::cerr << "cuOccupancyMaxActiveBlocksPerMultiprocessor failed\n";
    return 1.0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  double activeThreads = double(maxBlocksPerSM * blockSize);
  double theoreticalOcc = activeThreads / double(props.maxThreadsPerMultiProcessor);
  if (theoreticalOcc > 1.0) theoreticalOcc = 1.0;

  double registerLimit = 1.0;
  {
    if (numRegs > 0) {
      registerLimit = double(props.regsPerBlock) / double(blockSize * numRegs);
      if (registerLimit > 1.0) registerLimit = 1.0;
    }
  }
  double smemLimit = 1.0;
  {
    double needed = double(staticShmem + kernel.sharedMemUsage());
    if (needed > 0) {
      smemLimit = double(props.sharedMemPerBlock) / needed;
      if (smemLimit > 1.0) smemLimit = 1.0;
    }
  }
  double finalOcc = std::min({theoreticalOcc, registerLimit, smemLimit});
  return (finalOcc < 0.05) ? 0.05 : finalOcc;
}

double estimateCoalescingScore(const QuantumGate& gate, int nQubits) {
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
    const QuantumGate& gate, int precision, int blockSize) const {
    
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

double CUDACostModel::computeGiBTime(const QuantumGate& gate, int precision, int) const {
  const int gateNQubits = gate.nQubits();
  const double gateOpCount = gate.opCount(zeroTol);
  // We'll hold a pointer to an exactMatch if we find one
  const CUDAPerformanceCache::Item* exactItem = nullptr;
  for (auto &item : cache->items) {
      if (item.nQubits   == gateNQubits &&
          item.precision == precision   &&
          item.blockSize == currentBlockSize)
      {
          exactItem = &item;
          break;
      }
  }

  if (exactItem) {
      double speed = exactItem->memUpdateSpeed; // GiB/s
      double baseTimePerGiB = 1.0 / std::max(speed, minGibTimeCap);
      
      // TODO: use calculation functions!
      double occupancyPenalty  = 1.0 / (exactItem->occupancy * exactItem->occupancy);
      double coalescingPenalty = std::pow(1.0 / exactItem->coalescingScore, 1.5);

      constexpr double kernelLaunchOverheadSec = 3.0e-6;

      double totalTime = baseTimePerGiB * gateOpCount * occupancyPenalty * coalescingPenalty
                        + kernelLaunchOverheadSec;
      return totalTime;
  }

  // No exact match: find the "closest" item
  const CUDAPerformanceCache::Item* bestItem = nullptr;
  double bestScore = -1.0;
  for (auto &item : cache->items) {
      if (item.precision != precision) {
          continue;
      }
      double score = (item.blockSize == currentBlockSize) ? 1.0 : 0.25;
      int dq = std::abs(item.nQubits - gateNQubits);
      score += 1.0 / double(1 + dq);

      if (score > bestScore) {
          bestScore = score;
          bestItem  = &item;
      }
  }
  // fallback
  if (!bestItem) {
      return 1e8;
  }

  // sublinear scaling to adapt from bestItem to gateNQubits
  // and from bestItem->blockSize to currentBlockSize, 
  // plus occupancy & coalescing penalty
  double blockSizeRatio = double(bestItem->blockSize) / double(currentBlockSize);
  double blockSizeScale = std::pow(blockSizeRatio, 0.85); // pow(std::min(blockSizeRatio, 4.0), 0.5);

  double occupancyPenalty  = 1.0 / (bestItem->occupancy * bestItem->occupancy);
  double coalescingPenalty = std::pow(1.0 / bestItem->coalescingScore, 1.5);

  // mild exponential penalty if gateNQubits differs 
  // from bestItem->nQubits
  int diffQ = gateNQubits - bestItem->nQubits;
  // if diffQ=2 => sizePenalty = 2^|2|=4 - can improve on this
  double sizePenalty = (diffQ == 0) ? 1.0 : std::pow(2.0, std::abs(diffQ));

  double speed = bestItem->memUpdateSpeed * blockSizeScale;
  speed = std::max(speed, 1e-6);

  double baseTimePerGiB = 1.0 / std::max(speed, minGibTimeCap);
  double totalComputeTime = baseTimePerGiB * gateOpCount
                          * occupancyPenalty * coalescingPenalty
                          * sizePenalty;

  // Add constant kernel overhead
  totalComputeTime += 3.0e-6;

  return totalComputeTime;
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
  std::vector<std::shared_ptr<QuantumGate>> gates;
  // Rethink this
  // gates.reserve(nRuns);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disFloat(0.0, 1.0);
  std::uniform_int_distribution<int> disInt(0, nQubits - 1);
  float prob = 1.0f;

  const auto randFloat = [&]() { return disFloat(gen); };
  const auto randRemove = [&](QuantumGate& gate) {
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
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a)));
    randRemove(*gates.back());
  };

  const auto addRandU2q = [&]() {
    int a,b;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b)));
    randRemove(*gates.back());
  };

  const auto addRandU3q = [&]() {
    int a,b,c;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c)));
    randRemove(*gates.back());
  };

  const auto addRandU4q = [&]() {
    int a,b,c,d;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d)));
    randRemove(*gates.back());
  };

  const auto addRandU5q = [&]() {
    int a,b,c,d,e;
    a = disInt(gen);
    do { b = disInt(gen); } while (b == a);
    do { c = disInt(gen); } while (c == a || c == b);
    do { d = disInt(gen); } while (d == a || d == b || d == c);
    do { e = disInt(gen); } while (e == a || e == b || e == c || e == d);
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e)));
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
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e, f)));
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
    gates.emplace_back(std::make_shared<QuantumGate>(
      QuantumGate::RandomUnitary(a, b, c, d, e, f, g)));
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
      occupancy,
      coalescing
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
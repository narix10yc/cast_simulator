#include "cast/CostModel.h"
#include "cast/QuantumGate.h"
#include "simulation/StatevectorCPU.h"
#include "utils/Formats.h"
#include "timeit/timeit.h"

#include <fstream>
#include <iomanip>

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

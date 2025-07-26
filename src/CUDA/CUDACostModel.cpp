// #include "cast/CUDA/CUDACostModel.h"
// #include "cast/CUDA/CUDAKernelManager.h"
// #include "cast/CUDA/CUDAStatevector.h"

// #include "utils/cuda_api_call.h"

// #include <random>

// using namespace cast;

// namespace {

// inline void
// driverGetFuncAttributes(CUfunction func, int& numRegs, int& sharedSizeBytes) {
//   // Number of registers used
//   CUresult rc = cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
//   if (rc != CUDA_SUCCESS) {
//     std::cerr << "cuFuncGetAttribute(NUM_REGS) failed\n";
//     numRegs = 0;
//   }
//   // Static shared memory usage
//   rc = cuFuncGetAttribute(
//       &sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
//   if (rc != CUDA_SUCCESS) {
//     std::cerr << "cuFuncGetAttribute(SHARED_SIZE_BYTES) failed\n";
//     sharedSizeBytes = 0;
//   }
// }

// double calculateMemUpdateSpeedCuda(int nQubits,
//                                    int precision,
//                                    double t,
//                                    double coalescingScore,
//                                    double occupancy) {
//   assert(nQubits >= 0);
//   assert(precision == 32 || precision == 64);
//   assert(t > 0.0);
//   assert(coalescingScore > 0.0 && coalescingScore <= 1.0);
//   assert(occupancy > 0.0 && occupancy <= 1.0);

//   const double bytesTouched =
//       (precision == 32 ? 8.0 : 16.0) // 8 or 16 bytes per amplitude
//       * std::pow(2.0, nQubits);

//   return bytesTouched / (static_cast<double>(1u << 30) // bytes GiB
//                          * t                           // divide by time
//                          * coalescingScore             // coalescing penalty
//                          * std::sqrt(occupancy));      // occupancy penalty
// }

// double estimateOccupancy(const CUDAKernelInfo& k, int blockSize) {
//   // Device properties (unchanged for the whole run, so probably should cache)
//   static cudaDeviceProp props;
//   static bool propsInit = false;
//   if (!propsInit) {
//     CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
//     propsInit = true;
//   }

//   // Fast path: kernel has been JITed -> ask the driver
//   if (k.kernelFunction()) {
//     int maxBlocksPerSM = 0;
//     CUresult res = cuOccupancyMaxActiveBlocksPerMultiprocessor(
//         &maxBlocksPerSM,
//         k.kernelFunction(),
//         blockSize,
//         k.sharedMemUsage()); // dynamic SMEM
//     if (res != CUDA_SUCCESS) {
//       std::cerr << "cuOccupancyMaxActiveBlocksPerMultiprocessor failed\n";
//       return 1.0;
//     }

//     double occ = double(maxBlocksPerSM * blockSize) /
//                  double(props.maxThreadsPerMultiProcessor);
//     return std::clamp(occ, 0.05, 1.0);
//   }

//   // Metadata‑only path : have an estimate of regs & SMEM
//   unsigned regsPerThr = k.registerUsage();
//   size_t dynSmem = k.sharedMemUsage();

//   unsigned maxBlocksThreads = props.maxThreadsPerMultiProcessor / blockSize;

//   unsigned maxBlocksRegs = (regsPerThr == 0)
//                                ? maxBlocksThreads
//                                : props.regsPerBlock / (regsPerThr * blockSize);

//   unsigned maxBlocksSmem =
//       (dynSmem == 0) ? maxBlocksThreads : props.sharedMemPerBlock / dynSmem;

//   unsigned activeBlocks =
//       std::max(1u, std::min({maxBlocksThreads, maxBlocksRegs, maxBlocksSmem}));

//   double occ = double(activeBlocks * blockSize) /
//                double(props.maxThreadsPerMultiProcessor);

//   return std::clamp(occ, 0.05, 1.0);
// }

// double estimateCoalescingScore(const QuantumGate& gate, int nQubits) {
//   const auto& targets = gate.qubits;
//   const std::vector<int> controls = {}; // Assuming no controls for now

//   // Check for perfect coalescing (all strides = 1)
//   bool perfect_coalescing = true;
//   int max_stride = 1;
//   std::unordered_set<int> unique_strides;

//   for (int t : targets) {
//     int stride = 1 << t;
//     unique_strides.insert(stride);
//     if (stride != 1)
//       perfect_coalescing = false;
//     if (stride > max_stride)
//       max_stride = stride;
//   }

//   if (perfect_coalescing) {
//     return 1.0; // Ideal case
//   }

//   double score = 1.0;
//   // Stride-based penalties
//   if (max_stride >= 32) {
//     score *= 0.2; // Worst case for 32-byte transactions
//   } else if (max_stride >= 8) {
//     score *= 0.4;
//   } else if (max_stride >= 2) {
//     score *= 0.7;
//   }

//   // Control qubits cause broadcast patterns
//   if (!controls.empty()) {
//     score *=
//         0.6 * pow(0.9, controls.size()); // Base penalty + diminishing returns
//   }
//   // Multiple strides hurt coalescing
//   if (unique_strides.size() > 1) {
//     score *= 0.8 / log2(unique_strides.size() + 1);
//   }
//   // Warp divergence penalty for conditional gates
//   // if (gate.hasConditionalOperations()) {
//   //     score *= 0.7;
//   // }

//   // Ensure score stays in valid range
//   return std::clamp(score, 0.05, 1.0);
// }

// } // anonymous namespace

// // Not included: Warp Efficiency Metrics, Shared Memory Bank Conflict Detection,
// // L1/L2 Cache Modeling
// CUDAPerformanceCache
// CUDAPerformanceCache::LoadFromCSV(const std::string& filename) {
//   CUDAPerformanceCache cache;
//   std::ifstream file(filename);
//   if (!file)
//     throw std::runtime_error("Cannot open file: " + filename);

//   std::string line;
//   std::getline(file, line); // Header
//   if (line != CUDAPerformanceCache::CSV_HEADER)
//     throw std::runtime_error("Invalid CSV header");

//   while (std::getline(file, line)) {
//     if (line.empty())
//       continue;
//     std::istringstream iss(line);
//     CUDAPerformanceCache::Item item;
//     char comma;
//     iss >> item.nQubits >> comma >> item.opCount >> comma >> item.precision >>
//         comma >> item.blockSize >> comma >> item.occupancy >> comma >>
//         item.coalescingScore >> comma >> item.memUpdateSpeed;
//     cache.items.push_back(item);
//   }
//   return cache;
// }

// void CUDAPerformanceCache::writeResults(std::ostream& os) const {
//   for (const auto& item : items) {
//     os << item.nQubits << "," << item.opCount << "," << item.precision << ","
//        << item.blockSize << "," << item.occupancy << "," << item.coalescingScore
//        << "," << std::scientific << std::setw(6) << item.memUpdateSpeed << "\n";
//   }
// }

// void CUDAPerformanceCache::writeResults(const std::string& filename) const {
//   std::ofstream out(filename);
//   out << "nQubits,opCount,precision,blockSize,occupancy,coalescing,memSpd\n";
//   writeResults(out);
// }

// const CUDAPerformanceCache::Item* CUDAPerformanceCache::findClosestMatch(
//     const QuantumGate& gate, int precision, int blockSize) const {

//   const Item* bestMatch = nullptr;
//   double bestScore = -1.0;

//   for (const auto& item : items) {
//     if (item.precision != precision)
//       continue;

//     double score = 0.0;
//     // Prefer same block size
//     score += (item.blockSize == blockSize) ? 1.0 : 0.3;
//     // Prefer similar gate size
//     score += 1.0 / (1.0 + abs(item.nQubits - gate.nQubits()));
//     // Prefer similar op count
//     score += 1.0 / (1.0 + abs(item.opCount - gate.opCount(1e-8)));

//     if (score > bestScore) {
//       bestScore = score;
//       bestMatch = &item;
//     }
//   }

//   return bestMatch;
// }

// inline double regPenalty(int regsPerThr, int blk, int regsPerSM = 65536) {
//   if (regsPerThr == 0)
//     return 1.0; // unknown -> no penalty
//   double maxWarps = double(regsPerSM) / (regsPerThr * (blk / 32.0));
//   double occLimit = std::min(1.0, maxWarps / 64.0); // 64 warps = 100 %
//   return 1.0 / std::max(0.05, occLimit);
// }

// double CUDACostModel::computeGiBTime(const QuantumGate& gate,
//                                      int precision,
//                                      int) const {
//   const double gpuPeakTFLOPs = (precision == 32 ? 35.6 : 0.556); // RTX‑3090
//   const double launchOverhead = 3.0e-6;
//   const double opCnt = gate.opCount(zeroTol);
//   const int nQ = gate.nQubits();

//   // pick the best anchor (same precision, closest nQ & blockSize)
//   const CUDAPerformanceCache::Item* anchor = nullptr;
//   double bestScore = -1.0;
//   for (const auto& it : cache->items) {
//     if (it.precision != precision)
//       continue;
//     double s = (it.blockSize == currentBlockSize ? 1.0 : 0.25);
//     s += 1.0 / (1.0 + std::abs(it.nQubits - nQ));
//     if (s > bestScore) {
//       bestScore = s;
//       anchor = &it;
//     }
//   }
//   if (!anchor)
//     return 1e8; // no data → discourage fuse

//   // live resource estimates for the fused candidate gate
//   CUDAKernelInfo tmp;
//   tmp.setGate(&gate);
//   tmp.setKernelFunction(nullptr); // metadata only

//   auto estSmemBytes = [&](const QuantumGate& g) {
//     int localQ = g.nQubits();
//     const int bytesPerAmp = (precision == 32 ? 8 : 16);
//     return (localQ <= 5 ? 0 : (std::size_t{1} << localQ) * bytesPerAmp);
//   };
//   auto estRegisters = [&](const QuantumGate& g) {
//     return 32 + 4 * g.nQubits();
//   };

//   tmp.setSharedMemUsage(estSmemBytes(gate));
//   tmp.setRegisterUsage(estRegisters(gate));

//   double occLive =
//       std::clamp(estimateOccupancy(tmp, currentBlockSize), 0.05, 1.0);
//   double coalLive = std::clamp(estimateCoalescingScore(gate, nQ), 0.05, 1.0);

//   // scale the anchor bandwidth by live penalties
//   double blkScale = std::pow(double(anchor->blockSize) / currentBlockSize, 0.5);
//   double occScale = occLive / std::max(0.05, anchor->occupancy);
//   double coalScale =
//       std::pow(coalLive / std::max(0.05, anchor->coalescingScore), 1.5);

//   double effGiBps = anchor->memUpdateSpeed * occScale * blkScale * coalScale;
//   effGiBps = std::max(effGiBps, 1.0 / minGibTimeCap); // clamp

//   // turn bandwidth into time (memory + FLOP + launch const)
//   const double bytesPerAmp = (precision == 32 ? 8.0 : 16.0);
//   double memGiB = bytesPerAmp * opCnt / (1ULL << 30);
//   double memTime = memGiB / effGiBps;
//   double flopTime = opCnt * 2.0 / (gpuPeakTFLOPs * 1e12);

//   constexpr double gpuBW_Bytes = 840.0 * (1ULL << 30);
//   double ridgeAI = (gpuPeakTFLOPs * 1e12) / gpuBW_Bytes;
//   double ai = 2.0 / bytesPerAmp;
//   double execCore = (ai < ridgeAI) ? memTime : flopTime;

//   return execCore + launchOverhead;
// }

// double CUDACostModel::calculateOccupancyPenalty(
//     const CUDAPerformanceCache::Item& item) const {
//   // Quadratic penalty
//   return 1.0 / (item.occupancy * item.occupancy);
// }

// double CUDACostModel::calculateCoalescingPenalty(
//     const CUDAPerformanceCache::Item& item) const {
//   // Exponential penalty for memory access patterns
//   const double exponent = 1.5;
//   return pow(1.0 / item.coalescingScore, exponent);
// }

// void CUDAPerformanceCache::runExperiments(const CUDAKernelGenConfig& gpuConfig,
//                                           int nQubits,
//                                           int blockSize,
//                                           int nRuns,
//                                           int nWorkerThreads) {
//   std::vector<std::shared_ptr<QuantumGate>> gates;
//   // Rethink this
//   // gates.reserve(nRuns);

//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<float> disFloat(0.0, 1.0);
//   std::uniform_int_distribution<int> disInt(0, nQubits - 1);
//   float prob = 1.0f;

//   const auto randFloat = [&]() { return disFloat(gen); };
//   const auto randRemove = [&](QuantumGate& gate) {
//     if (prob >= 1.0)
//       return;
//     auto* cMat = gate.gateMatrix.getConstantMatrix();
//     assert(cMat != nullptr);
//     for (size_t i = 0; i < cMat->size(); ++i) {
//       if (randFloat() > prob)
//         cMat->data()[i].real(0.0);
//       if (randFloat() > prob)
//         cMat->data()[i].imag(0.0);
//     }
//   };

//   // nQubitsWeights[q] denotes the weight for n-qubit gates
//   std::array<int, 8> nQubitsWeights;

//   const auto addRandU1q = [&]() {
//     auto a = disInt(gen);
//     gates.emplace_back(
//         std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(a)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU2q = [&]() {
//     int a, b;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     gates.emplace_back(
//         std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(a, b)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU3q = [&]() {
//     int a, b, c;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     do {
//       c = disInt(gen);
//     } while (c == a || c == b);
//     gates.emplace_back(
//         std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(a, b, c)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU4q = [&]() {
//     int a, b, c, d;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     do {
//       c = disInt(gen);
//     } while (c == a || c == b);
//     do {
//       d = disInt(gen);
//     } while (d == a || d == b || d == c);
//     gates.emplace_back(
//         std::make_shared<QuantumGate>(QuantumGate::RandomUnitary(a, b, c, d)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU5q = [&]() {
//     int a, b, c, d, e;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     do {
//       c = disInt(gen);
//     } while (c == a || c == b);
//     do {
//       d = disInt(gen);
//     } while (d == a || d == b || d == c);
//     do {
//       e = disInt(gen);
//     } while (e == a || e == b || e == c || e == d);
//     gates.emplace_back(std::make_shared<QuantumGate>(
//         QuantumGate::RandomUnitary(a, b, c, d, e)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU6q = [&]() {
//     int a, b, c, d, e, f;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     do {
//       c = disInt(gen);
//     } while (c == a || c == b);
//     do {
//       d = disInt(gen);
//     } while (d == a || d == b || d == c);
//     do {
//       e = disInt(gen);
//     } while (e == a || e == b || e == c || e == d);
//     do {
//       f = disInt(gen);
//     } while (f == a || f == b || f == c || f == d || f == e);
//     gates.emplace_back(std::make_shared<QuantumGate>(
//         QuantumGate::RandomUnitary(a, b, c, d, e, f)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU7q = [&]() {
//     int a, b, c, d, e, f, g;
//     a = disInt(gen);
//     do {
//       b = disInt(gen);
//     } while (b == a);
//     do {
//       c = disInt(gen);
//     } while (c == a || c == b);
//     do {
//       d = disInt(gen);
//     } while (d == a || d == b || d == c);
//     do {
//       e = disInt(gen);
//     } while (e == a || e == b || e == c || e == d);
//     do {
//       f = disInt(gen);
//     } while (f == a || f == b || f == c || f == d || f == e);
//     do {
//       g = disInt(gen);
//     } while (g == a || g == b || g == c || g == d || g == e || g == f);
//     gates.emplace_back(std::make_shared<QuantumGate>(
//         QuantumGate::RandomUnitary(a, b, c, d, e, f, g)));
//     randRemove(*gates.back());
//   };

//   const auto addRandU = [&](int nQubits) {
//     assert(nQubits != 0);
//     switch (nQubits) {
//     case 1:
//       addRandU1q();
//       break;
//     case 2:
//       addRandU2q();
//       break;
//     case 3:
//       addRandU3q();
//       break;
//     case 4:
//       addRandU4q();
//       break;
//     case 5:
//       addRandU5q();
//       break;
//     case 6:
//       addRandU6q();
//       break;
//     case 7:
//       addRandU7q();
//       break;
//     default:
//       assert(false && "Unknown nQubits");
//     }
//   };

//   const auto randAdd = [&]() {
//     int sum = 0;
//     for (auto weight : nQubitsWeights)
//       sum += weight;
//     assert(sum > 0 && "nQubitsWeight is empty");
//     std::uniform_int_distribution<int> dist(0, sum - 1);
//     int r = dist(gen);
//     int acc = 0;
//     for (int i = 1; i < nQubitsWeights.size(); ++i) {
//       acc += nQubitsWeights[i];
//       if (r <= acc)
//         return addRandU(i);
//     }
//   };

//   // Initial deterministic gates (1-5 qubits)
//   prob = 1.0f;
//   for (int n = 1; n <= 5; ++n) {
//     addRandU(n);
//     addRandU(n);
//   }

//   // Add randomized gates with varying sparsity
//   nQubitsWeights = {0, 1, 2, 3, 5, 5, 3, 2};
//   int initialRuns = gates.size();
//   for (int run = 0; run < nRuns - initialRuns; ++run) {
//     float ratio = static_cast<float>(run) / (nRuns - initialRuns);
//     prob = ratio * 1.0f + (1.0f - ratio) * 0.25f;
//     randAdd();
//   }

//   CUDAKernelManager kernelMgr;

//   // Generate kernels
//   utils::timedExecute(
//       [&]() {
//         int i = 0;
//         for (const auto& gate : gates)
//           kernelMgr.genStandardaloneGate(
//               gpuConfig, gate, "gate_" + std::to_string(i++), nQubits);
//       },
//       "CUDA Kernel Generation");

//   // Initialize JIT with GPU-specific parameters
//   utils::timedExecute(
//       [&]() {
//         kernelMgr.emitPTX(nWorkerThreads, llvm::OptimizationLevel::O3);
//         kernelMgr.initCUJIT(nWorkerThreads, 1); // 1 stream
//       },
//       "JIT Compilation");

//   // Prepare statevector
//   utils::StatevectorCUDA<double> sv(nQubits);
//   utils::timedExecute([&]() { sv.randomize(); }, "Initialize statevector");

//   for (auto& kernel : kernelMgr.kernels()) {
//     // Warmup
//     for (int i = 0; i < 3; ++i) {
//       kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), kernel, blockSize);
//     }
//     cudaDeviceSynchronize();

//     // Timed measurement using CUDA events
//     cudaEvent_t start, stop;
//     CUDA_CHECK(cudaEventCreate(&start));
//     CUDA_CHECK(cudaEventCreate(&stop));

//     // Time multiple runs for better accuracy
//     const int measurementRuns = 15;
//     CUDA_CHECK(cudaEventRecord(start));
//     for (int i = 0; i < measurementRuns; ++i) {
//       kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), kernel, blockSize);
//     }
//     CUDA_CHECK(cudaEventRecord(stop));
//     CUDA_CHECK(cudaEventSynchronize(stop));

//     float milliseconds = 0.0f;
//     CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
//     double avgTimeSec = milliseconds * 1e-3 / measurementRuns;

//     // Calculate GPU-specific metrics
//     double occupancy = estimateOccupancy(kernel, blockSize);
//     double coalescing = estimateCoalescingScore(*kernel.gate, nQubits);
//     double memSpeed = calculateMemUpdateSpeedCuda(
//         nQubits, kernel.precision, avgTimeSec, coalescing, occupancy);

//     items.emplace_back(Item{kernel.gate->nQubits(),
//                             kernel.opCount,
//                             kernel.precision,
//                             blockSize,
//                             occupancy,
//                             coalescing,
//                             memSpeed});

//     CUDA_CHECK(cudaEventDestroy(start));
//     CUDA_CHECK(cudaEventDestroy(stop));

//     std::cerr << "Benchmarked gate @ ";
//     utils::printArray(
//         std::cerr,
//         ArrayRef(kernel.gate->qubits.begin(), kernel.gate->qubits.size()));
//     std::cerr << ": " << memSpeed << " GiB/s, "
//               << "occupancy=" << occupancy << ", "
//               << "coalescing=" << coalescing << "\n";
//   }
// }

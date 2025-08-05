// // #include <algorithm>
// // #include <cmath>
// // #include <iostream>
// // #include <memory>
// // #include <string>
// // #include <vector>

// // #include "cast/CUDA/CUDAStatevector.h"
// // #include "cast/Core/KernelManager.h"
// // #include "cast/Fusion.h"
// // #include "cast/Legacy/CircuitGraph.h"
// // #include "openqasm/ast.h"
// // #include "openqasm/parser.h"
// // #include "timeit/timeit.h"
// // #include "llvm/Passes/OptimizationLevel.h"

// // using namespace cast;

// // enum class FusionStrategy { NoFuse, Naive, Adaptive, Cuda };

// // static const char* toString(FusionStrategy fs) {
// //   switch (fs) {
// //   case FusionStrategy::NoFuse:
// //     return "NoFuse";
// //   case FusionStrategy::Naive:
// //     return "Naive";
// //   case FusionStrategy::Adaptive:
// //     return "Adaptive";
// //   case FusionStrategy::Cuda:
// //     return "Cuda";
// //   default:
// //     return "Unknown";
// //   }
// // }

// // static void printTimingStats(const std::string& label,
// //                              const timeit::TimingResult& tr) {
// //   if (tr.tArr.empty()) {
// //     std::cout << label << ": No data\n";
// //     return;
// //   }
// //   double sum = 0.0;
// //   for (auto& t : tr.tArr)
// //     sum += t;
// //   double mean = sum / tr.tArr.size();
// //   double sumsq = 0.0;
// //   for (auto& t : tr.tArr) {
// //     double diff = t - mean;
// //     sumsq += diff * diff;
// //   }
// //   double var = (tr.tArr.size() > 1) ? (sumsq / tr.tArr.size()) : 0.0;
// //   double stdev = std::sqrt(var);

// //   auto computeMedian = [&](const std::vector<double>& arr) {
// //     if (arr.empty())
// //       return 0.0;
// //     std::vector<double> sorted(arr);
// //     std::sort(sorted.begin(), sorted.end());
// //     size_t n = sorted.size();
// //     if (n % 2 == 1)
// //       return sorted[n / 2];
// //     return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
// //   };
// //   double med = computeMedian(tr.tArr);

// //   double meanMs = mean * 1e3;
// //   double stdevMs = stdev * 1e3;
// //   double medianMs = med * 1e3;

// //   std::cout << label << ": mean=" << meanMs << " ms ± " << stdevMs
// //             << " ms (median=" << medianMs << " ms, n=" << tr.tArr.size()
// //             << ")\n";
// // }

// // int main() {
// //   using namespace timeit;

// //   // std::string qasmFile = "../examples/qft/qft-16-cp.qasm";
// //   // std::string qasmFile = "../examples/rqc/q12_189_128.qasm";
// //   // std::string qasmFile = "../examples/rqc/q20_592_427.qasm";
// //   // std::string qasmFile = "../examples/rqc/q30_521_379.qasm";
// //   std::string qasmFile = "../examples/rqc/q30_4299_3272.qasm";
// //   std::string adaptiveModelPath = "cost_model_simd1.csv";
// //   std::string cudaModelPath = "cuda128test.csv";
// //   int naiveMaxK = 2;
// //   int nReps = 3;
// //   int blockSize = 256;
// //   int nThreads = 1;
// //   auto optLevel = llvm::OptimizationLevel::O1;
// //   FusionStrategy strategy = FusionStrategy::Cuda;

// //   // Parse Time (ephemeral parse used for timing)
// //   Timer parseTimer(nReps);
// //   TimingResult parseTR = parseTimer.timeit([&]() {
// //     openqasm::Parser parser(qasmFile, 0);
// //     auto r = parser.parse();
// //   });

// //   printTimingStats("1) Parse Time", parseTR);

// //   // Parse again for the real AST we will use
// //   openqasm::Parser parser(qasmFile, 0);
// //   auto root = parser.parse();
// //   if (!root) {
// //     std::cerr << "Error: Could not parse.\n";
// //     return 1;
// //   }

// //   // Build Time (ephemeral legacy::CircuitGraph)
// //   Timer buildTimer(nReps);
// //   TimingResult buildTR = buildTimer.timeit([&]() {
// //     auto tmp = std::make_unique<cast::legacy::CircuitGraph>();
// //     root->tolegacy::CircuitGraph(*tmp);
// //   });

// //   printTimingStats("2) Build Time", buildTR);

// //   // Build the real legacy::CircuitGraph we will keep
// //   auto graphPtr = std::make_unique<cast::legacy::CircuitGraph>();
// //   root->tolegacy::CircuitGraph(*graphPtr);

// //   // Prepare caches in unique_ptr so they remain alive
// //   std::unique_ptr<PerformanceCache> adaptiveCachePtr;
// //   std::unique_ptr<CUDAPerformanceCache> cudaCachePtr;
// //   if (!adaptiveModelPath.empty()) {
// //     // If LoadFromCSV() returns by value
// //     adaptiveCachePtr = std::make_unique<PerformanceCache>(
// //         PerformanceCache::LoadFromCSV(adaptiveModelPath));
// //   }
// //   if (!cudaModelPath.empty()) {
// //     cudaCachePtr = std::make_unique<CUDAPerformanceCache>(
// //         CUDAPerformanceCache::LoadFromCSV(cudaModelPath));
// //   }

// //   // Create cost-model objects outside ephemeral blocks so that
// //   // if the fusion or circuit references them, they do not go out of scope.
// //   NaiveCostModel naiveModel(naiveMaxK, -1, 1e-8);
// //   StandardCostModel* scModelPtr = nullptr;
// //   CUDACostModel* cudaModelPtr = nullptr;

// //   std::unique_ptr<StandardCostModel> scModel;
// //   std::unique_ptr<CUDACostModel> cudaModel;
// //   if (adaptiveCachePtr) {
// //     scModel = std::make_unique<StandardCostModel>(adaptiveCachePtr.get());
// //     scModelPtr = scModel.get();
// //   }
// //   if (cudaCachePtr) {
// //     cudaModel = std::make_unique<CUDACostModel>(cudaCachePtr.get());
// //     cudaModelPtr = cudaModel.get();
// //   }

// //   // Fusion Time (ephemeral circuit used only for timing)
// //   Timer fusionTimer(nReps);
// //   TimingResult fusionTR = fusionTimer.timeit([&]() {
// //     auto tmp = std::make_unique<cast::legacy::CircuitGraph>();
// //     root->tolegacy::CircuitGraph(*tmp);

// //     FusionConfig fusionConfig = FusionConfig::Aggressive;
// //     fusionConfig.nThreads = -1;
// //     fusionConfig.precision = 64;

// //     switch (strategy) {
// //     case FusionStrategy::NoFuse:
// //       // No fusion
// //       break;

// //     case FusionStrategy::Naive:
// //       applyGateFusion(fusionConfig, &naiveModel, *tmp);
// //       break;

// //     case FusionStrategy::Adaptive:
// //       if (scModelPtr) {
// //         applyGateFusion(fusionConfig, scModelPtr, *tmp);
// //       }
// //       break;

// //     case FusionStrategy::Cuda:
// //       if (cudaModelPtr) {
// //         applyGateFusion(fusionConfig, cudaModelPtr, *tmp);
// //       }
// //       break;
// //     }
// //   });

// //   printTimingStats("3) Fusion Time", fusionTR);

// //   {
// //     FusionConfig fusionConfig = FusionConfig::Aggressive;
// //     fusionConfig.nThreads = -1;
// //     fusionConfig.precision = 64;

// //     switch (strategy) {
// //     case FusionStrategy::NoFuse:
// //       break;

// //     case FusionStrategy::Naive:
// //       applyGateFusion(fusionConfig, &naiveModel, *graphPtr);
// //       break;

// //     case FusionStrategy::Adaptive:
// //       if (scModelPtr) {
// //         applyGateFusion(fusionConfig, scModelPtr, *graphPtr);
// //       }
// //       break;

// //     case FusionStrategy::Cuda:
// //       if (cudaModelPtr) {
// //         applyGateFusion(fusionConfig, cudaModelPtr, *graphPtr);
// //       }
// //       break;
// //     }
// //   }

// //   // Configure kernel generation
// //   CUDAKernelGenConfig genConfig;
// //   genConfig.blockSize = blockSize;
// //   genConfig.precision = 64;
// //   genConfig.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;

// //   // Kernel Gen Time (ephemeral test)
// //   Timer kernelGenTimer(nReps);
// //   TimingResult kernelGenTR = kernelGenTimer.timeit([&]() {
// //     CUDAKernelManager localKM;
// //     localKM.genCUDAGatesFromlegacy::CircuitGraph(
// //         genConfig, *graphPtr, "testCircuit");
// //   });

// //   printTimingStats("4) Kernel Gen Time", kernelGenTR);

// //   // PTX Time (ephemeral test)
// //   Timer ptxTimer(nReps);
// //   std::unique_ptr<CUDAKernelManager> localKMptx;

// //   TimingResult ptxTR = ptxTimer.timeitPartial(
// //       // preMethod:
// //       [&]() {
// //         localKMptx = std::make_unique<CUDAKernelManager>();
// //         localKMptx->genCUDAGatesFromlegacy::CircuitGraph(
// //             genConfig, *graphPtr, "testCircuit");
// //       },
// //       // timedMethod: measure initCUJIT
// //       [&]() { localKMptx->emitPTX(nThreads, optLevel, 0); },
// //       // postMethod:
// //       [&]() { localKMptx.reset(); },
// //       // setup:
// //       []() {},
// //       // teardown:
// //       []() {});

// //   printTimingStats("5) PTX Time", ptxTR);

// //   Timer jitTimer(nReps);
// //   std::unique_ptr<CUDAKernelManager> localKM;

// //   TimingResult jitTR = jitTimer.timeitPartial(
// //       // preMethod:
// //       [&]() {
// //         localKM = std::make_unique<CUDAKernelManager>();
// //         localKM->genCUDAGatesFromlegacy::CircuitGraph(
// //             genConfig, *graphPtr, "testCircuit");
// //         localKM->emitPTX(nThreads, optLevel, 0);
// //       },
// //       // timedMethod: measure initCUJIT
// //       [&]() { localKM->initCUJIT(nThreads, 0); },

// //       // postMethod:
// //       [&]() { localKM.reset(); },

// //       // setup:
// //       []() {},

// //       // teardown:
// //       []() {});

// //   printTimingStats("6) JIT Time", jitTR);

// //   std::cout << "=== Post-Fusion Block Summary ===\n";
// //   for (const auto& block : graphPtr->getAllBlocks()) {
// //     auto gate = block->quantumGate;
// //     unsigned fusedQubits = gate->qubits.size();
// //     std::cout << "Block " << block->id << ": covers " << fusedQubits
// //               << " qubits.\n";
// //     if (fusedQubits > 7) {
// //       std::cerr << "WARNING: Large fused gate with " << fusedQubits
// //                 << " qubits!\n";
// //     }
// //   }
// //   std::cout << "=================================\n";

// //   // Final kernel manager that we'll actually use for execution
// //   CUDAKernelManager kernelMgr;
// //   kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
// //       genConfig, *graphPtr, "testCircuit");

// //   auto& allKernels = kernelMgr.kernels(); // returns std::vector<CUDAKernelInfo>
// //   std::cout << "[LOG] Number of generated GPU kernels: " << allKernels.size()
// //             << std::endl;

// //   for (const auto& kInfo : allKernels) {
// //     std::cout << "  - Kernel name: " << kInfo.llvmFuncName << "\n";
// //   }

// //   kernelMgr.emitPTX(nThreads, optLevel, 0);

// //   kernelMgr.initCUJIT(nThreads, 0);

// //   auto kernels =
// //       kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("testCircuit");

// //   std::cerr << "Timing execution now!\n";

// //   // Execution Time
// //   Timer execTimer(nReps);
// //   TimingResult execTR = execTimer.timeit([&]() {
// //     cast::CUDAStatevector<double> sv(graphPtr->nQubits);
// //     sv.initialize();
// //     for (auto* k : kernels) {
// //       kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, blockSize);
// //     }
// //     cudaDeviceSynchronize();
// //   });

// //   printTimingStats("7) Execution Time", execTR);

// //   // Print timing stats
// //   std::cout << "======== Timing Results ========\n";
// //   printTimingStats("1) Parse Time", parseTR);
// //   printTimingStats("2) Build Time", buildTR);
// //   printTimingStats("3) Fusion Time", fusionTR);
// //   printTimingStats("4) Kernel Gen Time", kernelGenTR);
// //   printTimingStats("5) PTX Time", ptxTR);
// //   printTimingStats("6) JIT Time", jitTR);
// //   printTimingStats("7) Execution Time", execTR);
// //   std::cout << "================================\n";

// //   return 0;
// // }


// // cuda_bcmk_imm.cpp – FINAL version compatible with CAST trunk (Aug 2025)
// // Build:
// //   clang++ -std=c++20 -O2 cuda_bcmk_imm.cpp \
// //       `cast-config --cuda --cxxflags --ldflags`
// //
// // Example run:
// //   ./cuda_bcmk_imm -i examples/rqc/q20_592_427.qasm \
// //        --run-cuda-fuse --cuda-model=cuda128test.csv
// // -----------------------------------------------------------------------------

// // cuda_bcmk_imm.cpp — benchmark harness updated (fixes build errors)
// // Build:
// //   clang++ -std=c++20 -O2 cuda_bcmk_imm.cpp \
// //       `cast-config --cuda --cxxflags --ldflags`

// #include <algorithm>
// #include <cmath>
// #include <iostream>
// #include <memory>
// #include <numeric>
// #include <string>
// #include <vector>

// #include "cast/CUDA/CUDAKernelManager.h"
// #include "cast/CUDA/CUDAStatevector.h"
// #include "cast/CUDA/CUDACostModel.h"
// #include "cast/CUDA/CUDAFusionConfig.h"
// #include "cast/CUDA/CUDAOptimizer.h"

// #include "utils/iocolor.h"
// #include "timeit/timeit.h"

// #include "llvm/Support/CommandLine.h"

// using namespace cast;
// namespace cl = llvm::cl;

// // ───────────────────────────────────── CLI ───────────────────────────────────
// cl::opt<std::string> ArgInputFilename("i", cl::desc("Input QASM file"), cl::Positional, cl::Required);
// cl::opt<std::string> ArgCudaModel("cuda-model", cl::desc("CUDA cost‑model CSV"), cl::init(""));
// cl::opt<int>         ArgPrecision("precision", cl::desc("Precision (32/64)"),  cl::init(64));
// cl::opt<int>         ArgBlockSize("block",     cl::desc("CUDA block size"),      cl::init(256));
// cl::opt<int>         ArgNThreads ("T",         cl::desc("Host threads"),        cl::Prefix, cl::init(0));

// cl::opt<bool> ArgRunNoFuse   ("run-no-fuse",    cl::desc("Run no‑fuse circuit"),   cl::init(false));
// cl::opt<bool> ArgRunNaiveFuse("run-naive-fuse", cl::desc("Run size‑only fusion"),  cl::init(true));
// cl::opt<bool> ArgRunCudaFuse ("run-cuda-fuse",  cl::desc("Run CUDA fusion"),       cl::init(false));

// cl::opt<int> ArgNaiveMaxK("sizeonly-size", cl::desc("Max gate size for naive fusion"), cl::init(2));
// cl::opt<int> ArgReplication("replication",  cl::desc("Timing replications"),           cl::init(3));
// cl::opt<int> ArgVerbose   ("verbose",      cl::desc("Verbosity level"),              cl::init(1));

// // ────────────────────────────────── Helpers ──────────────────────────────────
// static void printTimingStats(const std::string &label, const timeit::TimingResult &tr) {
//   if (tr.tArr.empty()) { std::cout << label << ": No data\n"; return; }
//   const double mean   = std::accumulate(tr.tArr.begin(), tr.tArr.end(), 0.0) / tr.tArr.size();
//   const double var    = std::accumulate(tr.tArr.begin(), tr.tArr.end(), 0.0,
//                          [mean](double a,double v){return a+(v-mean)*(v-mean);}) / tr.tArr.size();
//   const double stdev  = std::sqrt(var);
//   std::vector<double> sorted = tr.tArr; std::sort(sorted.begin(), sorted.end());
//   const double median = (sorted.size() & 1) ? sorted[sorted.size()/2]
//                                             : 0.5*(sorted[sorted.size()/2-1]+sorted[sorted.size()/2]);
//   std::cout << label << ": mean=" << mean*1e3 << " ms ± " << stdev*1e3
//             << " ms (median=" << median*1e3 << " ms, n=" << tr.tArr.size() << ")\n";
// }

// static ir::CircuitNode parseOrDie(const std::string &path) {
//   auto cirOrErr = cast::parseCircuitFromQASMFile(path);
//   if (!cirOrErr) {
//     std::cerr << BOLDRED("[Err] ") << "Failed to parse " << path << "\n" << cirOrErr.takeError() << "\n";
//     std::exit(1);
//   }
//   return cirOrErr.takeValue();
// }

// // ─────────────────────────────────── Main ────────────────────────────────────
// int main(int argc,const char** argv){
//   cl::ParseCommandLineOptions(argc,argv);

//   const Precision precision = (ArgPrecision==32)?Precision::F32:Precision::F64;
//   const int nThreads        = (ArgNThreads>0)?ArgNThreads:cast::get_cpu_num_threads();

//   if(ArgRunCudaFuse && ArgCudaModel.empty()){
//     std::cerr<<BOLDRED("[Err] ")<<"--cuda-model must be provided for cuda fusion\n";return 1;}

//   // Parse circuit & clones
//   ir::CircuitNode base = parseOrDie(ArgInputFilename);
//   ir::CircuitNode noFuse("no_fuse");   base.body.deepcopyTo(noFuse.body);
//   ir::CircuitNode naive ("naive");     base.body.deepcopyTo(naive.body);
//   ir::CircuitNode cuda  ("cuda");      base.body.deepcopyTo(cuda.body);

//   utils::Logger logger(std::cerr,ArgVerbose);
//   using CUDAOpt = cast::CUDAOptimizer;

//   // Size‑only fusion
//   if(ArgRunNaiveFuse){
//     CUDAOpt opt; opt.setSizeOnlyFusionConfig(ArgNaiveMaxK).setPrecision(precision);
//     opt.run(naive,logger);
//   }
//   // CUDA cost‑model fusion
//   if(ArgRunCudaFuse){
//     auto pc = std::make_unique<CUDAPerformanceCache>(ArgCudaModel);
//     auto cm = std::make_unique<CUDACostModel>(std::move(pc));
//     auto cfg= std::make_unique<CUDAFusionConfig>(std::move(cm)); cfg->setPrecision(precision);
//     CUDAOpt opt; opt.setCUDAFusionConfig(std::move(cfg)).setPrecision(precision);
//     opt.run(cuda,logger);
//   }

//   // Kernel generation config
//   CUDAKernelGenConfig kcfg; kcfg.blockSize=ArgBlockSize; kcfg.precision=precision;
//   kcfg.matrixLoadMode=CUDAMatrixLoadMode::UseMatImmValues;

//   CUDAKernelManager km;
//   auto addGraph=[&](const std::string& name,ir::CircuitNode& c){
//     if(auto r=km.genGraphGates(kcfg,*c.getAllCircuitGraphs()[0],name);!r){
//       std::cerr<<BOLDRED("[Err] ")<<"Kernel gen failed for "<<name<<"\n"; std::exit(1);} };

//   if(ArgRunNoFuse)   addGraph("graphNoFuse",   noFuse);
//   if(ArgRunNaiveFuse)addGraph("graphNaiveFuse",naive);
//   if(ArgRunCudaFuse) addGraph("graphCudaFuse", cuda);

//   // JIT
//   km.initCUJIT(nThreads,1);

//   const std::string tgt = ArgRunCudaFuse?"graphCudaFuse":(ArgRunNaiveFuse?"graphNaiveFuse":"graphNoFuse");
//   const auto& kernels = km.getKernelsFromGraphName(tgt);
//   if(kernels.empty()){ std::cerr<<BOLDRED("[Err] ")<<"No kernels generated for "<<tgt<<"\n"; return 1; }

//   // Execute
//   CUDAStatevector<double> sv(base.getAllCircuitGraphs()[0]->nQubits()); sv.initialize();
//   timeit::Timer timer(ArgReplication);
//   auto tr = timer.timeit([&]{
//     for(const auto& k:kernels) km.launchCUDAKernel(sv.dData(),sv.nQubits(),*k,ArgBlockSize);
//     cudaDeviceSynchronize();
//   });
//   printTimingStats("Execution("+tgt+")",tr);
//   return 0;
// }

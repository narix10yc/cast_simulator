// #include "cast/CUDA/CUDAOptimizer.h"
// #include "cast/Core/ImplOptimize.h"   // applySizeTwoFusion / applyGateFusion
// #include "timeit/timeit.h"
// #include "utils/Formats.h"

// using namespace cast;

// CUDAOptimizer::CUDAOptimizer()
//     : fusionConfig(std::make_unique<SizeOnlyFusionConfig>(3)) {}

// CUDAOptimizer &CUDAOptimizer::setSizeOnlyFusionConfig(int size) {
//   fusionConfig = std::make_unique<SizeOnlyFusionConfig>(size);
//   return *this;
// }

// CUDAOptimizer &CUDAOptimizer::setCUDAFusionConfig(
//     std::unique_ptr<CUDAFusionConfig> config) {
//   fusionConfig = std::move(config);
//   return *this;
// }

// CUDAOptimizer &CUDAOptimizer::setPrecision(Precision precision) {
//   if (auto *cfg = llvm::dyn_cast<CUDAFusionConfig>(fusionConfig.get()))
//     cfg->setPrecision(precision);
//   return *this;
// }

// CUDAOptimizer &CUDAOptimizer::setZeroTol(double tol) {
//   if (fusionConfig)
//     fusionConfig->zeroTol = tol;
//   return *this;
// }

// CUDAOptimizer &CUDAOptimizer::setSwapTol(double tol) {
//   if (fusionConfig)
//     fusionConfig->swapTol = tol;
//   return *this;
// }

// void CUDAOptimizer::run(ir::CircuitNode &circuit,
//                         utils::Logger logger) const {
//   assert(fusionConfig != nullptr);

//   // 1. Canonicalisation
//   if (enableCanonicalization_) {
//     int nFused = 0;
//     double elapsed = timeit::once([&] {
//       for (auto *graph : circuit.getAllCircuitGraphs())
//         nFused = impl::applySizeTwoFusion(*graph, fusionConfig->swapTol);
//     });
//     logger.log(1) << "Canonicalisation Block-wise Fusion Pass: " << nFused
//                   << " gates fused in " << utils::fmt_time(elapsed) << "\n";

//     if (enableCFO_) {
//       double cfoTime = timeit::once([&] {
//         SizeOnlyFusionConfig cfoCfg(2);
//         cfoCfg.swapTol = fusionConfig->swapTol;
//         nFused = impl::applyCFOFusion(circuit, &cfoCfg, 2);
//       });
//       logger.log(1) << "Canonicalisation CFO Fusion Pass: " << nFused
//                     << " gates fused in " << utils::fmt_time(cfoTime) << "\n";
//     }
//   }

//   // 2. Agglomerative fusion
//   if (enableFusion_) {
//     int nFusedTotal = 0;
//     double elapsed = timeit::once([&] {
//       auto graphs = circuit.getAllCircuitGraphs();
//       for (int size = fusionConfig->sizeMin; size <= fusionConfig->sizeMax;
//            ++size) {
//         int nFusedAtSize = 0;
//         for (auto *g : graphs) {
//           int round = 0;
//           do {
//             round = impl::applyGateFusion(*g, fusionConfig.get(), size);
//             nFusedAtSize += round;
//           } while (fusionConfig->enableMultiTraverse && round > 0);
//         }
//         if (enableCFO_ && nFusedAtSize > 0)
//           nFusedAtSize +=
//               impl::applyCFOFusion(circuit, fusionConfig.get(), size);

//         nFusedTotal += nFusedAtSize;
//         logger.log(2) << "  At size " << size << ", fused "
//                       << nFusedAtSize << " gates.\n";
//       }
//     });
//     logger.log(1) << "Agglomerative Fusion Pass: Finished in "
//                   << utils::fmt_time(elapsed) << "\n";
//   }
// }

// void CUDAOptimizer::run(ir::CircuitGraphNode &graph,
//                         utils::Logger logger) const {
//   assert(fusionConfig != nullptr);

//   // 1. Canonicalisation
//   if (enableCanonicalization_) {
//     int nFused = impl::applySizeTwoFusion(graph, fusionConfig->swapTol);
//     logger.log(1) << "Canonicalisation Block-wise Fusion Pass: " << nFused
//                   << " gates fused.\n";
//   }

//   // 2. Agglomerative fusion
//   if (enableFusion_) {
//     int nFusedTotal = 0;
//     double elapsed = timeit::once([&] {
//       for (int size = fusionConfig->sizeMin; size <= fusionConfig->sizeMax;
//            ++size) {
//         int nFused = impl::applyGateFusion(graph, fusionConfig.get(), size);
//         nFusedTotal += nFused;
//         logger.log(2) << "  At size " << size << ", fused "
//                       << nFused << " gates.\n";
//       }
//     });
//     logger.log(1) << "Agglomerative Fusion Pass: Finished in "
//                   << utils::fmt_time(elapsed) << "\n";
//   }
// }
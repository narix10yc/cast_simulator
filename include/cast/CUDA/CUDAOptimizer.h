// #ifndef CAST_CUDA_CUDAOPTIMIZER_H
// #define CAST_CUDA_CUDAOPTIMIZER_H

// #include "cast/CUDA/CUDAFusionConfig.h"
// #include "cast/Core/Optimizer.h"
// #include "utils/MaybeError.h"

// #include "llvm/Support/Casting.h"

// namespace cast {

// class CUDAOptimizer : public Optimizer {
//   // Allowed configs: SizeOnlyFusionConfig, CUDAFusionConfig.
//   std::unique_ptr<FusionConfig> fusionConfig;

//   bool enableCanonicalization_ = true;
//   bool enableFusion_           = true;
//   bool enableCFO_              = false; // disabled until CUDA CFO is ready

// public:
//   CUDAOptimizer() : fusionConfig(std::make_unique<SizeOnlyFusionConfig>(3)) {}

//   CUDAOptimizer &setSizeOnlyFusionConfig(int size) {
//     fusionConfig = std::make_unique<SizeOnlyFusionConfig>(size);
//     return *this;
//   }

//   CUDAOptimizer &setCUDAFusionConfig(std::unique_ptr<CUDAFusionConfig> config) {
//     fusionConfig = std::move(config);
//     return *this;
//   }

//   // Only meaningful for CUDAFusionConfig
//   CUDAOptimizer &setPrecision(Precision precision) {
//     if (auto *cfg = llvm::dyn_cast<CUDAFusionConfig>(fusionConfig.get()))
//       cfg->setPrecision(precision);
//     return *this;
//   }

//   // Generic FusionConfig knobs
//   CUDAOptimizer &setZeroTol(double tol) {
//     if (fusionConfig)
//       fusionConfig->zeroTol = tol;
//     return *this;
//   }

//   CUDAOptimizer &setSwapTol(double tol) {
//     if (fusionConfig)
//       fusionConfig->swapTol = tol;
//     return *this;
//   }

//   CUDAOptimizer &disableCanonicalization() {
//     enableCanonicalization_ = false;
//     return *this;
//   }
//   CUDAOptimizer &enableCanonicalization() {
//     enableCanonicalization_ = true;
//     return *this;
//   }

//   CUDAOptimizer &disableFusion() {
//     enableFusion_ = false;
//     return *this;
//   }
//   CUDAOptimizer &enableFusion() {
//     enableFusion_ = true;
//     return *this;
//   }

//   CUDAOptimizer &disableCFO() {
//     enableCFO_ = false;
//     return *this;
//   }
//   CUDAOptimizer &enableCFO() {
//     enableCFO_ = true;
//     return *this;
//   }

//   void run(ir::CircuitNode &circuit,
//            utils::Logger logger = nullptr) const override;

//   void run(ir::CircuitGraphNode &graph,
//            utils::Logger logger = nullptr) const override;
// }; // class CUDAOptimizer

// } // namespace cast

// #endif // CAST_CUDA_CUDAOPTIMIZER_H

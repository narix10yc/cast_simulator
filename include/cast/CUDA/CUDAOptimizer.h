#ifndef CAST_CUDA_CUDAOPTIMIZER_H
#define CAST_CUDA_CUDAOPTIMIZER_H

#include "cast/Core/Optimizer.h"
#include "utils/InfoLogger.h"

namespace cast {

class CUDAOptimizer : public Optimizer<CUDAOptimizer> {
public:
  CUDAOptimizer() = default;

  llvm::Error loadCUDACostModelFromFile(const std::string& filename,
                                        Precision queryPrecision);

  void displayInfo(utils::InfoLogger logger) const;
};

} // namespace cast

#endif // CAST_CUDA_CUDAOPTIMIZER_H
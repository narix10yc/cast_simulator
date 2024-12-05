#ifndef SIMULATION_KERNELGEN_H
#define SIMULATION_KERNELGEN_H

#include <llvm/IR/Module.h>
#include <memory>

namespace saot {

class QuantumGate;

struct CPUKernelGenConfig {
  enum AmpFormat { AltFormat, SepFormat };
  enum MatrixLoadMode { VectorInStack, ElemInStack, InMainBody };

  int simdS = 2;
  int precision = 64;
  AmpFormat ampFormat = AltFormat;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposite from BMI2
  bool usePDEP = false;
  bool useImmValues = true;
  bool loadMatrixInEntry = true;
  bool loadVectorMatrix = false;
  bool forceDenseKernel = false;
  double zeroSkipThres = 1e-8;
  double shareMatrixElemThres = 0.0;
  double oneTol = 1e-8;
  bool shareMatrixElemUseImmValue = false;
  MatrixLoadMode matrixLoadMode;

  static const CPUKernelGenConfig NativeDefault;
};

llvm::Function* genCPUCode(llvm::Module& llvmModule,
                           const CPUKernelGenConfig& config,
                           const QuantumGate& gate,
                           const std::string& funcName);

} // namespace saot

#endif // SIMULATION_KERNELGEN_H
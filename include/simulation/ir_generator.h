#ifndef SIMULATION_IR_GENERATOR_H
#define SIMULATION_IR_GENERATOR_H

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <array>
#include <vector>

#include "cast/LegacyCircuitGraph.h"
#include "cast/LegacyQuantumGate.h"

namespace simulation {

struct CUDAGenerationConfig {
  int precision;
  double zeroTol;
  double oneTol;
  bool useImmValues;
  bool useConstantMemSpaceForMatPtrArg;
  bool forceDenseKernel;
};

class ParamValueFeeder {
public:
  llvm::Value* basePtrV;
  std::vector<llvm::Value*> cache;
  ParamValueFeeder(llvm::Value* basePtrV)
      : basePtrV(basePtrV), cache(128, nullptr) {}

  llvm::Value* get(int v, llvm::IRBuilder<>& B, llvm::Type* Ty);
};

struct IRGeneratorConfig {
  enum AmpFormat { AltFormat, SepFormat };

  int simd_s = 2;
  int precision = 64;
  AmpFormat ampFormat = AltFormat;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposite from BMI2
  bool usePDEP = true;
  bool loadMatrixInEntry = true;
  bool loadVectorMatrix = false;
  bool forceDenseKernel = false;
  double zeroSkipThres = 1e-8;
  double shareMatrixElemThres = 0.0;
  bool shareMatrixElemUseImmValue = false;

  bool checkConfliction(std::ostream& os) const;

  std::ostream& display(int verbose = 1, bool title = true,
                        std::ostream& os = std::cerr) const;
};

/// @brief IR Generator.
class IRGenerator {
private:
  std::unique_ptr<llvm::orc::LLJIT> _jitter;

public:
  std::unique_ptr<llvm::LLVMContext> _context;
  std::unique_ptr<llvm::Module> _module;
  llvm::IRBuilder<> builder;
  IRGeneratorConfig _config;

  using AmpFormat = IRGeneratorConfig::AmpFormat;

public:
  IRGenerator(const std::string& moduleName = "myModule")
      : _jitter(nullptr), _context(std::make_unique<llvm::LLVMContext>()),
        _module(std::make_unique<llvm::Module>(moduleName, *_context)),
        builder(*_context), _config() {}

  IRGenerator(const IRGeneratorConfig& irConfig,
              const std::string& moduleName = "myModule")
      : _jitter(nullptr), _context(std::make_unique<llvm::LLVMContext>()),
        _module(std::make_unique<llvm::Module>(moduleName, *_context)),
        builder(*_context), _config(irConfig) {}

  const llvm::LLVMContext* getContext() const {
    assert(_context);
    return _context.get();
  }
  llvm::LLVMContext* getContext() {
    assert(_context);
    return _context.get();
  }

  const llvm::Module* getModule() const {
    assert(_module);
    return _module.get();
  }
  llvm::Module* getModule() {
    assert(_module);
    return _module.get();
  }

  const llvm::orc::LLJIT* getJitter() const {
    assert(_jitter && "Call 'createJitSession' first");
    return _jitter.get();
  }
  llvm::orc::LLJIT* getJitter() {
    assert(_jitter && "Call 'createJitSession' first");
    return _jitter.get();
  }

  llvm::IRBuilder<>& getBuilder() { return builder; }

  IRGeneratorConfig& config() { return _config; }
  const IRGeneratorConfig& config() const { return _config; }

  void loadFromFile(const std::string& fileName);

  void applyLLVMOptimization(const llvm::OptimizationLevel&);

  void dumpToStderr() const { _module->print(llvm::errs(), nullptr); }

  void createJitSession();

  llvm::Type* getScalarTy() {
    if (_config.precision == 32)
      return builder.getFloatTy();
    if (_config.precision == 64)
      return builder.getDoubleTy();
    llvm_unreachable("Unsupported precision");
  }

  /// @brief Generate the IR that applies new_aa = aa + bb * cc
  /// @param aa can be nullptr. In such case, new_aa will be assigned to bb * cc
  /// @param bbFlag special values are +1, -1, or 0
  /// @return aa + bb * cc. Possible nullptr, when aa is nullptr and bbFlag = 0
  llvm::Value* genMulAdd(llvm::Value* aa, llvm::Value* bb, llvm::Value* cc,
                         int bbFlag, const llvm::Twine& bbccName = "",
                         const llvm::Twine& aaName = "");

  /// @brief Generate the IR that applies new_aa = aa - bb * cc
  /// @param aa can be nullptr. In such case, new_aa will be assigned to -bb* 
  /// cc
  /// @param bbFlag special values are +1, -1, or 0
  /// @return aa - bb * cc. Possible nullptr, when aa is nullptr and bbFlag = 0
  llvm::Value* genMulSub(llvm::Value* aa, llvm::Value* bb, llvm::Value* cc,
                         int bbFlag, const llvm::Twine& bbccName = "",
                         const llvm::Twine& aaName = "");

  // fadd, accepting nullable inputs
  llvm::Value* genFAdd(llvm::Value* a, llvm::Value* b);
  // fsub, accepting nullable inputs
  llvm::Value* genFSub(llvm::Value* a, llvm::Value* b);
  // fmul, accepting nullable inputs
  llvm::Value* genFMul(llvm::Value* a, llvm::Value* b);

  std::pair<llvm::Value*, llvm::Value*>
  genComplexMultiply(const std::pair<llvm::Value*, llvm::Value*>& ,
                     const std::pair<llvm::Value*, llvm::Value*>&);

  std::pair<llvm::Value*, llvm::Value*>
  genComplexDotProduct(const std::vector<llvm::Value*>& aRe,
                       const std::vector<llvm::Value*>& aIm,
                       const std::vector<llvm::Value*>& bRe,
                       const std::vector<llvm::Value*>& bIm);

  llvm::Function* generateKernel(const cast::LegacyQuantumGate& gate,
                                 const std::string& funcName = "") {
    return generateKernelDebug(gate, 0, funcName);
  }

  llvm::Function* generateKernelDebug(const cast::LegacyQuantumGate& gate,
                                      int debugLevel,
                                      const std::string& funcName = "");

  llvm::Function* generateCUDAKernel(const cast::LegacyQuantumGate& gate,
                                     const CUDAGenerationConfig& config,
                                     const std::string& funcName = "");

  std::pair<llvm::Value*, llvm::Value*>
  generatePolynomial(const cast::Polynomial& polynomial,
                     ParamValueFeeder &feeder);

  // Generate a function that prepares matrices in simulation.
  // @return A function void(void* param, void* matrix).
  llvm::Function* generatePrepareParameter(const cast::LegacyCircuitGraph& graph);
};

} // namespace simulation

#endif // SIMULATION_IR_GENERATOR_H
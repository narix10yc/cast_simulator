#include "cast/Core/ir_generator.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"

#include "utils/iocolor.h"

using namespace IOColor;
using namespace llvm;
using namespace simulation;

Value* ParamValueFeeder::get(int v, IRBuilder<>& B, Type* Ty) {
  if (v >= cache.size())
    cache.resize(v + 1);

  if (cache[v] != nullptr)
    return cache[v];

  Value* ptr =
      B.CreateConstGEP1_32(Ty, basePtrV, v, "p.param." + std::to_string(v));
  return cache[v] = B.CreateLoad(Ty, ptr, "param." + std::to_string(v));
}

bool IRGeneratorConfig::checkConfliction(std::ostream& os) const {
  bool check = true;
  const auto warn = [&os]() -> std::ostream&  {
    return os << YELLOW_FG << BOLD << "Config warning: " << RESET;
  };
  const auto error = [&os, &check]() -> std::ostream&  {
    check = false;
    return os << RED_FG << BOLD << "Config error: " << RESET;
  };

  if (shareMatrixElemThres < 0.0)
    warn() << "Set 'shareMatrixElemThres' to a negative value has no effect\n";

  if (forceDenseKernel) {
    if (zeroSkipThres > 0.0)
      warn() << "'forceDenseKernel' is ON, 'zeroSkipThres' has no effect\n";
    if (shareMatrixElemThres > 0.0)
      warn()
          << "'forceDenseKernel' is ON, 'shareMatrixElemThres' has no effect\n";
  }

  if (shareMatrixElemThres <= 0.0 && shareMatrixElemUseImmValue)
    warn() << "'shareMatrixElemUseImmValue' is only effective when "
              "'shareMatrixElemThres' is positive (turned on)\n";

  return check;
}

std::ostream& IRGeneratorConfig::display(int verbose, bool title,
                                         std::ostream& os) const {
  if (title)
    os << CYAN_FG << "===== IR Generator Config =====\n" << RESET;

  const char* ON = "\033[32mon\033[0m";
  const char* OFF = "\033[31moff\033[0m";

  os << "simd s:               " << simd_s << "\n"
     << "precision:            " << "f" << precision << "\n"
     << "amp format:           "
     << ((ampFormat == IRGeneratorConfig::AltFormat) ? "Alt" : "Sep") << "\n"
     << "FMA " << ((useFMA) ? ON : OFF) << ", FMS " << ((useFMS) ? ON : OFF)
     << ", PDEP " << ((usePDEP) ? ON : OFF) << "\n"
     << "loadMatrixInEntry:    " << ((loadMatrixInEntry) ? "true" : "false")
     << "\n"
     << "loadVectorMatrix:     " << ((loadVectorMatrix) ? "true" : "false")
     << "\n"
     << "forceDenseKernel:     " << ((forceDenseKernel) ? "true" : "false")
     << '\n'
     << "zeroSkipThres:        " << std::scientific << zeroSkipThres << " "
     << ((zeroSkipThres > 0.0) ? ON : OFF) << "\n"
     << "shareMatrixElemThres: " << std::scientific << shareMatrixElemThres
     << " " << ((shareMatrixElemThres > 0.0) ? ON : OFF) << "\n"
     << "shareMatrixElemUseImmValue: "
     << ((shareMatrixElemUseImmValue) ? ON : OFF) << "\n";

  if (title)
    os << CYAN_FG << "===============================\n" << RESET;
  return os;
}

void IRGenerator::loadFromFile(const std::string& fileName) {
  SMDiagnostic err;
  _module = std::move(parseIRFile(fileName, err, *_context));
  if (_module == nullptr) {
    err.print("IRGenerator::loadFromFile", llvm::errs());
    llvm_unreachable("Failed to load from file");
  }
}

void IRGenerator::applyLLVMOptimization(const OptimizationLevel &level) {
  // These must be declared in this order so that they are destroyed in the
  // correct order due to inter-analysis-manager references.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  PassBuilder PB;

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create the pass manager.
  // This one corresponds to a typical -O2 optimization pipeline.
  ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(level);

  // Optimize the IR!
  MPM.run(*_module, MAM);
}

Value* IRGenerator::genMulAdd(Value* aa, Value* bb, Value* cc, int bbFlag,
                              const Twine& bbccName, const Twine& aaName) {
  if (bbFlag == 0)
    return aa;

  // new_aa = aa + cc
  if (bbFlag == 1) {
    if (aa == nullptr)
      return cc;
    return builder.CreateFAdd(aa, cc, aaName);
  }

  // new_aa = aa - cc
  if (bbFlag == -1) {
    if (aa == nullptr)
      return builder.CreateFNeg(cc, aaName);
    return builder.CreateFSub(aa, cc, aaName);
  }

  // bb is non-special
  if (aa == nullptr)
    return builder.CreateFMul(bb, cc, aaName);

  // new_aa = aa + bb * cc
  if (_config.useFMA)
    return builder.CreateIntrinsic(bb->getType(), Intrinsic::fmuladd,
                                   {bb, cc, aa}, nullptr, aaName);
  // not use FMA
  auto* bbcc = builder.CreateFMul(bb, cc, bbccName);
  return builder.CreateFAdd(aa, bbcc, aaName);
}

Value* IRGenerator::genMulSub(Value* aa, Value* bb, Value* cc, int bbFlag,
                              const Twine& bbccName, const Twine& aaName) {
  if (bbFlag == 0)
    return aa;

  auto* ccNeg = builder.CreateFNeg(cc, "ccNeg");
  // new_aa = aa - cc
  if (bbFlag == 1) {
    if (aa == nullptr)
      return ccNeg;
    return builder.CreateFSub(aa, cc, aaName);
  }

  // new_aa = aa + cc
  if (bbFlag == -1) {
    if (aa == nullptr)
      return cc;
    return builder.CreateFAdd(aa, cc, aaName);
  }

  // bb is non-special
  // new_aa = aa - bb * cc
  if (aa == nullptr)
    return builder.CreateFMul(bb, ccNeg, aaName);

  if (_config.useFMS)
    return builder.CreateIntrinsic(bb->getType(), Intrinsic::fmuladd,
                                   {bb, ccNeg, aa}, nullptr, aaName);
  // not use FMS
  auto* bbccNeg = builder.CreateFMul(bb, ccNeg, bbccName + "Neg");
  return builder.CreateFAdd(aa, bbccNeg, aaName);
}

Value* IRGenerator::genFAdd(Value* a, Value* b) {
  if (a && b)
    return builder.CreateFAdd(a, b);
  if (a)
    return a;
  return b;
}

Value* IRGenerator::genFSub(Value* a, Value* b) {
  if (a && b)
    return builder.CreateFSub(a, b);
  if (a)
    return a;
  if (b)
    return builder.CreateFNeg(b);
  return nullptr;
}

Value* IRGenerator::genFMul(Value* a, Value* b) {
  if (a && b)
    return builder.CreateFMul(a, b);
  return nullptr;
}

std::pair<Value*, Value*> IRGenerator::genComplexMultiply(
    const std::pair<llvm::Value*, llvm::Value*>& a,
    const std::pair<llvm::Value*, llvm::Value*>& b) {
  return {genFSub(genFMul(a.first, b.first), genFMul(a.second, b.second)),
          genFAdd(genFMul(a.first, b.second), genFMul(a.second, b.first))};
}

std::pair<Value*, Value*> IRGenerator::genComplexDotProduct(
    const std::vector<Value*>& aRe, const std::vector<Value*>& aIm,
    const std::vector<Value*>& bRe, const std::vector<Value*>& bIm) {
  auto length = aRe.size();
  assert(aRe.size() == aIm.size());
  assert(aIm.size() == bRe.size());
  assert(bRe.size() == bIm.size());

  auto* ty = aRe[0]->getType();
  auto* re = builder.CreateFMul(aRe[0], bRe[0]);
  auto* im = builder.CreateFMul(aRe[0], bIm[0]);
  for (unsigned i = 1; i < length; i++) {
    re = builder.CreateIntrinsic(ty, Intrinsic::fmuladd, {aRe[i], bRe[i], re});
    im = builder.CreateIntrinsic(ty, Intrinsic::fmuladd, {aRe[i], bIm[i], im});
  }

  auto ree = builder.CreateFMul(aIm[0], bIm[0]);
  im = builder.CreateIntrinsic(ty, Intrinsic::fmuladd, {aIm[0], bRe[0], im});
  for (unsigned i = 1; i < length; i++) {
    ree =
        builder.CreateIntrinsic(ty, Intrinsic::fmuladd, {aRe[i], bRe[i], ree});
    im = builder.CreateIntrinsic(ty, Intrinsic::fmuladd, {aIm[i], bRe[i], im});
  }

  re = builder.CreateFSub(re, ree);
  return {re, im};
}
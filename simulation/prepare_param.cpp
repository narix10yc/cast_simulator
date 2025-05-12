#include "simulation/ir_generator.h"

#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace simulation;
using namespace cast;

std::pair<Value*, Value*>
IRGenerator::generatePolynomial(const Polynomial& P, ParamValueFeeder& feeder) {
  P.print(std::cerr << "generating polynomial: ") << "\n";
  const auto generateMonomial = [&](const Monomial& M) {
    std::pair<Value*, Value*> resultV{
        (M.coef.real() == 0.0) ? nullptr
                               : ConstantFP::get(getScalarTy(), M.coef.real()),
        (M.coef.imag() == 0.0) ? nullptr
                               : ConstantFP::get(getScalarTy(), M.coef.imag())};

    assert((resultV.first || resultV.second) && "coef should not be 0");
    // mul terms
    Value* mulTermsV = nullptr;
    for (const auto& T : M.mulTerms()) {
      assert(!T.vars.empty() && "should be absorbed into M.coef");
      auto it = T.vars.cbegin();
      mulTermsV = feeder.get(*it, builder, getScalarTy());
      while (++it != T.vars.cend())
        mulTermsV = builder.CreateFAdd(mulTermsV,
                                       feeder.get(*it, builder, getScalarTy()));

      if (T.constant != 0.0) {
        // errs() << "T.constant is non-zero " << T.constant << "\n";
        mulTermsV = builder.CreateFAdd(
            mulTermsV, ConstantFP::get(getScalarTy(), T.constant));
      }

      if (T.op == VariableSumNode::CosOp)
        mulTermsV = builder.CreateUnaryIntrinsic(Intrinsic::cos, mulTermsV);
      else if (T.op == VariableSumNode::SinOp)
        mulTermsV = builder.CreateUnaryIntrinsic(Intrinsic::sin, mulTermsV);
    }

    resultV = genComplexMultiply(resultV, {mulTermsV, nullptr});
    // expi terms
    if (M.expiVars().empty())
      return resultV;

    auto it = M.expiVars().cbegin();
    Value* expiVarV = feeder.get(it->var, builder, getScalarTy());
    if (!it->isPlus)
      expiVarV = builder.CreateFNeg(expiVarV);
    while (++it != M.expiVars().cend()) {
      Value* tmp = feeder.get(it->var, builder, getScalarTy());
      if (it->isPlus)
        expiVarV = builder.CreateFAdd(expiVarV, tmp);
      else
        expiVarV = builder.CreateFSub(expiVarV, tmp);
    }
    return genComplexMultiply(
        resultV, {builder.CreateUnaryIntrinsic(Intrinsic::cos, expiVarV),
                  builder.CreateUnaryIntrinsic(Intrinsic::sin, expiVarV)});
  };

  if (P.monomials().empty())
    return {nullptr, nullptr};
  auto it = P.monomials().cbegin();
  auto polyV = generateMonomial(*it);
  while (++it != P.monomials().cend()) {
    auto tmp = generateMonomial(*it);
    polyV.first = genFAdd(polyV.first, tmp.first);
    polyV.second = genFAdd(polyV.second, tmp.second);
  }
  return polyV;
}

Function* IRGenerator::generatePrepareParameter(const CircuitGraph& graph) {
  Type* scalarTy = getScalarTy();

  auto* funcTy = FunctionType::get(
      builder.getVoidTy(), {builder.getPtrTy(), builder.getPtrTy()}, false);
  Function* func = Function::Create(funcTy, Function::ExternalLinkage,
                                    "prepare_function", getModule());

  auto* paramArgV = func->getArg(0);
  paramArgV->setName("param.ptr");
  auto* matrixArgV = func->getArg(1);
  matrixArgV->setName("matrix.ptr");
  ParamValueFeeder feeder(paramArgV);

  BasicBlock* entryBB = BasicBlock::Create(*_context, "entry", func);
  builder.SetInsertPoint(entryBB);

  uint64_t startIndex = 0;
  const auto allBlocks = graph.getAllBlocks();
  for (unsigned i = 0; i < allBlocks.size(); i++) {
    GateBlock* gateBlock = allBlocks[i];
    uint64_t numCompMatrixEntries = (1ULL << (2 * gateBlock->nQubits()));

    const LegacyGateMatrix& gateMatrix = gateBlock->quantumGate->gateMatrix;
    if (const auto* cMat = gateMatrix.getConstantMatrix()) {
      const auto& cData = cMat->data();
      for (uint64_t d = 0; d < numCompMatrixEntries; d++) {
        std::string gepName;
        Value* matPtrV;
        // real part
        gepName = "m.block" + std::to_string(i) + ".re" + std::to_string(d);
        matPtrV = builder.CreateConstInBoundsGEP1_64(
            scalarTy, matrixArgV, startIndex + 2 * d, gepName);
        builder.CreateStore(ConstantFP::get(scalarTy, cData[d].real()),
                            matPtrV);

        // imag part
        gepName = "m.block" + std::to_string(i) + ".im" + std::to_string(d);
        matPtrV = builder.CreateConstInBoundsGEP1_64(
            scalarTy, matrixArgV, startIndex + 2 * d + 1, gepName);
        builder.CreateStore(ConstantFP::get(scalarTy, cData[d].imag()),
                            matPtrV);
      }
    } else {
      const auto* pData = gateMatrix.getParametrizedMatrix().data();
      for (uint64_t d = 0; d < numCompMatrixEntries; d++) {
        auto polyV = generatePolynomial(pData[d], feeder);
        std::string gepName;
        Value* matPtrV;

        // real part
        gepName = "m.block" + std::to_string(i) + ".re" + std::to_string(d);
        matPtrV = builder.CreateConstInBoundsGEP1_64(
            scalarTy, matrixArgV, startIndex + 2 * d, gepName);
        if (polyV.first)
          builder.CreateStore(polyV.first, matPtrV);
        else
          builder.CreateStore(ConstantFP::get(getScalarTy(), 0.0), matPtrV);

        // imag part
        gepName = "m.block" + std::to_string(i) + ".im" + std::to_string(d);
        matPtrV = builder.CreateConstInBoundsGEP1_64(
            scalarTy, matrixArgV, startIndex + 2 * d + 1, gepName);
        if (polyV.second)
          builder.CreateStore(polyV.second, matPtrV);
        else
          builder.CreateStore(ConstantFP::get(getScalarTy(), 0.0), matPtrV);
        // func->print(errs());
      }
    }

    startIndex += 2 * numCompMatrixEntries;
  }
  builder.CreateRetVoid();

  return func;
}
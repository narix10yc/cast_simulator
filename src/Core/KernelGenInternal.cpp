#include "cast/Core/KernelGenInternal.h"

using namespace cast;
using namespace llvm;

Value* cast::internal::genMulAdd(
    IRBuilder<>& B, Value* a, Value* b, Value* c,
    ScalarKind aKind, const Twine& name) {
  assert(b && "operand b cannot be null when calling genMulAdd");
  switch (aKind) {
  case SK_General: {
    // a * b + c
    assert(a != nullptr && "General kind 'a' operand cannot be null");
    if (c)
      return B.CreateIntrinsic(
        a->getType(), Intrinsic::fmuladd, {a, b, c}, nullptr, name);
    return B.CreateFMul(a, b, name);
  }
  case SK_ImmValue: {
    // (imm value) a * b + c
    assert(a && "ImmValue 'a' must be non-null");
    assert(a->getType() == b->getType());
    if (c)
      return B.CreateIntrinsic(
        a->getType(), Intrinsic::fmuladd, {a, b, c}, nullptr, name);
    return B.CreateFMul(a, b, name);
  }
  case SK_One: {
    // b + c
    if (c)
      return B.CreateFAdd(b, c, name);
    return b;
  }
  case SK_MinusOne: {
    // -b + c
    if (c)
      return B.CreateFSub(c, b, name);
    return B.CreateFNeg(b, name);
  }
  case SK_Zero:
    return c;
  default:
    llvm_unreachable("Unknown ScalarKind");
    return nullptr;
  }
}

Value* cast::internal::genNegMulAdd(
    IRBuilder<>& B, Value* a, Value* b, Value* c,
    ScalarKind aKind, const Twine& name) {
  assert(b && "operand b cannot be null when calling genNegMulAdd");
  // special-a cases
  switch (aKind) {
  case SK_One: {
    // -b + c
    if (c)
      return B.CreateFSub(c, b, name);
    return B.CreateFNeg(b, name);
  }
  case SK_MinusOne: {
    // b + c
    if (c)
      return B.CreateFAdd(b, c, name);
    return b;
  }
  case SK_Zero:
    return c;
  default:
    break;
  }

  auto* aNeg = B.CreateFNeg(a, "a.neg");
  switch (aKind) {
  case SK_General: {
    assert(a && "General kind 'a' operand cannot be null");
    assert(a->getType() == b->getType());
    // -a * b + c
    if (c)
      return B.CreateIntrinsic(
        aNeg->getType(), Intrinsic::fmuladd, {aNeg, b, c}, nullptr, name);
    return B.CreateFMul(aNeg, b, name);
  }
  case SK_ImmValue: {
    // -(imm value)a * b + c
    assert(a && "ImmValue 'a' must be non-null");
    if (c)
      return B.CreateIntrinsic(
        aNeg->getType(), Intrinsic::fmuladd, {aNeg, b, c}, nullptr, name);
    return B.CreateFMul(aNeg, b, name);
  }
  default:
    llvm_unreachable("Unknown ScalarKind");
    return nullptr;
  }
}


std::pair<Value*, Value*> cast::internal::genComplexInnerProduct(
    IRBuilder<>& B,
    const std::vector<Value*>& aVec, const std::vector<Value*>& bVec,
    const Twine& name, cast::internal::FusedOpKind foKind) {
  assert(aVec.size() == bVec.size());
  unsigned size = aVec.size();
  Value* re;
  Value* im;

  assert(0 && "Not implemented");
  return {nullptr, nullptr};
}

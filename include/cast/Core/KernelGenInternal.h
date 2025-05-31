#ifndef CAST_CORE_KERNEL_MANAGER_INTERNAL_H
#define CAST_CORE_KERNEL_MANAGER_INTERNAL_H

#include "llvm/IR/IRBuilder.h"
#include "cast/Core/ScalarKind.h"

#include <vector>

namespace cast::internal {

enum FusedOpKind {
  FO_None,      // do not use fused operations
  FO_FMA_Only,  // use fma only
  FO_FMA_FMS,   // use fma and fms
};

// genMulAdd: generate multiply-add operation a * b + c.
// @param b cannot be nullptr
// @return a * b + c
llvm::Value* genMulAdd(
    llvm::IRBuilder<>& B, llvm::Value* a, llvm::Value* b, llvm::Value* c,
    ScalarKind aKind, const llvm::Twine& name = "");

// genFMA: generate negate-multiply-add operation -a * b + c.
// @param b cannot be nullptr
// @return -a * b + c
llvm::Value* genNegMulAdd(
    llvm::IRBuilder<>& B, llvm::Value* a, llvm::Value* b, llvm::Value* c,
    ScalarKind aKind, const llvm::Twine& name = "");

std::pair<llvm::Value*, llvm::Value*> genComplexInnerProduct(
    llvm::IRBuilder<>& B, const std::vector<llvm::Value*>& aVec,
    const std::vector<llvm::Value*>& bVec, const llvm::Twine& name = "",
    FusedOpKind foKind = FO_FMA_FMS);

}; // namespace cast::internal

#endif // CAST_CORE_KERNEL_MANAGER_INTERNAL_H
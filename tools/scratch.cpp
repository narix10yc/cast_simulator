#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
  // Initialize LLVM native targets (for JIT)
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  // Create LLVM context and module
  auto Context = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("my_module", *Context);
  IRBuilder<> Builder(*Context);

  // Define: int add(int a, int b)
  FunctionType *FT = FunctionType::get(Builder.getInt32Ty(),
                                       {Builder.getInt32Ty(), Builder.getInt32Ty()},
                                       false);
  Function *AddFn = Function::Create(FT, Function::ExternalLinkage, "add", M.get());

  // Build function body: return a + b;
  BasicBlock *BB = BasicBlock::Create(*Context, "entry", AddFn);
  Builder.SetInsertPoint(BB);
  auto Args = AddFn->args().begin();
  Value *A = Args++;
  Value *B = Args;
  Value *Sum = Builder.CreateAdd(A, B, "sum");
  Builder.CreateRet(Sum);

  // Create the JIT
  auto JIT = cantFail(orc::LLJITBuilder().create());

  // Add the module to the JIT
  cantFail(JIT->addIRModule(orc::ThreadSafeModule(std::move(M), std::move(Context))));

  // Look up the symbol "add"
  auto Sym = cantFail(JIT->lookup("add"));

  auto Add = Sym.toPtr<int(int, int)>();

  int result = Add(40, 2);
  outs() << "add(40, 2) = " << result << "\n";

  return 0;
}
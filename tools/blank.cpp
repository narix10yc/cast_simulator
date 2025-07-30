#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

void get() {
  auto name = llvm::Twine("A") + llvm::Twine(1) + llvm::Twine("B");
  llvm::outs() << name << "\n";
}
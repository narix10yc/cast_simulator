#include "cast/CPU/CPUOptimizerBuilder.h"

using namespace cast;

MaybeError<Optimizer> CPUOptimizerBuilder::build() {
  assert(data != nullptr && "CPUOptimizerBuilder is not initialized");
  
  return cast::makeError<Optimizer>("Not Implemented yet.");
}
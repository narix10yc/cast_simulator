#include "cast/CPU/CPUOptimizerBuilder.h"

using namespace cast;

int main() {
  CPUOptimizerBuilder builder;
  auto opt = builder.setSizeOnlyFusion(3).build();
  if (!opt) {
    std::cerr << "Failed to build optimizer: " << opt.takeError() << std::endl;
    return 1;
  }
  return 0;
}
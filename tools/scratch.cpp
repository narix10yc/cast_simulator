#include "utils/MaybeError.h"
#include <iostream>

using namespace cast;

MaybeError<void> funcMayThrow(int x) {
  if (x < 0)
    return makeError("Negative value error");
  return {}; // success
}

int main(int argc, char** argv) {
  if (auto r = funcMayThrow(-1)) {
    std::cerr << "Success\n";
  } else {
    std::cerr << "Error: " << r.what() << "\n";
  }
  return 0;
}
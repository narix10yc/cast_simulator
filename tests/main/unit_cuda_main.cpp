#include "tests/TestKit.h"

int main() {
  bool ok = true;
  ok = cast::test::test_statevectorCUDA() && ok;
  ok = cast::test::test_cudaU() && ok;
  ok = cast::test::test_cudaU2() && ok;
  return ok ? 0 : 1;
}

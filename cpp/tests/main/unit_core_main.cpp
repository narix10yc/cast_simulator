#include "tests/TestKit.h"

int main() {
  bool ok = true;
  ok = cast::test::test_complexSquareMatrix() && ok;
  ok = cast::test::test_quantumGate() && ok;
  ok = cast::test::test_quantumChannel() && ok;
  return ok ? 0 : 1;
}

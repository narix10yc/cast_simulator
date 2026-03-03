#include "tests/TestKit.h"

int main() {
  bool ok = true;
  ok = cast::test::test_cpuH() && ok;
  ok = cast::test::test_cpuU() && ok;
  ok = cast::test::test_fusionCPU() && ok;
  return ok ? 0 : 1;
}

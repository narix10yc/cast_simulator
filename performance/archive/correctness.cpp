#include "gen_file.h"
#include "timeit/timeit.h"
#include "utils/iocolor.h"
#include "simulation/StatevectorCPU.h"
#include <iomanip>
#include <iostream>
#include <cast/simulate.h>

#ifdef USING_F32
using real_t = float;
#else
using real_t = double;
#endif

using namespace timeit;
using namespace IOColor;
using namespace cast::simulate;
using namespace utils::statevector;
using namespace cast;

int main(int argc, char* *argv) {
  assert(argc > 1);
  unsigned targetQ = std::stoi(argv[1]);

  const int nQubits = 10;

  auto mat = GateMatrix::FromName("u3", {0.92, 0.46, 0.22});
  // auto mat = GateMatrix::FromName("h");
  auto gate = QuantumGate(mat, {targetQ});
  gate = gate.lmatmul({mat, {targetQ + 1}});
  // gate = gate.lmatmul({ mat , {9}});

  StatevectorComp<real_t> sv_ref(nQubits);
#ifdef USING_ALT_KERNEL
  StatevectorAlt<real_t, SIMD_S> sv_test(nQubits);
#else
  StatevectorSep<real_t> sv_test(nQubits);
#endif

  sv_ref.randomize();
  // for (unsigned i = 0; i < sv_ref.N; i++)
  // sv_ref.data[i] = {1.0, 1.0};

  for (unsigned i = 0; i < sv_ref.N; i++) {
#ifdef USING_ALT_KERNEL
    sv_test.real(i) = sv_ref.data[i].real();
    sv_test.imag(i) = sv_ref.data[i].imag();
#else
    sv_test.real[i] = sv_ref.data[i].real();
    sv_test.imag[i] = sv_ref.data[i].imag();
#endif
  }

  applyGeneral(sv_ref.data, gate.gateMatrix, gate.qubits, nQubits);
  // sv_ref.print(std::cerr) << "\n";

  uint64_t idxMax =
      1ULL << (sv_test.nQubits - SIMD_S - _metaData[targetQ].nQubits);
// uint64_t idxMax = 1;
#ifdef USING_ALT_KERNEL
  _metaData[targetQ].func(sv_test.data, 0, idxMax, _metaData[targetQ].mPtr);
#else
  _metaData[targetQ].func(sv_test.real, sv_test.imag, 0, idxMax,
                          _metaData[targetQ].mPtr);
#endif

  // sv_test.print(std::cerr) << "\n";

  for (unsigned i = 0; i < sv_test.N; i++) {
#ifdef USING_ALT_KERNEL
    if (std::abs(sv_test.real(i) - sv_ref.data[i].real()) > 1e-5)
      std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i
                << " real\n";
    if (std::abs(sv_test.imag(i) - sv_ref.data[i].imag()) > 1e-5)
      std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i
                << " imag\n";
#else
    if (std::abs(sv_test.real[i] - sv_ref.data[i].real()) > 1e-5)
      std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i
                << " real\n";
    if (std::abs(sv_test.imag[i] - sv_ref.data[i].imag()) > 1e-5)
      std::cerr << RED_FG << "Unmatch: " << RESET << "position " << i
                << " imag\n";
#endif
  }
  std::cerr << GREEN_FG << "All done\n" << RESET << "\n";

  return 0;
}

#include "cast/KrausFormat.h"

using namespace cast;

int main(int argc, char** argv) {
  
  KrausFormat kf(3);
  kf.setKrausOperator(0, GateMatrix::H(), 0.5);
  
  kf.display(std::cerr);
  return 0;
}
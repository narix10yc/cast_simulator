#include "cast/CPU/CPUFusionConfig.h"
#include "utils/iocolor.h"

using namespace cast;

std::ostream& CPUFusionConfig::displayInfo(std::ostream& os,
                                           int verbose) const {
  os << BOLDCYAN("===== Info of CPUFusionConfig @ " << this << "=====\n");
  
  os << BOLDCYAN("===== End of Info =====\n");
  return os;
}
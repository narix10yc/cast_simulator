#include "cast/CPU/CPUFusionConfig.h"
#include "utils/iocolor.h"

using namespace cast;

void CPUFusionConfig::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUFusionConfig");
  auto l = logger.indent();
  FusionConfig::displayInfo(l);
}

#include "cast/CPU/CPUFusionConfig.h"
#include "utils/iocolor.h"

using namespace cast;

void CPUFusionConfig::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUFusionConfig");
  FusionConfig::displayInfo(logger.indent());
}

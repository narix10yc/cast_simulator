#include "utils/Logger.h"

void myFunc(utils::Logger logger) {
  logger.log(-1) << "This is a log message.\n";
}

int main() {
  utils::Logger logger(std::cerr, 2);
  myFunc(logger);
  return 0;
}
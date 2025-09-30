#include "utils/InfoLogger.h"
#include <iostream>

int main() {
  utils::InfoLogger logger(std::cerr, 2);
  logger.display(std::array<std::string_view, 2>{"Name", "Age"}, "Alice", 30);

  return 0;
}
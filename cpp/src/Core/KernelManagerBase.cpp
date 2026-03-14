#include "cast/Core/KernelManager.h"

using namespace cast;

std::string cast::internal::mangleGraphName(const std::string& graphName) {
  return "G" + std::to_string(graphName.length()) + graphName;
}

std::string cast::internal::demangleGraphName(const std::string& mangledName) {
  const auto* p = mangledName.data();
  const auto* e = mangledName.data() + mangledName.size();
  assert(p != e);
  assert(*p == 'G' && "Mangled graph name must start with 'G'");
  ++p;
  assert(p != e);
  auto p0 = p;
  while ('0' <= *p && *p <= '9') {
    ++p;
    assert(p != e);
  }
  auto l = std::stoi(std::string(p0, p));
  assert(p + l <= e);
  return std::string(p, p + l);
}

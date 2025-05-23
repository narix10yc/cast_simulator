#include "new_parser/ASTContext.h"

using namespace cast::draft::ast;

std::ostream& ASTContext::displayLineTable(std::ostream& os) const {
  os << "Line table:\n";
  const auto& table = sourceManager.lineTable;
  if (table.empty())
    return os << "  Line table is empty.\n";
  auto nLines = sourceManager.lineTable.size() - 1;
  for (size_t i = 0; i < nLines; ++i) {
    os << "Line " << i + 1 << " @ "
       << static_cast<const void*>(table[i]) << " | "
       << IOColor::CYAN_FG;
    os.write(table[i], table[i + 1] - table[i]);
    os << IOColor::RESET;
  }
  os << "EoF @ " << static_cast<const void*>(table[nLines]) << "\n";
  return os;
}
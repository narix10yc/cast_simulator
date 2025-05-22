#include "new_parser/Parser.h"

using namespace cast::draft::ast;

GateChainStmt* Parser::parseGateChainStmt() {
  llvm::SmallVector<GateApplyStmt*> gates;
  while (true) {
    if (curToken.is(tk_Semicolon))
      break;
    auto* gate = parseGateApplyStmt();
    assert(gate != nullptr);
    gates.push_back(gate);
    if (curToken.is(tk_AtSymbol)) {
      advance(tk_AtSymbol);
      continue;
    }
    requireCurTokenIs(tk_Semicolon, "Expect ';' to finish a GateChain");
  }
  advance(tk_Semicolon);

  return new (ctx) GateChainStmt(
    ctx.createSpan(gates.data(), gates.size())
  );
}

std::ostream& GateChainStmt::print(std::ostream& os) const {
  for (size_t i = 0, size = gates.size(); i < size; ++i) {
    gates[i]->print(os);
    os << ((i == size - 1) ? ";" : "\n@ ");
  }
  return os;
}

void GateChainStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << ": " << gates.size() << " gates\n";
  p.setState(indent, gates.size());
  for (size_t i = 0; i < gates.size(); ++i)
    gates[i]->prettyPrint(p, indent + 1);
}
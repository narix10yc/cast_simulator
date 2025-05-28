#include "cast/Core/AST/Parser.h"

using namespace cast::draft::ast;

RootNode* Parser::parse() {
  llvm::SmallVector<Stmt*> stmts;
  pushScope();

  while (curToken.isNot(tk_Eof)) {
    switch (curToken.kind) {
      case tk_Circuit: {
        auto* stmt = parseCircuitStmt();
        assert(stmt != nullptr);
        stmts.push_back(stmt);
        addSymbol(stmt->name, stmt);
        break;
      }
      case tk_Channel: {
        auto* stmt = parseChannelStmt();
        assert(stmt != nullptr);
        stmts.push_back(stmt);
        addSymbol(stmt->name, stmt);
        break;
      }
      default: {
        logErrHere("Expecting a top-level expression, which could be a "
                   "Circuit or Channel.");
        failAndExit();
      }
    }
  }
  popScope();
  return new (ctx) RootNode(
    ctx.createSpan(stmts.data(), stmts.size())
  );
}

std::ostream& RootNode::print(std::ostream& os) const {
  for (const auto* stmt : stmts) {
    stmt->print(os);
    os << '\n';
  }
  return os;
}

void RootNode::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = stmts.size();
  p.write(indent) << getKindName() << ": " << size << " stmts\n";
  p.setState(indent, size);
  for (unsigned i = 0; i < size; ++i)
    stmts[i]->prettyPrint(p, indent+1);
}
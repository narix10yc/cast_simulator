#include "new_parser/Parser.h"

using namespace cast::draft;

ast::CircuitStmt* Parser::parseCircuitStmt() {
  assert(curToken.is(tk_Circuit) &&
         "parseCircuitStmt expects to be called with a 'Circuit' token");
  advance(tk_Circuit);

  auto* attr = parseAttribute();
  requireCurTokenIs(tk_Identifier, "Expect a circuit name");
  auto name = ctx.createIdentifier(curToken.toStringView());
  auto nameLoc = curToken.loc;
  advance(tk_Identifier);

  // circuit body
  requireCurTokenIs(tk_L_CurlyBracket, "Expect '{' to start circuit body");
  advance(tk_L_CurlyBracket);
  pushScope();
  llvm::SmallVector<ast::Stmt*> body;
  while (true) {
    auto* stmt = parseCircuitLevelStmt();
    if (stmt == nullptr)
      break;
    body.push_back(stmt);
  }

  requireCurTokenIs(tk_R_CurlyBracket, "Expect '}' to end circuit body");
  advance(tk_R_CurlyBracket);
  popScope();
  // end of circuit body

  return new (ctx) ast::CircuitStmt(
    name,
    nameLoc,
    attr,
    ctx.createSpan(body.data(), body.size())
  );
}

std::ostream& ast::CircuitStmt::print(std::ostream& os) const {
  os << "Circuit";
  if (attr != nullptr)
    attr->print(os);
  os << " " << name << " {\n";
  for (const auto& stmt : body)
    stmt->print(os << "  ") << "\n";
  os << "}\n";
  return os;
}

void ast::CircuitStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = body.size();
  p.write(indent) << getKindName() << "(" << name << "): "
                  << size << " stmts\n";
  if (attr != nullptr)
    p.os << "Attr\n";
  p.setState(indent, size);
  for (unsigned i = 0; i < size; ++i)
    body[i]->prettyPrint(p, indent + 1);
}

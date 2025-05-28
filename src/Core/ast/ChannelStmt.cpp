#include "cast/Core/AST/Parser.h"

using namespace cast::draft::ast;

ChannelStmt* Parser::parseChannelStmt() {
  assert(curToken.is(tk_Channel) && "Expect 'Channel' to start a channel");
  advance(tk_Channel);
  requireCurTokenIs(tk_Identifier, "Expect a channel name");
  auto name = ctx.createIdentifier(curToken.toStringView());
  auto nameLoc = curToken.loc;
  advance(tk_Identifier);
  auto* paramDecl = parseParameterDecl();
  std::vector<PauliComponentStmt*> body;
  requireCurTokenIs(tk_L_CurlyBracket, "Expect '{' to start channel body");
  advance(tk_L_CurlyBracket);
  while (true) {
    if (curToken.is(tk_Identifier)) {
      auto* stmt = parsePauliComponentStmt();
      assert(stmt != nullptr && "pauli string expr should not be nullptr");
      body.push_back(stmt);
      continue;
    }
    if (curToken.is(tk_R_CurlyBracket))
      break;
    logErrHere("Unexpected token in channel body");
    failAndExit();
  }
  requireCurTokenIs(tk_R_CurlyBracket, "Expect '}' to end channel body");
  advance(tk_R_CurlyBracket);
  return new (ctx) ChannelStmt(
    name, nameLoc, paramDecl, ctx.createSpan(body.data(), body.size()));
}

std::ostream& ChannelStmt::print(std::ostream& os) const {
  os << "Channel " << name;
  if (paramDecl != nullptr)
    paramDecl->print(os);
  os << " {\n";
  for (const auto* stmt : components) {
    stmt->print(os << "  ");
    os << "\n";
  }
  return os << "}";
}

void ChannelStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = components.size();
  p.write(indent) << getKindName() << ": " << name;
  if (paramDecl != nullptr)
    paramDecl->print(p.os);
  p.os << "\n";
  p.setState(indent, size);
  for (unsigned i = 0; i < size; ++i)
    components[i]->prettyPrint(p, indent + 1);
}
#include "new_parser/Parser.h"

using namespace cast::draft;

ast::PauliComponentStmt* Parser::parsePauliComponentStmt() {
  assert(curToken.is(tk_Identifier) && 
    "parsePauliComponentStmt expects to be called with an identifier");
  auto str = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  auto* weight = parseExpr();
  if (weight == nullptr) {
    logErrHere("Expect a weight after the Pauli component");
    failAndExit();
  }
  if (curToken.isNot(tk_Semicolon)) {
    logErrHere("Expect ';' to end a Pauli component");
    failAndExit();
  }
  advance(tk_Semicolon);
  return new (ctx) ast::PauliComponentStmt(str, weight);
}

ast::ChannelStmt* Parser::parseChannelStmt() {
  assert(curToken.is(tk_Channel) && "Expect 'Channel' to start a channel");
  advance(tk_Channel);
  auto* attr = parseAttribute();
  requireCurTokenIs(tk_Identifier, "Expect a channel name");
  auto name = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  auto* paramDecl = parseParameterDecl();
  std::vector<ast::PauliComponentStmt*> body;
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
  return new (ctx) ast::ChannelStmt(
    attr, name, paramDecl, ctx.createSpan(body.data(), body.size()));
}

std::ostream& ast::ChannelStmt::print(std::ostream& os) const {
  os << "Channel ";
  if (attr != nullptr)
    attr->print(os);
  os << name;
  if (paramDecl != nullptr)
    paramDecl->print(os);
  os << "{\n";
  for (const auto* stmt : body) {
    stmt->print(os << "  ");
    os << "\n";
  }
  return os << "}";
} // ChannelStmt::print
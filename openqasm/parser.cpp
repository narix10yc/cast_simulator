#include "openqasm/parser.h"

using namespace openqasm;

std::unique_ptr<ast::RootNode> Parser::parse() {
  auto root = std::make_unique<ast::RootNode>();

  if (!lexer->openFile()) {
    logError("Unable to open file");
    return nullptr;
  }
  nextToken();

  while (curToken.type != TokenTy::Eof) {
    logDebug(1, "Ready to PARSE!");
    auto stmt = parseStmt();
    if (!stmt) {
      logError("Failed to parse stmt");
      return nullptr;
    }
    logDebug(1, "Parsed Stmt " + stmt->toString());
    root->addStmt(std::move(stmt));
  }

  logDebug(1, "Reached the end of file");
  lexer->closeFile();
  return root;
}

std::unique_ptr<ast::Statement> Parser::parseStmt() {
  std::unique_ptr<ast::Statement> stmt;
  skipSeparators();

  // skip include stmt
  if (curToken.type == TokenTy::Include) {
    nextToken();
    while (curToken.type != TokenTy::LineFeed)
      nextToken();
    skipSeparators();
  }

  if (curToken.type == TokenTy::If) {
    stmt = parseIfThenElseStmt();
  } else if (curToken.type == TokenTy::Openqasm) {
    logDebug(1, "ready to parse openqasm");
    stmt = parseVersionStmt();
  } else if (curToken.type == TokenTy::Qreg) {
    logDebug(1, "ready to parse qreg");
    stmt = parseQRegStmt();
  } else if (curToken.type == TokenTy::Creg) {
    logDebug(1, "ready to parse creg");
    stmt = parseCRegStmt();
  } else if (curToken.type == TokenTy::Identifier) {
    logDebug(1, "ready to parse gate apply");
    stmt = parseGateApplyStmt();
  } else {
    logError("Unknown token type: " + tokenTypetoString(curToken.type));
  }

  skipSeparators();
  return stmt;
}

std::unique_ptr<ast::VersionStmt> Parser::parseVersionStmt() {
  nextToken();
  if (curToken.type != TokenTy::Numeric) {
    logError("Expect version string to be a numerics");
    return nullptr;
  }
  auto value = curToken.str;
  nextToken();
  return std::make_unique<ast::VersionStmt>(value);
}

std::unique_ptr<ast::QRegStmt> Parser::parseQRegStmt() {
  if (!expectNextToken(TokenTy::Identifier))
    return nullptr;

  auto name = curToken.str;
  if (!expectNextToken(TokenTy::L_SquareBracket))
    return nullptr;
  if (!expectNextToken(TokenTy::Numeric))
    return nullptr;

  int size = std::stoi(curToken.str);
  if (!expectNextToken(TokenTy::R_SquareBracket))
    return nullptr;

  auto stmt = std::make_unique<ast::QRegStmt>(name, size);
  nextToken();
  return stmt;
}

std::unique_ptr<ast::CRegStmt> Parser::parseCRegStmt() {
  if (!expectNextToken(TokenTy::Identifier))
    return nullptr;

  auto name = curToken.str;
  if (!expectNextToken(TokenTy::L_SquareBracket))
    return nullptr;
  if (!expectNextToken(TokenTy::Numeric))
    return nullptr;

  int size = std::stoi(curToken.str);
  if (!expectNextToken(TokenTy::R_SquareBracket))
    return nullptr;
  nextToken();

  auto stmt = std::make_unique<ast::CRegStmt>(name, size);
  nextToken();
  return stmt;
}

std::unique_ptr<ast::GateApplyStmt> Parser::parseGateApplyStmt() {
  auto stmt = std::make_unique<ast::GateApplyStmt>(curToken.str);
  nextToken(); // eat the identifier
  logDebug(2, "parsing GateApplyStmt " + stmt->name);
  if (curToken.type == TokenTy::L_RoundBracket) {
    // parameters
    nextToken(); // eat '('
    while (true) {
      if (curToken.type == TokenTy::R_RoundBracket) {
        nextToken();
        break;
      }
      if (curToken.type == TokenTy::LineFeed ||
          curToken.type == TokenTy::CarriageReturn) {
        logError("Expect ')'");
        return nullptr;
      }
      if (curToken.type == TokenTy::Comma) {
        nextToken();
        continue;
      }
      auto param = parseExpr();

      if (!param) {
        logError("GateApply: failed to parse parameter");
        return nullptr;
      }
      stmt->addParameter(std::move(param));
    }
  }
  logDebug(2, "GateApplyStmt " + stmt->name + ": " +
                  std::to_string(stmt->parameters.size()) +
                  " parameters parsed");

  logDebug(1, "Current Token: " + tokenTypetoString(curToken.type));
  while (true) {
    if (curToken.type == TokenTy::Semicolon ||
        curToken.type == TokenTy::LineFeed ||
        curToken.type == TokenTy::CarriageReturn ||
        curToken.type == TokenTy::Eof)
      break;
    if (curToken.type == TokenTy::Comma) {
      nextToken();
      continue;
    }
    auto targ = parseSubscriptExpr();
    if (!targ) {
      logError("GateApply: failed to parse target");
      return nullptr;
    }
    stmt->addTarget(std::move(targ));
  }

  logDebug(2, std::to_string(stmt->targets.size()) + " targets parsed");

  return stmt;
}

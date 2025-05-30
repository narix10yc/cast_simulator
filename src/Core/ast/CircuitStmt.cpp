#include "cast/Core/AST/Parser.h"

using namespace cast::ast;

CircuitStmt* Parser::parseCircuitStmt() {
  assert(curToken.is(tk_Circuit) &&
         "parseCircuitStmt expects to be called with a 'Circuit' token");
  advance(tk_Circuit);

  // circuit attributes
  CircuitAttribute attr;
  if (optionalAdvance(tk_Less)) {
    std::string key;
    while (true) {
      if (curToken.is(tk_Greater)) {
        break;
      }
      requireCurTokenIs(tk_Identifier, "Expect a key");
      key = curToken.toString();
      advance(tk_Identifier);
      requireCurTokenIs(tk_Equal, "Expect '=' after key");
      advance(tk_Equal);
      if (key == "nqubits") {
        if (!curToken.convertibleToInt()) {
          logErrHere("'nqubits' must be an integer");
          failAndExit();
        }
        attr.nQubits = curToken.toInt();
        advance(tk_Numeric);
      }
      else if (key == "nparams") {
        if (!curToken.convertibleToInt()) {
          logErrHere("'nparams' must be an integer");
          failAndExit();
        }
        attr.nParams = curToken.toInt();
        advance(tk_Numeric);
      }
      else if (key == "phase") {
        attr.phase = parseExpr();
        if (attr.phase == nullptr) {
          logErrHere("Expect a phase expression");
          failAndExit();
        }
      }
      else if (key == "noise") {
        attr.noise = parseExpr();
        if (attr.noise == nullptr) {
          logErrHere("Expect a noise expression");
          failAndExit();
        }
      }
      else {
        logErrHere("Unknown attribute key");
        failAndExit();
      }
      key.clear();
      if (curToken.is(tk_Comma))
        advance(tk_Comma);
    } // end of while loop
    advance(tk_Greater);
  }

  // name and parameter declaration
  requireCurTokenIs(tk_Identifier, "Expect a circuit name");
  auto name = ctx.createIdentifier(curToken.toStringView());
  auto nameLoc = curToken.loc;
  advance(tk_Identifier);
  auto* paramDecl = parseParameterDecl();

  // circuit body
  requireCurTokenIs(tk_L_CurlyBracket, "Expect '{' to start circuit body");
  advance(tk_L_CurlyBracket);
  pushScope();
  llvm::SmallVector<Stmt*> body;
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

  auto* circuit = new (ctx) CircuitStmt(
    name,
    nameLoc,
    paramDecl,
    attr,
    ctx.createSpan(body.data(), body.size())
  );

  circuit->updateAttribute();
  return circuit;
}

void CircuitStmt::updateAttribute() {
  // Not implemented yet
}

std::ostream& CircuitStmt::print(std::ostream& os) const {
  os << "Circuit";
  bool hasAttr = attr.isInited();
  if (hasAttr) {
    os << "<";
    bool needComma = false;
    if (attr.nQubits != -1) {
      os << "nqubits=" << attr.nQubits;
      needComma = true;
    }
    if (attr.nParams != -1) {
      if (needComma)
        os << ",";
      os << "nparams=" << attr.nParams;
      needComma = true;
    }
    if (attr.phase != nullptr) {
      if (needComma)
        os << ",";
      attr.phase->print(os << "phase=");
      needComma = true;
    }
    if (attr.noise != nullptr) {
      if (needComma)
        os << ",";
      attr.noise->print(os << "noise=");
      needComma = true;
    }
    os << ">";
  }
  os << " " << name << " {\n";
  for (const auto& stmt : body)
    stmt->print(os << "  ") << "\n";
  os << "}\n";
  return os;
}

void CircuitStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = body.size();
  p.write(indent) << getKindName() << "(" << name << "): "
                  << size << " stmts\n";
  int val = size;
  if (attr.phase != nullptr)
    val++;
  if (attr.noise != nullptr)
    val++;
  p.setState(indent, val);
  if (attr.phase != nullptr) {
    p.setPrefix("phase: ");
    attr.phase->prettyPrint(p, indent + 1);
  }
  if (attr.noise != nullptr) {
    p.setPrefix("noise: ");
    attr.noise->prettyPrint(p, indent + 1);
  }
  for (unsigned i = 0; i < size; ++i)
    body[i]->prettyPrint(p, indent + 1);
}

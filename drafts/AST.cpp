#include "new_parser/Parser.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "cast/CircuitGraph.h"

using namespace cast::draft;

/*
    NK_Stmt,
      NK_Stmt_GateApply,
      NK_Stmt_GateChain,
      NK_Stmt_GateBlock,
      NK_Stmt_Measure,
      NK_Stmt_If,
      NK_Stmt_Repeat,
      NK_Stmt_Circuit,
      NK_Stmt_PauliComponent,
      NK_Stmt_Channel,
      _NK_Stmt_End,
    NK_Expr,
      NK_Expr_Identifier,
      NK_Expr_ParameterDecl,
      NK_Expr_Call,
      NK_Expr_SimpleNumeric,
        NK_Expr_IntegerLiteral,
        NK_Expr_FloatingLiteral,
        NK_Expr_FractionLiteral,
        NK_Expr_FractionPiLiteral,
        _NK_Expr_SimpleNumeric_End,
      NK_Expr_Measure,
      NK_Expr_All,
      NK_Expr_Parameter,
      NK_Expr_BinaryOp,
      NK_Expr_MinusOp,
      _NK_Expr_End,
    NK_Root
*/
std::string ast::Node::_getKindName(ast::Node::NodeKind k) {
  switch (k) {
    case NK_Stmt_GateApply: return "GateApplyStmt";
    case NK_Stmt_GateChain: return "GateChainStmt";
    case NK_Stmt_GateBlock: return "GateBlockStmt";
    case NK_Stmt_Measure: return "MeasureStmt";
    case NK_Stmt_If: return "IfStmt";
    case NK_Stmt_Repeat: return "RepeatStmt";
    case NK_Stmt_Circuit: return "CircuitStmt";
    case NK_Stmt_PauliComponent: return "PauliComponentStmt";
    case NK_Stmt_Channel: return "ChannelStmt";
    case NK_Expr_Identifier: return "IdentifierExpr";
    case NK_Expr_ParameterDecl: return "ParameterDeclExpr";
    case NK_Expr_Call: return "CallExpr";
    case NK_Expr_SimpleNumeric: return "SimpleNumericExpr";
    case NK_Expr_IntegerLiteral: return "IntegerLiteral";
    case NK_Expr_FloatingLiteral: return "FloatingLiteral";
    case NK_Expr_FractionLiteral: return "FractionLiteral";
    case NK_Expr_FractionPiLiteral: return "FractionPiLiteral";
    case NK_Expr_Measure: return "MeasureExpr";
    case NK_Expr_All: return "AllExpr";
    case NK_Expr_Parameter: return "ParameterExpr";
    case NK_Expr_BinaryOp: return "BinaryOpExpr";
    case NK_Expr_MinusOp: return "MinusOpExpr";
    case NK_Root: return "RootNode";
    default:
      return "<KindName>";
  }
}

void ast::CircuitStmt::toCircuitGraph(cast::CircuitGraph& graph) const {
  assert(false && "Not implemented yet");
}

std::ostream& ast::Attribute::print(std::ostream& os) const {
  os << "<";
  bool needComma = false;
  if (nQubits != nullptr) {
    nQubits->print(os << "nqubits=");
    needComma = true;
  }
  if (nParams != nullptr) {
    if (needComma)
      os << ", ";
    nParams->print(os << "nparams=");
    needComma = true;
  }
  if (phase != nullptr) {
    if (needComma)
      os << ", ";
    phase->print(os << "phase=");
  }
  return os << ">";
}

std::ostream& ast::ParameterDeclExpr::print(std::ostream& os) const {
  os << "(";
  utils::printSpanWithPrinterNoBracket(
    os, parameters, 
    [](std::ostream& os, ast::IdentifierExpr* param) {
      param->print(os);
    }
  );
  return os << ") ";
}

ast::CircuitStmt* ast::RootNode::lookupCircuit(const std::string& name) {
  for (auto& stmt : stmts) {
    if (auto* circuit = llvm::dyn_cast<CircuitStmt>(stmt)) {
      if (name.empty() || circuit->name.str == name)
        return circuit;
    }
  }
  return nullptr;
}
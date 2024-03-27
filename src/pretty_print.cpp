#include "ast.h"

using namespace ast;

#define INDENT f << std::string(2*depth, ' ')
#define INDENT2 f << std::string(2*depth + 2, ' ')

void RootNode::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "RootNode" << "\n";
    INDENT2 << "(Statements:)" << "\n";
    for (const auto& item : stmts) {
        item->prettyPrint(f, depth+1);
    }
}

void IfThenElseStmt::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "IfThenElse" << "\n";
    INDENT2 << "(If:)" << "\n";
    ifExpr->prettyPrint(f, depth+1);
    INDENT2 << "(Then:)" << "\n";
    for (const auto& item : thenBody) {
        item->prettyPrint(f, depth+1);
    }
    INDENT2 << "(Else:)" << "\n";
    for (const auto& item : elseBody) {
        item->prettyPrint(f, depth+1);
    }
}

void VersionStmt::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "Version: " << getVersion() << "\n";
}

void QRegStmt::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "QReg: " << getName() << "[" << getSize() << "]\n";
}

void CRegStmt::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "CReg: " << getName() << "[" << getSize() << "]\n";
}

void NumericExpr::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "Numeric[" << value << "]";
    f << "\n";
}

void VariableExpr::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "Variable[" << name << "]";
    f << "\n";
}

void UnaryExpr::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "UnaryExpr" << "\n";
}

void BinaryExpr::prettyPrint(std::ofstream &f, int depth) const {
    INDENT << "BinaryExpr" << "\n";
    INDENT2 << "(Op:) " << "\n";
    INDENT2 << "(LHS:)" << "\n";
    lhs->prettyPrint(f, depth+1);
    INDENT2 << "(RHS:)" << "\n";
    rhs->prettyPrint(f, depth+1);
}


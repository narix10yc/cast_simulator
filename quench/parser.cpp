#include "quench/parser.h"
#include <cassert>

using namespace quench;
using namespace quench::ast;
using namespace quench::cas;

int Parser::readLine() {
    if (file.eof()) {
        file.close();
        return -1;
    }
    tokenVec.clear();
    do {
        std::getline(file, currentLine);
        lineNumber++;
        lineLength = currentLine.size();
    } while (!file.eof() && lineLength == 0);

    if (file.eof()) {
        file.close();
        return -1;
    }

    int col = 0;
    while (col < lineLength) {
        tokenVec.push_back(parseToken(col));
        col = tokenVec.back().colEnd;
    }
    tokenIt = tokenVec.cbegin();

    std::cerr << Color::CYAN_FG << lineNumber << " | " << currentLine << "\n";
    for (auto it = tokenVec.cbegin(); it != tokenVec.cend(); it++)
        std::cerr << "col " << it->colStart << "-" << it->colEnd << "  " << (*it) << "\n";
    std::cerr << Color::RESET;

    return tokenVec.size();
}

Token Parser::parseToken(int col) {
    if (col >= lineLength)
        // end of line
        return Token(TokenTy::EndOfLine, "", lineLength, lineLength+1);

    int curCol = col;
    char c = currentLine[curCol];
    while (c == ' ')
        c = currentLine[++curCol];
    int colStart = curCol;

    if (std::isdigit(c) || c == '.') {
        // numeric
        std::string str;
        while (true) {
            if (std::isdigit(c) || c == '.') {
                str += c;
                c = currentLine[++curCol];
                continue;
            }
            break;
        }
        return Token(TokenTy::Numeric, str, colStart, curCol);
    }
    if (std::isalpha(c)) {
        // identifier
        std::string str;
        while (true) {
            if (std::isalnum(c) || c == '_') {
                str += c;
                c = currentLine[++curCol];
                continue;
            }
            break;
        }
        if (str == "circuit")
            return Token(TokenTy::Circuit, "", colStart, curCol);
        else 
            return Token(TokenTy::Identifier, str, colStart, curCol);
    }

    char cnext = currentLine[curCol+1];
    // std::cerr << "next is " << next << "\n";
    switch (c) {
    // operators
    case '+':
        return Token(TokenTy::Add, "", colStart, colStart+1);
    case '-':
        return Token(TokenTy::Sub, "", colStart, colStart+1);
    case '*': // '**' or '*/' or '*'
        if (cnext == '*')
            return Token(TokenTy::Sub, "", colStart, colStart+2);
        if (cnext == '/')
            return Token(TokenTy::CommentEnd, "", colStart, colStart+2);
        return Token(TokenTy::Mul, "", colStart, colStart+1);
    case '/': // '//' or '/*' or '/'
        if (cnext == '/')
            return Token(TokenTy::Comment, "", colStart, colStart+2);
        if (cnext == '*')
            return Token(TokenTy::CommentStart, "", colStart, colStart+2);
        return Token(TokenTy::Div, "", colStart, colStart+1);
    case '=': // '==' or '='
        if (cnext == '=')
            return Token(TokenTy::EqualEqual, "", colStart, colStart+2);
        return Token(TokenTy::Equal, "", colStart, colStart+1);
    case '>': // '>=' or '>'
        if (cnext == '=')
            return Token(TokenTy::GreaterEqual, "", colStart, colStart+2);
        return Token(TokenTy::Greater, "", colStart, colStart+1);
    case '<': // '<=' or '<'
        if (cnext == '=')
            return Token(TokenTy::LessEqual, "", colStart, colStart+2);
        return Token(TokenTy::Less, "", colStart, colStart+1);
    // symbols
    case ',':
        return Token(TokenTy::Comma, "", colStart, colStart+1);
    case ';':
        return Token(TokenTy::Semicolon, "", colStart, colStart+1);
    case '(':
        return Token(TokenTy::L_RoundBraket, "", colStart, colStart+1);
    case ')':
        return Token(TokenTy::R_RoundBraket, "", colStart, colStart+1);
    case '[':
        return Token(TokenTy::L_SquareBraket, "", colStart, colStart+1);
    case ']':
        return Token(TokenTy::R_SquareBraket, "", colStart, colStart+1);
    case '{':
        return Token(TokenTy::L_CurlyBraket, "", colStart, colStart+1);
    case '}':
        return Token(TokenTy::R_CurlyBraket, "", colStart, colStart+1);
    case '\'':
        return Token(TokenTy::SingleQuote, "", colStart, colStart+1);
    case '\"':
        return Token(TokenTy::DoubleQuote, "", colStart, colStart+1);
    case '@':
        return Token(TokenTy::AtSymbol, "", colStart, colStart+1);
    case '%':
        return Token(TokenTy::Percent, "", colStart, colStart+1);
    case '#':
        return Token(TokenTy::Hash, "", colStart, colStart+1);
    case '\\':
        return Token(TokenTy::Backslash, "", colStart, colStart+1);
    case '\n':
        assert(false && "parsed LineFeed Token?");
        return Token(TokenTy::LineFeed, "", colStart, colStart+1);
    default:
        throwParserError("Unknown char " + std::to_string(c));
        assert(false && "Unknown char");
        return Token(TokenTy::Unknown, "", colStart, colStart+1);
    }
}

RootNode* Parser::parse() {
    readLine();
    auto* root = new RootNode();
    while (true) {
    // parse circuit
    if (tokenIt->type == TokenTy::Circuit) {
        displayParserLog("ready to parse circuit");
        proceedWithType(TokenTy::Identifier);
        root->circuit.name = tokenIt->str;
        proceedWithType(TokenTy::L_CurlyBraket, true);

        while (true) {
            proceed();
            if (tokenIt->type == TokenTy::Identifier) {
                GateChainStmt chain;
                while (true) {
                    chain.gates.push_back(_parseGateApply());
                    proceed();
                    if (tokenIt->type == TokenTy::AtSymbol) {
                        proceed();
                        continue;
                    }
                    if (tokenIt->type == TokenTy::Semicolon)
                        break;
                    throwParserError("Unexpected token type " + TokenTyToString(tokenIt->type)
                                    + " when expecting either AtSymbol or Semicolon");
                }
                root->circuit.addGateChain(chain);
                continue;
            }
            break;
        } 
        
        if (tokenIt->type != TokenTy::R_CurlyBraket) {
            throwParserError("Unexpected token " + tokenIt->to_string());
        }
        proceed(); // eat '}'
        displayParserLog("Parsed a circuit with " + std::to_string(root->circuit.stmts.size()) + " chains");
        continue;
    }
    if (tokenIt->type == TokenTy::Hash) {
        root->paramDefs.push_back(_parseParameterDefStmt(root->casContext));
    }
    
    break;
    }
    return root;
}

quench::quantum_gate::GateParameter Parser::_parseGateParameter() {
    if (tokenIt->type == TokenTy::Percent) {
        proceedWithType(TokenTy::Numeric);
        int i = convertCurTokenToInt();
        proceed();
        return { "%" + std::to_string(i) };
    }
    return { _parseComplexNumber() };
}

GateApplyStmt Parser::_parseGateApply() {
    assert(tokenIt->type == TokenTy::Identifier);
    GateApplyStmt gate(tokenIt->str);

    if (optionalProceedWithType(TokenTy::L_RoundBraket)) {
        if (optionalProceedWithType(TokenTy::Hash)) {
            proceedWithType(TokenTy::Numeric);
            gate.paramRefNumber = convertCurTokenToInt();
            proceedWithType(TokenTy::R_RoundBraket);
        }
        else {
            proceed(); // eat '('
            while (true) {
                gate.params.push_back(_parseGateParameter());
                if (tokenIt->type == TokenTy::Comma) {
                    proceed();
                    continue;
                }
                if (tokenIt->type == TokenTy::Numeric || tokenIt->type == TokenTy::Percent)
                    continue;
                if (tokenIt->type == TokenTy::R_RoundBraket)
                    break;
                throwParserError("Unexpected token " + TokenTyToString(tokenIt->type));
            }
        }
    }

    // parse target qubits
    while (true) {
        if (optionalProceedWithType(TokenTy::Numeric)) {
            gate.qubits.push_back(convertCurTokenToInt());
            optionalProceedWithType(TokenTy::Comma);
            continue;
        }
        break;
    }
    
    if (gate.qubits.empty())
        throwParserError("Gate " + gate.name + " has no target");
    
    displayParserLog("Parsed gate " + gate.name + " with " +
                     std::to_string(gate.qubits.size()) + " targets");
    return gate;
}

ParameterDefStmt Parser::_parseParameterDefStmt(cas::Context& casContext) {
    assert(tokenIt->type == TokenTy::Hash);

    proceedWithType(TokenTy::Numeric);
    ParameterDefStmt def(convertCurTokenToInt());
    displayParserLog("Ready to parse ParameterDef #" + std::to_string(def.refNumber));

    proceedWithType(TokenTy::Equal);
    proceedWithType(TokenTy::L_CurlyBraket);
    proceed();

    std::vector<cas::Polynomial> polyMatrix;
    while (true) {
        polyMatrix.push_back(_parsePolynomial(casContext));
        if (tokenIt->type == TokenTy::Comma) {
            proceed();
        }
        if (tokenIt->type == TokenTy::R_CurlyBraket) {
            proceed();
            break;
        }
    }

    return def;
}

quench::cas::Polynomial Parser::_parsePolynomial(cas::Context& casContext) {
    const auto parseExponent = [&]() -> int {
        if (optionalProceedWithType(TokenTy::Pow)) {
            proceedWithType(TokenTy::Numeric);
            return convertCurTokenToInt();
        }
        return 1;
    };

    const auto parseAtom = [&]() -> cas::CASNode* {
        if (tokenIt->type == TokenTy::Percent) {
            proceedWithType(TokenTy::Numeric);
            return casContext.getVar("%" + std::to_string(convertCurTokenToInt()));
        }
        if (tokenIt->type == TokenTy::L_RoundBraket) {
            proceedWithType(TokenTy::Percent);
            proceedWithType(TokenTy::Numeric);
            auto* varLHS = casContext.getVar("%" + std::to_string(convertCurTokenToInt()));
            proceedWithType(TokenTy::Add);
            proceedWithType(TokenTy::Percent);
            proceedWithType(TokenTy::Numeric);
            auto* varRHS = casContext.getVar("%" + std::to_string(convertCurTokenToInt()));
            auto* varAdd = casContext.createAddNode(varLHS, varRHS);
            proceedWithType(TokenTy::R_RoundBraket);
            return varAdd;
        }
        return nullptr;
    };

    const auto parseCoef = [&]() -> std::complex<double> {
        if (tokenIt->type == TokenTy::Percent)
            return { 1.0, 0.0 };     
        if (tokenIt->type == TokenTy::Identifier && tokenIt->str != "i")
            return { 1.0, 0.0 };
        return _parseComplexNumber();
    };

    /// Before: at the first token;
    /// After: exit the last token
    const auto parseMonomial = [&]() -> cas::Polynomial::monomial_t {
        Polynomial::monomial_t monomial;
        monomial.coef = parseCoef();
        while (true) {
            if (tokenIt->type == TokenTy::Mul)
                proceed();

            if (tokenIt->type == TokenTy::Identifier) {
                int flag;
                if (tokenIt->str == "cos")
                    flag = 1;
                else if (tokenIt->str == "sin")
                    flag = 2;
                else if (tokenIt->str == "cexp")
                    flag = 3;
                else
                    throwParserError("Available operators are 'cos', 'sin', or 'cexp'");
                
                proceed();
                auto* atom = parseAtom();
                if (atom == nullptr)
                    throwParserError("Expect an atom");
                int exponent = parseExponent();
                if (flag == 1)
                    monomial.powers.push_back({casContext.createCosNode(atom), exponent});
                else if (flag == 2)
                    monomial.powers.push_back({casContext.createSinNode(atom), exponent});
                else if (flag == 3)
                    monomial.powers.push_back({casContext.createCompExpNode(atom), exponent});
                proceed();
                continue;
            }
            if (auto* atom = parseAtom()) {
                monomial.powers.push_back({atom, parseExponent()});
                proceed();
                continue;
            }
            break;
        }
        return monomial;
    };

    Polynomial poly;
    while (true) {
        poly.insertMonomial(parseMonomial());
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    displayParserLog("Parsed polynomial " + poly.str());
    return poly;
}

std::complex<double> Parser::_parseComplexNumber() {
    double m = 1.0;
    while (true) {
        if (tokenIt->type == TokenTy::Sub) {
            proceed();
            m *= -1.0;
            continue;
        }
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    double real = m;
    if (tokenIt->type == TokenTy::Numeric) {
        real *= convertCurTokenToFloat();
        proceed();
    }
    else if (tokenIt->type == TokenTy::Identifier) {
        if (tokenIt->str == "i") {
            proceed();
            return { 0.0, real };
        }
        throwParserError("Expect purely imaginary number to end with 'i'");
    }
    // just one part (pure real or pure imag)
    if (tokenIt->type != TokenTy::Add && tokenIt->type != TokenTy::Sub)
        return { real, 0.0 };

    m = 1.0;
    while (true) {
        if (tokenIt->type == TokenTy::Sub) {
            proceed();
            m *= -1.0;
            continue;
        }
        if (tokenIt->type == TokenTy::Add) {
            proceed();
            continue;
        }
        break;
    }
    double imag = m * convertCurTokenToFloat();
    proceed();
    if (!(tokenIt->type == TokenTy::Identifier && tokenIt->str == "i"))
        throwParserError("Expect complex number to end with 'i'");

    proceed();
    return { real, imag };
}

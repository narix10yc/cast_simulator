#include "cast/Core/Polynomial.h"
#include <algorithm>
#include <cassert>

using namespace cast;

std::ostream& VariableSumNode::print(std::ostream& os) const {
  assert(!vars.empty() || constant != 0.0);

  if (vars.empty()) {
    if (op == None)
      return os << constant;
    if (op == CosOp)
      return os << "cos(" << constant << ")";
    assert(op == SinOp);
    return os << "sin(" << constant << ")";
  }

  if (op == CosOp)
    os << "cos";
  else if (op == SinOp)
    os << "sin";

  if (vars.size() == 1) {
    if (constant == 0.0)
      return os << "%" << vars[0];
    return os << "(" << "%" << vars[0] << "+" << constant << ")";
  }

  os << "(";
  for (unsigned i = 0; i < vars.size() - 1; i++)
    os << "%" << vars[i] << "+";
  os << "%" << vars.back();

  if (constant != 0.0)
    os << "+" << constant;

  return os << ")";
}

std::ostream& Monomial::print(std::ostream& os) const {
  // coef
  bool coefFlag = (coef.real() != 0.0 && coef.imag() != 0.0);
  bool mulSign = true;
  if (coefFlag)
    os << "(";
  if (coef.real() == 0.0 && coef.imag() == 0.0)
    os << "0.0";
  else if (coef.imag() == 0.0) {
    if (coef.real() == 1.0) {
      if (_mulTerms.empty() && _expiVars.empty())
        return os << "1.0";
      mulSign = false;
    } else if (coef.real() == -1.0) {
      os << "-";
      mulSign = false;
    } else
      os << coef.real();
  } else if (coef.real() == 0.0) {
    if (coef.imag() == 1.0)
      os << "i";
    else if (coef.imag() == -1.0)
      os << "-i";
    else
      os << coef.imag() << "i";
  } else {
    os << coef.real();
    if (coef.imag() > 0.0)
      os << " + " << coef.imag() << "i";
    else
      os << " - " << -coef.imag() << "i";
  }
  if (coefFlag)
    os << ")";

  // mul terms
  if (!_mulTerms.empty()) {
    auto it = _mulTerms.cbegin();
    if (mulSign)
      os << "*";
    mulSign = true;
    it->print(os);
    while (++it != _mulTerms.cend())
      it->print(os << "*");
  }

  // expi terms
  if (!_expiVars.empty()) {
    if (mulSign)
      os << "*";
    auto it = _expiVars.cbegin();
    os << "expi(" << (it->isPlus ? "%" : "-%") << it->var;
    while (++it != _expiVars.cend())
      os << (it->isPlus ? "+%" : "-%") << it->var;
    os << ")";
  }

  return os;
}

std::ostream& Polynomial::print(std::ostream& os) const {
  if (_monomials.empty())
    return os << "0";
  _monomials[0].print(os);
  for (unsigned i = 1; i < _monomials.size(); i++)
    _monomials[i].print(os << " + ");

  return os;
}

int VariableSumNode::compare(const VariableSumNode& other) const {
  if (op < other.op)
    return -1;
  if (op > other.op)
    return +1;

  auto aSize = vars.size();
  auto bSize = other.vars.size();
  if (aSize < bSize)
    return -1;
  if (aSize > bSize)
    return +1;

  for (unsigned i = 0; i < aSize; i++) {
    if (vars[i] < other.vars[i])
      return -1;
    if (vars[i] > other.vars[i])
      return +1;
  }
  if (constant < other.constant)
    return -1;
  if (constant > other.constant)
    return +1;
  return 0;
}

bool VariableSumNode::operator==(const VariableSumNode& N) const {
  if (constant != N.constant)
    return false;
  if (op != N.op)
    return false;

  auto vSize = vars.size();
  if (vSize != N.vars.size())
    return false;

  for (size_t i = 0; i < vSize; i++) {
    if (vars[i] != N.vars[i])
      return false;
  }
  return true;
}

bool VariableSumNode::operator!=(const VariableSumNode& N) const {
  if (constant != N.constant)
    return true;
  if (op != N.op)
    return true;

  auto vSize = vars.size();
  if (vSize != N.vars.size())
    return true;

  for (size_t i = 0; i < vSize; i++) {
    if (vars[i] != N.vars[i])
      return true;
  }
  return false;
}

int Monomial::compare(const Monomial& other) const {
  size_t aSize, bSize;
  aSize = _mulTerms.size();
  bSize = other._mulTerms.size();
  if (aSize < bSize)
    return -1;
  if (aSize > bSize)
    return +1;

  aSize = _expiVars.size();
  bSize = other._expiVars.size();
  if (aSize < bSize)
    return -1;
  if (aSize > bSize)
    return +1;

  int c;
  for (unsigned i = 0; i < _mulTerms.size(); i++) {
    if ((c = _mulTerms[i].compare(other._mulTerms[i])) != 0)
      return c;
  }

  for (unsigned i = 0; i < _expiVars.size(); i++) {
    if (_expiVars[i] < other._expiVars[i])
      return -1;
    if (_expiVars[i] > other._expiVars[i])
      return +1;
  }
  return 0;
}

bool Monomial::mergeable(const Monomial& M) const {
  auto mSize = _mulTerms.size();
  if (mSize != M._mulTerms.size())
    return false;
  auto eSize = _expiVars.size();
  if (eSize != M._expiVars.size())
    return false;

  for (size_t i = 0; i < mSize; i++) {
    if (_mulTerms[i] != M._mulTerms[i])
      return false;
  }
  for (size_t i = 0; i < eSize; i++) {
    if (_expiVars[i] != M._expiVars[i])
      return false;
  }
  return true;
}

Polynomial& Polynomial::operator+=(const Monomial& M) {
  auto it = std::lower_bound(_monomials.begin(), _monomials.end(), M);
  if (it == _monomials.end()) {
    _monomials.push_back(M);
    return *this;
  }

  if (it->mergeable(M)) {
    it->coef += M.coef;
    return *this;
  }

  _monomials.insert(it, M);
  return *this;
}

Monomial& Monomial::operator*=(const Monomial& monomial) {
  coef *= monomial.coef;
  for (const auto& t : monomial._mulTerms)
    insertMulTerm(t);
  for (const auto& v : monomial._expiVars)
    insertExpiVar(v);
  return *this;
}

VariableSumNode& VariableSumNode::simplify(
    const std::vector<std::pair<int, double>>& varValues) {
  std::vector<int> updatedVars;
  for (const int var : vars) {
    auto it = std::ranges::find_if(varValues,
      [var](const std::pair<int, double>& p) { return p.first == var; });
    if (it == varValues.cend())
      updatedVars.push_back(var);
    else
      constant += it->second;
  }
  if (updatedVars.empty()) {
    vars.clear();
    if (op == CosOp) {
      constant = std::cos(constant);
      op = None;
    } else if (op == SinOp) {
      constant = std::sin(constant);
      op = None;
    }
  } else
    vars = std::move(updatedVars);

  return *this;
}

Monomial& 
Monomial::simplify(const std::vector<std::pair<int, double>>& varValues) {
  if (coef == std::complex<double>(0.0, 0.0)) {
    _mulTerms.clear();
    _expiVars.clear();
    return *this;
  }
  for (auto& M : _mulTerms)
    M.simplify(varValues);

  std::vector<VariableSumNode> updatedMulTerms;
  for (const auto& M : _mulTerms) {
    if (M.op == VariableSumNode::None && M.vars.empty())
      coef *= M.constant;
    else
      updatedMulTerms.push_back(M);
  }
  _mulTerms = std::move(updatedMulTerms);

  std::erase_if(_expiVars, [&varValues, this](const ExpiVar& E) {
    auto it = varValues.cbegin();
    while (true) {
      if (it->first == E.var) {
        coef *= std::complex<double>(
          std::cos(it->second),
          (E.isPlus) ? std::sin(it->second)
            : -std::sin(it->second));
        return true;
      }
      if (++it == varValues.cend())
        return false;
    }
  });
  return *this;
}

Polynomial& Polynomial::removeSmallMonomials(double thres) {
  std::erase_if(_monomials, [thres](const Monomial& M) {
    return std::abs(M.coef) < thres;
  });
  return *this;
}

Polynomial& 
Polynomial::simplifySelf(const std::vector<std::pair<int, double>>& varValues) {
  if (varValues.empty())
    return *this;
  for (auto& M : _monomials)
    M.simplify(varValues);

  std::complex<double> cons(0.0, 0.0);
  std::erase_if(_monomials, [&cons](const Monomial& M) {
      if (M.isConstant()) {
        cons += M.coef;
        return true;
      }
      return false;
    });

  if (cons != std::complex<double>(0.0, 0.0))
    return (*this) += Monomial::Constant(cons);
  return *this;
}

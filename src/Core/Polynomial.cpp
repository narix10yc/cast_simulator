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
  bool coefFlag = (coef_.real() != 0.0 && coef_.imag() != 0.0);
  bool mulSign = true;
  if (coefFlag)
    os << "(";
  if (coef_.real() == 0.0 && coef_.imag() == 0.0)
    os << "0.0";
  else if (coef_.imag() == 0.0) {
    if (coef_.real() == 1.0) {
      if (mulTerms_.empty() && expiVars_.empty())
        return os << "1.0";
      mulSign = false;
    } else if (coef_.real() == -1.0) {
      os << "-";
      mulSign = false;
    } else
      os << coef_.real();
  } else if (coef_.real() == 0.0) {
    if (coef_.imag() == 1.0)
      os << "i";
    else if (coef_.imag() == -1.0)
      os << "-i";
    else
      os << coef_.imag() << "i";
  } else {
    os << coef_.real();
    if (coef_.imag() > 0.0)
      os << " + " << coef_.imag() << "i";
    else
      os << " - " << -coef_.imag() << "i";
  }
  if (coefFlag)
    os << ")";

  // mul terms
  if (!mulTerms_.empty()) {
    auto it = mulTerms_.cbegin();
    if (mulSign)
      os << "*";
    mulSign = true;
    it->print(os);
    while (++it != mulTerms_.cend())
      it->print(os << "*");
  }

  // expi terms
  if (!expiVars_.empty()) {
    if (mulSign)
      os << "*";
    auto it = expiVars_.cbegin();
    os << "expi(" << (it->isPlus ? "%" : "-%") << it->var;
    while (++it != expiVars_.cend())
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
  aSize = mulTerms_.size();
  bSize = other.mulTerms_.size();
  if (aSize < bSize)
    return -1;
  if (aSize > bSize)
    return +1;

  aSize = expiVars_.size();
  bSize = other.expiVars_.size();
  if (aSize < bSize)
    return -1;
  if (aSize > bSize)
    return +1;

  int c;
  for (unsigned i = 0; i < mulTerms_.size(); i++) {
    if ((c = mulTerms_[i].compare(other.mulTerms_[i])) != 0)
      return c;
  }

  for (unsigned i = 0; i < expiVars_.size(); i++) {
    if (expiVars_[i] < other.expiVars_[i])
      return -1;
    if (expiVars_[i] > other.expiVars_[i])
      return +1;
  }
  return 0;
}

bool Monomial::mergeable(const Monomial& M) const {
  auto mSize = mulTerms_.size();
  if (mSize != M.mulTerms_.size())
    return false;
  auto eSize = expiVars_.size();
  if (eSize != M.expiVars_.size())
    return false;

  for (size_t i = 0; i < mSize; i++) {
    if (mulTerms_[i] != M.mulTerms_[i])
      return false;
  }
  for (size_t i = 0; i < eSize; i++) {
    if (expiVars_[i] != M.expiVars_[i])
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
    it->coef() += M.coef();
    return *this;
  }

  _monomials.insert(it, M);
  return *this;
}

Monomial& Monomial::operator*=(const Monomial& M) {
  coef_ *= M.coef();
  for (const auto& t : M.mulTerms_)
    insertMulTerm(t);
  for (const auto& v : M.expiVars_)
    insertExpiVar(v);
  return *this;
}

VariableSumNode& VariableSumNode::simplify(
    const std::vector<std::pair<int, double>>& varValues) {
  std::vector<int> updatedVars;
  for (const int var : vars) {
    auto it =
        std::ranges::find_if(varValues, [var](const std::pair<int, double>& p) {
          return p.first == var;
        });
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
  if (coef_ == std::complex<double>(0.0, 0.0)) {
    mulTerms_.clear();
    expiVars_.clear();
    return *this;
  }
  for (auto& M : mulTerms_)
    M.simplify(varValues);

  std::vector<VariableSumNode> updatedMulTerms;
  for (const auto& M : mulTerms_) {
    if (M.op == VariableSumNode::None && M.vars.empty())
      coef_ *= M.constant;
    else
      updatedMulTerms.push_back(M);
  }
  mulTerms_ = std::move(updatedMulTerms);

  std::erase_if(expiVars_, [&varValues, this](const ExpiVar& E) {
    auto it = varValues.cbegin();
    while (true) {
      if (it->first == E.var) {
        coef_ *= std::complex<double>(std::cos(it->second),
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
    return std::abs(M.coef()) < thres;
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
      cons += M.coef();
      return true;
    }
    return false;
  });

  if (cons != std::complex<double>(0.0, 0.0))
    return (*this) += Monomial::Constant(cons);
  return *this;
}

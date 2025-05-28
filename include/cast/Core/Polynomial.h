#ifndef CAST_CORE_POLYNOMIAL_H
#define CAST_CORE_POLYNOMIAL_H

#include <algorithm>
#include <complex>
#include <vector>

namespace cast {

class CasNode {
public:
  virtual std::ostream& print(std::ostream&) const = 0;
  virtual ~CasNode() = default;
};

/// @brief op(vars[0] + vars[1] + <other vars> + constant)
/// @param op one of None, CosOp, SinOp
class VariableSumNode : public CasNode {
public:
  enum Operator { None, CosOp, SinOp };

  double constant;
  std::vector<int> vars;
  Operator op;

  VariableSumNode(
      std::initializer_list<int> variables = {},
      double constant = 0.0, Operator op = None)
    : constant(constant), vars(), op(op) {
    for (const auto& v : variables)
      addVar(v);
  }

  static VariableSumNode Cosine(int var, double constant = 0.0) {
    return VariableSumNode({var}, constant, CosOp);
  }

  static VariableSumNode Cosine(std::initializer_list<int> vars,
                                double constant = 0.0) {
    return VariableSumNode(vars, constant, CosOp);
  }

  static VariableSumNode Sine(int var, double constant = 0.0) {
    return VariableSumNode({var}, constant, SinOp);
  }

  static VariableSumNode Sine(
      std::initializer_list<int> vars, double constant = 0.0) {
    return VariableSumNode(vars, constant, SinOp);
  }

  void addVar(int v) {
    auto it = std::ranges::lower_bound(vars, v);
    vars.insert(it, v);
  }

  int compare(const VariableSumNode&) const;
  bool operator<(const VariableSumNode& N) const { return compare(N) < 0; }
  bool operator>(const VariableSumNode& N) const { return compare(N) > 0; }

  bool operator==(const VariableSumNode&) const;
  bool operator!=(const VariableSumNode&) const;

  std::ostream& print(std::ostream& ) const override;

  VariableSumNode& 
  simplify(const std::vector<std::pair<int, double>>& varValues);
};

class Monomial : public CasNode {
public:
  class ExpiVar {
  public:
    int var;
    bool isPlus;
    ExpiVar(int var, bool isPlus = true) : var(var), isPlus(isPlus) {}

    bool operator<(const ExpiVar& E) const {
      if (var < E.var)
        return true;
      if (isPlus < E.isPlus)
        return true;
      return false;
    }

    bool operator>(const ExpiVar& E) const {
      if (var > E.var)
        return true;
      if (isPlus > E.isPlus)
        return true;
      return false;
    }

    bool operator==(const ExpiVar& E) const {
      return var == E.var && isPlus == E.isPlus;
    }

    bool operator!=(const ExpiVar& E) const {
      return var != E.var || isPlus != E.isPlus;
    }
  };

private:
  std::vector<VariableSumNode> _mulTerms;
  std::vector<ExpiVar> _expiVars;

public:
  std::complex<double> coef;

  Monomial() : coef(1.0, 0.0), _mulTerms(), _expiVars() {}

  Monomial(double re, double im) : coef(re, im), _mulTerms(), _expiVars() {}

  Monomial(const std::complex<double>& coef)
      : coef(coef), _mulTerms(), _expiVars() {}

  Monomial(const std::complex<double>& coef,
           const std::vector<VariableSumNode>& mulTerms,
           const std::vector<ExpiVar>& expiVars)
      : coef(coef), _mulTerms(), _expiVars() {
    for (const auto& M : mulTerms)
      insertMulTerm(M);
    for (const auto& V : expiVars)
      insertExpiVar(V);
  }

  static Monomial Constant(const std::complex<double>& v) {
    return Monomial(v, {}, {});
  }

  // cos(var)
  static Monomial Cosine(int var) {
    return Monomial({1.0, 0.0}, {VariableSumNode::Cosine(var)}, {});
  }

  // sin(var)
  static Monomial Sine(int var) {
    return Monomial({1.0, 0.0}, {VariableSumNode::Sine(var)}, {});
  }

  // expi(var)
  static Monomial Expi(int var) { return Monomial({1.0, 0.0}, {}, {var}); }

  bool isConstant() const { return _expiVars.empty() && _mulTerms.empty(); }
  int compare(const Monomial&) const;
  bool operator<(const Monomial& M) const { return compare(M) < 0; }
  bool operator>(const Monomial& M) const { return compare(M) > 0; }

  bool mergeable(const Monomial&) const;

  std::vector<VariableSumNode>& mulTerms() { return _mulTerms; }
  const std::vector<VariableSumNode>& mulTerms() const { return _mulTerms; }

  std::vector<ExpiVar>& expiVars() { return _expiVars; }
  const std::vector<ExpiVar>& expiVars() const { return _expiVars; }

  void insertMulTerm(const VariableSumNode& node) {
    if (_mulTerms.empty()) {
      _mulTerms.push_back(node);
      return;
    }
    auto it = std::lower_bound(_mulTerms.begin(), _mulTerms.end(), node);
    _mulTerms.insert(it, node);
  }

  void insertExpiVar(const ExpiVar& E) {
    if (_expiVars.empty()) {
      _expiVars.push_back(E);
      return;
    }
    auto it = std::upper_bound(
      _expiVars.begin(), _expiVars.end(), ExpiVar(E.var, true));
    --it;
    if (it->var == E.var && (it->isPlus ^ E.isPlus)) {
      _expiVars.erase(it);
      return;
    }

    _expiVars.insert(++it, E);
  }
  void insertExpiVar(int v, bool isPlus = true) {
    return insertExpiVar(ExpiVar(v, isPlus));
  }

  Monomial& operator*=(const std::complex<double>& v) {
    coef *= v;
    return *this;
  }

  Monomial& operator*=(const Monomial&);

  Monomial operator*(const std::complex<double>& v) const {
    return Monomial(*this) *= v;
  }

  Monomial operator*(const Monomial& M) const { return Monomial(*this) *= M; }

  std::ostream& print(std::ostream& ) const override;

  Monomial& simplify(const std::vector<std::pair<int, double>>& varValues);
};

class Polynomial : public CasNode {
  std::vector<Monomial> _monomials;

public:
  Polynomial() : _monomials() {}

  Polynomial(double re, double im) : _monomials({{re, im}}) {}

  Polynomial(const std::complex<double>& value) : _monomials({{value}}) {}

  Polynomial(const Monomial& M) : _monomials({M}) {}

  Polynomial(std::initializer_list<Monomial> Ms) : _monomials() {
    for (const auto& M : Ms)
      insertMonomial(M);
  }

  std::vector<Monomial>& monomials() { return _monomials; }

  const std::vector<Monomial>& monomials() const { return _monomials; }

  void insertMonomial(const Monomial& M) {
    auto it = std::lower_bound(_monomials.begin(), _monomials.end(), M);
    _monomials.insert(it, M);
  }

  static Polynomial Constant(const std::complex<double>& c) {
    Polynomial P;
    P.insertMonomial(Monomial(c));
    return P;
  }

  /// @return <isConstant, value>
  std::pair<bool, std::complex<double>> getValue() const {
    if (_monomials.size() == 1 && _monomials[0].mulTerms().empty() &&
        _monomials[0].expiVars().empty()) {
      return {true, _monomials[0].coef};
    }
    return {false, {0.0, 0.0}};
  }

  std::ostream& print(std::ostream& ) const override;

  /// @brief Remove Monomials whose coefficient is less than a given threshold
  /// @return updated* this
  Polynomial& removeSmallMonomials(double thres = 1e-8);

  Polynomial& simplifySelf(
    const std::vector<std::pair<int, double>>& varValues = {});

  Polynomial& operator+=(const Monomial&);

  Polynomial& operator+=(const Polynomial& P) {
    for (const auto& M : P._monomials)
      operator+=(M);
    return *this;
  }

  Polynomial operator+(const Polynomial& P) const {
    return Polynomial(*this) += P;
  }

  Polynomial& operator*=(const std::complex<double>& v) {
    for (auto& m : _monomials)
      m *= v;
    return *this;
  }

  Polynomial& operator*=(const Monomial& M) {
    auto size = _monomials.size();
    for (unsigned i = 0; i < size; i++)
      _monomials[i] *= M;
    return *this;
  }

  Polynomial& operator*=(const Polynomial& P) {
    for (const auto& M : P._monomials)
      operator*=(M);
    return *this;
  }

  Polynomial operator*(const std::complex<double>& v) const {
    return Polynomial(*this) *= v;
  }

  Polynomial operator*(const Monomial& M) const {
    return Polynomial(*this) *= M;
  }

  Polynomial operator*(const Polynomial& P) const {
    return Polynomial(*this) *= P;
  }
};

} // namespace cast

#endif // CAST_CORE_POLYNOMIAL_H
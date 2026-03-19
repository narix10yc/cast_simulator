use super::{Angle, Circuit, Gate};
use crate::types::Rational;
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Names and numeric literals
    Identifier(String), // foo, q, cx
    NumericsI(i64),     // 42
    NumericsF(f64),     // 3.14
    Pi,                 // pi

    // Delimiters
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    Comma,     // ,
    Semicolon, // ;

    // Arithmetic and assignment operators
    Plus,  // +
    Minus, // -
    Star,  // *
    Slash, // /
    Caret, // ^
    Equal, // =

    Eof,

    // Fallback for not-yet-supported single characters
    Char(char), // any unsupported single char
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Integer(i64),
    Float(f64),
    Pi,
    UnaryMinus(Box<Expr>),
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub fn parse_qasm(input: &str) -> anyhow::Result<Circuit> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_circuit()
}

/// A simple recursive-descent parser. `Parser` is an internal type. Users should use the function
/// `parse_qasm`.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    qreg_name: Option<String>,
    qreg_size: Option<u32>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            qreg_name: None,
            qreg_size: None,
        }
    }

    fn peek(&self) -> &Token {
        let idx = self.pos.min(self.tokens.len().saturating_sub(1));
        &self.tokens[idx]
    }

    fn bump(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos < self.tokens.len().saturating_sub(1) {
            self.pos += 1;
        }
        tok
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    /// Returns error if the next token is not the same variant as `expected`.
    fn expect(&mut self, expected: Token) -> anyhow::Result<()> {
        let found = self.peek().clone();
        if same_variant(&found, &expected) {
            self.bump();
            Ok(())
        } else {
            anyhow::bail!(
                "expected {}, found {}",
                token_label(&expected),
                token_label(&found)
            );
        }
    }

    fn parse_circuit(&mut self) -> anyhow::Result<Circuit> {
        let mut circuit = Circuit::new();
        while !self.is_eof() {
            if matches!(self.peek(), Token::Semicolon) {
                self.bump();
                continue;
            }
            if self.try_parse_preamble_or_declaration()? {
                continue;
            }
            let gate = self.parse_gate_statement()?;
            circuit.gates.push(gate);
        }
        if self.qreg_size.is_none() {
            eprintln!("warning: no qreg declaration found; qubit bounds are not checked");
        }
        self.expect(Token::Eof)?;
        Ok(circuit)
    }

    fn try_parse_preamble_or_declaration(&mut self) -> anyhow::Result<bool> {
        let Some(keyword) = self.peek_identifier().map(str::to_string) else {
            return Ok(false);
        };

        if keyword.eq_ignore_ascii_case("openqasm") {
            self.bump();
            self.consume_until_semicolon("OPENQASM header")?;
            return Ok(true);
        }
        if keyword.eq_ignore_ascii_case("include") {
            self.bump();
            self.consume_until_semicolon("include statement")?;
            return Ok(true);
        }
        if keyword.eq_ignore_ascii_case("qreg") {
            self.parse_qreg_declaration()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn parse_qreg_declaration(&mut self) -> anyhow::Result<()> {
        self.bump(); // `qreg`

        if self.qreg_size.is_some() {
            anyhow::bail!("multiple qreg declarations are not supported in this parser");
        }

        let name = self.expect_identifier()?;
        self.expect(Token::LBracket)?;
        let size = match self.peek() {
            Token::NumericsI(raw) => {
                let size = to_u32_index(*raw)?;
                self.bump();
                size
            }
            other => anyhow::bail!("expected integer qreg size, found {}", token_label(other)),
        };
        self.expect(Token::RBracket)?;
        self.expect(Token::Semicolon)?;

        self.qreg_name = Some(name);
        self.qreg_size = Some(size);
        Ok(())
    }

    fn consume_until_semicolon(&mut self, context: &str) -> anyhow::Result<()> {
        loop {
            match self.peek() {
                Token::Semicolon => {
                    self.bump();
                    return Ok(());
                }
                Token::Eof => {
                    anyhow::bail!("expected `;` to terminate {context}, found end-of-input")
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    fn peek_identifier(&self) -> Option<&str> {
        match self.peek() {
            Token::Identifier(name) => Some(name),
            _ => None,
        }
    }

    fn parse_gate_statement(&mut self) -> anyhow::Result<Gate> {
        let gate_name = self.expect_identifier()?;
        let gate = match gate_name.as_str() {
            "x" => Gate::X(self.parse_qubit_operand()?),
            "y" => Gate::Y(self.parse_qubit_operand()?),
            "z" => Gate::Z(self.parse_qubit_operand()?),
            "h" => Gate::H(self.parse_qubit_operand()?),
            "s" => Gate::S(self.parse_qubit_operand()?),
            "t" => Gate::T(self.parse_qubit_operand()?),
            "cx" => {
                let ctrl = self.parse_qubit_operand()?;
                self.expect(Token::Comma)?;
                let targ = self.parse_qubit_operand()?;
                Gate::CX(ctrl, targ)
            }
            "cz" => {
                let ctrl = self.parse_qubit_operand()?;
                self.expect(Token::Comma)?;
                let targ = self.parse_qubit_operand()?;
                Gate::CZ(ctrl, targ)
            }
            "swap" => {
                let q0 = self.parse_qubit_operand()?;
                self.expect(Token::Comma)?;
                let q1 = self.parse_qubit_operand()?;
                Gate::SWAP(q0, q1)
            }
            "ccx" => {
                let ctrl1 = self.parse_qubit_operand()?;
                self.expect(Token::Comma)?;
                let ctrl2 = self.parse_qubit_operand()?;
                self.expect(Token::Comma)?;
                let targ = self.parse_qubit_operand()?;
                Gate::CCX(ctrl1, ctrl2, targ)
            }
            "rx" => {
                let angle = self.parse_angle_argument()?;
                let q = self.parse_qubit_operand()?;
                Gate::RX(angle, q)
            }
            "ry" => {
                let angle = self.parse_angle_argument()?;
                let q = self.parse_qubit_operand()?;
                Gate::RY(angle, q)
            }
            "rz" => {
                let angle = self.parse_angle_argument()?;
                let q = self.parse_qubit_operand()?;
                Gate::RZ(angle, q)
            }
            "u3" => {
                self.expect(Token::LParen)?;
                let theta = expr_to_angle(self.parse_expression()?)?;
                self.expect(Token::Comma)?;
                let phi = expr_to_angle(self.parse_expression()?)?;
                self.expect(Token::Comma)?;
                let lambda = expr_to_angle(self.parse_expression()?)?;
                self.expect(Token::RParen)?;
                let q = self.parse_qubit_operand()?;
                Gate::U3(theta, phi, lambda, q)
            }
            _ => anyhow::bail!("unsupported gate `{gate_name}`"),
        };

        self.expect(Token::Semicolon)?;
        Ok(gate)
    }

    fn parse_angle_argument(&mut self) -> anyhow::Result<Angle> {
        self.expect(Token::LParen)?;
        let expr = self.parse_expression()?;
        self.expect(Token::RParen)?;
        expr_to_angle(expr)
    }

    fn parse_qubit_operand(&mut self) -> anyhow::Result<u32> {
        match self.peek() {
            Token::NumericsI(value) => {
                let raw = *value;
                self.bump();
                let idx = to_u32_index(raw)?;
                self.check_qreg_bounds(idx)?;
                Ok(idx)
            }
            Token::Identifier(_) => {
                let ident = self.expect_identifier()?;
                if matches!(self.peek(), Token::LBracket) {
                    self.bump();
                    let idx = match self.peek() {
                        Token::NumericsI(value) => {
                            let idx = to_u32_index(*value)?;
                            self.bump();
                            idx
                        }
                        other => {
                            anyhow::bail!(
                                "expected integer qubit index, found {}",
                                token_label(other)
                            )
                        }
                    };
                    self.expect(Token::RBracket)?;
                    if let Some(expected_name) = &self.qreg_name {
                        if ident != *expected_name {
                            anyhow::bail!("unknown qreg `{ident}`; expected `{expected_name}`");
                        }
                    }
                    self.check_qreg_bounds(idx)?;
                    Ok(idx)
                } else if let Some(idx) = trailing_digits_to_u32(&ident)? {
                    self.check_qreg_bounds(idx)?;
                    Ok(idx)
                } else {
                    anyhow::bail!("expected qubit like `q[0]` or `q0`, found identifier `{ident}`")
                }
            }
            other => anyhow::bail!("expected qubit operand, found {}", token_label(other)),
        }
    }

    fn check_qreg_bounds(&self, idx: u32) -> anyhow::Result<()> {
        if let Some(size) = self.qreg_size {
            if idx >= size {
                anyhow::bail!("qubit index `{idx}` is out of bounds for qreg size `{size}`");
            }
        }
        Ok(())
    }

    fn expect_identifier(&mut self) -> anyhow::Result<String> {
        match self.peek() {
            Token::Identifier(name) => {
                let out = name.clone();
                self.bump();
                Ok(out)
            }
            other => anyhow::bail!("expected identifier, found {}", token_label(other)),
        }
    }

    fn parse_expression(&mut self) -> anyhow::Result<Expr> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> anyhow::Result<Expr> {
        let mut lhs = self.parse_mul_div()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul_div()?;
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_mul_div(&mut self) -> anyhow::Result<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> anyhow::Result<Expr> {
        if matches!(self.peek(), Token::Minus) {
            self.bump();
            let expr = self.parse_unary()?;
            Ok(Expr::UnaryMinus(Box::new(expr)))
        } else {
            self.parse_power()
        }
    }

    fn parse_power(&mut self) -> anyhow::Result<Expr> {
        let base = self.parse_primary()?;
        if matches!(self.peek(), Token::Caret) {
            self.bump();
            let exponent = self.parse_unary()?;
            Ok(Expr::Binary {
                op: BinaryOp::Pow,
                left: Box::new(base),
                right: Box::new(exponent),
            })
        } else {
            Ok(base)
        }
    }

    fn parse_primary(&mut self) -> anyhow::Result<Expr> {
        match self.peek() {
            Token::NumericsI(value) => {
                let value = *value;
                self.bump();
                Ok(Expr::Integer(value))
            }
            Token::NumericsF(value) => {
                let value = *value;
                self.bump();
                Ok(Expr::Float(value))
            }
            Token::Pi => {
                self.bump();
                Ok(Expr::Pi)
            }
            Token::LParen => {
                self.bump();
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            other => anyhow::bail!("expected expression term, found {}", token_label(other)),
        }
    }
}

fn same_variant(lhs: &Token, rhs: &Token) -> bool {
    std::mem::discriminant(lhs) == std::mem::discriminant(rhs)
}

/// Used in error messages.
fn token_label(token: &Token) -> String {
    match token {
        Token::Identifier(name) => format!("identifier `{name}`"),
        Token::NumericsI(value) => format!("integer `{value}`"),
        Token::NumericsF(value) => format!("float `{value}`"),
        Token::Pi => "`pi`".to_string(),
        Token::LParen => "`(`".to_string(),
        Token::RParen => "`)`".to_string(),
        Token::LBracket => "`[`".to_string(),
        Token::RBracket => "`]`".to_string(),
        Token::Comma => "`,`".to_string(),
        Token::Semicolon => "`;`".to_string(),
        Token::Plus => "`+`".to_string(),
        Token::Minus => "`-`".to_string(),
        Token::Star => "`*`".to_string(),
        Token::Slash => "`/`".to_string(),
        Token::Caret => "`^`".to_string(),
        Token::Equal => "`=`".to_string(),
        Token::Eof => "<end-of-input>".to_string(),
        Token::Char(ch) => format!("unsupported character `{ch}`"),
    }
}

fn tokenize(input: &str) -> anyhow::Result<Vec<Token>> {
    let mut chars = input.chars().peekable();
    let mut tokens = Vec::new();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '/' => {
                chars.next();
                if matches!(chars.peek(), Some('/')) {
                    chars.next();
                    for c in chars.by_ref() {
                        if c == '\n' {
                            break;
                        }
                    }
                } else {
                    tokens.push(Token::Slash);
                }
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            '[' => {
                chars.next();
                tokens.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                tokens.push(Token::RBracket);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            ';' => {
                chars.next();
                tokens.push(Token::Semicolon);
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            '^' => {
                chars.next();
                tokens.push(Token::Caret);
            }
            '=' => {
                chars.next();
                tokens.push(Token::Equal);
            }
            '.' => {
                let mut lookahead = chars.clone();
                lookahead.next();
                if matches!(lookahead.peek(), Some(next) if next.is_ascii_digit()) {
                    let first = chars.next().expect("peeked '.' must exist");
                    tokens.push(lex_number(first, &mut chars)?);
                } else {
                    chars.next();
                    tokens.push(Token::Char('.'));
                }
            }
            _ if ch.is_ascii_digit() => {
                let first = chars.next().expect("peeked digit must exist");
                tokens.push(lex_number(first, &mut chars)?);
            }
            _ if is_ident_start(ch) => {
                let first = chars.next().expect("peeked identifier start must exist");
                tokens.push(lex_identifier_or_keyword(first, &mut chars));
            }
            _ => {
                chars.next();
                tokens.push(Token::Char(ch));
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

/// a-Z and underscore
fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

/// a-Z, 0-9, and underscore
fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn lex_identifier_or_keyword(first: char, chars: &mut Peekable<Chars<'_>>) -> Token {
    let mut ident = String::from(first);
    while let Some(&next) = chars.peek() {
        if !is_ident_continue(next) {
            break;
        }
        ident.push(next);
        chars.next();
    }

    if ident == "pi" {
        Token::Pi
    } else {
        Token::Identifier(ident)
    }
}

fn lex_number(first: char, chars: &mut Peekable<Chars<'_>>) -> anyhow::Result<Token> {
    let mut number = String::from(first);
    let mut has_dot = first == '.';
    let mut has_exponent = false;

    while let Some(&next) = chars.peek() {
        if next.is_ascii_digit() {
            number.push(next);
            chars.next();
            continue;
        }
        if next == '.' && !has_dot {
            has_dot = true;
            number.push(next);
            chars.next();
            continue;
        }
        break;
    }

    // scientific notation like `1e-6` or `2.5E+3`
    if matches!(chars.peek(), Some('e') | Some('E')) {
        let mut lookahead = chars.clone();
        lookahead.next(); // e/E
        if matches!(lookahead.peek(), Some('+') | Some('-')) {
            lookahead.next();
        }
        if !matches!(lookahead.peek(), Some(next) if next.is_ascii_digit()) {
            anyhow::bail!("invalid scientific literal `{number}`");
        }

        has_exponent = true;
        number.push(chars.next().expect("peeked exponent marker must exist"));
        if matches!(chars.peek(), Some('+') | Some('-')) {
            number.push(chars.next().expect("peeked exponent sign must exist"));
        }
        while let Some(&digit) = chars.peek() {
            if digit.is_ascii_digit() {
                number.push(digit);
                chars.next();
            } else {
                break;
            }
        }
    }

    if has_dot || has_exponent {
        let value = number
            .parse::<f64>()
            .map_err(|e| anyhow::anyhow!("invalid float literal `{number}`: {e}"))?;
        Ok(Token::NumericsF(value))
    } else {
        let value = number
            .parse::<i64>()
            .map_err(|e| anyhow::anyhow!("invalid integer literal `{number}`: {e}"))?;
        Ok(Token::NumericsI(value))
    }
}

fn expr_to_angle(expr: Expr) -> anyhow::Result<Angle> {
    if let Some(rational_pi) = eval_rational_pi_multiple(&expr)? {
        return Ok(Angle::RationalPi(rational_pi));
    }

    let numeric = eval_expr_f64(&expr)?;
    Ok(Angle::Number(numeric))
}

fn eval_expr_f64(expr: &Expr) -> anyhow::Result<f64> {
    match expr {
        Expr::Integer(value) => Ok(*value as f64),
        Expr::Float(value) => Ok(*value),
        Expr::Pi => Ok(std::f64::consts::PI),
        Expr::UnaryMinus(inner) => Ok(-eval_expr_f64(inner)?),
        Expr::Binary { op, left, right } => {
            let lhs = eval_expr_f64(left)?;
            let rhs = eval_expr_f64(right)?;
            let value = match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
                BinaryOp::Pow => lhs.powf(rhs),
            };
            Ok(value)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbolic {
    Rational(Rational),
    RationalPi(Rational),
}

fn eval_rational_pi_multiple(expr: &Expr) -> anyhow::Result<Option<Rational>> {
    let symbolic = match try_eval_symbolic(expr)? {
        Some(value) => value,
        None => return Ok(None),
    };

    match symbolic {
        Symbolic::RationalPi(coeff) => Ok(Some(coeff)),
        Symbolic::Rational(_) => Ok(None),
    }
}

fn try_eval_symbolic(expr: &Expr) -> anyhow::Result<Option<Symbolic>> {
    match expr {
        Expr::Integer(v) => {
            let int =
                i32::try_from(*v).map_err(|_| anyhow::anyhow!("integer `{v}` is too large"))?;
            Ok(Some(Symbolic::Rational(Rational::from_integer(int))))
        }
        Expr::Float(_) => Ok(None),
        Expr::Pi => Ok(Some(Symbolic::RationalPi(Rational::ONE))),
        Expr::UnaryMinus(inner) => {
            let Some(value) = try_eval_symbolic(inner)? else {
                return Ok(None);
            };
            let negated = match value {
                Symbolic::Rational(r) => Symbolic::Rational(-r),
                Symbolic::RationalPi(r) => Symbolic::RationalPi(-r),
            };
            Ok(Some(negated))
        }
        Expr::Binary { op, left, right } => {
            let Some(lhs) = try_eval_symbolic(left)? else {
                return Ok(None);
            };
            let Some(rhs) = try_eval_symbolic(right)? else {
                return Ok(None);
            };

            let evaluated = match op {
                BinaryOp::Add => match (lhs, rhs) {
                    (Symbolic::Rational(a), Symbolic::Rational(b)) => Symbolic::Rational(a + b),
                    (Symbolic::RationalPi(a), Symbolic::RationalPi(b)) => {
                        Symbolic::RationalPi(a + b)
                    }
                    _ => return Ok(None),
                },
                BinaryOp::Sub => match (lhs, rhs) {
                    (Symbolic::Rational(a), Symbolic::Rational(b)) => Symbolic::Rational(a - b),
                    (Symbolic::RationalPi(a), Symbolic::RationalPi(b)) => {
                        Symbolic::RationalPi(a - b)
                    }
                    _ => return Ok(None),
                },
                BinaryOp::Mul => match (lhs, rhs) {
                    (Symbolic::Rational(a), Symbolic::Rational(b)) => Symbolic::Rational(a * b),
                    (Symbolic::Rational(a), Symbolic::RationalPi(b))
                    | (Symbolic::RationalPi(b), Symbolic::Rational(a)) => {
                        Symbolic::RationalPi(a * b)
                    }
                    (Symbolic::RationalPi(_), Symbolic::RationalPi(_)) => return Ok(None),
                },
                BinaryOp::Div => match (lhs, rhs) {
                    (Symbolic::Rational(_), Symbolic::Rational(den)) if den.numerator == 0 => {
                        anyhow::bail!("division by zero in angle expression")
                    }
                    (Symbolic::RationalPi(_), Symbolic::Rational(den)) if den.numerator == 0 => {
                        anyhow::bail!("division by zero in angle expression")
                    }
                    (Symbolic::Rational(num), Symbolic::Rational(den)) => {
                        Symbolic::Rational(num / den)
                    }
                    (Symbolic::RationalPi(num), Symbolic::Rational(den)) => {
                        Symbolic::RationalPi(num / den)
                    }
                    (Symbolic::RationalPi(num), Symbolic::RationalPi(den)) => {
                        if den.numerator == 0 {
                            anyhow::bail!("division by zero in angle expression");
                        }
                        Symbolic::Rational(num / den)
                    }
                    (Symbolic::Rational(_), Symbolic::RationalPi(_)) => return Ok(None),
                },
                BinaryOp::Pow => return Ok(None),
            };

            Ok(Some(evaluated))
        }
    }
}

fn to_u32_index(value: i64) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| {
        anyhow::anyhow!("qubit index must be a non-negative 32-bit integer, found `{value}`")
    })
}

fn trailing_digits_to_u32(ident: &str) -> anyhow::Result<Option<u32>> {
    let non_digit_len = ident.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    if non_digit_len == ident.len() {
        return Ok(None);
    }

    let digits = &ident[non_digit_len..];
    let idx = digits
        .parse::<u32>()
        .map_err(|e| anyhow::anyhow!("invalid trailing index in identifier `{ident}`: {e}"))?;
    Ok(Some(idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate_kind(gate: &Gate) -> &'static str {
        match gate {
            Gate::X(_) => "x",
            Gate::Y(_) => "y",
            Gate::Z(_) => "z",
            Gate::H(_) => "h",
            Gate::S(_) => "s",
            Gate::T(_) => "t",
            Gate::RX(..) => "rx",
            Gate::RY(..) => "ry",
            Gate::RZ(..) => "rz",
            Gate::U3(..) => "u3",
            Gate::CX(..) => "cx",
            Gate::CZ(..) => "cz",
            Gate::SWAP(..) => "swap",
            Gate::CCX(..) => "ccx",
        }
    }

    #[test]
    fn parses_program_without_qreg_declaration() {
        let circuit = parse_qasm("x q0; // comment\nrz(2*pi/3) q1;\ncx q0,q1;")
            .expect("parse should succeed");
        let kinds: Vec<_> = circuit.gates.iter().map(gate_kind).collect();
        assert_eq!(kinds, vec!["x", "rz", "cx"]);
    }

    #[test]
    fn parses_openqasm_include_and_qreg_preamble() {
        let src = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; h q[0]; rx(pi/2) q[1]; ccx q[0],q[1],q[2];";
        let circuit = parse_qasm(src).expect("parse should succeed");
        let kinds: Vec<_> = circuit.gates.iter().map(gate_kind).collect();
        assert_eq!(kinds, vec!["h", "rx", "ccx"]);
    }

    #[test]
    fn parses_parametric_gates_and_angles() {
        let circuit = parse_qasm("qreg q[2]; rx(2*pi/3) q[0]; u3(pi/2,0.25,1) q[1];")
            .expect("parse should succeed");
        assert_eq!(circuit.gates.len(), 2);
        match &circuit.gates[0] {
            Gate::RX(angle, q) => {
                assert_eq!(*q, 0);
                match angle {
                    Angle::RationalPi(r) => assert_eq!(*r, Rational::new(2, 3)),
                    Angle::Number(_) => panic!("expected RationalPi for rx"),
                }
            }
            _ => panic!("expected RX gate"),
        }
        match &circuit.gates[1] {
            Gate::U3(_, _, _, q) => assert_eq!(*q, 1),
            _ => panic!("expected U3 gate"),
        }
    }

    #[test]
    fn parses_scientific_notation_angles() {
        let circuit =
            parse_qasm("qreg q[2]; rx(1e-6) q[0]; rz(0.3e-1) q[1];").expect("parse should succeed");
        assert_eq!(circuit.gates.len(), 2);
        match &circuit.gates[0] {
            Gate::RX(angle, q) => {
                assert_eq!(*q, 0);
                match angle {
                    Angle::Number(v) => assert!((*v - 1e-6).abs() < 1e-16),
                    Angle::RationalPi(_) => panic!("expected Number angle"),
                }
            }
            _ => panic!("expected RX gate"),
        }
        match &circuit.gates[1] {
            Gate::RZ(angle, q) => {
                assert_eq!(*q, 1);
                match angle {
                    Angle::Number(v) => assert!((*v - 3e-2).abs() < 1e-16),
                    Angle::RationalPi(_) => panic!("expected Number angle"),
                }
            }
            _ => panic!("expected RZ gate"),
        }
    }

    #[test]
    fn errors_on_multiple_qreg_declarations() {
        let msg = match parse_qasm("qreg q[2]; qreg r[3]; x q[0];") {
            Ok(_) => panic!("parse should fail"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("multiple qreg declarations"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn errors_on_qreg_out_of_bounds_access() {
        let msg = match parse_qasm("qreg q[2]; x q[2];") {
            Ok(_) => panic!("parse should fail"),
            Err(err) => err.to_string(),
        };
        assert!(msg.contains("out of bounds"), "unexpected: {msg}");
    }

    #[test]
    fn errors_on_unknown_qreg_name() {
        let msg = match parse_qasm("qreg q[2]; x r[0];") {
            Ok(_) => panic!("parse should fail"),
            Err(err) => err.to_string(),
        };
        assert!(msg.contains("unknown qreg"), "unexpected: {msg}");
    }

    #[test]
    fn errors_on_unsupported_gate_name() {
        let msg = match parse_qasm("qreg q[2]; foo q[0];") {
            Ok(_) => panic!("parse should fail"),
            Err(err) => err.to_string(),
        };
        assert!(msg.contains("unsupported gate"), "unexpected: {msg}");
    }
}

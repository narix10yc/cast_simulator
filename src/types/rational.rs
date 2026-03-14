use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    pub numerator: i32,
    pub denominator: i32,
}

impl Rational {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn new(numerator: i32, denominator: i32) -> Self {
        Self::from_i64(numerator as i64, denominator as i64)
    }

    fn from_i64(numerator: i64, denominator: i64) -> Self {
        assert_ne!(denominator, 0, "Denominator cannot be zero");

        if numerator == 0 {
            return Self::ZERO;
        }

        let mut n = numerator;
        let mut d = denominator;
        if d < 0 {
            n = -n;
            d = -d;
        }

        let gcd = Self::gcd_i64(n, d);
        n /= gcd;
        d /= gcd;

        Self {
            numerator: i32::try_from(n)
                .expect("Rational overflow: normalized numerator does not fit in i32"),
            denominator: i32::try_from(d)
                .expect("Rational overflow: normalized denominator does not fit in i32"),
        }
    }

    pub fn from_integer(value: i32) -> Self {
        Self::new(value, 1)
    }

    pub fn reciprocal(self) -> Self {
        assert_ne!(self.numerator, 0, "Cannot take reciprocal of zero");
        Self::from_i64(self.denominator as i64, self.numerator as i64)
    }

    pub fn to_f64(self) -> f64 {
        (self.numerator as f64) / (self.denominator as f64)
    }

    fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }
}

impl From<i32> for Rational {
    fn from(value: i32) -> Self {
        Self::from_integer(value)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = (self.numerator as i64) * (other.denominator as i64);
        let rhs = (other.numerator as i64) * (self.denominator as i64);
        lhs.cmp(&rhs)
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_i64(-(self.numerator as i64), self.denominator as i64)
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let numerator = (self.numerator as i64) * (rhs.denominator as i64)
            + (rhs.numerator as i64) * (self.denominator as i64);
        let denominator = (self.denominator as i64) * (rhs.denominator as i64);
        Self::from_i64(numerator, denominator)
    }
}

impl AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for Rational {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let numerator = (self.numerator as i64) * (rhs.numerator as i64);
        let denominator = (self.denominator as i64) * (rhs.denominator as i64);
        Self::from_i64(numerator, denominator)
    }
}

impl MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.reciprocal()
    }
}

impl DivAssign for Rational {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::Rational;

    #[test]
    fn constructor_normalizes() {
        assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
        assert_eq!(Rational::new(-2, 4), Rational::new(-1, 2));
        assert_eq!(Rational::new(2, -4), Rational::new(-1, 2));
        assert_eq!(Rational::new(-2, -4), Rational::new(1, 2));
        assert_eq!(Rational::new(0, 7), Rational::ZERO);
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn constructor_rejects_zero_denominator() {
        let _ = Rational::new(1, 0);
    }

    #[test]
    fn arithmetic_works() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        assert_eq!(a + b, Rational::new(5, 6));
        assert_eq!(a - b, Rational::new(1, 6));
        assert_eq!(a * b, Rational::new(1, 6));
        assert_eq!(a / b, Rational::new(3, 2));
    }

    #[test]
    fn ordering_works() {
        assert!(Rational::new(1, 2) < Rational::new(2, 3));
        assert!(Rational::new(-3, 4) < Rational::new(-1, 3));
        assert_eq!(Rational::new(2, 6), Rational::new(1, 3));
    }

    #[test]
    fn display_works() {
        assert_eq!(format!("{}", Rational::new(6, 2)), "3");
        assert_eq!(format!("{}", Rational::new(2, 3)), "2/3");
    }
}

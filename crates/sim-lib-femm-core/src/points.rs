//! Shared value/expression-to-scalar decoding for the FEMM crates.
//!
//! These helpers are deliberately physics-free: they only translate kernel
//! `Value`/`Expr` data into the `f64` and `[f64; 2]` shapes the physics crates
//! consume. They live here as the single shared copy with one set of
//! numeric/error semantics.
//!
//! Note: the cell-to-`f64` decoders accept a rational `num/den` literal form in
//! addition to a plain decimal literal. `sim_value::access::as_f64` only handles
//! the plain decimal case, so it is intentionally not used here -- doing so
//! would change behavior for `num/den` inputs.

use sim_kernel::{Cx, DefaultFactory, Expr, Value};

use crate::implementation::{FemmError, FemmResult, ParamSet, parse_finite_number, value_as_f64};

/// Operators accepted by FEMM scalar expression evaluators.
///
/// This is the shared policy for primal and sensitivity evaluation. Integer
/// powers use integer exponent semantics and accept non-positive bases when the
/// mathematical result is defined; non-integer real powers require a positive
/// base and fail closed instead of returning `NaN`.
pub const FEMM_EXPR_OPERATORS: &[&str] =
    &["+", "*", "-", "/", "pow", "sin", "cos", "exp", "ln", "sqrt"];

/// Normalize accepted FEMM scalar expression syntax into canonical call form.
///
/// The FEMM scalar evaluators dispatch on `Expr::Call`, but codecs may produce
/// infix, prefix, or postfix operator nodes for the same arithmetic surface.
/// This helper gives every FEMM crate one normalization policy.
pub fn normalize_femm_expr(expr: &Expr) -> FemmResult<Expr> {
    match expr {
        Expr::Infix {
            operator,
            left,
            right,
        } => Ok(Expr::Call {
            operator: Box::new(Expr::Symbol(operator.clone())),
            args: vec![normalize_femm_expr(left)?, normalize_femm_expr(right)?],
        }),
        Expr::Prefix { operator, arg } | Expr::Postfix { operator, arg } => Ok(Expr::Call {
            operator: Box::new(Expr::Symbol(operator.clone())),
            args: vec![normalize_femm_expr(arg)?],
        }),
        Expr::Call { operator, args } => Ok(Expr::Call {
            operator: Box::new(normalize_femm_expr(operator)?),
            args: args
                .iter()
                .map(normalize_femm_expr)
                .collect::<FemmResult<Vec<_>>>()?,
        }),
        other => Ok(other.clone()),
    }
}

/// Decode a two-element `[x y]` point literal into `[f64; 2]`.
///
/// Each coordinate must be a numeric literal (plain decimal or `num/den`
/// rational). Anything else fails with [`FemmError::FieldOutOfDomain`].
pub fn decode_point2(points: &Value) -> FemmResult<[f64; 2]> {
    let mut cx = Cx::new(
        std::sync::Arc::new(sim_kernel::EagerPolicy),
        std::sync::Arc::new(DefaultFactory),
    );
    let expr = points
        .object()
        .as_expr(&mut cx)
        .map_err(|err| FemmError::FieldOutOfDomain(err.to_string()))?;
    match expr {
        Expr::List(items) if items.len() == 2 => {
            Ok([expr_cell_as_f64(&items[0])?, expr_cell_as_f64(&items[1])?])
        }
        _ => Err(FemmError::FieldOutOfDomain(
            "expected [x y] point".to_owned(),
        )),
    }
}

fn expr_cell_as_f64(expr: &Expr) -> FemmResult<f64> {
    match expr {
        Expr::Number(number) => parse_finite_number(&number.canonical)
            .ok_or_else(|| FemmError::FieldOutOfDomain(format!("bad point coordinate {expr:?}"))),
        _ => Err(FemmError::FieldOutOfDomain(
            "point coordinates must be numeric literals".to_owned(),
        )),
    }
}

/// Evaluate an expression to `f64` against a parameter set and coordinate
/// bindings.
///
/// Supports numeric literals (plain decimal or `num/den` rational), symbol
/// lookups (coordinate bindings first, then params), and
/// [`FEMM_EXPR_OPERATORS`]. Non-integer real powers require a positive base and
/// return [`FemmError::InvalidGeometry`] otherwise.
pub fn eval_expr_f64(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    coords: &[(&str, f64)],
) -> FemmResult<f64> {
    let expr = normalize_femm_expr(expr)?;
    eval_canonical_expr_f64(cx, &expr, params, coords)
}

fn eval_canonical_expr_f64(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    coords: &[(&str, f64)],
) -> FemmResult<f64> {
    match expr {
        Expr::Number(number) => parse_finite_number(&number.canonical)
            .ok_or_else(|| FemmError::InvalidGeometry(format!("bad number {}", number.canonical))),
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                return finite_scalar(*value, "non-finite coordinate binding");
            }
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
            value_as_f64(cx, value)
        }
        Expr::Call { operator, args } => {
            let Expr::Symbol(symbol) = operator.as_ref() else {
                return Err(FemmError::InvalidGeometry(
                    "unsupported operator".to_owned(),
                ));
            };
            let values = args
                .iter()
                .map(|arg| eval_canonical_expr_f64(cx, arg, params, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            match symbol.name.as_ref() {
                "+" => finite_scalar(values.into_iter().sum(), "non-finite scalar addition"),
                "*" => finite_scalar(
                    values.into_iter().product(),
                    "non-finite scalar multiplication",
                ),
                "-" if values.len() == 1 => finite_scalar(-values[0], "non-finite scalar negation"),
                "-" if values.len() == 2 => {
                    finite_scalar(values[0] - values[1], "non-finite scalar subtraction")
                }
                "/" if values.len() == 2 => {
                    if values[1] == 0.0 {
                        return Err(FemmError::InvalidGeometry(
                            "division by zero in scalar expression".to_owned(),
                        ));
                    }
                    finite_scalar(values[0] / values[1], "non-finite scalar division")
                }
                "pow" if values.len() == 2 => eval_pow_f64(values[0], values[1]),
                "sin" if values.len() == 1 => finite_scalar(values[0].sin(), "non-finite sin"),
                "cos" if values.len() == 1 => finite_scalar(values[0].cos(), "non-finite cos"),
                "exp" if values.len() == 1 => finite_scalar(values[0].exp(), "non-finite exp"),
                "ln" if values.len() == 1 && values[0] > 0.0 => {
                    finite_scalar(values[0].ln(), "non-finite ln")
                }
                "sqrt" if values.len() == 1 && values[0] >= 0.0 => {
                    finite_scalar(values[0].sqrt(), "non-finite sqrt")
                }
                _ => Err(FemmError::InvalidGeometry(format!(
                    "unsupported operator {symbol}"
                ))),
            }
        }
        _ => Err(FemmError::InvalidGeometry(
            "unsupported expression".to_owned(),
        )),
    }
}

fn eval_pow_f64(base: f64, exponent: f64) -> FemmResult<f64> {
    if let Some(exponent) = integer_exponent(exponent) {
        if base == 0.0 && exponent < 0 {
            return Err(FemmError::InvalidGeometry(
                "pow with negative integer exponent requires nonzero base".to_owned(),
            ));
        }
        return finite_scalar(base.powi(exponent), "non-finite integer pow");
    }
    if base <= 0.0 {
        return Err(FemmError::InvalidGeometry(
            "pow with non-integer exponent requires positive base".to_owned(),
        ));
    }
    finite_scalar(base.powf(exponent), "non-finite pow")
}

/// Returns `Some(n)` when `value` is exactly representable as an `i32` exponent.
pub fn integer_exponent(value: f64) -> Option<i32> {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= f64::from(i32::MIN)
        && value <= f64::from(i32::MAX)
    {
        Some(value as i32)
    } else {
        None
    }
}

fn finite_scalar(value: f64, context: &str) -> FemmResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FemmError::InvalidGeometry(context.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Expr, NumberLiteral, Symbol};

    use super::*;

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    fn num(canonical: &str) -> Expr {
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "f64"),
            canonical: canonical.to_owned(),
        })
    }

    fn call(operator: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new(operator))),
            args,
        }
    }

    fn infix(operator: &str, left: Expr, right: Expr) -> Expr {
        Expr::Infix {
            operator: Symbol::new(operator),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn prefix(operator: &str, arg: Expr) -> Expr {
        Expr::Prefix {
            operator: Symbol::new(operator),
            arg: Box::new(arg),
        }
    }

    fn postfix(operator: &str, arg: Expr) -> Expr {
        Expr::Postfix {
            operator: Symbol::new(operator),
            arg: Box::new(arg),
        }
    }

    #[test]
    fn finite_number_parser_rejects_malformed_and_nonfinite_values() {
        assert_eq!(parse_finite_number("3/4"), Some(0.75));
        assert_eq!(parse_finite_number("not-a-number"), None);
        assert_eq!(parse_finite_number("1/0"), None);
        assert_eq!(parse_finite_number("inf"), None);
        assert_eq!(parse_finite_number("1e309"), None);
    }

    #[test]
    fn scalar_evaluation_rejects_bad_literals_and_nonfinite_arithmetic() {
        let mut cx = test_cx();
        let params = ParamSet::default();
        assert!(eval_expr_f64(&mut cx, &num("bad"), &params, &[]).is_err());
        assert!(
            eval_expr_f64(
                &mut cx,
                &call("/", vec![num("1.0"), num("0.0")]),
                &params,
                &[],
            )
            .is_err()
        );
        assert!(eval_expr_f64(&mut cx, &call("exp", vec![num("1000.0")]), &params, &[]).is_err());
    }

    #[test]
    fn expression_normalization_canonicalizes_operator_forms() {
        let expr = call(
            "+",
            vec![
                infix("*", num("2.0"), num("3.0")),
                prefix("-", postfix("sqrt", num("4.0"))),
            ],
        );
        assert_eq!(
            normalize_femm_expr(&expr).unwrap(),
            call(
                "+",
                vec![
                    call("*", vec![num("2.0"), num("3.0")]),
                    call("-", vec![call("sqrt", vec![num("4.0")])]),
                ],
            )
        );
    }

    #[test]
    fn scalar_evaluation_supports_all_femm_expression_operators() {
        let mut cx = test_cx();
        let params = ParamSet::default();
        let cases = [
            (call("+", vec![num("1.0"), num("2.0"), num("3.0")]), 6.0),
            (call("*", vec![num("2.0"), num("3.0"), num("4.0")]), 24.0),
            (prefix("-", num("2.0")), -2.0),
            (infix("-", num("5.0"), num("2.0")), 3.0),
            (infix("/", num("8.0"), num("4.0")), 2.0),
            (call("pow", vec![num("2.0"), num("3.0")]), 8.0),
            (prefix("sin", num("0.0")), 0.0),
            (prefix("cos", num("0.0")), 1.0),
            (prefix("exp", num("0.0")), 1.0),
            (prefix("ln", num("2.718281828459045")), 1.0),
            (postfix("sqrt", num("4.0")), 2.0),
        ];
        for (expr, expected) in cases {
            let value = eval_expr_f64(&mut cx, &expr, &params, &[]).unwrap();
            assert!((value - expected).abs() < 1.0e-12, "{expr:?}");
        }
    }
}

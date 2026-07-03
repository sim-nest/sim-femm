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

use crate::implementation::{FemmError, FemmResult, ParamSet, value_as_f64};

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
        Expr::Number(number) => number
            .canonical
            .parse::<f64>()
            .or_else(|_| {
                number
                    .canonical
                    .split_once('/')
                    .ok_or(())
                    .and_then(|(num, den)| {
                        num.parse::<f64>()
                            .and_then(|num| den.parse::<f64>().map(|den| num / den))
                            .map_err(|_| ())
                    })
            })
            .map_err(|_| FemmError::FieldOutOfDomain(format!("bad point coordinate {expr:?}"))),
        _ => Err(FemmError::FieldOutOfDomain(
            "point coordinates must be numeric literals".to_owned(),
        )),
    }
}

/// Evaluate an expression to `f64` against a parameter set and coordinate
/// bindings.
///
/// Supports numeric literals (plain decimal or `num/den` rational), symbol
/// lookups (coordinate bindings first, then params), and the small arithmetic
/// operator set (`+ * - / pow`) used by FEMM geometry expressions.
pub fn eval_expr_f64(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    coords: &[(&str, f64)],
) -> FemmResult<f64> {
    match expr {
        Expr::Number(number) => number
            .canonical
            .parse::<f64>()
            .or_else(|_| {
                number
                    .canonical
                    .split_once('/')
                    .map(|(num, den)| {
                        num.parse::<f64>()
                            .and_then(|num| den.parse::<f64>().map(|den| num / den))
                    })
                    .transpose()
                    .map(|value| value.unwrap_or_default())
            })
            .map_err(|_| FemmError::InvalidGeometry(format!("bad number {}", number.canonical))),
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                return Ok(*value);
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
                .map(|arg| eval_expr_f64(cx, arg, params, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            match symbol.name.as_ref() {
                "+" => Ok(values.into_iter().sum()),
                "*" => Ok(values.into_iter().product()),
                "-" if values.len() == 1 => Ok(-values[0]),
                "-" if values.len() == 2 => Ok(values[0] - values[1]),
                "/" if values.len() == 2 => Ok(values[0] / values[1]),
                "pow" if values.len() == 2 => Ok(values[0].powf(values[1])),
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

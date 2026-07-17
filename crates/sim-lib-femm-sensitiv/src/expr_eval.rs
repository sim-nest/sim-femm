//! Scalar expression evaluation for FEMM sensitivity paths.

use sim_kernel::{Cx, Expr, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult, ParamSet, integer_exponent};
use sim_lib_numbers_ad::{Dual, Scalarish, Tape, Var};

pub(crate) fn direct_expr_derivative(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<f64> {
    Ok(eval_expr_dual(cx, expr, params, Some(wrt), &[])?.d[0])
}

pub(crate) fn eval_expr_dual(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: Option<&Symbol>,
    coords: &[(&str, f64)],
) -> FemmResult<Dual<1>> {
    match expr {
        Expr::Number(number) => parse_number(&number.canonical).map(Dual::<1>::cst),
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                return finite_dual(Dual::<1>::cst(*value), "non-finite coordinate binding");
            }
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
            let scalar = sim_lib_femm_core::value_as_f64(cx, value)?;
            if wrt == Some(symbol) {
                Ok(Dual::<1>::var(scalar, 0))
            } else {
                Ok(Dual::<1>::cst(scalar))
            }
        }
        Expr::Call { operator, args } => {
            let Expr::Symbol(symbol) = operator.as_ref() else {
                return Err(FemmError::SensitivityUnavailable(
                    "unsupported non-symbol operator".to_owned(),
                ));
            };
            let values = args
                .iter()
                .map(|arg| eval_expr_dual(cx, arg, params, wrt, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            apply_dual_op(symbol, &values)
        }
        _ => Err(FemmError::SensitivityUnavailable(
            "unsupported expression in exact direct gradient".to_owned(),
        )),
    }
}

fn apply_dual_op(symbol: &Symbol, values: &[Dual<1>]) -> FemmResult<Dual<1>> {
    match (symbol.name.as_ref(), values) {
        ("+", []) => Ok(Dual::cst(0.0)),
        ("+", values) => finite_dual(
            values
                .iter()
                .copied()
                .fold(Dual::cst(0.0), |acc, value| acc + value),
            "non-finite scalar addition",
        ),
        ("*", []) => Ok(Dual::cst(1.0)),
        ("*", values) => finite_dual(
            values
                .iter()
                .copied()
                .fold(Dual::cst(1.0), |acc, value| acc * value),
            "non-finite scalar multiplication",
        ),
        ("-", [value]) => finite_dual(-*value, "non-finite scalar negation"),
        ("-", [left, right]) => finite_dual(*left - *right, "non-finite scalar subtraction"),
        ("/", [left, right]) => {
            if right.v == 0.0 {
                return Err(FemmError::SensitivityUnavailable(
                    "division by zero in scalar expression".to_owned(),
                ));
            }
            finite_dual(*left / *right, "non-finite scalar division")
        }
        ("pow", [base, exp]) => apply_dual_pow(*base, *exp),
        ("sin", [arg]) => finite_dual(arg.sin(), "non-finite sin"),
        ("cos", [arg]) => finite_dual(arg.cos(), "non-finite cos"),
        ("exp", [arg]) => finite_dual(arg.exp(), "non-finite exp"),
        ("ln", [arg]) if arg.v > 0.0 => finite_dual(arg.ln(), "non-finite ln"),
        ("sqrt", [arg]) if arg.v >= 0.0 => finite_dual(arg.sqrt(), "non-finite sqrt"),
        _ => Err(FemmError::SensitivityUnavailable(format!(
            "unsupported operator {symbol} in exact direct gradient"
        ))),
    }
}

pub(crate) fn reverse_expr_gradient(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &[Symbol],
) -> FemmResult<Vec<(Symbol, f64)>> {
    let mut tape = Tape::new();
    let inputs = wrt
        .iter()
        .enumerate()
        .map(|(slot, symbol)| {
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))
                .and_then(|value| sim_lib_femm_core::value_as_f64(cx, value))?;
            Ok((symbol.clone(), slot, tape.input(slot, value)))
        })
        .collect::<FemmResult<Vec<_>>>()?;
    let output = eval_expr_tape(cx, expr, params, &inputs, &mut tape, &[])?;
    let gradient = tape.grad(output.var, wrt.len());
    Ok(inputs
        .into_iter()
        .map(|(symbol, slot, _)| (symbol, gradient[slot]))
        .collect())
}

#[derive(Clone, Copy)]
struct TapeScalar {
    var: Var,
    depends_on_input: bool,
}

impl TapeScalar {
    fn constant(var: Var) -> Self {
        Self {
            var,
            depends_on_input: false,
        }
    }

    fn input(var: Var) -> Self {
        Self {
            var,
            depends_on_input: true,
        }
    }
}

fn eval_expr_tape(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &[(Symbol, usize, Var)],
    tape: &mut Tape,
    coords: &[(&str, f64)],
) -> FemmResult<TapeScalar> {
    match expr {
        Expr::Number(number) => {
            parse_number(&number.canonical).map(|value| TapeScalar::constant(tape.constant(value)))
        }
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                if !value.is_finite() {
                    return Err(FemmError::SensitivityUnavailable(
                        "non-finite coordinate binding".to_owned(),
                    ));
                }
                return Ok(TapeScalar::constant(tape.constant(*value)));
            }
            if let Some((_, _, var)) = wrt.iter().find(|(name, _, _)| name == symbol) {
                return Ok(TapeScalar::input(*var));
            }
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
            sim_lib_femm_core::value_as_f64(cx, value)
                .map(|value| TapeScalar::constant(tape.constant(value)))
        }
        Expr::Call { operator, args } => {
            let Expr::Symbol(symbol) = operator.as_ref() else {
                return Err(FemmError::SensitivityUnavailable(
                    "unsupported non-symbol operator".to_owned(),
                ));
            };
            let values = args
                .iter()
                .map(|arg| eval_expr_tape(cx, arg, params, wrt, tape, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            apply_tape_op(symbol, &values, tape)
        }
        _ => Err(FemmError::SensitivityUnavailable(
            "unsupported expression in exact adjoint gradient".to_owned(),
        )),
    }
}

fn apply_tape_op(
    symbol: &Symbol,
    values: &[TapeScalar],
    tape: &mut Tape,
) -> FemmResult<TapeScalar> {
    match (symbol.name.as_ref(), values) {
        ("+", []) => Ok(TapeScalar::constant(tape.constant(0.0))),
        ("+", values) => {
            let mut acc = tape.constant(0.0);
            let mut depends_on_input = false;
            for value in values {
                acc = tape.add(acc, value.var);
                depends_on_input |= value.depends_on_input;
            }
            finite_tape_scalar(acc, depends_on_input, tape, "non-finite scalar addition")
        }
        ("*", []) => Ok(TapeScalar::constant(tape.constant(1.0))),
        ("*", values) => {
            let mut acc = tape.constant(1.0);
            let mut depends_on_input = false;
            for value in values {
                acc = tape.mul(acc, value.var);
                depends_on_input |= value.depends_on_input;
            }
            finite_tape_scalar(
                acc,
                depends_on_input,
                tape,
                "non-finite scalar multiplication",
            )
        }
        ("-", [value]) => {
            let zero = tape.constant(0.0);
            finite_tape_scalar(
                tape.sub(zero, value.var),
                value.depends_on_input,
                tape,
                "non-finite scalar negation",
            )
        }
        ("-", [left, right]) => finite_tape_scalar(
            tape.sub(left.var, right.var),
            left.depends_on_input || right.depends_on_input,
            tape,
            "non-finite scalar subtraction",
        ),
        ("/", [left, right]) => {
            if tape.value(right.var) == 0.0 {
                return Err(FemmError::SensitivityUnavailable(
                    "division by zero in scalar expression".to_owned(),
                ));
            }
            finite_tape_scalar(
                tape.div(left.var, right.var),
                left.depends_on_input || right.depends_on_input,
                tape,
                "non-finite scalar division",
            )
        }
        ("pow", [base, exp]) => apply_tape_pow(*base, *exp, tape),
        ("sin", [arg]) => finite_tape_scalar(
            tape.sin(arg.var),
            arg.depends_on_input,
            tape,
            "non-finite sin",
        ),
        ("cos", [arg]) => finite_tape_scalar(
            tape.cos(arg.var),
            arg.depends_on_input,
            tape,
            "non-finite cos",
        ),
        ("exp", [arg]) => finite_tape_scalar(
            tape.exp(arg.var),
            arg.depends_on_input,
            tape,
            "non-finite exp",
        ),
        ("ln", [arg]) if tape.value(arg.var) > 0.0 => finite_tape_scalar(
            tape.ln(arg.var),
            arg.depends_on_input,
            tape,
            "non-finite ln",
        ),
        ("sqrt", [arg]) if tape.value(arg.var) >= 0.0 => finite_tape_scalar(
            tape.sqrt(arg.var),
            arg.depends_on_input,
            tape,
            "non-finite sqrt",
        ),
        _ => Err(FemmError::SensitivityUnavailable(format!(
            "unsupported operator {symbol} in exact adjoint gradient"
        ))),
    }
}

fn apply_dual_pow(base: Dual<1>, exp: Dual<1>) -> FemmResult<Dual<1>> {
    if exp.d.iter().all(|derivative| *derivative == 0.0)
        && let Some(exponent) = integer_exponent(exp.v)
    {
        return dual_powi(base, exponent);
    }
    if base.v <= 0.0 {
        return Err(FemmError::SensitivityUnavailable(
            "pow with non-integer exponent requires positive base".to_owned(),
        ));
    }
    finite_dual((base.ln() * exp).exp(), "non-finite pow")
}

fn dual_powi(base: Dual<1>, exponent: i32) -> FemmResult<Dual<1>> {
    if base.v == 0.0 && exponent < 0 {
        return Err(FemmError::SensitivityUnavailable(
            "pow with negative integer exponent requires nonzero base".to_owned(),
        ));
    }
    let value = base.v.powi(exponent);
    let scale = match exponent {
        0 => 0.0,
        1 => 1.0,
        _ if base.v == 0.0 => 0.0,
        _ => f64::from(exponent) * value / base.v,
    };
    finite_dual(
        Dual {
            v: value,
            d: [base.d[0] * scale],
        },
        "non-finite integer pow",
    )
}

fn apply_tape_pow(base: TapeScalar, exp: TapeScalar, tape: &mut Tape) -> FemmResult<TapeScalar> {
    if !exp.depends_on_input
        && let Some(exponent) = integer_exponent(tape.value(exp.var))
    {
        return tape_powi(base, exponent, tape);
    }
    if tape.value(base.var) <= 0.0 {
        return Err(FemmError::SensitivityUnavailable(
            "pow with non-integer exponent requires positive base".to_owned(),
        ));
    }
    let ln_base = tape.ln(base.var);
    let scaled = tape.mul(ln_base, exp.var);
    finite_tape_scalar(
        tape.exp(scaled),
        base.depends_on_input || exp.depends_on_input,
        tape,
        "non-finite pow",
    )
}

fn tape_powi(base: TapeScalar, exponent: i32, tape: &mut Tape) -> FemmResult<TapeScalar> {
    if tape.value(base.var) == 0.0 && exponent < 0 {
        return Err(FemmError::SensitivityUnavailable(
            "pow with negative integer exponent requires nonzero base".to_owned(),
        ));
    }
    let magnitude = if exponent < 0 {
        (-i64::from(exponent)) as u32
    } else {
        exponent as u32
    };
    let positive = tape_powi_nonnegative(base.var, magnitude, tape);
    let var = if exponent < 0 {
        let one = tape.constant(1.0);
        tape.div(one, positive)
    } else {
        positive
    };
    finite_tape_scalar(var, base.depends_on_input, tape, "non-finite integer pow")
}

fn tape_powi_nonnegative(base: Var, mut exponent: u32, tape: &mut Tape) -> Var {
    let mut acc = tape.constant(1.0);
    let mut factor = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            acc = tape.mul(acc, factor);
        }
        exponent >>= 1;
        if exponent > 0 {
            factor = tape.mul(factor, factor);
        }
    }
    acc
}

fn parse_number(text: &str) -> FemmResult<f64> {
    sim_lib_femm_core::parse_displayed_number(text)
        .ok_or_else(|| FemmError::SensitivityUnavailable(format!("bad number literal {text}")))
}

fn finite_dual(value: Dual<1>, context: &str) -> FemmResult<Dual<1>> {
    if value.v.is_finite() && value.d.iter().all(|derivative| derivative.is_finite()) {
        Ok(value)
    } else {
        Err(FemmError::SensitivityUnavailable(context.to_owned()))
    }
}

fn finite_tape_scalar(
    var: Var,
    depends_on_input: bool,
    tape: &Tape,
    context: &str,
) -> FemmResult<TapeScalar> {
    if tape.value(var).is_finite() {
        Ok(TapeScalar {
            var,
            depends_on_input,
        })
    } else {
        Err(FemmError::SensitivityUnavailable(context.to_owned()))
    }
}

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
                return Ok(Dual::<1>::cst(*value));
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
        ("+", values) => Ok(values
            .iter()
            .copied()
            .fold(Dual::cst(0.0), |acc, value| acc + value)),
        ("*", []) => Ok(Dual::cst(1.0)),
        ("*", values) => Ok(values
            .iter()
            .copied()
            .fold(Dual::cst(1.0), |acc, value| acc * value)),
        ("-", [value]) => Ok(-*value),
        ("-", [left, right]) => Ok(*left - *right),
        ("/", [left, right]) => Ok(*left / *right),
        ("pow", [base, exp]) => apply_dual_pow(*base, *exp),
        ("sin", [arg]) => Ok(arg.sin()),
        ("cos", [arg]) => Ok(arg.cos()),
        ("exp", [arg]) => Ok(arg.exp()),
        ("ln", [arg]) if arg.v > 0.0 => Ok(arg.ln()),
        ("sqrt", [arg]) if arg.v >= 0.0 => Ok(arg.sqrt()),
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
            Ok(TapeScalar {
                var: acc,
                depends_on_input,
            })
        }
        ("*", []) => Ok(TapeScalar::constant(tape.constant(1.0))),
        ("*", values) => {
            let mut acc = tape.constant(1.0);
            let mut depends_on_input = false;
            for value in values {
                acc = tape.mul(acc, value.var);
                depends_on_input |= value.depends_on_input;
            }
            Ok(TapeScalar {
                var: acc,
                depends_on_input,
            })
        }
        ("-", [value]) => {
            let zero = tape.constant(0.0);
            Ok(TapeScalar {
                var: tape.sub(zero, value.var),
                depends_on_input: value.depends_on_input,
            })
        }
        ("-", [left, right]) => Ok(TapeScalar {
            var: tape.sub(left.var, right.var),
            depends_on_input: left.depends_on_input || right.depends_on_input,
        }),
        ("/", [left, right]) => Ok(TapeScalar {
            var: tape.div(left.var, right.var),
            depends_on_input: left.depends_on_input || right.depends_on_input,
        }),
        ("pow", [base, exp]) => apply_tape_pow(*base, *exp, tape),
        ("sin", [arg]) => Ok(TapeScalar {
            var: tape.sin(arg.var),
            depends_on_input: arg.depends_on_input,
        }),
        ("cos", [arg]) => Ok(TapeScalar {
            var: tape.cos(arg.var),
            depends_on_input: arg.depends_on_input,
        }),
        ("exp", [arg]) => Ok(TapeScalar {
            var: tape.exp(arg.var),
            depends_on_input: arg.depends_on_input,
        }),
        ("ln", [arg]) if tape.value(arg.var) > 0.0 => Ok(TapeScalar {
            var: tape.ln(arg.var),
            depends_on_input: arg.depends_on_input,
        }),
        ("sqrt", [arg]) if tape.value(arg.var) >= 0.0 => Ok(TapeScalar {
            var: tape.sqrt(arg.var),
            depends_on_input: arg.depends_on_input,
        }),
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
    Ok((base.ln() * exp).exp())
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
    Ok(Dual {
        v: value,
        d: [base.d[0] * scale],
    })
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
    Ok(TapeScalar {
        var: tape.exp(scaled),
        depends_on_input: base.depends_on_input || exp.depends_on_input,
    })
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
    Ok(TapeScalar {
        var,
        depends_on_input: base.depends_on_input,
    })
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

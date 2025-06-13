use std::iter::Peekable;
use std::iter::Rev;

use pest::Parser;
use pest::iterators::Pair;
use pest::iterators::Pairs;
use pest_derive::Parser;

use super::error::ParseError;
use super::error::ParseResult;
use super::error::Result;

use super::types::PartialType;
use super::types::PartialValue;

use super::types::label::{LabelInterner, Labels};

use super::types::Literal;
use super::types::Span;

use super::types::raw_ast::{
    RawArithmeticExpr, RawBoolExpr, RawExpr, RawExprKind, RawFunctionExpr, RawProgram, RawStmt,
    RawStmtKind,
};

#[derive(Parser)]
#[grammar = "compiler/parser/grammar.pest"]
pub struct CoupParser;

pub fn parse_program(source: &str) -> Result<RawProgram> {
    if source.is_empty() {
        return Ok(RawProgram::new(Vec::new()));
    }

    let mut parsed =
        CoupParser::parse(Rule::program, source).map_err(|x| ParseError::PestError(x))?;
    let program = parsed.next().ok_or(ParseError::MissingRootNode)?;

    let mut statements = Vec::new();
    for pair in program.into_inner() {
        if pair.as_rule() == Rule::EOI {
            break;
        }

        let pos = pair.line_col();
        let text = pair.as_str().to_string();

        statements.push(RawStmt::new(
            parse_statement(pair)?,
            (),
            Span::new(pos, text),
        ));
    }

    Ok(RawProgram::new(statements))
}

fn parse_statement(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let stmt = match child.as_rule() {
        Rule::skip_stm => RawStmtKind::Skip,
        Rule::let_stm => parse_let(child)?,
        Rule::if_stm => parse_if(child)?,
        Rule::while_stm => parse_while(child)?,
        Rule::expr_stm => parse_expr_stmt(child)?,
        Rule::return_stm => parse_return(child)?,
        r => return Err(ParseError::InvalidNode(r, pos)),
    };

    Ok(stmt)
}

fn parse_let(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var_decl = children.next().ok_or(ParseError::MissingNode(pos))?;
    let (var, ty) = parse_var_declration(var_decl)?;
    let expr = children.next().ok_or(ParseError::MissingNode(pos))?;
    let value = parse_expr(expr)?;

    Ok(RawStmtKind::Let { var, ty, value })
}

fn parse_if(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let condition = parse_expr(children.next().ok_or(ParseError::MissingNode(pos))?)?;
    let then_p = children.next().ok_or(ParseError::MissingNode(pos))?;
    let line_col = then_p.line_col();
    let text = then_p.as_str().to_string();
    let then_branch = RawExpr::new(parse_blk(then_p)?, (), Span::new(line_col, text));

    let mut else_if_branches = Vec::new();
    let mut else_branch = None;
    while let Some(next) = children.next() {
        match next.as_rule() {
            // If else branch
            Rule::expr => {
                let cond = parse_expr(next)?;
                let then_p = children.next().ok_or(ParseError::MissingNode(pos))?;
                let line_col = then_p.line_col();
                let text = then_p.as_str().to_string();
                let then = RawExpr::new(parse_blk(then_p)?, (), Span::new(line_col, text));
                else_if_branches.push((cond, then));
            }
            // Else branch
            Rule::blk_expr => {
                let line_col = next.line_col();
                let text = next.as_str().to_string();
                else_branch = Some(RawExpr::new(
                    parse_blk(next)?,
                    (),
                    Span::new(line_col, text),
                ));
            }
            r => return Err(ParseError::InvalidNode(r, pos)),
        }
    }

    Ok(RawStmtKind::If {
        condition,
        then_branch,
        else_if_branches,
        else_branch,
    })
}

fn parse_while(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let condition = parse_expr(children.next().ok_or(ParseError::MissingNode(pos))?)?;
    let body_p = children.next().ok_or(ParseError::MissingNode(pos))?;
    let line_col = body_p.line_col();
    let text = body_p.as_str().to_string();
    let body = RawExpr::new(parse_blk(body_p)?, (), Span::new(line_col, text));

    Ok(RawStmtKind::While { condition, body })
}

fn parse_expr_stmt(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    Ok(RawStmtKind::Expr(parse_expr(child)?))
}

fn parse_return(p: Pair<'_, Rule>) -> ParseResult<RawStmtKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    Ok(RawStmtKind::Return(parse_expr(child)?))
}

fn parse_expr(p: Pair<'_, Rule>) -> ParseResult<RawExpr> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::bool_expr => parse_bool(child)?,
        Rule::arithmetic_expr => parse_arithmetic(child)?,
        Rule::function_expr => parse_function(child)?,
        Rule::blk_expr => parse_blk(child)?,
        Rule::literal => parse_literal(child)?,
        Rule::var => parse_var(child)?,
        r => return Err(ParseError::InvalidNode(r, pos)),
    };

    Ok(RawExpr::new(expr, (), Span::new(pos, text)))
}

fn parse_bool(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::and_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::And(l, r)
        }
        Rule::or_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Or(l, r)
        }
        Rule::eq_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Eq(l, r)
        }
        Rule::le_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Le(l, r)
        }
        Rule::leq_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Leq(l, r)
        }
        Rule::ge_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Ge(l, r)
        }
        Rule::geq_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawBoolExpr::Geq(l, r)
        }
        Rule::neg_expr => {
            let expr = parse_expr(
                child
                    .into_inner()
                    .next()
                    .ok_or(ParseError::MissingNode(pos))?,
            )?;
            RawBoolExpr::Neg(expr)
        }
        r => return Err(ParseError::InvalidNode(r, pos)),
    };
    Ok(RawExprKind::Bool(Box::new(expr)))
}

fn parse_arithmetic(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::add_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawArithmeticExpr::Add(l, r)
        }
        Rule::sub_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawArithmeticExpr::Sub(l, r)
        }
        Rule::mul_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawArithmeticExpr::Mul(l, r)
        }
        Rule::div_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawArithmeticExpr::Div(l, r)
        }
        Rule::mod_expr => {
            let (l, r) = bin_op_helper(child)?;
            RawArithmeticExpr::Mod(l, r)
        }
        r => return Err(ParseError::InvalidNode(r, pos)),
    };
    Ok(RawExprKind::Arithmetic(Box::new(expr)))
}

fn parse_function(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::lambda_expr => parse_lambda(child)?,
        Rule::apply_expr => parse_apply(child)?,
        r => return Err(ParseError::InvalidNode(r, pos)),
    };

    Ok(RawExprKind::Function(Box::new(expr)))
}

fn parse_lambda(p: Pair<'_, Rule>) -> ParseResult<RawFunctionExpr> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var_declrations = children.next().ok_or(ParseError::MissingNode(pos))?;
    let next = children.next().ok_or(ParseError::MissingNode(pos))?;
    match next.as_rule() {
        Rule::r#type => {
            let ty_p = next;
            let expr_p = children.next().ok_or(ParseError::MissingNode(pos))?;
            lambda_helper(pos, var_declrations.into_inner(), Some(ty_p), expr_p)
        }
        Rule::expr => {
            let expr_p = next;
            lambda_helper(pos, var_declrations.into_inner(), None, expr_p)
        }
        r => return Err(ParseError::InvalidNode(r, pos)),
    }
}

fn lambda_helper(
    pos: (usize, usize),
    mut vars: Pairs<'_, Rule>,
    ty_p: Option<Pair<'_, Rule>>,
    expr_p: Pair<'_, Rule>,
) -> ParseResult<RawFunctionExpr> {
    let var_declration = vars.next().ok_or(ParseError::MissingNode(pos))?;
    let line_col = var_declration.line_col();
    let text = var_declration.as_str().to_string();
    let (var, var_ty) = parse_var_declration(var_declration)?;

    if vars.peek().is_none() {
        let ret_ty = match ty_p {
            Some(p) => parse_type(p)?,
            None => PartialType::empty(),
        };
        let expr = parse_expr(expr_p)?;

        Ok(RawFunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr,
        })
    } else {
        let f_expr = lambda_helper(pos, vars, ty_p, expr_p)?;
        let ret_ty = match &f_expr {
            RawFunctionExpr::Lambda {
                var: _,
                var_ty,
                ret_ty,
                expr: _,
            } => {
                let p_t = PartialType {
                    value: Some(PartialValue::Function {
                        param: Box::new(var_ty.clone()),
                        return_type: Box::new(ret_ty.clone()),
                    }),
                    labels: None,
                };
                p_t
            }
            _ => unreachable!("fn lambdahelper produces apply statement"),
        };
        let expr = RawExpr::new(
            RawExprKind::Function(Box::new(f_expr)),
            (),
            Span::new(line_col, text),
        );

        Ok(RawFunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr,
        })
    }
}

fn parse_apply(p: Pair<'_, Rule>) -> ParseResult<RawFunctionExpr> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var = children.next().ok_or(ParseError::MissingNode(pos))?;
    let exprs = children.next().ok_or(ParseError::MissingNode(pos))?;
    apply_helper(pos, exprs.into_inner().rev().peekable(), var)
}

fn apply_helper(
    pos: (usize, usize),
    mut exprs: Peekable<Rev<Pairs<'_, Rule>>>,
    var: Pair<'_, Rule>,
) -> ParseResult<RawFunctionExpr> {
    let expr_p = exprs.next().ok_or(ParseError::MissingNode(pos))?;
    let line_col = expr_p.line_col();
    let text = expr_p.as_str().to_string();
    let arg = parse_expr(expr_p)?;
    if exprs.peek().is_none() {
        let line_col = var.line_col();
        let text = var.as_str().to_string();
        let fun = RawExpr::new(parse_var(var)?, (), Span::new(line_col, text));
        Ok(RawFunctionExpr::Apply { fun, arg })
    } else {
        let fun = RawExpr::new(
            RawExprKind::Function(Box::new(apply_helper(pos, exprs, var)?)),
            (),
            Span::new(line_col, text),
        );
        Ok(RawFunctionExpr::Apply { fun, arg })
    }
}

fn parse_blk(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let mut statements = Vec::new();
    for stmt in p.into_inner() {
        let pos = stmt.line_col();
        let text = stmt.as_str().to_string();
        statements.push(RawStmt::new(
            parse_statement(stmt)?,
            (),
            Span::new(pos, text),
        ));
    }

    Ok(RawExprKind::Block(statements))
}

fn parse_literal(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let literal_part = children.next().ok_or(ParseError::MissingNode(pos))?;
    let literal = literal_helper(literal_part)?;

    Ok(RawExprKind::Literal(literal))
}

fn literal_helper(p: Pair<'_, Rule>) -> ParseResult<Literal> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let lit = children.next().ok_or(ParseError::MissingNode(pos))?;
    match lit.as_rule() {
        Rule::int_lit => Ok(Literal::Int(lit.as_str().parse::<i64>()?)),
        Rule::float_lit => Ok(Literal::Float(lit.as_str().parse::<f64>()?)),
        Rule::bool_lit => Ok(Literal::Bool(lit.as_str().parse::<bool>()?)),
        Rule::unit_lit => Ok(Literal::Unit),
        r => Err(ParseError::InvalidNode(r, pos)),
    }
}

fn parse_var(p: Pair<'_, Rule>) -> ParseResult<RawExprKind> {
    let pos = p.line_col();
    match p.as_rule() {
        Rule::var => Ok(RawExprKind::Var(p.as_str().to_string())),
        r => Err(ParseError::InvalidNode(r, pos)),
    }
}

fn bin_op_helper(p: Pair<'_, Rule>) -> ParseResult<(RawExpr, RawExpr)> {
    let pos = p.line_col();
    let mut components = p.into_inner();
    let first = components.next().ok_or(ParseError::MissingNode(pos))?;
    let line_col = first.line_col();
    let text = first.as_str().to_string();
    let left = match first.as_rule() {
        Rule::literal => RawExpr::new(parse_literal(first)?, (), Span::new(line_col, text)),
        Rule::var => RawExpr::new(
            RawExprKind::Var(first.as_str().to_string()),
            (),
            Span::new(line_col, text),
        ),
        Rule::expr => parse_expr(first)?,
        r => return Err(ParseError::InvalidNode(r, line_col)),
    };
    let second = components.next().ok_or(ParseError::MissingNode(pos))?;
    let right = parse_expr(second)?;

    Ok((left, right))
}

fn parse_var_declration(p: Pair<'_, Rule>) -> ParseResult<(String, PartialType)> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var = children
        .next()
        .ok_or(ParseError::MissingNode(pos))?
        .as_str()
        .to_string();

    let ty = if let Some(typing) = children.next() {
        let t = typing
            .into_inner()
            .next()
            .ok_or(ParseError::MissingNode(pos))?;
        parse_type(t)?
    } else {
        PartialType::empty()
    };

    Ok((var, ty))
}

fn parse_type(p: Pair<'_, Rule>) -> ParseResult<PartialType> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let (value, label) = match child.as_rule() {
        Rule::type_full => {
            let mut children = child.into_inner();
            let value_type = children.next().ok_or(ParseError::MissingNode(pos))?;
            let labels = children.next().ok_or(ParseError::MissingNode(pos))?;

            (
                Some(parse_value_type(value_type)?),
                Some(parse_labels(labels)?),
            )
        }
        Rule::type_val => {
            let value_type = child
                .into_inner()
                .next()
                .ok_or(ParseError::MissingNode(pos))?;
            (Some(parse_value_type(value_type)?), None)
        }
        Rule::type_label => {
            let labels = child
                .into_inner()
                .next()
                .ok_or(ParseError::MissingNode(pos))?;
            (None, Some(parse_labels(labels)?))
        }
        r => return Err(ParseError::InvalidNode(r, pos)),
    };

    Ok(PartialType {
        value,
        labels: label,
    })
}

fn parse_value_type(p: Pair<'_, Rule>) -> ParseResult<PartialValue> {
    let pos = p.line_col();
    let child = p.into_inner().next().ok_or(ParseError::MissingNode(pos))?;
    let val = match child.as_rule() {
        Rule::int_type => PartialValue::Int,
        Rule::float_type => PartialValue::Float,
        Rule::bool_type => PartialValue::Bool,
        Rule::fun_type => parse_fun_type(child)?,
        r => return Err(ParseError::InvalidNode(r, pos)),
    };

    Ok(val)
}

fn parse_fun_type(p: Pair<'_, Rule>) -> ParseResult<PartialValue> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let mut children = p.into_inner();
    let value_type = children.next().ok_or(ParseError::MissingNode(pos))?;
    let value = parse_value_type(value_type)?;
    let next = children.next().ok_or(ParseError::MissingNode(pos))?;
    match next.as_rule() {
        Rule::labels => {
            let label = parse_labels(next)?;
            let param = Box::new(PartialType {
                value: Some(value),
                labels: Some(label),
            });
            let t = children.next().ok_or(ParseError::MissingNode(pos))?;
            let ty = parse_type(t)?;

            Ok(PartialValue::Function {
                param,
                return_type: Box::new(ty),
            })
        }
        Rule::r#type => {
            let param = Box::new(PartialType {
                value: Some(value),
                labels: None,
            });
            let ty = parse_type(next)?;
            Ok(PartialValue::Function {
                param,
                return_type: Box::new(ty),
            })
        }
        r => Err(ParseError::InvalidNode(r, pos)),
    }
}

fn parse_labels(p: Pair<'_, Rule>) -> ParseResult<Labels> {
    let pos = p.line_col();
    let mut labels = Vec::new();

    for label in p.into_inner() {
        if label.as_rule() != Rule::label {
            return Err(ParseError::InvalidNode(label.as_rule(), pos));
        }

        labels.push(LabelInterner::intern_label(
            label
                .into_inner()
                .next()
                .ok_or(ParseError::MissingNode(pos))?
                .as_str(),
        ));
    }

    Ok(Labels::new(labels))
}

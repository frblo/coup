use std::iter::Peekable;
use std::iter::Rev;

use pest::Parser;
use pest::iterators::Pair;
use pest::iterators::Pairs;
use pest_derive::Parser;

use super::error::CompilerError;
use super::error::Result;

use super::types::ast;
use super::types::PartialValue;
use super::types::spanned::Spanned;
use super::types::PartialType;
use super::types::label;
use super::types::label::Labels;

#[derive(Parser)]
#[grammar = "compiler/parser/grammar.pest"]
pub struct CoupParser;

pub fn parse_program(source: &str) -> Result<ast::Program> {
    if source.is_empty() {
        return Ok(ast::Program(Vec::new()));
    }

    let mut parsed = CoupParser::parse(Rule::program, source)?;
    let program = parsed.next().ok_or(CompilerError::MissingRootNode)?;

    let mut statements = Vec::new();
    for pair in program.into_inner() {
        if pair.as_rule() == Rule::EOI {
            break;
        }

        let pos = pair.line_col();
        let text = pair.as_str().to_string();

        statements.push(Spanned::new(parse_statement(pair)?, pos, text));
    }

    Ok(ast::Program(statements))
}

fn parse_statement(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let stmt = match child.as_rule() {
        Rule::skip_stm => ast::Stmt::Skip,
        Rule::let_stm => parse_let(child)?,
        Rule::if_stm => parse_if(child)?,
        Rule::while_stm => parse_while(child)?,
        Rule::expr_stm => parse_expr_stmt(child)?,
        Rule::return_stm => parse_return(child)?,
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };

    Ok(stmt)
}

fn parse_let(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var_decl = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let (var, ty) = parse_var_declration(var_decl)?;
    let expr = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let value = parse_expr(expr)?;

    Ok(ast::Stmt::Let { var, ty, value })
}

fn parse_if(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let condition = parse_expr(children.next().ok_or(CompilerError::MissingNode(pos))?)?;
    let then_p = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let line_col = then_p.line_col();
    let text = then_p.as_str().to_string();
    let then_branch = Spanned::new(parse_blk(then_p)?, line_col, text);

    let mut else_if_branches = Vec::new();
    let mut else_branch = None;
    while let Some(next) = children.next() {
        match next.as_rule() {
            // If else branch
            Rule::expr => {
                let cond = parse_expr(next)?;
                let then_p = children.next().ok_or(CompilerError::MissingNode(pos))?;
                let line_col = then_p.line_col();
                let text = then_p.as_str().to_string();
                let then = Spanned::new(parse_blk(then_p)?, line_col, text);
                else_if_branches.push((cond, then));
            },
            // Else branch
            Rule::blk_expr => {
                let line_col = next.line_col();
                let text = next.as_str().to_string();
                else_branch = Some(Spanned::new(parse_blk(next)?, line_col, text));
            },
            r => return Err(CompilerError::InvalidNode(r, pos)),
        }
    }

    Ok(ast::Stmt::If {
        condition,
        then_branch,
        else_if_branches,
        else_branch,
    })
}

fn parse_while(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let condition = parse_expr(children.next().ok_or(CompilerError::MissingNode(pos))?)?;
    let body_p = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let line_col = body_p.line_col();
    let text = body_p.as_str().to_string();
    let body = Spanned::new(parse_blk(body_p)?, line_col, text);

    Ok(ast::Stmt::While { condition, body })
}

fn parse_expr_stmt(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    Ok(ast::Stmt::Expr(parse_expr(child)?))
}

fn parse_return(p: Pair<'_, Rule>) -> Result<ast::Stmt> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    Ok(ast::Stmt::Return(parse_expr(child)?))
}

fn parse_expr(p: Pair<'_, Rule>) -> Result<Spanned<ast::Expr>> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::bool_expr => parse_bool(child)?,
        Rule::arithmetic_expr => parse_arithmetic(child)?,
        Rule::function_expr => parse_function(child)?,
        Rule::blk_expr => parse_blk(child)?,
        Rule::literal => parse_literal(child)?,
        Rule::var => parse_var(child)?,
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };

    Ok(Spanned::new(expr, pos, text))
}

fn parse_bool(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::and_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::And(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::or_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Or(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::eq_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Eq(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::le_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Le(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::leq_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Leq(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::ge_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Ge(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::geq_expr => {
            let (l, r) = bin_op_helper(child)?;
            let b_expr = ast::BoolExpr::Geq(l, r);
            ast::Expr::Bool(Box::new(b_expr))
        }
        Rule::neg_expr => {
            let expr = parse_expr(
                child
                    .into_inner()
                    .next()
                    .ok_or(CompilerError::MissingNode(pos))?,
            )?;
            ast::Expr::Bool(Box::new(ast::BoolExpr::Neg(expr)))
        }
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };
    Ok(expr)
}

fn parse_arithmetic(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::add_expr => {
            let (l, r) = bin_op_helper(child)?;
            let a_expr = ast::ArithmeticExpr::Add(l, r);
            ast::Expr::Arithmetic(Box::new(a_expr))
        }
        Rule::sub_expr => {
            let (l, r) = bin_op_helper(child)?;
            let a_expr = ast::ArithmeticExpr::Sub(l, r);
            ast::Expr::Arithmetic(Box::new(a_expr))
        }
        Rule::mul_expr => {
            let (l, r) = bin_op_helper(child)?;
            let a_expr = ast::ArithmeticExpr::Mul(l, r);
            ast::Expr::Arithmetic(Box::new(a_expr))
        }
        Rule::div_expr => {
            let (l, r) = bin_op_helper(child)?;
            let a_expr = ast::ArithmeticExpr::Div(l, r);
            ast::Expr::Arithmetic(Box::new(a_expr))
        }
        Rule::mod_expr => {
            let (l, r) = bin_op_helper(child)?;
            let a_expr = ast::ArithmeticExpr::Mod(l, r);
            ast::Expr::Arithmetic(Box::new(a_expr))
        }
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };
    Ok(expr)
}

fn parse_function(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let expr = match child.as_rule() {
        Rule::lambda_expr => parse_lambda(child)?,
        Rule::apply_expr => parse_apply(child)?,
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };

    Ok(ast::Expr::Function(Box::new(expr)))
}

fn parse_lambda(p: Pair<'_, Rule>) -> Result<ast::FunctionExpr> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var_declrations = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let next = children.next().ok_or(CompilerError::MissingNode(pos))?;
    match next.as_rule() {
        Rule::r#type => {
            let ty_p = next;
            let expr_p = children.next().ok_or(CompilerError::MissingNode(pos))?;
            lambda_helper(pos, var_declrations.into_inner(), Some(ty_p), expr_p)
        }
        Rule::expr => {
            let expr_p = next;
            lambda_helper(pos, var_declrations.into_inner(), None, expr_p)
        }
        r => return Err(CompilerError::InvalidNode(r, pos)),
    }
}

fn lambda_helper(
    pos: (usize, usize),
    mut vars: Pairs<'_, Rule>,
    ty_p: Option<Pair<'_, Rule>>,
    expr_p: Pair<'_, Rule>,
) -> Result<ast::FunctionExpr> {
    let var_declration = vars.next().ok_or(CompilerError::MissingNode(pos))?;
    let line_col = var_declration.line_col();
    let text = var_declration.as_str().to_string();
    let (var, var_ty) = parse_var_declration(var_declration)?;

    if vars.peek().is_none() {
        let ret_ty = match ty_p {
            Some(p) => parse_type(p)?,
            None => Spanned::new(PartialType::empty(), line_col, "".to_string()),
        };
        let expr = parse_expr(expr_p)?;

        Ok(ast::FunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr,
        })
    } else {
        let f_expr = lambda_helper(pos, vars, ty_p, expr_p)?;
        let ret_ty = match &f_expr {
            ast::FunctionExpr::Lambda { var: _, var_ty, ret_ty, expr: _ } => {
                let p_t = PartialType {
                    value: Some(PartialValue::Function { param: Box::new(var_ty.node.clone()), return_type: Box::new(ret_ty.node.clone()) }),
                    labels: None,
                };
                Spanned::new(p_t, line_col, "".to_string())
            },
            _ => unreachable!("fn lambdahelper produces apply statement"),
        };
        let expr = Spanned::new(ast::Expr::Function(Box::new(f_expr)), line_col, text);

        Ok(ast::FunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr
        })
    }
}

fn parse_apply(p: Pair<'_, Rule>) -> Result<ast::FunctionExpr> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let exprs = children.next().ok_or(CompilerError::MissingNode(pos))?;
    apply_helper(pos, exprs.into_inner().rev().peekable(), var)
}

fn apply_helper(
    pos: (usize, usize),
    mut exprs: Peekable<Rev<Pairs<'_, Rule>>>,
    var: Pair<'_, Rule>,
) -> Result<ast::FunctionExpr> {
    let expr_p = exprs.next().ok_or(CompilerError::MissingNode(pos))?;
    let line_col = expr_p.line_col();
    let text = expr_p.as_str().to_string();
    let arg = parse_expr(expr_p)?;
    if exprs.peek().is_none() {
        let line_col = var.line_col();
        let text = var.as_str().to_string();
        let fun = Spanned::new(parse_var(var)?, line_col, text);
        Ok(ast::FunctionExpr::Apply { fun, arg })
    } else {
        let fun = Spanned::new(ast::Expr::Function(Box::new(apply_helper(pos, exprs, var)?)), line_col, text);
        Ok(ast::FunctionExpr::Apply { fun, arg })
    }
}

fn parse_blk(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let mut statements = Vec::new();
    for stmt in p.into_inner() {
        let pos = stmt.line_col();
        let text = stmt.as_str().to_string();
        statements.push(Spanned::new(parse_statement(stmt)?, pos, text));
    }

    Ok(ast::Expr::Block(statements))
}

fn parse_literal(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let mut children = p.into_inner();
    let literal_part = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let literal = literal_helper(literal_part)?;

    let value = match literal {
        ast::Literal::Int(_) => PartialValue::Int,
        ast::Literal::Float(_) => PartialValue::Float,
        ast::Literal::Bool(_) => PartialValue::Bool,
        ast::Literal::Unit => PartialValue::Unit,
    };

    let ty = if let Some(labels_p) = children.next() {
        let label = parse_labels(labels_p)?;
        Spanned::new(PartialType {
            value: Some(value),
            labels: Some(label),
        }, pos, text)
    } else {
        Spanned::new(PartialType {
            value: Some(value),
            labels: None,
        }, pos, text)
    };

    Ok(ast::Expr::Literal(literal, ty))
}

fn literal_helper(p: Pair<'_, Rule>) -> Result<ast::Literal> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let lit = children.next().ok_or(CompilerError::MissingNode(pos))?;
    match lit.as_rule() {
        Rule::int_lit => Ok(ast::Literal::Int(lit.as_str().parse::<i64>()?)),
        Rule::float_lit => Ok(ast::Literal::Float(lit.as_str().parse::<f64>()?)),
        Rule::bool_lit => Ok(ast::Literal::Bool(lit.as_str().parse::<bool>()?)),
        Rule::unit_lit => Ok(ast::Literal::Unit),
        r => Err(CompilerError::InvalidNode(r, pos)),
    }
}

fn parse_var(p: Pair<'_, Rule>) -> Result<ast::Expr> {
    let pos = p.line_col();
    match p.as_rule() {
        Rule::var => Ok(ast::Expr::Var(p.as_str().to_string())),
        r => Err(CompilerError::InvalidNode(r, pos)),
    }
}

fn bin_op_helper(p: Pair<'_, Rule>) -> Result<(Spanned<ast::Expr>, Spanned<ast::Expr>)> {
    let pos = p.line_col();
    let mut components = p.into_inner();
    let first = components.next().ok_or(CompilerError::MissingNode(pos))?;
    let line_col = first.line_col();
    let text = first.as_str().to_string();
    let left = match first.as_rule() {
        Rule::literal => Spanned::new(parse_literal(first)?, line_col, text),
        Rule::var => Spanned::new(ast::Expr::Var(first.as_str().to_string()), line_col, text),
        Rule::expr => parse_expr(first)?,
        r => return Err(CompilerError::InvalidNode(r, line_col)),
    };
    let second = components.next().ok_or(CompilerError::MissingNode(pos))?;
    let right = parse_expr(second)?;

    Ok((left, right))
}

fn parse_var_declration(p: Pair<'_, Rule>) -> Result<(String, Spanned<PartialType>)> {
    let pos = p.line_col();
    let mut children = p.into_inner();
    let var = children
        .next()
        .ok_or(CompilerError::MissingNode(pos))?
        .as_str()
        .to_string();

    let ty = if let Some(typing) = children.next() {
        let t = typing
            .into_inner()
            .next()
            .ok_or(CompilerError::MissingNode(pos))?;
        parse_type(t)?
    } else {
        Spanned::new(PartialType::empty(), pos, "".to_string())
    };

    Ok((var, ty))
}

fn parse_type(p: Pair<'_, Rule>) -> Result<Spanned<PartialType>> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let (value, label) = match child.as_rule() {
        Rule::type_full => {
            let mut children = child.into_inner();
            let value_type = children.next().ok_or(CompilerError::MissingNode(pos))?;
            let labels = children.next().ok_or(CompilerError::MissingNode(pos))?;

            (
                Some(parse_value_type(value_type)?),
                Some(parse_labels(labels)?),
            )
        }
        Rule::type_val => {
            let value_type = child
                .into_inner()
                .next()
                .ok_or(CompilerError::MissingNode(pos))?;
            (Some(parse_value_type(value_type)?), None)
        }
        Rule::type_label => {
            let labels = child
                .into_inner()
                .next()
                .ok_or(CompilerError::MissingNode(pos))?;
            (None, Some(parse_labels(labels)?))
        }
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };

    Ok(Spanned::new(PartialType { value, labels: label }, pos, text))
}

fn parse_value_type(p: Pair<'_, Rule>) -> Result<PartialValue> {
    let pos = p.line_col();
    let child = p
        .into_inner()
        .next()
        .ok_or(CompilerError::MissingNode(pos))?;
    let val = match child.as_rule() {
        Rule::int_type => PartialValue::Int,
        Rule::float_type => PartialValue::Float,
        Rule::bool_type => PartialValue::Bool,
        Rule::fun_type => parse_fun_type(child)?,
        r => return Err(CompilerError::InvalidNode(r, pos)),
    };

    Ok(val)
}

fn parse_fun_type(p: Pair<'_, Rule>) -> Result<PartialValue> {
    let pos = p.line_col();
    let text = p.as_str().to_string();
    let mut children = p.into_inner();
    let value_type = children.next().ok_or(CompilerError::MissingNode(pos))?;
    let value = parse_value_type(value_type)?;
    let next = children.next().ok_or(CompilerError::MissingNode(pos))?;
    match next.as_rule() {
        Rule::labels => {
            let label = parse_labels(next)?;
            let param = Box::new(PartialType {
                value: Some(value),
                labels: Some(label),
            });
            let t = children.next().ok_or(CompilerError::MissingNode(pos))?;
            let ty = parse_type(t)?;

            Ok(PartialValue::Function {
                param,
                return_type: Box::new(ty.node),
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
                return_type: Box::new(ty.node),
            })
        }
        r => Err(CompilerError::InvalidNode(r, pos)),
    }
}

fn parse_labels(p: Pair<'_, Rule>) -> Result<Labels> {
    let pos = p.line_col();
    let mut labels = Vec::new();

    for label in p.into_inner() {
        if label.as_rule() != Rule::label {
            return Err(CompilerError::InvalidNode(label.as_rule(), pos));
        }

        labels.push(
            label::intern_label(label
                .into_inner()
                .next()
                .ok_or(CompilerError::MissingNode(pos))?
                .as_str()
            )
        );
    }

    Ok(Labels::new(labels))
}

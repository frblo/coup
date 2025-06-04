use std::collections::HashMap;

use crate::compiler::types::unify;

use super::error::Result;
use super::error::TypeError;
use super::error::TypeResult;

use super::types::inferred_ast::{
    InferredArithmeticExpr, InferredBoolExpr, InferredExpr, InferredExprKind, InferredFunctionExpr,
    InferredProgram, InferredStmt, InferredStmtKind,
};

use super::types::label::Labels;
use super::types::label::flows_to;
use super::types::label::join_labels;
use super::types::raw_ast::{
    RawArithmeticExpr, RawBoolExpr, RawExpr, RawExprKind, RawFunctionExpr, RawProgram, RawStmt,
    RawStmtKind,
};

use super::types::Literal;
use super::types::PartialType;
use super::types::PartialValue;
use super::types::Span;

type TypeEnv = HashMap<String, PartialType>;

struct Context {
    type_env: TypeEnv,
    pc: Labels,
}

pub fn infer_program(program: &mut RawProgram) -> Result<InferredProgram> {
    let type_env = TypeEnv::new();
    let pc = Labels::new(Vec::new());
    let mut ctx = Context { type_env, pc };

    let mut inferred_stmts = Vec::new();
    for stmt in &mut program.0 {
        inferred_stmts.push(infer_stmt(stmt, &mut ctx)?);
    }

    Ok(InferredProgram::new(inferred_stmts))
}

fn infer_stmt(stmt: &mut RawStmt, ctx: &mut Context) -> TypeResult<InferredStmt> {
    let span = &mut stmt.meta;
    match &mut stmt.stmt {
        RawStmtKind::Skip => Ok(InferredStmt {
            stmt: InferredStmtKind::Skip,
            ty: PartialType {
                value: Some(PartialValue::Unit),
                labels: None,
            },
            meta: span.to_owned(),
        }),
        RawStmtKind::Let { var, ty, value } => infer_let(var, ty, value, ctx, span),
        RawStmtKind::If {
            condition,
            then_branch,
            else_if_branches,
            else_branch,
        } => infer_if(
            condition,
            then_branch,
            else_if_branches,
            else_branch,
            ctx,
            span,
        ),
        RawStmtKind::While { condition, body } => infer_while(condition, body, ctx, span),
        RawStmtKind::Expr(expr) => {
            let expr = infer_expr(expr, ctx)?;
            Ok(InferredStmt {
                stmt: InferredStmtKind::Expr(expr),
                ty: PartialType {
                    value: Some(PartialValue::Unit),
                    labels: None,
                },
                meta: span.to_owned(),
            })
        }
        RawStmtKind::Return(expr) => {
            let expr = infer_expr(expr, ctx)?;
            let ty = PartialType {
                value: expr.ty.value.clone(),
                labels: expr.ty.labels.clone(),
            };
            Ok(InferredStmt {
                stmt: InferredStmtKind::Return(expr),
                ty,
                meta: span.to_owned(),
            })
        }
    }
}

fn infer_let(
    var: &mut str,
    var_ty: &mut PartialType,
    value: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredStmt> {
    let mut expr = infer_expr(value, ctx)?;

    unify(&mut var_ty.value, &mut expr.ty.value)?;

    if var_ty.labels.is_none() {
        ctx.type_env.insert(var.to_string(), expr.ty.clone());

        Ok(InferredStmt {
            stmt: InferredStmtKind::Let {
                var: var.to_string(),
                ty: var_ty.clone(),
                value: expr,
            },
            ty: PartialType {
                value: Some(PartialValue::Unit),
                labels: None,
            },
            meta: span.to_owned(),
        })
    } else if flows_to(&mut expr.ty.labels, &mut var_ty.labels)
        .ok_or(TypeError::MissingType(span.clone()))?
    {
        ctx.type_env.insert(var.to_string(), var_ty.clone());

        Ok(InferredStmt {
            stmt: InferredStmtKind::Let {
                var: var.to_string(),
                ty: var_ty.clone(),
                value: expr,
            },
            ty: PartialType {
                value: Some(PartialValue::Unit),
                labels: None,
            },
            meta: span.to_owned(),
        })
    } else {
        Err(TypeError::InvalidFlow(
            expr.ty.labels.unwrap(),
            var_ty.labels.clone().unwrap(),
            span.clone(),
        ))
    }
}

fn infer_if(
    condition: &mut RawExpr,
    then_branch: &mut RawExpr,
    else_if_branches: &mut Vec<(RawExpr, RawExpr)>,
    else_branch: &mut Option<RawExpr>,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredStmt> {
    let mut condition = infer_expr(condition, ctx)?;
    unify(&mut condition.ty.value, &mut Some(PartialValue::Bool))?;

    let old_pc = ctx.pc.clone();
    ctx.pc = ctx.pc.join_labels(
        condition
            .ty
            .labels
            .as_ref()
            .ok_or(TypeError::MissingType(span.clone()))?,
    );

    let then_branch = infer_expr(then_branch, ctx)?;

    let mut else_ifs = Vec::new();
    for (cond, then) in else_if_branches {
        let mut cond = infer_expr(cond, ctx)?;
        unify(&mut cond.ty.value, &mut Some(PartialValue::Bool))?;

        let old_pc = ctx.pc.clone();
        ctx.pc = ctx.pc.join_labels(
            cond.ty
                .labels
                .as_ref()
                .ok_or(TypeError::MissingType(span.clone()))?,
        );

        let then = infer_expr(then, ctx)?;

        ctx.pc = old_pc;

        else_ifs.push((cond, then));
    }

    let else_branch = if let Some(b) = else_branch.as_mut() {
        Some(infer_expr(b, ctx)?)
    } else {
        None
    };

    ctx.pc = old_pc;

    Ok(InferredStmt {
        stmt: InferredStmtKind::If {
            condition,
            then_branch,
            else_if_branches: else_ifs,
            else_branch,
        },
        ty: PartialType {
            value: Some(PartialValue::Unit),
            labels: None,
        },
        meta: span.to_owned(),
    })
}

fn infer_while(
    condition: &mut RawExpr,
    body: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredStmt> {
    let mut condition = infer_expr(condition, ctx)?;
    unify(&mut condition.ty.value, &mut Some(PartialValue::Bool))?;

    let old_pc = ctx.pc.clone();
    ctx.pc = ctx.pc.join_labels(
        condition
            .ty
            .labels
            .as_ref()
            .ok_or(TypeError::MissingType(span.clone()))?,
    );

    let body = infer_expr(body, ctx)?;

    ctx.pc = old_pc;

    Ok(InferredStmt {
        stmt: InferredStmtKind::While { condition, body },
        ty: PartialType {
            value: Some(PartialValue::Unit),
            labels: None,
        },
        meta: span.to_owned(),
    })
}

fn infer_expr(expr: &mut RawExpr, ctx: &mut Context) -> TypeResult<InferredExpr> {
    let span = &mut expr.meta;
    match &mut expr.expr {
        RawExprKind::Bool(b) => infer_bool(b, ctx, span),
        RawExprKind::Arithmetic(a) => infer_arithmetic(a, ctx, span),
        RawExprKind::Function(f) => infer_function(f, ctx, span),
        RawExprKind::Block(b) => infer_block(b, ctx, span),
        RawExprKind::Var(v) => infer_var(v, ctx, span),
        RawExprKind::Literal(l) => infer_literal(l, span),
    }
}

fn infer_bool(
    expr: &mut RawBoolExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    match expr {
        RawBoolExpr::And(lhs, rhs) => {
            let (lhs, rhs, ty) = bin_op_helper(
                lhs,
                rhs,
                vec![PartialValue::Int, PartialValue::Float],
                ctx,
                span,
            )?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::And(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Or(lhs, rhs) => {
            let (lhs, rhs, ty) = bin_op_helper(
                lhs,
                rhs,
                vec![PartialValue::Int, PartialValue::Float],
                ctx,
                span,
            )?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Or(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Eq(lhs, rhs) => {
            let (lhs, rhs, ty) = bin_op_helper(
                lhs,
                rhs,
                vec![PartialValue::Int, PartialValue::Float],
                ctx,
                span,
            )?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Eq(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Le(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_boolean_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Le(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Leq(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_boolean_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Leq(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Ge(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_boolean_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Ge(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Geq(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_boolean_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Geq(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawBoolExpr::Neg(lhs) => {
            let mut lhs = infer_expr(lhs, ctx)?;

            if unify(&mut lhs.ty.value, &mut Some(PartialValue::Bool)).is_err() {
                return Err(TypeError::ExpectedBoolean(
                    lhs.ty.value.unwrap(),
                    span.clone(),
                ));
            }

            let ty = PartialType {
                value: Some(
                    lhs.ty
                        .value
                        .as_ref()
                        .ok_or(TypeError::MissingType(span.clone()))?
                        .clone(),
                ),
                labels: Some(
                    lhs.ty
                        .labels
                        .as_ref()
                        .ok_or(TypeError::MissingType(span.clone()))?
                        .clone(),
                ),
            };

            Ok(InferredExpr::new(
                InferredExprKind::Bool(Box::new(InferredBoolExpr::Neg(lhs))),
                ty,
                span.to_owned(),
            ))
        }
    }
}

fn infer_arithmetic(
    expr: &mut RawArithmeticExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    match expr {
        RawArithmeticExpr::Add(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Arithmetic(Box::new(InferredArithmeticExpr::Add(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawArithmeticExpr::Sub(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Arithmetic(Box::new(InferredArithmeticExpr::Sub(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawArithmeticExpr::Mul(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Arithmetic(Box::new(InferredArithmeticExpr::Mul(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawArithmeticExpr::Div(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Arithmetic(Box::new(InferredArithmeticExpr::Div(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
        RawArithmeticExpr::Mod(lhs, rhs) => {
            let (lhs, rhs, ty) = arithmetic_helper(lhs, rhs, ctx, span)?;
            Ok(InferredExpr::new(
                InferredExprKind::Arithmetic(Box::new(InferredArithmeticExpr::Mod(lhs, rhs))),
                ty,
                span.to_owned(),
            ))
        }
    }
}

fn arithmetic_helper(
    lhs: &mut RawExpr,
    rhs: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<(InferredExpr, InferredExpr, PartialType)> {
    bin_op_helper(
        lhs,
        rhs,
        vec![PartialValue::Int, PartialValue::Float],
        ctx,
        span,
    )
    .map_err(|err| match err {
        TypeError::Missmatch(lhs, _) => TypeError::ExpectedNumeric(lhs, span.clone()),
        x => x,
    })
}

fn bool_helper(
    lhs: &mut RawExpr,
    rhs: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<(InferredExpr, InferredExpr, PartialType)> {
    bin_op_helper(lhs, rhs, vec![PartialValue::Bool], ctx, span).map_err(|err| match err {
        TypeError::Missmatch(lhs, _) => TypeError::ExpectedBoolean(lhs, span.clone()),
        x => x,
    })
}

fn arithmetic_boolean_helper(
    lhs: &mut RawExpr,
    rhs: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<(InferredExpr, InferredExpr, PartialType)> {
    let expected = vec![PartialValue::Int, PartialValue::Float];
    let mut lhs = infer_expr(lhs, ctx)?;
    let mut rhs = infer_expr(rhs, ctx)?;

    if expected
        .into_iter()
        .all(|expect| unify(&mut lhs.ty.value, &mut Some(expect)).is_err())
    {
        return Err(TypeError::ExpectedNumeric(
            lhs.ty.value.unwrap(),
            span.clone(),
        ));
    }

    unify(&mut lhs.ty.value, &mut rhs.ty.value)?;

    let ty = PartialType {
        value: Some(PartialValue::Bool),
        labels: Some(join_labels(&lhs.ty.labels, &mut rhs.ty.labels)),
    };

    Ok((lhs, rhs, ty))
}

fn bin_op_helper(
    lhs: &mut RawExpr,
    rhs: &mut RawExpr,
    expected: Vec<PartialValue>,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<(InferredExpr, InferredExpr, PartialType)> {
    let mut lhs = infer_expr(lhs, ctx)?;
    let mut rhs = infer_expr(rhs, ctx)?;

    let exp = expected[0].clone();
    if expected
        .into_iter()
        .all(|expect| unify(&mut lhs.ty.value, &mut Some(expect)).is_err())
    {
        return Err(TypeError::Missmatch(lhs.ty.value.unwrap(), exp));
    }

    unify(&mut lhs.ty.value, &mut rhs.ty.value)?;

    let ty = PartialType {
        value: Some(
            lhs.ty
                .value
                .as_ref()
                .ok_or(TypeError::MissingType(span.clone()))?
                .clone(),
        ),
        labels: Some(join_labels(&mut lhs.ty.labels, &rhs.ty.labels)),
    };

    Ok((lhs, rhs, ty))
}

fn infer_function(
    expr: &mut RawFunctionExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    match expr {
        RawFunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr,
        } => infer_lambda(var, var_ty, ret_ty, expr, ctx, span),
        RawFunctionExpr::Apply { fun, arg } => infer_apply(fun, arg, ctx, span),
    }
}

fn infer_lambda(
    var: &mut String,
    var_ty: &mut PartialType,
    ret_ty: &mut PartialType,
    expr: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    // TODO:
    // NEED TO FIX LABEL INFERENCE.

    ctx.type_env.insert(var.clone(), var_ty.clone());

    let mut body = infer_expr(expr, ctx)?;
    unify(&mut ret_ty.value, &mut body.ty.value)?;
    let labels = join_labels(&ret_ty.labels, &body.ty.labels);

    let fun_type = PartialValue::Function {
        param: Box::new(var_ty.clone()),
        return_type: Box::new(body.ty.clone()),
    };

    Ok(InferredExpr {
        expr: InferredExprKind::Function(Box::new(InferredFunctionExpr::Lambda {
            var: var.clone(),
            var_ty: var_ty.clone(),
            ret_ty: body.ty.clone(),
            expr: body,
        })),
        ty: PartialType {
            value: Some(fun_type),
            labels: Some(labels),
        },
        meta: span.to_owned(),
    })
}

fn infer_apply(
    fun: &mut RawExpr,
    arg: &mut RawExpr,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    let fun = infer_expr(fun, ctx)?;
    let mut arg = infer_expr(arg, ctx)?;

    let PartialValue::Function {
        mut param,
        return_type,
    } = fun
        .ty
        .value
        .clone()
        .ok_or(TypeError::MissingType(span.clone()))?
    else {
        return Err(TypeError::ExpectedFunction(
            fun.ty.value.unwrap(),
            span.clone(),
        ));
    };

    unify(&mut arg.ty.value, &mut param.value)?;

    let labels = join_labels(&fun.ty.labels, &arg.ty.labels);

    Ok(InferredExpr {
        expr: InferredExprKind::Function(Box::new(InferredFunctionExpr::Apply { fun, arg })),
        ty: PartialType {
            value: return_type.value,
            labels: Some(labels),
        },
        meta: span.to_owned(),
    })
}

fn infer_block(
    stmts: &mut Vec<RawStmt>,
    ctx: &mut Context,
    span: &mut Span,
) -> TypeResult<InferredExpr> {
    let mut inferred_stmts = Vec::new();
    for stmt in stmts {
        inferred_stmts.push(infer_stmt(stmt, ctx)?);
    }

    let ty = if let Some(InferredStmt {
        stmt: InferredStmtKind::Return(_),
        ty,
        meta: _,
    }) = inferred_stmts.last()
    {
        ty.clone()
    } else {
        PartialType {
            value: Some(PartialValue::Unit),
            labels: None,
        }
    };

    Ok(InferredExpr {
        expr: InferredExprKind::Block(inferred_stmts),
        ty,
        meta: span.to_owned(),
    })
}

fn infer_var(var: &mut str, ctx: &mut Context, span: &mut Span) -> TypeResult<InferredExpr> {
    if let Some(ty) = ctx.type_env.get(var) {
        Ok(InferredExpr {
            expr: InferredExprKind::Var(var.to_string()),
            ty: ty.clone(),
            meta: span.to_owned(),
        })
    } else {
        Err(TypeError::UndeclaredVar(var.to_string(), span.clone()))
    }
}

fn infer_literal(lit: &mut Literal, span: &mut Span) -> TypeResult<InferredExpr> {
    let value = match lit {
        Literal::Int(_) => Some(PartialValue::Int),
        Literal::Float(_) => Some(PartialValue::Float),
        Literal::Bool(_) => Some(PartialValue::Bool),
        Literal::Unit => Some(PartialValue::Unit),
    };

    let ty = PartialType {
        value,
        labels: Some(Labels::new(vec![])),
    };

    Ok(InferredExpr::new(
        InferredExprKind::Literal(lit.clone()),
        ty,
        span.to_owned(),
    ))
}

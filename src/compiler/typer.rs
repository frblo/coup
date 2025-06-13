use super::error::{Result, TypeError, TypeResult};

use super::types::inferred_ast::{
    InferredArithmeticExpr, InferredBoolExpr, InferredExpr, InferredExprKind, InferredFunctionExpr,
    InferredProgram, InferredStmt, InferredStmtKind,
};

use super::types::typed_ast::{
    TypedArithmeticExpr, TypedBoolExpr, TypedExpr, TypedExprKind, TypedFunctionExpr, TypedProgram,
    TypedStmt, TypedStmtKind,
};

pub fn type_program(program: InferredProgram) -> Result<TypedProgram> {
    let mut typed_stmts = Vec::new();
    for stmt in program.0 {
        typed_stmts.push(type_stmt(stmt)?);
    }

    Ok(TypedProgram::new(typed_stmts))
}

fn type_stmt(stmt: InferredStmt) -> TypeResult<TypedStmt> {
    let span = stmt.meta;
    let full_ty = stmt
        .ty
        .to_full()
        .ok_or(TypeError::MissingType(span.clone()))?;

    match stmt.stmt {
        InferredStmtKind::Skip => Ok(TypedStmt {
            stmt: TypedStmtKind::Skip,
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::Let { var, ty, value } => Ok(TypedStmt {
            stmt: TypedStmtKind::Let {
                var,
                ty,
                value: type_expr(value)?,
            },
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::If {
            condition,
            then_branch,
            else_if_branches,
            else_branch,
        } => Ok(TypedStmt {
            stmt: TypedStmtKind::If {
                condition: type_expr(condition)?,
                then_branch: type_expr(then_branch)?,
                else_if_branches: else_if_branches
                    .into_iter()
                    .map(|(cond, then)| {
                        let cond = type_expr(cond)?;
                        let then = type_expr(then)?;
                        Ok((cond, then))
                    })
                    .collect::<TypeResult<Vec<_>>>()?,
                else_branch: else_branch.map(|e| type_expr(e)).transpose()?,
            },
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::While { condition, body } => Ok(TypedStmt {
            stmt: TypedStmtKind::While {
                condition: type_expr(condition)?,
                body: type_expr(body)?,
            },
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::Expr(expr) => Ok(TypedStmt {
            stmt: TypedStmtKind::Expr(type_expr(expr)?),
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::Return(expr) => Ok(TypedStmt {
            stmt: TypedStmtKind::Return(type_expr(expr)?),
            ty: full_ty,
            meta: (),
        }),
    }
}

fn type_expr(expr: InferredExpr) -> TypeResult<TypedExpr> {
    let span = expr.meta;
    let ty = expr
        .ty
        .to_full()
        .ok_or(TypeError::MissingType(span.clone()))?;

    let expr = match expr.expr {
        InferredExprKind::Bool(b) => TypedExprKind::Bool(Box::new(type_bool(*b)?)),
        InferredExprKind::Arithmetic(a) => {
            TypedExprKind::Arithmetic(Box::new(type_arithmetic(*a)?))
        }
        InferredExprKind::Function(f) => TypedExprKind::Function(Box::new(type_function(*f)?)),
        InferredExprKind::Block(b) => TypedExprKind::Block(type_block(b)?),
        InferredExprKind::Var(v) => TypedExprKind::Var(v),
        InferredExprKind::Literal(l) => TypedExprKind::Literal(l),
    };

    Ok(TypedExpr { expr, ty, meta: () })
}

fn type_bool(expr: InferredBoolExpr) -> TypeResult<TypedBoolExpr> {
    match expr {
        InferredBoolExpr::And(lhs, rhs) => Ok(TypedBoolExpr::And(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Or(lhs, rhs) => Ok(TypedBoolExpr::Or(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Eq(lhs, rhs) => Ok(TypedBoolExpr::Eq(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Le(lhs, rhs) => Ok(TypedBoolExpr::Le(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Leq(lhs, rhs) => Ok(TypedBoolExpr::Leq(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Ge(lhs, rhs) => Ok(TypedBoolExpr::Ge(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Geq(lhs, rhs) => Ok(TypedBoolExpr::Geq(type_expr(lhs)?, type_expr(rhs)?)),
        InferredBoolExpr::Neg(expr) => Ok(TypedBoolExpr::Neg(type_expr(expr)?)),
    }
}

fn type_arithmetic(expr: InferredArithmeticExpr) -> TypeResult<TypedArithmeticExpr> {
    match expr {
        InferredArithmeticExpr::Add(lhs, rhs) => {
            Ok(TypedArithmeticExpr::Add(type_expr(lhs)?, type_expr(rhs)?))
        }
        InferredArithmeticExpr::Sub(lhs, rhs) => {
            Ok(TypedArithmeticExpr::Sub(type_expr(lhs)?, type_expr(rhs)?))
        }
        InferredArithmeticExpr::Mul(lhs, rhs) => {
            Ok(TypedArithmeticExpr::Mul(type_expr(lhs)?, type_expr(rhs)?))
        }
        InferredArithmeticExpr::Div(lhs, rhs) => {
            Ok(TypedArithmeticExpr::Div(type_expr(lhs)?, type_expr(rhs)?))
        }
        InferredArithmeticExpr::Mod(lhs, rhs) => {
            Ok(TypedArithmeticExpr::Mod(type_expr(lhs)?, type_expr(rhs)?))
        }
    }
}

fn type_function(expr: InferredFunctionExpr) -> TypeResult<TypedFunctionExpr> {
    match expr {
        InferredFunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr,
        } => Ok(TypedFunctionExpr::Lambda {
            var,
            var_ty,
            ret_ty,
            expr: type_expr(expr)?,
        }),
        InferredFunctionExpr::Apply { fun, arg } => Ok(TypedFunctionExpr::Apply {
            fun: type_expr(fun)?,
            arg: type_expr(arg)?,
        }),
    }
}

fn type_block(expr: Vec<InferredStmt>) -> TypeResult<Vec<TypedStmt>> {
    expr.into_iter()
        .map(|stmt| {
            let stmt = type_stmt(stmt)?;
            Ok(stmt)
        })
        .collect::<TypeResult<Vec<_>>>()
}

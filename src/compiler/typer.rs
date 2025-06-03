use super::error::{TypeError, TypeResult};
use super::types::inferred_ast::InferredStmtKind;
use super::types::typed_ast::TypedStmtKind;
use super::types::{ExprWith, PartialType, Span};
use super::types::{
    inferred_ast::{InferredProgram, InferredStmt},
    typed_ast::{TypedProgram, TypedStmt},
};

pub fn type_program(program: &mut InferredProgram) -> TypeResult<TypedProgram> {
    let mut typed_stmts = Vec::new();
    for stmt in &mut program.0 {
        typed_stmts.push(type_stmt(stmt)?);
    }

    Ok(TypedProgram::new(typed_stmts))
}

fn type_stmt(stmt: &mut InferredStmt) -> TypeResult<TypedStmt> {
    let span = &mut stmt.meta;
    let full_ty = match stmt.ty.to_full() {
        Some(ty) => ty,
        None => return Err(TypeError::MissingType(span.clone())),
    };

    match &mut stmt.stmt {
        InferredStmtKind::Skip => Ok(TypedStmt {
            stmt: TypedStmtKind::Skip,
            ty: full_ty,
            meta: (),
        }),
        InferredStmtKind::Let { var, ty, value } => type_let(var, ty, full_ty, value, span),
        // InferredStmtKind::If {
        //     condition,
        //     then_branch,
        //     else_if_branches,
        //     else_branch,
        // } => type_if(
        //     condition,
        //     then_branch,
        //     else_if_branches,
        //     else_branch,
        //     ctx,
        //     span,
        // ),
        // InferredStmtKind::While { condition, body } => type_while(condition, body, ctx, span),
        // InferredStmtKind::Expr(expr) => {
        //     let expr = infer_expr(expr, ctx)?;
        //     Ok(TypedStmt {
        //         stmt: TypedStmtKind::Expr(expr),
        //         ty: PartialType {
        //             value: Some(PartialValue::Unit),
        //             labels: None,
        //         },
        //         meta: (),
        //     })
        // }
        // InferredStmtKind::Return(expr) => {
        //     let expr = infer_expr(expr, ctx)?;
        //     let ty = PartialType {
        //         value: expr.ty.value.clone(),
        //         labels: expr.ty.labels.clone(),
        //     };
        //     Ok(TypedStmt {
        //         stmt: TypedStmtKind::Return(expr),
        //         ty,
        //         meta: (),
        //     })
        // }
    }
}

fn type_let(
    var: &str,
    ty: &mut PartialType,
    full_ty: super::types::Type,
    value: &mut ExprWith<PartialType, Span>,
    span: &mut Span,
) -> TypeResult<TypedStmt> {
    // let expr_ty = match value.expr.to_full() {
    //     Some(ty) => ty,
    //     None => return Err(TypeError::MissingType(span.clone())),
    // };
    Ok(TypedStmt {
        stmt: TypedStmtKind::Let {
            var: var.to_owned(),
            ty: ty.to_owned(),
            value: ExprWith {
                expr: value.expr,
                ty: full_ty,
                meta: (),
            },
        },
        ty: full_ty,
        meta: (),
    })
}

use std::collections::HashMap;

use crate::compiler::types::{typed_ast::{
    TypedArithmeticExpr, TypedBoolExpr, TypedExpr, TypedExprKind, TypedFunctionExpr, TypedProgram,
    TypedStmt, TypedStmtKind,
}, Literal};

#[derive(Debug, PartialEq, Clone)]
enum Value {
    Int(i64),
    Bool(bool),
    Float(f64),
    Function(Function),
    Unit,
}

#[derive(Debug, Clone)]
struct Function {
    param: String,
    body: TypedExpr
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}

struct Context {
    vars: HashMap<String, Value>
}

pub fn interpret_program(ast: TypedProgram) {
    let mut ctx = Context { vars: HashMap::new() };
    for stmt in &ast.0 {
        let res = interpret_stmt(stmt, &mut ctx);
        if let Value::Unit = res {
            continue;
        } else {
            dbg!(res);
        };
    }
}

fn interpret_stmt(stmt: &TypedStmt, ctx: &mut Context) -> Value {
    use TypedStmtKind as TS;
    match &stmt.stmt {
        TS::Skip => (),
        TS::Let { var, ty: _, value } => {
            let res = interpret_expr(value, ctx);
            ctx.vars.insert(var.clone(), res);
        },
        TS::If { condition, then_branch, else_if_branches, else_branch } => {
            if let Value::Bool(true) = interpret_expr(condition, ctx) {
                let _ = interpret_expr(then_branch, ctx);
                return Value::Unit;
            }

            for (cond, body) in else_if_branches {
                if let Value::Bool(true) = interpret_expr(cond, ctx) {
                    let _ = interpret_expr(body, ctx);
                    return Value::Unit;
                }
            }

            if let Some(e) = else_branch {
                let _ = interpret_expr(e, ctx);
            }
        },
        TS::Expr(e) => {
            let _ = interpret_expr(e, ctx);
        },
        TS::While { condition, body } => {
            while let Value::Bool(true) = interpret_expr(condition, ctx) {
                let _ = interpret_expr(body, ctx);
            }
        },
        TS::Return(e) => {
            return interpret_expr(e, ctx);
        }
    }

    Value::Unit
}

fn interpret_expr(expr: &TypedExpr, ctx: &mut Context) -> Value {
    use TypedExprKind as TE;
    match &expr.expr {
        TE::Bool(b) => interpret_bool(b, ctx),
        TE::Arithmetic(a) => interpret_arithmetic(a, ctx),
        TE::Function(f) => interpret_function(f, ctx),
        TE::Var(v) => ctx.vars.get(v).unwrap_or(&Value::Unit).clone(),
        TE::Literal(l) => match l {
            Literal::Int(i) => Value::Int(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Unit => Value::Unit,
        },
        TE::Block(blk) => interpret_block(blk, ctx),
    }
}

fn interpret_bool(expr: &TypedBoolExpr, ctx: &mut Context) -> Value {
    use TypedBoolExpr as TB;
    match expr {
        TB::And(l, r) => {
            let Value::Bool(lb) = interpret_expr(l, ctx) else {
                return Value::Unit;
            };
            let Value::Bool(rb) = interpret_expr(r, ctx) else {
                return Value::Unit;
            };

            return Value::Bool(lb && rb);
        },
        TB::Or(l, r) => {
            let Value::Bool(lb) = interpret_expr(l, ctx) else {
                return Value::Unit;
            };
            let Value::Bool(rb) = interpret_expr(r, ctx) else {
                return Value::Unit;
            };

            return Value::Bool(lb || rb);
        },
        TB::Eq(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            return Value::Bool(l == r);
        },
        TB::Leq(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Bool(x <= y),
                (Value::Float(x), Value::Float(y)) => return Value::Bool(x <= y),
                _ => return Value::Unit
            }
        },
        TB::Geq(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Bool(x >= y),
                (Value::Float(x), Value::Float(y)) => return Value::Bool(x >= y),
                _ => return Value::Unit
            }
        },
        TB::Le(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Bool(x < y),
                (Value::Float(x), Value::Float(y)) => return Value::Bool(x < y),
                _ => return Value::Unit
            }
        },
        TB::Ge(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Bool(x > y),
                (Value::Float(x), Value::Float(y)) => return Value::Bool(x > y),
                _ => return Value::Unit
            }
        },
        TB::Neg(e) => {
            let Value::Bool(b) = interpret_expr(e, ctx) else {
                return Value::Unit
            };

            return Value::Bool(!b);
        }
    }
}

fn interpret_arithmetic(expr: &TypedArithmeticExpr, ctx: &mut Context) -> Value {
    use TypedArithmeticExpr as TA;
    match expr {
        TA::Add(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Int(x + y),
                (Value::Float(x), Value::Float(y)) => return Value::Float(x + y),
                _ => return Value::Unit
            }
        },
        TA::Sub(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Int(x - y),
                (Value::Float(x), Value::Float(y)) => return Value::Float(x - y),
                _ => return Value::Unit
            }
        },
        TA::Mul(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Int(x * y),
                (Value::Float(x), Value::Float(y)) => return Value::Float(x * y),
                _ => return Value::Unit
            }
        },
        TA::Div(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Int(x / y),
                (Value::Float(x), Value::Float(y)) => return Value::Float(x / y),
                _ => return Value::Unit
            }
        },
        TA::Mod(l, r) => {
            let l = interpret_expr(l, ctx);
            let r = interpret_expr(r, ctx);

            match (l, r) {
                (Value::Int(x), Value::Int(y)) => return Value::Int(x % y),
                (Value::Float(x), Value::Float(y)) => return Value::Float(x % y),
                _ => return Value::Unit
            }
        },
    }
}

fn interpret_function(expr: &TypedFunctionExpr, ctx: &mut Context) -> Value {
    use TypedFunctionExpr as TF;
    match expr {
        TF::Apply { fun, arg } => {
            let Value::Function(f) = interpret_expr(fun, ctx) else {
                return Value::Unit;
            };

            let arg = interpret_expr(arg, ctx);

            let prev = ctx.vars.insert(f.param.clone(), arg);
            let res = interpret_expr(&f.body, ctx);

            if let Some(prev) = prev {
                ctx.vars.insert(f.param.clone(), prev);
            }

            res
        },
        TF::Lambda { var, var_ty: _, ret_ty: _, expr } => {
            Value::Function(Function { param: var.clone(), body: expr.clone() })
        },
    }
}

fn interpret_block(blk: &Vec<TypedStmt>, ctx: &mut Context) -> Value {
    for stmt in blk {
        let res = interpret_stmt(stmt, ctx);
        if let Value::Unit = res {
            continue;
        } else {
            return res;
        }
    }
    Value::Unit
}

use super::Type;

#[derive(Debug)]
pub struct TypedProgram(pub Vec<TypedStmt>);

#[derive(Debug)]
pub enum TypedStmt {
    Skip,
    Let {
        var: String,
        ty: Type,
        value: TypedExpr,
    },
    If {
        condition: TypedExpr,
        then_branch: TypedExpr,
        else_if_branches: Vec<(TypedExpr, TypedExpr)>,
        else_branch: Option<TypedExpr>,
    },
    While {
        condition: TypedExpr,
        body: TypedExpr,
    },
    Expr(TypedExpr),
    Return(TypedExpr),
}

#[derive(Debug)]
pub enum TypedExpr {
    Bool(Box<TypedBoolExpr>, Type),
    Arithmetic(Box<TypedArithmeticExpr>, Type),
    Function(Box<TypedFunctionExpr>, Type),
    Block(Vec<TypedStmt>, Type),
    Literal(Literal, Type),
    Var(String, Type),
}

#[derive(Debug)]
pub enum TypedBoolExpr {
    And(TypedExpr, TypedExpr),
    Or(TypedExpr, TypedExpr),
    Eq(TypedExpr, TypedExpr),
    Neg(TypedExpr),
    Le(TypedExpr, TypedExpr),
    Leq(TypedExpr, TypedExpr),
    Ge(TypedExpr, TypedExpr),
    Geq(TypedExpr, TypedExpr),
}

#[derive(Debug)]
pub enum TypedArithmeticExpr {
    Add(TypedExpr, TypedExpr),
    Sub(TypedExpr, TypedExpr),
    Mul(TypedExpr, TypedExpr),
    Div(TypedExpr, TypedExpr),
    Mod(TypedExpr, TypedExpr),
}

#[derive(Debug)]
pub enum TypedFunctionExpr {
    Lambda {
        var: String,
        var_ty: Type,
        ret_ty: Type,
        expr: TypedExpr,
    },
    Apply {
        fun: TypedExpr,
        arg: TypedExpr,
    },
}

#[derive(Debug)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    Float(f64),
    Unit,
}


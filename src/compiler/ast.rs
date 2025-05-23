#[derive(Debug)]
pub struct Program(pub Vec<Stmt>);

#[derive(Debug)]
pub enum Stmt {
    Skip,
    Let {
        var: String,
        ty: Type,
        value: Expr,
    },
    If {
        condition: Expr,
        then_branch: Expr,
        else_if_branches: Vec<(Expr, Expr)>,
        else_branch: Option<Expr>,
    },
    While {
        condition: Expr,
        body: Expr,
    },
    Expr(Expr),
    Return(Expr),
}

#[derive(Debug)]
pub enum Expr {
    Bool(Box<BoolExpr>),
    Arithmetic(Box<ArithmeticExpr>),
    Function(Box<FunctionExpr>),
    Block(Vec<Stmt>),
    Literal(Literal, Type),
    Var(String),
}

#[derive(Debug)]
pub enum BoolExpr {
    And(Expr, Expr),
    Or(Expr, Expr),
    Eq(Expr, Expr),
    Neg(Expr),
    Le(Expr, Expr),
    Leq(Expr, Expr),
    Ge(Expr, Expr),
    Geq(Expr, Expr),
}

#[derive(Debug)]
pub enum ArithmeticExpr {
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Mod(Expr, Expr),
}

#[derive(Debug)]
pub enum FunctionExpr {
    Lambda {
        var: String,
        var_ty: Type,
        ret_ty: Type,
        expr: Expr,
    },
    Apply {
        fun: Expr,
        arg: Expr,
    },
}

#[derive(Debug)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    Float(f64),
    Unit,
}

#[derive(Debug)]
pub struct Type {
    pub value: Option<Value>,
    pub label: Option<Vec<String>>,
}

impl Type {
    pub fn empty() -> Self {
        Self {
            value: None,
            label: None,
        }
    }
}

#[derive(Debug)]
pub enum Value {
    Int,
    Bool,
    Float,
    Unit,
    Function {
        param: Box<Type>,
        return_type: Box<Type>,
    },
}

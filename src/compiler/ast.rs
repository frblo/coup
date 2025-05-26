#[derive(Debug)]
pub struct Spanned<T> {
    pub text: String,
    pub node: T,
    pub line: usize,
    pub col: usize,
}

impl<T> Spanned<T> {
    pub fn new(node: T, line_col: (usize, usize), text: String) -> Self {
        Self { node, line: line_col.0, col: line_col.1, text }
    }
}

#[derive(Debug)]
pub struct Program(pub Vec<Spanned<Stmt>>);

#[derive(Debug)]
pub enum Stmt {
    Skip,
    Let {
        var: String,
        ty: Type,
        value: Spanned<Expr>,
    },
    If {
        condition: Spanned<Expr>,
        then_branch: Spanned<Expr>,
        else_if_branches: Vec<(Spanned<Expr>, Spanned<Expr>)>,
        else_branch: Option<Spanned<Expr>>,
    },
    While {
        condition: Spanned<Expr>,
        body: Spanned<Expr>,
    },
    Expr(Spanned<Expr>),
    Return(Spanned<Expr>),
}

#[derive(Debug)]
pub enum Expr {
    Bool(Box<BoolExpr>),
    Arithmetic(Box<ArithmeticExpr>),
    Function(Box<FunctionExpr>),
    Block(Vec<Spanned<Stmt>>),
    Literal(Literal, Type),
    Var(String),
}

#[derive(Debug)]
pub enum BoolExpr {
    And(Spanned<Expr>, Spanned<Expr>),
    Or(Spanned<Expr>, Spanned<Expr>),
    Eq(Spanned<Expr>, Spanned<Expr>),
    Neg(Spanned<Expr>),
    Le(Spanned<Expr>, Spanned<Expr>),
    Leq(Spanned<Expr>, Spanned<Expr>),
    Ge(Spanned<Expr>, Spanned<Expr>),
    Geq(Spanned<Expr>, Spanned<Expr>),
}

#[derive(Debug)]
pub enum ArithmeticExpr {
    Add(Spanned<Expr>, Spanned<Expr>),
    Sub(Spanned<Expr>, Spanned<Expr>),
    Mul(Spanned<Expr>, Spanned<Expr>),
    Div(Spanned<Expr>, Spanned<Expr>),
    Mod(Spanned<Expr>, Spanned<Expr>),
}

#[derive(Debug)]
pub enum FunctionExpr {
    Lambda {
        var: String,
        var_ty: Type,
        ret_ty: Type,
        expr: Spanned<Expr>,
    },
    Apply {
        fun: Spanned<Expr>,
        arg: Spanned<Expr>,
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
    pub value: Option<Spanned<Value>>,
    pub label: Option<Spanned<Vec<String>>>,
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


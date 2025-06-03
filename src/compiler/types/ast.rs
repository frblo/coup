use super::PartialType;

#[derive(Debug, Clone)]
pub struct Span {
    pub text: String,
    pub line: usize,
    pub col: usize,
}

impl Span {
    pub fn new(line_col: (usize, usize), text: String) -> Self {
        Self {
            line: line_col.0,
            col: line_col.1,
            text,
        }
    }
}

#[derive(Debug)]
pub struct ProgramWith<T, M>(pub Vec<StmtWith<T, M>>);

impl<T, M> ProgramWith<T, M> {
    pub fn new(v: Vec<StmtWith<T, M>>) -> Self {
        Self(v)
    }
}

#[derive(Debug)]
pub struct StmtWith<T, M> {
    pub stmt: StmtKind<T, M>,
    pub ty: T,
    pub meta: M,
}

impl<T, M> StmtWith<T, M> {
    pub fn new(stmt: StmtKind<T, M>, ty: T, meta: M) -> Self {
        Self { stmt, ty, meta }
    }
}

#[derive(Debug)]
pub enum StmtKind<T, M> {
    Skip,
    Let {
        var: String,
        ty: PartialType,
        value: ExprWith<T, M>,
    },
    If {
        condition: ExprWith<T, M>,
        then_branch: ExprWith<T, M>,
        else_if_branches: Vec<(ExprWith<T, M>, ExprWith<T, M>)>,
        else_branch: Option<ExprWith<T, M>>,
    },
    While {
        condition: ExprWith<T, M>,
        body: ExprWith<T, M>,
    },
    Expr(ExprWith<T, M>),
    Return(ExprWith<T, M>),
}

#[derive(Debug)]
pub struct ExprWith<T, M> {
    pub expr: ExprKind<T, M>,
    pub ty: T,
    pub meta: M,
}

impl<T, M> ExprWith<T, M> {
    pub fn new(expr: ExprKind<T, M>, ty: T, meta: M) -> Self {
        Self { expr, ty, meta }
    }
}

#[derive(Debug)]
pub enum ExprKind<T, M> {
    Bool(Box<BoolExprWith<T, M>>),
    Arithmetic(Box<ArithmeticExprWith<T, M>>),
    Function(Box<FunctionExprWith<T, M>>),
    Block(Vec<StmtWith<T, M>>),
    Literal(Literal),
    Var(String),
}

#[derive(Debug)]
pub enum BoolExprWith<T, M> {
    And(ExprWith<T, M>, ExprWith<T, M>),
    Or(ExprWith<T, M>, ExprWith<T, M>),
    Eq(ExprWith<T, M>, ExprWith<T, M>),
    Neg(ExprWith<T, M>),
    Le(ExprWith<T, M>, ExprWith<T, M>),
    Leq(ExprWith<T, M>, ExprWith<T, M>),
    Ge(ExprWith<T, M>, ExprWith<T, M>),
    Geq(ExprWith<T, M>, ExprWith<T, M>),
}

#[derive(Debug)]
pub enum ArithmeticExprWith<T, M> {
    Add(ExprWith<T, M>, ExprWith<T, M>),
    Sub(ExprWith<T, M>, ExprWith<T, M>),
    Mul(ExprWith<T, M>, ExprWith<T, M>),
    Div(ExprWith<T, M>, ExprWith<T, M>),
    Mod(ExprWith<T, M>, ExprWith<T, M>),
}

#[derive(Debug)]
pub enum FunctionExprWith<T, M> {
    Lambda {
        var: String,
        var_ty: PartialType,
        ret_ty: PartialType,
        expr: ExprWith<T, M>,
    },
    Apply {
        fun: ExprWith<T, M>,
        arg: ExprWith<T, M>,
    },
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    Float(f64),
    Unit,
}

mod ast;
pub mod label;
mod value;

use label::Labels;
use value::Value;

use super::error::{CompilerError, Result, TypeError, TypeResult};

#[derive(Debug, Clone)]
pub struct Type {
    pub value: ConcreteValue,
    pub labels: Labels,
}

impl Type {
    pub fn to_partial(self) -> PartialType {
        let value = Some(self.value.to_partial());
        let labels = Some(self.labels);

        PartialType { value, labels }
    }
}

#[derive(Debug, Clone)]
pub struct PartialType {
    pub value: Option<PartialValue>,
    pub labels: Option<Labels>,
}

pub fn unify(
    mut t1: &mut Option<PartialValue>,
    mut t2: &mut Option<PartialValue>,
) -> TypeResult<()> {
    match (&mut t1, &mut t2) {
        (_, None) => *t2 = t1.clone(),
        (None, _) => *t1 = t2.clone(),
        (
            Some(PartialValue::Function {
                param: p1,
                return_type: rt1,
            }),
            Some(PartialValue::Function {
                param: p2,
                return_type: rt2,
            }),
        ) => {
            unify(&mut p1.value, &mut p2.value)?;
            unify(&mut rt1.value, &mut rt2.value)?;
        }
        (a, b) if a == b => (),
        _ => {
            return Err(TypeError::Missmatch(
                t1.clone().unwrap(),
                t2.clone().unwrap(),
            ));
        }
    }

    Ok(())
}

impl PartialType {
    pub fn empty() -> Self {
        Self {
            value: None,
            labels: None,
        }
    }

    pub fn to_full(self) -> Option<Type> {
        let value = self.value?.to_concrete()?;
        let labels = self.labels?;

        Some(Type { value, labels })
    }
}

impl PartialEq for PartialType {
    fn eq(&self, other: &Self) -> bool {
        (match (&self.value, &other.value) {
            (Some(v1), Some(v2)) => v1 == v2,
            (None, None) => true,
            _ => false,
        }) && (match (&self.labels, &other.labels) {
            (Some(l1), Some(l2)) => l1 == l2,
            (None, None) => true,
            _ => false,
        })
    }
}

impl Eq for PartialType {}

pub mod raw_ast {
    use super::ast;
    pub type RawProgram = ast::ProgramWith<(), ast::Span>;

    pub type RawStmt = ast::StmtWith<(), ast::Span>;
    pub type RawStmtKind = ast::StmtKind<(), ast::Span>;

    pub type RawExpr = ast::ExprWith<(), ast::Span>;
    pub type RawExprKind = ast::ExprKind<(), ast::Span>;
    pub type RawBoolExpr = ast::BoolExprWith<(), ast::Span>;
    pub type RawArithmeticExpr = ast::ArithmeticExprWith<(), ast::Span>;
    pub type RawFunctionExpr = ast::FunctionExprWith<(), ast::Span>;
}

pub mod inferred_ast {
    use super::{PartialType, ast};
    pub type InferredProgram = ast::ProgramWith<PartialType, ast::Span>;

    pub type InferredStmt = ast::StmtWith<PartialType, ast::Span>;
    pub type InferredStmtKind = ast::StmtKind<PartialType, ast::Span>;

    pub type InferredExpr = ast::ExprWith<PartialType, ast::Span>;
    pub type InferredExprKind = ast::ExprKind<PartialType, ast::Span>;
    pub type InferredBoolExpr = ast::BoolExprWith<PartialType, ast::Span>;
    pub type InferredArithmeticExpr = ast::ArithmeticExprWith<PartialType, ast::Span>;
    pub type InferredFunctionExpr = ast::FunctionExprWith<PartialType, ast::Span>;
}

pub mod typed_ast {
    use super::{Type, ast};
    pub type TypedProgram = ast::ProgramWith<Type, ()>;

    pub type TypedStmt = ast::StmtWith<Type, ()>;
    pub type TypedStmtKind = ast::StmtKind<Type, ()>;

    pub type TypedExpr = ast::ExprWith<Type, ()>;
    pub type TypedExprKind = ast::ExprKind<Type, ()>;
    pub type TypedBoolExpr = ast::BoolExprWith<Type, ()>;
    pub type TypedArithmeticExpr = ast::ArithmeticExprWith<Type, ()>;
    pub type TypedFunctionExpr = ast::FunctionExprWith<Type, ()>;
}

pub type Literal = ast::Literal;
pub type Span = ast::Span;

pub type ExprWith<T, M> = ast::ExprWith<T, M>;

pub type PartialValue = Value<PartialType>;
impl PartialValue {
    pub fn to_concrete(self) -> Option<ConcreteValue> {
        self.map(PartialType::to_full)
    }
}

pub type ConcreteValue = Value<Type>;
impl ConcreteValue {
    pub fn to_partial(self) -> PartialValue {
        let wrapper = |x| Some(Type::to_partial(x));
        self.map(wrapper).unwrap()
    }
}

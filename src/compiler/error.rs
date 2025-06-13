use thiserror::Error;

use super::{parser::Rule, types::PartialValue, types::Span, types::label::Labels};

pub type Result<T> = core::result::Result<T, CompilerError>;
pub type ParseResult<T> = core::result::Result<T, ParseError>;
pub type TypeResult<T> = core::result::Result<T, TypeError>;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Parsing failed with:\n{0}")]
    ParseError(#[from] ParseError),

    #[error("Typing failed with:\n{0}")]
    TypeError(#[from] TypeError),
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("{0}")]
    PestError(#[from] pest::error::Error<Rule>),

    #[error("{0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("{0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),

    #[error("{0}")]
    ParseBoolError(#[from] std::str::ParseBoolError),

    #[error("Missing Root node")]
    MissingRootNode,

    #[error("Missing node after {}:{}", _0.0, _0.1)]
    MissingNode((usize, usize)),

    #[error("Invalid node ({:?}) after {}:{}", _0, _1.0, _1.1)]
    InvalidNode(Rule, (usize, usize)),
}

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("Can't coherce types\n{0:#?}\n\nand\n\n{1:#?}")]
    Missmatch(PartialValue, PartialValue),

    #[error("Variable with labels {0:?} doesn't flow to {1:?}\nSpan: {2:#?}")]
    InvalidFlow(Labels, Labels, Span),

    #[error("Expected a numeric value but got {0:?}\nSpan: {1:#?}")]
    ExpectedNumeric(PartialValue, Span),

    #[error("Expected a boolean value but got {0:?}\nSpan: {1:#?}")]
    ExpectedBoolean(PartialValue, Span),

    #[error("Expected a function value but got {0:?}\nSpan: {1:#?}")]
    ExpectedFunction(PartialValue, Span),

    #[error("Use of undeclared variable: {0:?}\nSpan: {1:#?}")]
    UndeclaredVar(String, Span),

    #[error("Expected to be Some but None was encountered\nSpan: {0:#?}")]
    MissingType(Span),
}

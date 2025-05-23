use thiserror::Error;

use super::parser::Rule;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Parsing failed with:\n{0}")]
    ParseError(#[from] pest::error::Error<Rule>),

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

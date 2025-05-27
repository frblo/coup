pub mod spanned;
pub mod value;
pub mod ast;
pub mod typed_ast;
pub mod label;

use label::Labels;
use value::Value;

pub type PartialValue = Value<PartialType>;
pub type ConcreteValue = Value<Type>;

#[derive(Debug, Clone)]
pub struct Type {
    pub value: ConcreteValue,
    pub labels: Labels,
}

#[derive(Debug, Clone)]
pub struct PartialType {
    pub value: Option<PartialValue>,
    pub labels: Option<Labels>,
}

impl PartialType{
    pub fn empty() -> Self {
        Self {
            value: None,
            labels: None,
        }
    }
}

impl PartialEq for PartialType {
    fn eq(&self, other: &Self) -> bool {
        (
            match (&self.value, &other.value) {
                (Some(v1), Some(v2)) => v1 == v2,
                (None, None) => true,
                _ => false
            }
        ) && (
            match (&self.labels, &other.labels) {
                (Some(l1), Some(l2)) => l1 == l2,
                (None, None) => true,
                _ => false,
            }
        )
    }
}

impl Eq for PartialType {}

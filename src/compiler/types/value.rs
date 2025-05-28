#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value<T> {
    Int,
    Bool,
    Float,
    Unit,
    Function {
        param: Box<T>,
        return_type: Box<T>,
    },
}

impl<T> Value<T> {
    pub fn map<U>(self, f: impl Fn(T) -> Option<U>) -> Option<Value<U>> {
        Some(
            match self {
                Value::Int => Value::Int,
                Value::Bool => Value::Bool,
                Value::Float => Value::Float,
                Value::Unit => Value::Unit,
                Value::Function { param, return_type } => Value::Function {
                    param: Box::new(f(*param)?),
                    return_type: Box::new(f(*return_type)?)
                },
            }
        )
    }
}

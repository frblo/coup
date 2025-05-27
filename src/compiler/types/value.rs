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

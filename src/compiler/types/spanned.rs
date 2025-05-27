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


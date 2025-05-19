use compiler::parser::parse_output;

mod compiler;

use std::io::{stdin, Read};

fn main() {
    let mut file = String::new();
    stdin().lock().read_to_string(&mut file).unwrap();
    parse_output(&file).unwrap();
}

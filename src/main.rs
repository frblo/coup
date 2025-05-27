mod compiler;

use std::io::{Read, stdin};
use compiler::parser::parse_program;

fn main() {
    let mut file = String::new();
    stdin().lock().read_to_string(&mut file).unwrap();
    match parse_program(&file) {
        Err(err) => eprintln!("{:#}", err),
        Ok(ast) => {
            for stmt in &ast.0 {
                println!("{:#?}", stmt);
            }
        }
    }
}

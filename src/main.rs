mod compiler;

use compiler::{inferer::infer_program, parser::parse_program};
use std::io::{Read, stdin};

fn main() {
    let mut file = String::new();
    stdin().lock().read_to_string(&mut file).unwrap();
    match parse_program(&file) {
        Err(err) => eprintln!("{:#}", err),
        Ok(mut ast) => {
            println!("\nParse Result");
            for stmt in &ast.0 {
                println!("{:#?}", stmt);
            }

            println!("\nInfer Result");
            match infer_program(&mut ast) {
                Err(err) => eprintln!("{:#}", err),
                Ok(i_ast) => {
                    for stmt in &i_ast.0 {
                        println!("{:#?}", stmt);
                    }
                }
            }
        }
    }
}

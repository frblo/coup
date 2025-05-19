use pest::Parser;
use pest::pratt_parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "compiler/parser/grammar.pest"]
pub struct CoupParser;

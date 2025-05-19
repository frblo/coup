use pest::error::Error;
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "compiler/parser/grammar.pest"]
pub struct CoupParser;

pub fn parse_output(file: &str) -> Result<(), Error<Rule>> {
    let program = CoupParser::parse(Rule::program, file)?.next().unwrap();
    for pair in program.into_inner() {
        print_pair(pair);
        println!("---------");
    }
    Ok(())
}

fn print_pair(p: Pair<'_, Rule>) {
    println!("Rule:    {:?}", p.as_rule());
    println!("Text:    {}", p.as_str());
    println!("");

    // A pair can be converted to an iterator of the tokens which make it up:
    for inner_pair in p.into_inner() {
        print_pair(inner_pair);
    }
}

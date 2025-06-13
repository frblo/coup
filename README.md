# coup

## How to use
1. Write a `Coup` program in a `.coup` file.

    To understand the syntax of the language you can read the grammar and type inference rules in the accompanying essay.
    And/Or look at the example programs in `examples/`.

2. Run the compiler with `cargo run` supplying the program as input on `stdin`

    For example `cargo run < examples/higher-order-add.coup`.

3. The program will output the AST in three different stages (as text). The final one is the TypedAST meaning the program will be safe to run. The interperter will then run the AST. To get any output when running one can use `return` statement outside of a block which will be displayed in the console. If the program doesn't compile for some reason there will be an error message meant to give some hint to what went wrong along with the part of the code where something went wrong. However, these error messages can be a bit unclear at times as we lacked the time to make them pretty.

# NN-MLIR Compiler: Final Project Writeup

**Author**: Aditya Pandey  
**Date**: April 24, 2026

## Project Overview

This project implements a small domain-specific neural-network compiler using MLIR. The compiler accepts a simple `.nn` source language for feed-forward neural networks, parses that source into an in-memory representation, builds MLIR using a custom `nn` dialect, and lowers that high-level IR into standard MLIR dialects such as `linalg`, `tensor`, and `arith`.

## Motivation

In a compilers course, many projects follow a traditional structure: parse source text into an AST and then emit a target representation directly. That model is useful, but it compresses most of compilation into a single jump from source-level constructs to low-level output.

MLIR is different because it uses a more modern compiler architecture. Instead of a single fixed IR, it supports multiple IR levels, custom dialects, and progressive lowering. Neural-network operations are a strong fit for this style because operations such as dense layers and activations have clear high-level representations in the form of matrices and vectors that are worth preserving before lowering them into more standard linear algebra operations.

This project was designed to explore the following compiler ideas:

- designing a tiny domain-specific source language
- defining a custom MLIR dialect
- constructing IR programmatically with `OpBuilder`
- representing data flow explicitly with SSA values
- lowering high-level semantic operations into standard MLIR dialects

## Traditional Compilers vs. MLIR

A traditional educational compiler such as the Strix compiler from CSE 450 Project 3 can be described with a pipeline like this:

```text
source text -> lexer/token stream -> parser -> AST -> direct code generation
```

In Strix specifically, `Project3.cpp` tokenizes the source, uses precedence-climbing logic in `Parse_Expression`, builds an AST made of classes such as `ASTNode_NumLit`, `ASTNode_Var`, `ASTNode_Operator2`, and `ASTNode_Return`, and then emits WebAssembly text by calling `ToWAT()` on those AST nodes. The `SymbolTable` manages variable IDs, function information, and scope information that the AST and code generator rely on.

In that style of compiler, the AST is usually the main internal representation, and code generation walks the tree directly to emit the target program. Representation and transformation are therefore closely tied together: in Strix, an `ASTNode_Operator2` both represents a binary operator and knows how to emit the corresponding WAT instruction.

This project follows a different model:

```text
.nn source -> parser -> frontend representation -> MLIR with nn dialect -> lowering pass -> MLIR with standard dialects
```

The main conceptual difference is that MLIR separates high-level meaning from low-level implementation. Instead of immediately translating dense layers into lower-level linear algebra operations, the frontend first emits semantic operations such as `nn.dense` and `nn.relu`. A later pass then lowers those operations into standard MLIR constructs. Compared to Strix, this inserts an explicit compiler-transformation stage between parsing and lower-level output, rather than having each tree node directly know how to generate the final representation.

## MLIR Concepts Explained From a Traditional Compiler Perspective

Because MLIR uses its own terminology, it is useful to map the main concepts in this project to ideas from a more traditional compiler pipeline.

### Dialect

In a traditional compiler, the IR usually comes with a fixed vocabulary of instructions or node types. In Strix, that vocabulary appears as AST classes such as `ASTNode_NumLit`, `ASTNode_Var`, `ASTNode_Operator2`, and `ASTNode_Return`, and then later as WAT instructions such as `f64.add`, `f64.sub`, and `return`.

In MLIR, a **dialect** is a named collection of operations that belong together. A useful analogy is to think of a dialect as a custom IR namespace or a domain-specific instruction set. Instead of forcing every compiler to use one universal fixed IR vocabulary, MLIR allows a compiler to define its own operations when that is the clearest representation for a problem domain. If Strix had been designed in a more MLIR-like style, its high-level language constructs would not all need to live inside one fixed AST vocabulary.

In this project, the custom `nn` dialect defines:

- `nn.dense`
- `nn.relu`

These are higher-level than generic arithmetic or loop instructions. They preserve the semantic meaning of neural-network operations until a later lowering pass translates them into standard MLIR operations.

### Operation

An **operation** in MLIR is the basic unit of IR, roughly analogous to:

- an AST node in a high-level representation, or
- an instruction in a lower-level IR

For example, `nn.dense` is one MLIR operation. So are `linalg.matmul`, `arith.addf`, and `func.return`.

Unlike many traditional IRs, MLIR operations are extremely general. An operation can have:

- operands
- results
- attributes
- regions containing nested operations

In Strix terms, an MLIR operation can sometimes feel like an AST node and sometimes feel like a low-level instruction, but it is more general than either. This flexibility is one reason MLIR can represent many different abstraction levels in one framework.

### MLIRContext

A traditional compiler usually has some global or compiler-wide state describing the active language constructs, symbols, or IR storage environment. In the Strix project, pieces of that role are spread across the parser object, the lexer, and the `SymbolTable`.

In MLIR, the **`MLIRContext`** plays that role. It is the central object that owns and manages compiler-wide IR state. It is responsible for things such as:

- holding registered dialects
- uniquing types and attributes
- providing the environment in which operations are created and understood

In this project, the frontend creates an `MLIRContext`, registers the relevant dialects, and then uses that context while building the IR. Conceptually, the context is the “world” in which MLIR objects exist. If the Strix compiler had one object responsible for “what IR objects and language constructs exist in this compilation,” that would be the closest analogy.

### ModuleOp

In a traditional compiler, there is usually a top-level container for the whole program or compilation unit. Depending on the compiler, this might be called a module, translation unit, compilation unit, or program node. In Strix, the closest analogue is the full `(module ...)` wrapper emitted by `ToWAT()`, which contains imported functions, generated functions, and exports.

In MLIR, **`ModuleOp`** is that top-level container. It is itself an operation, but its role is to contain the rest of the IR for a compilation unit.

In this project, the frontend begins IR construction by creating a `ModuleOp`. The generated function and all later operations live inside that module. A useful analogy is:

- source file / compilation unit in a traditional compiler
- `ModuleOp` in MLIR

### OpBuilder

In a traditional compiler, when the parser or semantic phase constructs IR, it often uses helper routines such as “create this node,” “emit this instruction,” or “append this instruction to the current basic block.” In Strix, helpers such as `MakeVarNode`, `MakeOpNode`, and the `ASTNode` constructors play part of that frontend-building role, while `ToWAT()` is the emission side.

In MLIR, **`OpBuilder`** is the API object used to create operations programmatically. It is the mechanism that turns frontend decisions into actual IR objects.

For example, instead of emitting text like:

```text
%0 = nn.dense ...
```

the compiler calls `builder.create<DenseOp>(...)`.

That is an important design point in this project. The frontend does not generate MLIR by string concatenation. It uses `OpBuilder` to construct real IR objects directly, which is much closer to how a real compiler frontend should work. This is a useful contrast with Strix, where WAT is ultimately emitted as text by walking the AST.

### SSA Values

Many traditional educational compilers hide data flow inside:

- AST structure
- temporary variables in code generation
- or a stack-machine target such as WebAssembly text

MLIR is strongly based on **SSA** (Static Single Assignment) form. Every operation result produces a value, and later operations consume those values explicitly.

So instead of thinking “evaluate this subtree, then evaluate the next subtree,” the frontend thinks “this operation produced a value, and the next operation uses that value.”

That is why the frontend in this project keeps a variable named `current`: it represents the current SSA value flowing through the neural network from one layer to the next. This is a major difference from Strix. In Strix, `ASTNode_Operator2::ToWAT()` recursively emits its children and relies on WAT stack discipline. In the MLIR project, data flow is not implicit in evaluation order or the target stack; it is explicit in SSA values.

### Pass

In a traditional compiler, a **pass** is any transformation or analysis that runs over an intermediate representation, such as constant folding, dead-code elimination, or instruction selection.

The same idea exists in MLIR. A **pass** is a transformation over IR. In this project, the key pass is the lowering pass that rewrites `nn` dialect operations into standard MLIR dialect operations. A useful contrast with Strix is that Strix does not really insert a separate lowering pass between AST construction and WAT emission; code generation is largely embedded inside the AST classes themselves.

### Dialect Conversion

Traditional compilers often have phases that lower one IR into another. For example:

- AST -> three-address code
- high-level IR -> low-level IR
- machine-independent IR -> target-specific IR

In MLIR, **dialect conversion** is a structured framework for performing that kind of lowering. Instead of writing ad hoc rewrites, the compiler can formally declare:

- which dialects are legal after a pass
- which dialects are illegal after a pass
- which rewrite patterns should replace the illegal operations

This project uses dialect conversion to say, in effect:

- `nn` operations are not allowed to remain after lowering
- `arith`, `tensor`, `linalg`, and `func` operations are allowed

That makes the lowering phase explicit and mechanically checkable. Relative to Strix, this is the formal version of splitting what would otherwise be one direct “AST to output” step into a dedicated transformation stage with legality rules.

### Why These Concepts Matter In This Project

The most important MLIR-specific idea in this project is that the compiler does not jump directly from `.nn` source text to low-level linear algebra code. Instead, it:

1. parses the source language
2. builds a custom high-level IR in the `nn` dialect
3. applies a lowering pass to convert that high-level IR into standard MLIR operations

This is the central MLIR design pattern: represent the program first in the form that best matches its semantics, then lower it step by step into more general and more executable representations.

## Source Language

The implemented source language is intentionally minimal. It supports only the constructs needed to describe a simple feed-forward network:

```text
network mlp
input 784
dense 256 relu
dense 10
```

The meaning of each line is:

- `network <name>` defines the network name, which is also used as the generated function name.
- `input <size>` defines the input feature dimension.
- `dense <units>` adds a dense layer with the given output width.
- `relu` may optionally appear after a dense layer declaration to request a ReLU activation after that layer.

This language is deliberately small. It does not attempt to model training, backpropagation, control flow, or dynamic shapes. The goal is to keep the source language simple enough that the main focus remains on compiler structure and MLIR usage.

## High-Level Architecture

The compiler pipeline implemented in this project is:

```text
.nn source
  -> frontend parser
  -> in-memory network representation
  -> MLIR module with nn.dense / nn.relu
  -> lowering pass
  -> MLIR with linalg / tensor / arith
```

This pipeline has four main stages.

First, the frontend parser reads the `.nn` file and converts it into a small in-memory representation of the network structure. This is analogous to the role played by `Parse_Function`, `Parse_Block`, and `Parse_Expression` in the Strix compiler, except the source language here is far simpler and line-oriented.  
Second, the frontend IR builder uses MLIR APIs to build a `module` containing a `func.func` and a sequence of `nn.dense` and `nn.relu` operations.  
Third, the `nn` dialect preserves the high-level semantics of neural-network operations.  
Fourth, a lowering pass rewrites the custom `nn` operations into standard MLIR operations in the `linalg`, `tensor`, and `arith` dialects.

## Dialect Design

In MLIR, a dialect is a namespace of operations. This project defines a custom `nn` dialect containing two operations:

- `nn.dense`
- `nn.relu`

These operations are defined declaratively in `nn-mlir/include/NN/NNOps.td` using TableGen.

`nn.dense` represents a dense layer: matrix multiplication followed by bias addition.  
`nn.relu` represents an elementwise rectified linear unit activation.

The main reason for using a custom dialect instead of emitting `linalg` directly from the frontend is that it preserves high-level meaning. The frontend can talk in terms of neural-network operations, while the lowering pass becomes an explicit compiler stage responsible for translating those operations into standard structured linear algebra. This separation makes the project architecturally cleaner and better illustrates how MLIR supports progressive lowering.

TableGen is useful here because it lets the dialect be described declaratively. From the `.td` file, MLIR generates much of the parser, printer, and operation boilerplate automatically. That keeps the project focused on the compiler logic rather than repetitive C++ scaffolding.

## Frontend Implementation

### Parsing

The source language parser is implemented in `nn-mlir/parser/NNParser.h`.

The parser reads the file line by line, strips comments and whitespace, and enforces a simple ordering discipline:

1. a `network` declaration must appear first
2. an `input` declaration must appear next
3. one or more `dense` lines must follow

Rather than building MLIR directly during parsing, the parser first builds a small frontend representation. This keeps the frontend structured in the same way as a traditional compiler: parsing and IR construction are separate steps. In Strix, parsing builds an AST made of `ASTNode` subclasses. Here, parsing builds a much smaller frontend representation because the language is only a linear list of layers.

### Frontend Data Structures

The parser builds two main data structures:

- `Network`, which stores the network name, input size, and layer list
- `DenseLayer`, which stores the layer width and whether a ReLU should follow

This representation functions as a minimal frontend IR. It is not a full AST hierarchy because the language is linear and does not require nested expressions or control flow. Where Strix needs a richer tree structure to represent nested binary operators and statements, this project only needs enough structure to remember the ordered list of layers and activations.

### MLIR Construction

MLIR construction happens in `nn-mlir/tools/nn-compiler.cpp`.

The compiler:

- creates an `MLIRContext`
- registers the necessary dialects
- creates a top-level `ModuleOp`
- creates a `func.func` whose name matches the network
- models the network input, weights, and biases as function arguments
- emits `nn.dense` and `nn.relu` operations using `OpBuilder`

One of the most important implementation ideas is SSA threading. The frontend maintains a `current` value representing the current tensor flowing through the network. Each newly created operation consumes the previous `current` value and produces a new result, which then becomes the next `current` value. This is the MLIR equivalent of explicitly representing data flow between layers.

Weights and biases are modeled as function arguments rather than being parsed from the source file. This was a deliberate simplification that keeps the frontend focused on network structure rather than parameter storage.

At a traditional compiler level, this part of the project is best understood as the IR-building phase. The parser has already decided what the program means; `nn-compiler.cpp` is the stage that turns that meaning into actual compiler IR objects inside an `MLIRContext`, rooted in a `ModuleOp`, using `OpBuilder` calls. The closest Strix analogy is the transition from parsing to later AST use, except that in Strix the AST is ultimately walked directly by `ToWAT()`, while here the frontend produces a separate MLIR representation that will later be transformed by a lowering pass.

## Lowering Pass

The lowering pass is implemented in `nn-mlir/lib/Conversion/NNToStandard.cpp`.

The purpose of the pass is to eliminate the custom `nn` dialect operations and replace them with standard MLIR operations. This is the central compiler-transformation step in the project.

### Lowering `nn.relu`

`nn.relu` is lowered into a `linalg.generic` operation. The body of the generic op computes the elementwise maximum of each input value and zero using `arith.maxf` for floating-point values. This makes the ReLU semantics explicit in standard dialect operations while preserving the elementwise structure of the computation.

### Lowering `nn.dense`

`nn.dense` is lowered in multiple steps:

1. create an output tensor destination
2. fill that destination with zeros
3. perform `linalg.matmul` for matrix multiplication
4. add the bias using a `linalg.generic` operation with broadcast over the last dimension

This decomposition reflects the actual semantics of a dense layer more transparently than a single opaque operation would. It also demonstrates the value of lowering from high-level domain-specific operations into standard MLIR building blocks.

### Dialect Conversion

The pass uses MLIR’s dialect conversion framework.

The conversion marks the `nn` dialect as illegal after the pass and marks the standard `arith`, `func`, `linalg`, and `tensor` dialects as legal. Rewrite patterns are then applied to replace each `nn` operation with its lowered equivalent. If any `nn` operations remain after conversion, the pass fails. This makes the lowering stage precise and mechanically checkable.

## Testing and Validation

The project is validated through several concrete checks.

First, a dialect round-trip test verifies that hand-written MLIR containing `nn.dense` and `nn.relu` can be parsed and printed correctly.

Second, a frontend validation test checks that a `.nn` file can be compiled into valid `nn` dialect MLIR.

Third, a lowering validation test checks that after running the lowering pass, the output contains standard dialect operations and no remaining `nn.*` operations.

In addition to the original basic frontend example, the repository now includes several more `.nn` test inputs:

- `frontend_single_layer.nn` exercises the simplest valid network shape
- `frontend_deep.nn` exercises a longer chain of dense and ReLU layers
- `frontend_comments.nn` verifies that comments and blank lines are ignored correctly by the parser

The main demo commands are:

```bash
cd nn-mlir
make
./nn-opt tests/dialect_roundtrip.mlir
./nn-compiler tests/frontend_basic.nn | ./nn-opt
./nn-compiler --lower tests/frontend_basic.nn
```

The distinction between the last two commands is important.

`./nn-compiler tests/frontend_basic.nn | ./nn-opt` validates the frontend path only. It shows that the `.nn` source file can be parsed, converted into `nn` dialect MLIR, and then successfully parsed and printed by `nn-opt`. No lowering occurs in that command.

By contrast, `./nn-compiler --lower ...` takes a `.nn` source file all the way through the lowering stage and prints lowered MLIR directly. This is the simplest end-to-end frontend-plus-lowering command in the finished project.

The lowering pass can also still be invoked explicitly through `nn-opt`:

```bash
./nn-opt --pass-pipeline='builtin.module(lower-nn-to-standard)' tests/lowering_basic.mlir
```

In that form, the pass pipeline tells `nn-opt` to apply the `lower-nn-to-standard` pass to the top-level `ModuleOp`. This is the explicit command-line form of running the lowering stage of the compiler, and it is what actually rewrites `nn.dense` and `nn.relu` into standard `linalg`, `tensor`, and `arith` operations.

Additional useful test commands are:

```bash
cd nn-mlir
./nn-compiler tests/frontend_single_layer.nn | ./nn-opt
./nn-compiler tests/frontend_deep.nn | ./nn-opt
./nn-compiler tests/frontend_comments.nn | ./nn-opt
./nn-compiler --lower tests/frontend_deep.nn
```

The first three of these additional commands demonstrate frontend emission and parse/print validation on several network shapes. The final command demonstrates that the lowering pass also works on a deeper network, not just on the smallest hand-written MLIR example.

## Design Decisions and Simplifications

This project intentionally makes several simplifications in order to keep the implementation focused and defensible within the scope of a course project.

- The source language supports only dense layers and optional ReLU activations.
- The frontend assumes a fixed batch dimension of 1.
- The compiler models weights and biases as function arguments instead of parsing or storing trained parameters.
- The dialect uses lightweight tensor constraints rather than deep semantic verification.
- The project stops after lowering into standard MLIR dialects and does not continue into LLVM IR or native machine code generation.

These choices are limitations, but they are deliberate ones. They keep the project centered on compiler structure: parsing, IR design, programmatic IR generation, and lowering.

## What I Learned

This project helped me understand several core compiler and MLIR concepts more concretely.

I learned how MLIR dialects are defined declaratively with TableGen and then connected to generated C++ code through dialect registration.  
I learned how to construct IR programmatically using `OpBuilder` instead of emitting textual output directly like we did in Strix.  
I learned how SSA values make data flow explicit and how that differs from a more traditional direct code-generation pipeline.  
I learned how lowering passes work in MLIR and how the dialect conversion framework formalizes the process of replacing high-level operations with lower-level legal ones.

More broadly, the project clarified the difference between a simple single-IR educational compiler and a multi-level compiler architecture.

## Use of AI Assistance

I used AI as a development and learning assistant during this project, but not as a substitute for understanding the compiler design. In practice, AI was most useful for:

- helping me break the project into stages
- explaining unfamiliar MLIR concepts and APIs
- suggesting code structure for the frontend and lowering pass
- helping draft and revise documentation
- identifying cleanup and testing steps near the end of the project

The most important design decisions still required my own understanding and review. In particular, I had to verify:

- what the source language should support
- how the frontend data should map into MLIR
- how `nn.dense` and `nn.relu` should lower semantically
- which simplifications were acceptable for the scope of the project
- whether the generated code and test commands actually worked in the repository

As a result, AI accelerated implementation and helped me learn the MLIR ecosystem more quickly, but the final responsibility for architecture, code review, testing, debugging, and explanation remained mine.

## Future Work

There are several natural directions for extending this project:

- add stronger semantic verification for `nn.dense` and `nn.relu`
- support a richer source language with complex neural networks
- clean up the pass/build integration so the lowering pass is compiled and linked more conventionally
- continue lowering toward LLVM IR and native code generation, with code optimizations in the lowering passes

## Conclusion

This project demonstrates a complete mini-compiler pipeline built with MLIR. It includes a source language, a frontend parser, a custom dialect, programmatic IR generation, and a real lowering pass into standard MLIR dialects. The most important result is not the size of the source language, but the structure of the compiler: the project shows how high-level domain semantics can be represented explicitly and then progressively lowered using modern compiler infrastructure.

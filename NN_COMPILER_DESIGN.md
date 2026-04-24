# `.nn` File Format Design Document

## Purpose

This document explains the design choices behind the `.nn` file format and how it bridges from human-readable neural network descriptions to low-level MLIR IR.

---

## Problem Statement

We have:
- ✅ A working MLIR dialect (`nn.dense`, `nn.relu`)
- ✅ A tool to parse MLIR files (`nn-opt`)
- ❌ No way to write neural networks in a simple format

**Goal**: Create a simple, human-readable format for neural network definitions that compiles to MLIR.

---

## Design Philosophy

### 1. Simplicity Over Completeness

**Decision**: Only support `dense` and `relu` operations in Phase 1.

**Rationale**:
- A simple 2-operation language is easier to learn and implement
- Full neural network libraries (PyTorch, TensorFlow) have 1000+ operations
- We can add more operations later (convolution, pooling, normalization, etc.)
- Proves the compiler infrastructure works before adding complexity

**Trade-off**: Less expressive than production frameworks, but sufficient for educational purposes and demonstrating compiler technology.

---

### 2. Static Types With Explicit Shape Declaration

**Decision**: All tensor shapes are specified at definition time, not inferred.

```
input: x: 2x3xf32
dense_1 = dense(x, W, b) -> 2x4xf32
```

**Rationale**:
- Explicit types make compilation simple: no need for shape inference
- Mirrors MLIR's approach (all types must be specified)
- Easy to type-check: no ambiguity about what shapes are
- Catches errors early: if shapes don't match, compiler rejects the program

**Trade-off**: More verbose than Python frameworks (where shapes are inferred), but more explicit and easier to debug.

---

### 3. Linear Computation Graph

**Decision**: No branching, loops, or complex control flow.

```
input -> op1 -> op2 -> op3 -> output
```

**Rationale**:
- Matches the computation pattern of feed-forward neural networks
- Simple to parse and compile
- Avoids control flow complexity (CFG, dominators, etc.)
- Can extend later with conditional branches if needed

**Trade-off**: Can't express models with skip connections, dynamic shapes, or loops in Phase 1.

---

### 4. Declarative, Not Imperative

**Decision**: Specify *what* the network does, not *how* to compute it.

```
network SimpleMLP {         # Declarative: describe the network
  input: x: 2x3xf32
  dense_1 = dense(...) -> 2x4xf32
  output: result: 2x4xf32
}
```

NOT:

```
// Imperative pseudocode (what we're NOT doing)
function compute(x, W, b) {
  result = zeros([2, 4])
  for i in 0..2:
    for j in 0..4:
      result[i][j] = sum(x[i][k] * W[k][j] for k in 0..3) + b[j]
  return result
}
```

**Rationale**:
- Declarative programs are easier to optimize (compiler has freedom)
- Declarative programs are easier to compile to different targets
- MLIR is declarative, so our format matches MLIR's philosophy
- Imperative would require control flow, which we want to avoid

---

### 5. Single Network Per File

**Decision**: One `.nn` file = one network definition.

```
file: networks/simple_mlp.nn
contains: exactly 1 network definition
```

**Rationale**:
- Simplifies parsing: one entry point per file
- Mirrors common compilation practice (one class per file in Java, etc.)
- Easier to organize large projects (one model = one file)
- Can compose networks later with a "module system" if needed

**Trade-off**: Can't define multiple networks in one file, but that's rarely needed in practice.

---

## Format Design Decisions

### Why These Syntax Choices?

#### `network { ... }` Block

```
network SimpleMLP {
  input: x: 2x3xf32
  dense_1 = dense(x, W, b) -> 2x4xf32
  output: result: 2x4xf32
}
```

**Why**:
- `network { }` clearly delineates the scope
- Inspired by C-like syntax (familiar to programmers)
- Easy to parse: brace-delimited blocks are standard

#### `identifier: name: type` Pattern

```
input: x: 2x3xf32
output: result: 2x4xf32
```

**Why**:
- First `input:` / `output:` is a keyword
- Second `:` separates name from type
- Makes explicit: "this is an input tensor named `x` of type `2x3xf32`"
- Prevents ambiguity (not `input x 2x3xf32`)

#### `result = op(...) -> type` Pattern

```
dense_1 = dense(x, W, b) -> 2x4xf32
relu_1 = relu(dense_1) -> 2x4xf32
```

**Why**:
- Left side: variable name (where result goes)
- Right side: operation call
- `->` indicates output type
- Similar to functional programming: `f(x, y) -> output_type`
- Mirrors MLIR's SSA form: `%0 = op(...)`

#### `dense(input, weight, bias)` Argument Order

```
result = dense(input, weight, bias) -> output_type
```

**Why**:
- Natural order: input first, then parameters (weight, bias)
- Input is the data being transformed
- Weights and bias are learned parameters (static)
- Easy to remember: data → transformation

---

## Mapping to MLIR

How `.nn` operations map to MLIR operations:

### Dense Operation

**`.nn` syntax**:
```
dense_out = dense(x, W, b) -> 2x4xf32
```

**MLIR equivalent**:
```mlir
%0 = nn.dense %x, %W, %b : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
```

**Compilation step**: IR Builder sees the `.nn` operation and emits `OpBuilder.create<DenseOp>(...)`.

### Relu Operation

**`.nn` syntax**:
```
relu_out = relu(dense_out) -> 2x4xf32
```

**MLIR equivalent**:
```mlir
%1 = nn.relu %0 : tensor<2x4xf32> -> tensor<2x4xf32>
```

### Full Network

**`.nn` file**:
```
network SimpleMLP {
  input: x: 2x3xf32
  dense_1 = dense(x, W1, b1) -> 2x4xf32
  relu_1 = relu(dense_1) -> 2x4xf32
  output: result: 2x4xf32
}
```

**Generated MLIR**:
```mlir
func.func @SimpleMLP(
  %arg0: tensor<2x3xf32>,  // x
  %arg1: tensor<3x4xf32>,  // W1
  %arg2: tensor<4xf32>     // b1
) -> tensor<2x4xf32> {
  %0 = nn.dense %arg0, %arg1, %arg2 : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
  %1 = nn.relu %0 : tensor<2x4xf32> -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}
```

**Key mapping**:
- Network name → Function name
- Input tensors → Function arguments
- Operations → MLIR operations
- Output tensor → Function return

---

## Type System Design

### Why Explicit Tensor Types?

```
input: x: 2x3xf32
```

Not just:
```
input: x
```

**Reasons**:

1. **No type inference complexity**: Compiler doesn't need to solve constraints
2. **Early error detection**: Type errors caught at parse time
3. **Specification clarity**: Reader knows exactly what shapes are expected
4. **MLIR compatibility**: MLIR requires all types to be explicit
5. **Compilation simplicity**: No need for separate type-checking phase

### Why These Element Types?

Supported: `f32`, `f64`, `i32`, `i64`

**Rationale**:
- `f32` and `f64`: Standard neural network precision
- `i32` and `i64`: Support for quantization, integer operations (future)
- Not supporting: `bf16`, `float8` (can add later)
- Keeps Phase 1 simple

---

## Compilation Pipeline

### `.nn` → MLIR Pipeline

```
.nn file
  ↓
[Lexer]           tokenizes into: KEYWORD, IDENTIFIER, TYPE, ...
  ↓
tokens
  ↓
[Parser]          builds: NetworkDef AST
  ↓
AST (Abstract Syntax Tree)
  ↓
[IR Builder]      converts: AST → MLIR operations
  ↓
MLIR IR (in-memory)
  ↓
[nn-opt]          parses/prints/optimizes MLIR
  ↓
MLIR output
```

### Three Compiler Phases

| Phase | Input | Output | Component |
|-------|-------|--------|-----------|
| **Lexical** | Raw text | Token stream | `Lexer.cpp` |
| **Syntactic** | Token stream | AST | `Parser.cpp` |
| **Semantic** | AST | MLIR IR | `IRBuilder.cpp` |

This mirrors traditional compilers:
- C source → Lexer → tokens
- tokens → Parser → AST
- AST → Codegen → assembly

---

## Implementation Strategy

### Phase 2a: Lexer (`src/Lexer.cpp`)

**Input**: `.nn` file text
**Output**: Token stream
**Complexity**: Low
- Straightforward character-by-character scanning
- Match keywords, identifiers, numbers, operators
- ~200-300 lines of C++

**Example**:
```
Input:  "dense(x, W, b) -> 2x4xf32"
Output: [dense, (x, comma, W, comma, b, ), arrow, 2, x, 4, x, f32]
```

### Phase 2b: Parser (`src/Parser.cpp`)

**Input**: Token stream
**Output**: AST (Abstract Syntax Tree)
**Complexity**: Medium
- Recursive descent parser (no parser generator needed)
- Match grammar rules from `NN_GRAMMAR.md`
- Build tree of AST nodes
- ~400-600 lines of C++

**Example**:
```
Input tokens:  [network, SimpleMLP, {, input, :, x, :, 2x3xf32, ..., }]
Output AST:
  NetworkDef {
    name: "SimpleMLP",
    input: InputNode("x", TensorType([2, 3], f32)),
    ops: [...],
    output: OutputNode("result", TensorType([2, 2], f32))
  }
```

### Phase 2c: IR Builder (`src/IRBuilder.cpp`)

**Input**: AST
**Output**: MLIR Module (in-memory IR)
**Complexity**: Medium-High
- Walk AST, creating MLIR operations
- Use MLIR's `OpBuilder` API
- Build function with correct signature
- Type-check as you go
- ~300-400 lines of C++

**Example**:
```cpp
// For each AST node:
for (auto& op : ast.operations) {
  if (op.type == "dense") {
    builder.create<nn::DenseOp>(...);
  }
}
```

---

## Error Handling Strategy

### Compile-Time Errors (Should Reject Programs)

1. **Syntax errors**: Invalid `.nn` syntax
   - `dense(x W b)` ← missing commas
   - Parser rejects with line/column info

2. **Type errors**: Mismatched tensor dimensions
   - `dense(2x3, 3x5, b)` → expects 5 in bias but gets 4xf32
   - Type checker rejects with error message

3. **Undefined variable**: Using undefined tensor
   - `relu(undefined_var)` → `undefined_var` never defined
   - Parser rejects at use site

4. **Missing input/output**: Network lacks input or output statement
   - Parser rejects if `input:` or `output:` statements missing

### Not Supporting Yet (Can Add Later)

- Compile-time shape inference
- Dynamic shapes
- Symbolic dimensions (e.g., `Bx3xf32` for batch size B)
- Type coercion (e.g., `i32` → `f32` automatic casting)

---

## Example: Walking Through Compilation

### Input File: `networks/mlp.nn`

```
network TwoLayerMLP {
  input: x: 2x3xf32

  hidden = dense(x, W1, b1) -> 2x4xf32
  activated = relu(hidden) -> 2x4xf32

  output: logits: 2x2xf32
}
```

### Lexer Output

```
NETWORK, IDENTIFIER(TwoLayerMLP), LBRACE,
INPUT, COLON, IDENTIFIER(x), COLON, 2, X, 3, X, f32, NEWLINE,
IDENTIFIER(hidden), EQUALS, dense, LPAREN, IDENTIFIER(x), COMMA, ...,
...
EOF
```

### Parser Output (AST)

```
NetworkDef {
  name: "TwoLayerMLP"
  input: InputStmt {
    name: "x"
    type: TensorType([2, 3], f32)
  }
  operations: [
    DenseOp {
      result: "hidden"
      input: "x"
      weight: "W1"
      bias: "b1"
      output_type: TensorType([2, 4], f32)
    },
    ReluOp {
      result: "activated"
      input: "hidden"
      output_type: TensorType([2, 4], f32)
    }
  ]
  output: OutputStmt {
    name: "logits"
    type: TensorType([2, 2], f32)
  }
}
```

### IR Builder Output (MLIR)

```mlir
module {
  func.func @TwoLayerMLP(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4x2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2xf32> {
    %0 = nn.dense %arg0, %arg1, %arg2 : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
    %1 = nn.relu %0 : tensor<2x4xf32> -> tensor<2x4xf32>
    return %1 : tensor<2x2xf32>
  }
}
```

### Using nn-opt

```bash
$ nn-compiler networks/mlp.nn > /tmp/mlp.mlir
$ nn-opt /tmp/mlp.mlir
module {
  func.func @TwoLayerMLP(...) -> tensor<2x2xf32> {
    %0 = nn.dense ...
    %1 = nn.relu ...
    return %1 : tensor<2x2xf32>
  }
}
```

---

## Why This Design Works

### ✅ Simplicity
- Grammar is 30 lines of EBNF
- Only 2 operations
- No control flow
- Easy to implement, learn, and extend

### ✅ Correctness
- Explicit types prevent errors
- Static shapes easy to verify
- Declarative form ensures determinism

### ✅ Extensibility
- New operations: just add to grammar + implement IR builder
- New types: add to type system
- Control flow: can extend parser later

### ✅ Compiler Technology
- Proves all three compiler phases work: lexer → parser → codegen
- Real parsing challenges (precedence, associativity) avoided but structurally sound
- Directly maps to MLIR (no intermediate transformations needed)

---

## Next Steps

1. **Ticket 2.1-2.3**: Implement Lexer
2. **Ticket 2.4-2.6**: Implement Parser
3. **Ticket 2.7-2.9**: Implement IR Builder
4. **Ticket 2.10**: Test end-to-end compilation

Ready to start implementing?

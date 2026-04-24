# `.nn` File Grammar Specification

## Formal Grammar (EBNF)

```ebnf
(* Top-level: A file contains a single network definition *)
program         = network ;

(* Network definition *)
network         = "network" identifier "{" statement_list "}" ;
statement_list  = { statement } ;
statement       = input_stmt
                | operation_stmt
                | output_stmt ;

(* Input statement: declares input tensors *)
input_stmt      = "input" ":" identifier ":" tensor_type ;

(* Operation statements: dense or relu *)
operation_stmt  = dense_stmt | relu_stmt ;

dense_stmt      = identifier "=" "dense" "(" identifier "," identifier "," identifier ")"
                  "->" tensor_type ;
relu_stmt       = identifier "=" "relu" "(" identifier ")" "->" tensor_type ;

(* Output statement: declares output tensor *)
output_stmt     = "output" ":" identifier ":" tensor_type ;

(* Tensor types *)
tensor_type     = dimension_list "x" element_type ;
dimension_list  = dimension ( "x" dimension )* ;
dimension       = integer ;
element_type    = "f32" | "f64" | "i32" | "i64" ;

(* Basic tokens *)
identifier      = letter ( letter | digit | "_" )* ;
integer         = digit+ ;
letter          = "a".."z" | "A".."Z" ;
digit           = "0".."9" ;
```

---

## Token Types

| Token Type | Examples | Description |
|-----------|----------|-------------|
| `KEYWORD` | `network`, `input`, `output`, `dense`, `relu` | Reserved words |
| `IDENTIFIER` | `SimpleNN`, `x`, `weight`, `bias` | Variable/network names |
| `INTEGER` | `2`, `3`, `4`, `32` | Dimension sizes, bit widths |
| `TYPE` | `f32`, `f64`, `i32`, `i64` | Element types |
| `LPAREN` | `(` | Left parenthesis |
| `RPAREN` | `)` | Right parenthesis |
| `LBRACE` | `{` | Left brace |
| `RBRACE` | `}` | Right brace |
| `COMMA` | `,` | Comma separator |
| `COLON` | `:` | Colon separator |
| `ARROW` | `->` | Type annotation |
| `EQUALS` | `=` | Assignment |
| `X` | `x` | Dimension separator |
| `EOF` | (implicit) | End of file |

---

## Type System

### Tensor Types

Tensors are multi-dimensional arrays with a fixed shape and element type.

**Syntax**: `dim1 x dim2 x ... x dimN x element_type`

**Examples**:
- `2x3xf32` — 2D tensor, 2 rows × 3 columns, float32 elements
- `4xf32` — 1D tensor (vector), 4 elements, float32
- `2x3x4xf32` — 3D tensor, shape (2, 3, 4), float32

### Supported Element Types

- `f32` — 32-bit floating point
- `f64` — 64-bit floating point
- `i32` — 32-bit signed integer
- `i64` — 64-bit signed integer

---

## Operations

### `dense` Operation

Matrix multiplication followed by bias addition.

**Syntax**:
```
result_var = dense(input_var, weight_var, bias_var) -> output_type
```

**Semantics**:
- `input_var`: Input tensor (shape: `m x n`)
- `weight_var`: Weight matrix (shape: `n x k`)
- `bias_var`: Bias vector (shape: `k`)
- **Result**: Output tensor (shape: `m x k`)
- **Computation**: `output = input @ weight + bias`

**Example**:
```
dense_out = dense(x, W, b) -> 2x4xf32
```
- `x`: `2x3xf32` (2 samples, 3 features)
- `W`: `3x4xf32` (3 input features, 4 output neurons)
- `b`: `4xf32` (4 output neurons)
- Result: `2x4xf32` (2 samples, 4 outputs)

### `relu` Operation

Rectified Linear Unit activation: `relu(x) = max(0, x)` applied element-wise.

**Syntax**:
```
result_var = relu(input_var) -> output_type
```

**Semantics**:
- `input_var`: Input tensor of any shape
- **Result**: Output tensor with same shape as input
- **Computation**: Element-wise `max(0, x)`

**Example**:
```
activated = relu(dense_out) -> 2x4xf32
```
- Takes `2x4xf32` tensor
- Returns `2x4xf32` tensor with negative values zeroed out

---

## Complete Example

### File: `networks/simple_mlp.nn`

```
network SimpleMLP {
  input: x: 2x3xf32

  dense_1 = dense(x, W1, b1) -> 2x4xf32
  relu_1 = relu(dense_1) -> 2x4xf32

  dense_2 = dense(relu_1, W2, b2) -> 2x2xf32
  relu_2 = relu(dense_2) -> 2x2xf32

  output: result: 2x2xf32
}
```

### Execution Flow

```
Inputs:
  x:  [2x3xf32]
  W1: [3x4xf32]
  b1: [4xf32]
  W2: [4x2xf32]
  b2: [2xf32]

Computation:
  dense_1 = x @ W1 + b1  →  [2x4xf32]
  relu_1 = max(0, dense_1)  →  [2x4xf32]

  dense_2 = relu_1 @ W2 + b2  →  [2x2xf32]
  relu_2 = max(0, dense_2)  →  [2x2xf32]

Output: result = relu_2  →  [2x2xf32]
```

---

## Constraints & Rules

1. **Network name must be unique** in the file (only one network per file)

2. **Identifier uniqueness**: Each variable/tensor must be defined before use
   - ❌ `relu(dense_1)` before `dense_1` is defined
   - ✅ Define `dense_1` first, then use it

3. **Type consistency**:
   - All tensor dimensions must match between operations
   - ❌ `dense(x, W, b)` where `x` is `2x3xf32` but `W` is `5x4xf32`
   - ✅ Dimensions match: `2x3xf32 @ 3x4xf32 → 2x4xf32`

4. **Dense operation requirements**:
   - 3 operands (input, weight, bias)
   - Input shape: `m x n`
   - Weight shape: `n x k`
   - Bias shape: `k` (must be 1D)
   - Output shape: `m x k`

5. **Relu operation requirements**:
   - 1 operand (any shape)
   - Output shape: same as input

6. **Input/Output declarations**:
   - Must have exactly 1 `input` statement
   - Must have exactly 1 `output` statement
   - Input must be the first operation
   - Output must reference the final computed tensor

---

## Lexical Rules

1. **Whitespace**: Ignored (spaces, tabs, newlines)
2. **Comments**: Not currently supported (can add later as `// comment`)
3. **Case-sensitive**: `dense` ≠ `Dense`
4. **Integers**: Positive integers only (no negative numbers)

---

## Error Cases

The compiler should reject these inputs:

```
(* Missing input *)
network BadNet {
  dense_1 = dense(x, W, b) -> 2x4xf32
  output: result: 2x4xf32
}

(* Missing output *)
network BadNet {
  input: x: 2x3xf32
  dense_1 = dense(x, W, b) -> 2x4xf32
}

(* Type mismatch *)
network BadNet {
  input: x: 2x3xf32
  dense_1 = dense(x, W, b) -> 2x5xf32  // Wrong output type
  output: result: 2x5xf32
}

(* Undefined variable *)
network BadNet {
  input: x: 2x3xf32
  relu_1 = relu(undefined_var) -> 2x3xf32  // undefined_var not defined
  output: result: 2x3xf32
}
```

---

## Next Steps (Implementation)

The lexer will tokenize this grammar:
- Read file character-by-character
- Produce stream of tokens: `NETWORK`, `IDENTIFIER("SimpleMLP")`, `LBRACE`, ...

The parser will build an AST:
```
NetworkDef {
  name: "SimpleMLP"
  input: Tensor("x", [2, 3], "f32")
  operations: [
    DenseOp("dense_1", "x", "W1", "b1", [2, 4], "f32"),
    ReluOp("relu_1", "dense_1", [2, 4], "f32"),
    ...
  ]
  output: Tensor("result", [2, 2], "f32")
}
```

The IR builder will convert to MLIR:
```mlir
func.func @SimpleMLP(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>, ...) -> tensor<2x2xf32> {
  %0 = nn.dense %arg0, %arg1, %arg2 : ...
  %1 = nn.relu %0 : ...
  ...
  return %1 : tensor<2x2xf32>
}
```

# TableGen Dialect Definition Guide

## What is a Dialect? (Concrete Version)

A dialect is a **collection of operations that belong together**.

Think of it like a **namespace in programming**:

```cpp
// This is like a C++ namespace
namespace nn {
  class DenseOp { ... };
  class ReluOp { ... };
}
```

In MLIR, we call this a **dialect**. Instead of `nn::DenseOp`, we write `nn.dense` in the IR.

**Example MLIR code that uses the nn dialect:**
```mlir
func.func @forward(%input: tensor<1x784xf32>) -> tensor<1x10xf32> {
  %w = arith.constant dense<0.1> : tensor<784x256xf32>
  %b = arith.constant dense<0.0> : tensor<256xf32>
  
  %h = nn.dense %input, %w, %b : tensor<1x784xf32>, tensor<784x256xf32>, tensor<256xf32> -> tensor<1x256xf32>
  
  %output = nn.relu %h : tensor<1x256xf32> -> tensor<1x256xf32>
  return %output : tensor<1x256xf32>
}
```

Notice:
- `func.func` — from the `func` dialect
- `arith.constant` — from the `arith` dialect
- `nn.dense`, `nn.relu` — from the `nn` dialect (which you're defining)

**Every operation belongs to exactly one dialect.**

---

## What is the Dialect For?

The dialect tells MLIR:
1. "These operations exist"
2. "Here's what arguments each operation takes"
3. "Here's what type each operation returns"
4. "Here's how to print/parse each operation"
5. "Here's how to verify each operation is valid"

---

## How to Define a Dialect in TableGen

A `.td` file has this structure:

```tablegen
// Step 1: Include MLIR base definitions
#include "mlir/IR/OpBase.td"

// Step 2: Define the dialect itself
def MyDialect : Dialect {
  let name = "mydialect";
  let summary = "A description";
  let cppNamespace = "::mlir::mydialect";
}

// Step 3: Define a base class for operations in this dialect
class MyOp<string mnemonic> : Op<MyDialect, mnemonic>;

// Step 4: Define each operation
def MyOp1 : MyOp<"op1"> {
  let arguments = (ins Type:$arg1);
  let results = (outs Type:$result);
  // ... more properties
}

def MyOp2 : MyOp<"op2"> {
  // ... similar
}
```

---

## Breaking Down Each Part

### Part 1: Dialect Declaration

```tablegen
def NN_Dialect : Dialect {
  let name = "nn";                        // Mnemonic used in IR (nn.dense, nn.relu)
  let summary = "Neural network dialect"; // One-line description
  let cppNamespace = "::mlir::nn";        // C++ namespace where the generated classes go
}
```

This tells MLIR:
- "I'm defining a dialect named `nn`"
- "Generated C++ code goes in the `mlir::nn` namespace"
- "When you see `nn.something` in MLIR, it belongs to this dialect"

### Part 2: Base Operation Class

```tablegen
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;
```

This is a **helper class** that you use to define individual operations. It says:
- "Every operation in the NN dialect inherits from this base class"
- "Operations take a mnemonic (name) and optional traits"
- "`Op<NN_Dialect, mnemonic, traits>` connects it to the dialect"

Think of it like a C++ base class:
```cpp
class NNOp : public Operation {
  // ...
};

class DenseOp : public NNOp { // inherits from NNOp
  // ...
};
```

### Part 3: Operation Definition

```tablegen
def DenseOp : NN_Op<"dense"> {
  let summary = "Dense layer: output = input @ weights + bias";
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    AnyRankedTensor:$bias
  );
  
  let results = (outs
    AnyRankedTensor:$output
  );
  
  let assemblyFormat = "$input `,` $weights `,` $bias attr-dict `:` type($input) `,` type($weights) `,` type($bias) `->` type($output)";
}
```

Breaking this down:

| Line | What it means |
|------|---------------|
| `def DenseOp : NN_Op<"dense">` | Define an op called `DenseOp` with mnemonic `"dense"` (in IR: `nn.dense`) |
| `let summary = "..."` | Human-readable description |
| `let arguments = (ins ...)` | The operation takes 3 inputs (operands): input, weights, bias |
| `AnyRankedTensor:$input` | An operand named `$input` of type `AnyRankedTensor` |
| `let results = (outs ...)` | The operation produces 1 output: a tensor |
| `let assemblyFormat = "..."` | How to print/parse this in MLIR text. See below. |

### Part 4: Assembly Format

```tablegen
let assemblyFormat = "$input `,` $weights `,` $bias attr-dict `:` type($input) `,` type($weights) `,` type($bias) `->` type($output)";
```

This tells MLIR how to print/parse the operation in text. It reads like:

```
$input          <- print the input operand
`,`             <- print a comma (literal)
$weights        <- print the weights operand
`,`             <- print a comma
$bias           <- print the bias operand
attr-dict       <- print any attributes (usually skipped)
`:`             <- print a colon (literal, backticks mean literal)
type($input)    <- print the type of the input operand
`,`             <- comma
type($weights)  <- type of weights
`,`             <- comma
type($bias)     <- type of bias
`->`            <- print arrow (literal)
type($output)   <- print the type of output
```

**Result**: When you print `nn.dense`, it looks like:
```mlir
%h = nn.dense %input, %weights, %bias : tensor<...>, tensor<...>, tensor<...> -> tensor<...>
```

---

## The Complete NNOps.td File

Here's exactly what you'll write in `include/NN/NNOps.td`:

```tablegen
//===- NNOps.td - NN dialect operation definitions -*-tablegen-*-===//
// This file defines the operations for the NN dialect.
//===----------------------------------------------------------------------===//

#ifndef NN_OPS
#define NN_OPS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NN Dialect Definition
//===----------------------------------------------------------------------===//

def NN_Dialect : Dialect {
  let name = "nn";
  let summary = "A neural network dialect for MLIR";
  let description = [{
    This dialect represents high-level neural network operations.
    Operations include dense layers and activation functions.
  }];
  let cppNamespace = "::mlir::nn";
}

//===----------------------------------------------------------------------===//
// Base NN Operation
//===----------------------------------------------------------------------===//

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NN Operations
//===----------------------------------------------------------------------===//

def DenseOp : NN_Op<"dense"> {
  let summary = "Dense (fully-connected) layer: output = input @ weights + bias";
  let description = [{
    Performs a dense layer computation:
    output = input @ weights + bias
    
    Arguments:
    - input:   tensor<batch x in_features x f32>
    - weights: tensor<in_features x out_features x f32>
    - bias:    tensor<out_features x f32>
    
    Result:
    - output:  tensor<batch x out_features x f32>
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    AnyRankedTensor:$bias
  );
  
  let results = (outs
    AnyRankedTensor:$output
  );

  let assemblyFormat = [{
    $input `,` $weights `,` $bias attr-dict `:`
    type($input) `,` type($weights) `,` type($bias) `->` type($output)
  }];
}

def ReluOp : NN_Op<"relu"> {
  let summary = "ReLU activation: output = max(0, input)";
  let description = [{
    Applies element-wise ReLU activation:
    output = max(0, input)
    
    Input and output have the same tensor type.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );
  
  let results = (outs
    AnyRankedTensor:$output
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // NN_OPS
```

---

## Key Concepts

### `def` vs `class`

- **`def`**: Concrete definition (creates an actual record). Use this for operations.
- **`class`**: Template/base class. Use this for reusable patterns.

In our file:
- `class NN_Op<...>` — template for all NN operations
- `def DenseOp : NN_Op<...>` — concrete DenseOp that inherits from template

### Type Names

- `AnyRankedTensor` — any ranked tensor type (e.g., `tensor<1x784xf32>`)
- `AnyType` — any type (tensors, memrefs, integers, floats, etc.)
- `F32` — specifically a 32-bit float
- `I32` — specifically a 32-bit integer

### Assembly Format Syntax

- `$operand` — print/parse the named operand
- `type($operand)` — print/parse the type of the operand
- `` `literal` `` — print a literal character (backticks mean it's literal)
- `attr-dict` — print/parse any attributes
- `optional(...)` — make something optional in parsing

---

## What Happens Next

Once you create this file:

1. **TableGen reads `NNOps.td`** and understands:
   - The `NN_Dialect` exists
   - Two operations exist: `DenseOp` and `ReluOp`
   - What their arguments and results are
   - How to parse/print them

2. **We run `mlir-tblgen`** which generates C++ code:
   - `NNDialect.h.inc` — the dialect class
   - `NNOps.h.inc` — the operation classes

3. **We include those `.inc` files** in our hand-written `.cpp` files

4. **The compiler** compiles everything together into the binary

---

## Next Steps

1. Create `/Users/adityapandey/Desktop/mlir/nn-mlir/include/NN/NNOps.td`
2. Copy the complete file above into it
3. Save it

Then we'll write the Makefile to run TableGen on this file.

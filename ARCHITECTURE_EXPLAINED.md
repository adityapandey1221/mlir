# NN-MLIR Compiler: Architecture Explained (From First Principles)

## The Goal

You're building a **compiler** that transforms neural network specifications into optimized machine code.

**Input**: A text file describing a neural network
```
network mlp
input 784
dense 256 relu
dense 10
```

**Output**: Machine code that computes the forward pass correctly and efficiently.

This is **exactly like compiling C code to machine code**, just for neural networks instead.

---

## The Classic Compiler Pipeline

You know from college that compilers follow this pattern:

```
Source code
    ↓
[Frontend] Lexer → Parser → AST → Type check
    ↓
[Optimization] Transform AST/IR
    ↓
[Backend] Lower to machine code
    ↓
Machine code
```

Our compiler follows the same pattern:

```
simple_mlp.nn
    ↓
[Lexer] Tokenize: "dense", "256", "relu"
[Parser] Build AST: NetworkAST { layers: [Dense(256, relu), Dense(10)] }
[IR Gen] Build MLIR: nn.dense ops in a module
    ↓
[Lowering Pass] Lower nn.dense → linalg.matmul + linalg.add
    ↓
[LLVM Pass] Lower to LLVM IR → machine code
    ↓
Executable
```

---

## Why MLIR Specifically?

In a standard compiler (like gcc), you have **one IR**:

```
C source code → [frontend] → LLVM IR → [optimization] → machine code
```

LLVM IR is **low-level** (close to assembly). Optimizations at this level are about register allocation, instruction scheduling, etc.

**The problem**: High-level optimizations (like "fuse these two matrix multiplies") are hard to express in low-level IR.

MLIR solves this with **multiple IRs at different levels**:

```
nn.dense (high-level, semantic)
    ↓ [lowering]
linalg.matmul (mid-level, algorithmic)
    ↓ [lowering]
scf.for + arith.* (low-level, loops + arithmetic)
    ↓ [lowering]
llvm.* (very low-level, almost assembly)
    ↓
machine code
```

**Key insight**: Each level is optimizable independently.
- At the `nn` level: "fuse consecutive dense layers"
- At the `linalg` level: "tile this matmul for cache locality"
- At the `scf` level: "vectorize this inner loop"

This is the **modern compiler approach** (used by IREE, torch-mlir, XLA, etc.).

---

## The 4 Phases You're Implementing

### **Phase 1: Define the Dialect (Day 1)**

Before you can build IR, MLIR needs to know what operations exist.

You define a **dialect** — a collection of operations:
- `nn.dense` — matrix multiply + bias
- `nn.relu` — ReLU activation

This is like writing the **IR definition** in a traditional compiler.

**Files created:**
- `NNOps.td` — TableGen spec (what operations exist, their types, how to parse/print them)
- `NNDialect.cpp`, `NNOps.cpp` — glue code that includes generated files
- `nn-opt` tool — validates that the dialect works

**Status**: The dialect is defined. You can now parse/print MLIR with `nn.dense` and `nn.relu`.

**Analogy**: This is like defining the instruction set for a new CPU. You're saying "here's what operations I support."

---

### **Phase 2: Build the Frontend (Day 2)**

Now you write the **frontend** that parses `.nn` files and generates IR.

**Pipeline:**
```
simple_mlp.nn
    ↓ [Lexer] → tokens
    ↓ [Parser] → AST
    ↓ [IR Builder] → MLIR module with nn.* ops
    ↓ [Print] → MLIR text
```

**Lexer**: Tokenizes input
```
"network mlp input 784 dense 256 relu dense 10"
    ↓
["network", "mlp", "input", "784", "dense", "256", "relu", "dense", "10"]
```

**Parser**: Builds AST
```
NetworkAST {
  name: "mlp",
  inputSize: 784,
  layers: [
    Layer(type="dense", units=256, activation="relu"),
    Layer(type="dense", units=10, activation="none")
  ]
}
```

**IR Builder**: Walks AST, creates MLIR operations
```cpp
// For each layer in AST:
auto dense_op = builder.create<nn::DenseOp>(
  loc, outputType, input, weights, bias
);
```

**Key API**: You use MLIR's `OpBuilder` to construct operations (not string concatenation).

This is crucial because:
- Type checking happens automatically
- SSA form is enforced
- Operations can be analyzed/transformed by other passes

**Analogy**: This is exactly how a C compiler's frontend works. Lex → Parse → Build IR.

---

### **Phase 3: Write the Lowering Pass (Day 3)**

Now you write **pattern-matching rules** that transform high-level ops into low-level ops.

**Before:**
```mlir
%h = nn.dense %input, %weights, %bias : ...
```

**After:**
```mlir
%h0 = linalg.matmul ins(%input, %weights) outs(%init)
%h1 = linalg.add %h0, %bias
```

**How it works:**
1. Match pattern: "Is this a `nn.dense` op?"
2. Extract operands and types
3. Create equivalent linalg operations
4. Replace the `nn.dense` with the new ops
5. Delete the old op

**Why it matters**: This is **the core of the compiler**. It shows you understand how compilers transform IR. Real ML compilers (IREE, torch-mlir) do this same thing.

**Analogy**: This is like a compiler optimization pass (e.g., loop unrolling, constant folding). You're rewriting the IR for efficiency.

---

### **Phase 4: Test and Polish (Day 4)**

Verify everything works end-to-end:
```
simple_mlp.nn → [parse] → [IR gen] → [lower] → valid MLIR with linalg ops
```

Tests:
- Can you parse the `.nn` file without errors?
- Does the generated MLIR have the right ops?
- Does lowering produce valid linalg ops?
- Can MLIR's built-in tools read the output?

---

## What You're Doing Right Now

**We're finishing Phase 1.**

We've:
1. ✅ Defined what `nn.dense` and `nn.relu` are (TableGen)
2. ✅ Generated the C++ boilerplate (TableGen → `.inc` files)
3. ✅ Written the minimal glue code (`NNDialect.cpp`, `NNOps.cpp`)
4. ⏳ Need to build `nn-opt` tool (so you can test parsing/printing)

Once `nn-opt` works, you know Phase 1 is complete. Then you move to Phase 2 (the frontend).

---

## The Key Mental Model

Think of it like this:

| Component | What it does | Example |
|-----------|------------|---------|
| **Dialect** | Define what operations exist | "There exists an `nn.dense` op that takes 3 tensors and produces 1" |
| **Parser** | Parse text → AST | "dense 256 relu" → `DenseLayer(256, "relu")` |
| **IR Builder** | AST → MLIR ops | Walk AST, call `builder.create<nn::DenseOp>(...)` |
| **Lowering** | Transform ops | Replace `nn.dense` with `linalg.matmul + linalg.add` |
| **Codegen** | MLIR → machine code | (Done by LLVM, you don't write this) |

---

## How It All Fits Together

```
┌────────────────────────────────────────────────────┐
│          NN-MLIR Compiler Architecture             │
└────────────────────────────────────────────────────┘

INPUT: simple_mlp.nn
    │
    ├─ [Phase 1: Dialect] ← YOU ARE HERE
    │   TableGen defines nn.dense, nn.relu
    │   nn-opt tool validates dialect
    │
    ├─ [Phase 2: Frontend]
    │   Lexer + Parser + IR Builder
    │   Builds MLIR with nn.* ops
    │
    ├─ [Phase 3: Lowering]
    │   Converts nn.* → linalg.* → scf.* → llvm.*
    │
    └─ [Phase 4: Testing]
        Verify end-to-end

OUTPUT: Machine code (via LLVM backend)
```

---

## Comparison to Real Compilers

Your architecture mirrors production ML compilers:

**IREE (Google):**
```
TensorFlow/PyTorch → IREE custom ops → linalg → memref → LLVM → machine code
```

**torch-mlir (PyTorch):**
```
PyTorch ops → torch dialect → linalg → memref → LLVM → machine code
```

**XLA (TensorFlow):**
```
TF ops → XLA IR → linalg → LLVM → machine code
```

Your compiler:
```
.nn file → nn dialect → linalg → LLVM → machine code
```

Same pattern. Same principles. You're learning real compiler architecture.

---

## Why This Matters for Your Project

1. **Educational**: You understand the full compilation pipeline from first principles
2. **Impressive**: This demonstrates compiler infrastructure knowledge
3. **Extensible**: Adding new features (convolution, batch norm) is straightforward
4. **Explainable**: You can explain every component to your professor

When they ask "why did you design it this way?" you can say:
> "I designed it following the MLIR architecture, which uses progressive lowering through multiple IR levels. This allows high-level optimizations (at the nn dialect) and low-level optimizations (at the llvm level) to be independent and composable."

That's the answer they want to hear.

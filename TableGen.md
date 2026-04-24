# TableGen: What It Is and Why We Use It

## What is TableGen?

TableGen is a **code generator** that comes with LLVM/MLIR. It's a tool that reads declarative specifications (`.td` files) and generates C++ code.

Think of it like a template engine, but specifically designed for compiler infrastructure.

---

## How TableGen Works (Simple Example)

### You write (declarative):
```tablegen
def DenseOp : NN_Op<"dense"> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weights, AnyRankedTensor:$bias);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input `,` $weights `,` $bias attr-dict `:` type($input) `,` type($weights) `,` type($bias) `->` type($output)";
}
```

### TableGen generates (C++ code):
```cpp
class DenseOp : public Operation {
public:
  // Constructor
  static DenseOp create(Location loc, Type resultType, Value input, Value weights, Value bias) { ... }

  // Accessors
  Value getInput() { return getOperand(0); }
  Value getWeights() { return getOperand(1); }
  Value getBias() { return getOperand(2); }

  // Parser (reads "nn.dense %input, %weights, %bias : ...")
  ParseResult parse(OpAsmParser &parser, OperationState &result) { ... }

  // Printer (writes "nn.dense %input, %weights, %bias : ...")
  void print(OpAsmPrinter &printer) { ... }

  // Verifier (checks operands are valid)
  LogicalResult verify() { ... }
};
```

This is **200+ lines of boilerplate C++ code** that TableGen writes for you.

---

## Why We Use TableGen for Dialects

### The Problem TableGen Solves

If you had to write that C++ by hand for every operation, you'd:
1. Write the same patterns over and over (tedious)
2. Make mistakes (bugs in parser, printer, verifier)
3. Maintain 10 different implementations of the same logic
4. Spend 80% of your time on boilerplate, 20% on actual logic

### The TableGen Solution

Write the **spec once**, TableGen generates the **C++ once**. You focus on the logic.

---

## Why TableGen Specifically for MLIR Dialects

MLIR operations have **standard parts**:
- Arguments (operands) with types
- Results with types
- Traits (properties like "pure", "commutative")
- Assembly format (how it prints/parses)
- Builders (convenient constructors)
- Verifiers (type checking)

All of these are **formulaic** — they follow patterns. TableGen automates these patterns.

---

## The Build Flow with TableGen

```
include/NN/NNOps.td  (You write: what the op is)
    ↓
  [mlir-tblgen --gen-op-decls]  (Tool runs)
    ↓
include/NN/NNOps.h.inc  (Generated: C++ op class declarations)
    ↓
lib/NN/NNOps.cpp includes "NNOps.h.inc"
    ↓
clang++ compiles it
    ↓
Binary contains your operations
```

---

## Why NOT Just Hand-Write C++?

You could write the C++ directly. Why not?

| Approach | Boilerplate | Extensibility | Maintainability |
|----------|------------|---------------|-----------------|
| **TableGen** | Minimal (you write 50 lines) | Easy (change `.td`, regenerate) | Easy (single source of truth) |
| **Hand-written C++** | Massive (200+ lines) | Hard (change C++, update everywhere) | Hard (multiple files, duplication) |

For a dialect with 2 operations, TableGen saves maybe 400 lines of C++. But more importantly, **it's the standard MLIR way**. Every dialect in MLIR (linalg, arith, vector, etc.) uses TableGen.

---

## What TableGen Actually Does (Under the Hood)

TableGen is a **record instantiation system**:

1. **Parsing**: Read `.td` file, understand the syntax
2. **Instantiation**: Create records from `def` declarations
3. **Inheritance**: Resolve class hierarchies (`DenseOp : NN_Op : Op`)
4. **Code generation**: Run backend (e.g., `--gen-op-decls`) to emit C++

The key insight: **Declarative input → Imperative C++ output**.

---

## Why We're NOT Using CMake's `add_mlir_dialect()`

CMake has a macro that automates the TableGen call:

```cmake
add_mlir_dialect(NN NNOps)  # Magic: finds .td, runs TableGen, compiles
```

We're **not using this** because we want you to see the TableGen step explicitly. The Makefile makes it clear:

```makefile
$(GEN_DIR)/NNOps.h.inc: include/NN/NNOps.td
	mlir-tblgen --gen-op-decls $(TBLGEN_FLAGS) $< -o $@
```

This way, you understand what's happening at each step.

---

## Summary

| Concept | What it is | Why we use it |
|---------|-----------|---------------|
| **TableGen** | Code generator for compiler specs | MLIR dialects are formulaic; automate the boilerplate |
| **`.td` file** | Declarative dialect spec | You write what the op is, not how to parse/print it |
| **`mlir-tblgen`** | The tool | Runs on `.td`, emits C++ |
| **`.inc` files** | Generated C++ | Included in your hand-written `.cpp` files |

---

## Key Takeaway

TableGen is **infrastructure automation for compiler dialects**. You specify the structure and semantics; TableGen generates the parsing, printing, and verification logic. This is the standard MLIR approach and saves hundreds of lines of boilerplate code.

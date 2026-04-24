# Strix (Traditional Compiler) → MLIR: Analogies & Design Feedback

This document captures the mapping between the Strix compiler (CSE 450 Project 3) and the NN-MLIR compiler, plus feedback on the COMPILER_DESIGN_SPEC.

---

## Part 1: The Strix Pipeline (What You Already Know)

The Strix compiler follows a classic **4-stage architecture**:

```
Strix Source (.strix)
    → [Lexer/DFA] → Token stream
    → [Parser (precedence climbing)] → AST (tree of ASTNode objects)
    → [ToWAT() tree-walk] → WebAssembly Text (.wat)
```

### Key Components

| Component | File | Role |
|---|---|---|
| **Lexer** | `lexer.hpp` | DFA-based tokenizer. Produces tokens like `ID`, `LIT_NUM`, `FUNCTION`, `RETURN`, operators |
| **Parser** | `Project3.cpp` | Recursive descent + precedence climbing (`Parse_Expression`). Builds an AST. |
| **AST** | `AST.hpp` | Tree of `ASTNode` subclasses: `ASTNode_NumLit`, `ASTNode_Var`, `ASTNode_Operator2`, `ASTNode_Return`, etc. |
| **Symbol Table** | `SymbolTable.hpp` | Tracks variables by unique ID, mutable values, scopes, function metadata |
| **Code Gen** | `ASTNode::ToWAT()` | Each AST node emits its own WebAssembly Text by walking its children recursively |

### Example: End-to-End

**Source** (`test-06.strix`):
```
function BinaryMath(double val1, double val2) : double {
  return 100.0 - val1 / val2;
}
```

**AST** (tree structure):
```
ASTNode_Return
  └─ ASTNode_Operator2 (-)
       ├─ ASTNode_NumLit (100.0)
       └─ ASTNode_Operator2 (/)
            ├─ ASTNode_Var (val1)
            └─ ASTNode_Var (val2)
```

**Output** (WAT — stack-based):
```wat
(module
  (func $Fun0 (param $var0 f64) (param $var1 f64) (result f64)
    (f64.const 100)
    (local.get $var0)
    (local.get $var1)
    (f64.div)
    (f64.sub)
    (return)
  )
  (export "BinaryMath" (func $Fun0))
)
```

### Architectural Observations

1. **Representation and transformation are fused.** `ASTNode_Operator2` both *stores* the operator data and *emits* the WAT code via `ToWAT()`. Adding a new target (e.g., LLVM IR) would require adding a new method to every `ASTNode` subclass.

2. **No optimization passes.** `100.0 - val1 / val2` with known constants still emits all instructions. There's nowhere to plug in constant folding, dead code elimination, or common subexpression elimination.

3. **Data flow is implicit via the WAT stack.** `ASTNode_Operator2::ToWAT()` emits left child, then right child, then the operator — trusting that WebAssembly's stack machine will wire them together correctly. There are no named values.

4. **Variables are mutable.** The `SymbolTable` allows `SetID(var_id, value)` to mutate a variable in-place. This works for interpretation but makes static analysis difficult.

---

## Part 2: MLIR Equivalents (Stage-by-Stage)

### The Pipeline Comparison

```
STRIX:  Source → Lexer → Parser → AST → ToWAT() → WAT
                                         ↑
                                    ONE BIG JUMP

MLIR:   Source → Lexer → Parser → AST → IR Builder → nn dialect → linalg → memref → scf → llvm → machine code
                                         ↑                                                  ↑
                                    BUILDS SSA IR                              PROGRESSIVE LOWERING
```

### Component Mapping

| Strix Compiler Concept | MLIR NN-Compiler Equivalent | Key Difference |
|---|---|---|
| `emplex::Lexer` (DFA tokenizer) | `.nn` file lexer | Same concept, simpler tokens (`network`, `dense`, `relu`, numbers) |
| `Parse_Function()` → `Parse_Block()` → `Parse_Statement()` | `Parser::parse()` building `NetworkAST` | `.nn` grammar is simpler — no expressions, just layer declarations |
| `ASTNode` class hierarchy (`ASTNode_Var`, `ASTNode_Operator2`, etc.) | `NetworkAST` with layer nodes | Strix AST is a **tree** (expressions nest); NN AST is a **flat list** (layers are sequential) |
| `SymbolTable` (tracks vars by ID, mutable) | **SSA values** (`mlir::Value`) — each op returns a `%result` | No mutation. `%0` is always `%0`. No `SetID()` equivalent. |
| `op_map` (operator precedence for parsing) | Not needed — `.nn` has no expressions | NN language has no operator precedence |
| `ASTNode::ToWAT()` (tree-walk code gen) | `builder.create<nn::DenseOp>(...)` in IR builder | Strix walks the tree emitting stack ops. MLIR builds flat SSA ops. |
| `std::cout << "(f64.add)"` (string emission) | `builder.create<arith::AddFOp>(...)` (typed API) | Strix prints strings. MLIR builds typed, verified IR objects. |
| `(module ...)` wrapper in `ToWAT()` | `mlir::ModuleOp` + `func::FuncOp` | Same concept — a container for functions |
| No optimization passes | `--canonicalize`, CSE, DCE, constant folding | The entire reason MLIR exists |
| Single target: WAT | Lowering pipeline: `nn` → `linalg` → `memref` → `llvm` | Strix does one jump. MLIR does progressive lowering. |
| `ASTNode_Operator2::ToWAT()` knows the target | Lowering patterns are **separate** from op definitions | `DenseOp` in `NNOps.td` says nothing about lowering — that's a separate pass file |

---

## Part 3: The Three Deepest Conceptual Shifts

### Shift 1: Tree → SSA (No More Implicit Data Flow)

**Strix** — data flow is implicit via the stack:
```
ASTNode_Return
  └─ ASTNode_Operator2 (-) ← "emit left child, emit right child, emit f64.sub"
       ├─ ASTNode_NumLit (100.0)
       └─ ASTNode_Operator2 (/)
            ├─ ASTNode_Var (val1)
            └─ ASTNode_Var (val2)
```
Each node calls `child->ToWAT()` recursively, trusting the WAT stack to wire values together.

**MLIR** — data flow is explicit via named SSA values:
```mlir
func.func @BinaryMath(%val1: f64, %val2: f64) -> f64 {
  %c100 = arith.constant 100.0 : f64
  %div  = arith.divf %val1, %val2 : f64
  %sub  = arith.subf %c100, %div : f64
  return %sub : f64
}
```
No tree. No stack. Every value has a name (`%div`), and consumers reference it explicitly.

### Shift 2: Representation ≠ Transformation

**Strix** — `ASTNode_Operator2` both *is* the representation and *does* the code generation:
```cpp
class ASTNode_Operator2 : public ASTNode {
  Token token;  // ← representation (what operator is this?)
  void ToWAT(...) override {  // ← transformation (how to emit WAT?)
    children[0]->ToWAT(symbols, prefix, false);
    children[1]->ToWAT(symbols, prefix, false);
    if (token.lexeme == "+") AddCode(prefix, "(f64.add)");
    // ... target-specific code baked into the node
  }
};
```

**MLIR** — `DenseOp` only defines what the op looks like. Lowering is separate:
```
NNOps.td         → defines nn.dense (inputs, outputs, types)
NNToLinalg.cpp   → defines HOW to lower nn.dense → linalg.matmul + bias add
```
You can add new lowering targets without touching the op definition at all.

### Shift 3: One Jump → Progressive Lowering

**Strix**: AST → WAT (one monolithic step)
- If you want LLVM IR output, you'd add `ToLLVM()` to every ASTNode subclass
- If you want optimization, there's nowhere to put it

**MLIR**: `nn` → `linalg` → `memref` → `scf` → `llvm` (multiple composable steps)
- Each level can be optimized independently
- At `nn` level: fuse consecutive dense layers
- At `linalg` level: tile loops for cache
- At `scf` level: vectorize inner loops
- At `llvm` level: register allocation
- You can inspect the IR at any level for debugging

---

## Part 4: Concrete MLIR Output (What Your Compiler Will Produce)

### Input `.nn` file:
```
network deep_mlp
input 784
dense 512 relu
dense 256 relu
dense 10
```

### After Phase 2 (IR Generation — `nn` dialect):
```mlir
module {
  func.func @forward(%input: tensor<1x784xf32>,
                     %w0: tensor<784x512xf32>, %b0: tensor<512xf32>,
                     %w1: tensor<512x256xf32>, %b1: tensor<256xf32>,
                     %w2: tensor<256x10xf32>,  %b2: tensor<10xf32>)
                     -> tensor<1x10xf32> {
    %0 = nn.dense %input, %w0, %b0 : tensor<1x784xf32>, tensor<784x512xf32>, tensor<512xf32> -> tensor<1x512xf32>
    %1 = nn.relu %0 : tensor<1x512xf32> -> tensor<1x512xf32>
    %2 = nn.dense %1, %w1, %b1 : tensor<1x512xf32>, tensor<512x256xf32>, tensor<256xf32> -> tensor<1x256xf32>
    %3 = nn.relu %2 : tensor<1x256xf32> -> tensor<1x256xf32>
    %4 = nn.dense %3, %w2, %b2 : tensor<1x256xf32>, tensor<256x10xf32>, tensor<10xf32> -> tensor<1x10xf32>
    return %4 : tensor<1x10xf32>
  }
}
```

### After Phase 3 (Lowering — `nn` ops eliminated):
```mlir
module {
  func.func @forward(%input: tensor<1x784xf32>, ...) -> tensor<1x10xf32> {
    // nn.dense lowered to linalg.matmul + bias add
    %init0 = tensor.empty() : tensor<1x512xf32>
    %zero0 = linalg.fill ins(%cst_zero) outs(%init0) -> tensor<1x512xf32>
    %matmul0 = linalg.matmul ins(%input, %w0) outs(%zero0) -> tensor<1x512xf32>
    %biased0 = linalg.generic {/* broadcast add %b0 */} -> tensor<1x512xf32>

    // nn.relu lowered to linalg.generic with max(0, x)
    %relu0 = linalg.generic {/* max(0, %biased0) */} -> tensor<1x512xf32>

    // ... repeat for remaining layers
    return %result : tensor<1x10xf32>
  }
}
```

---

## Part 5: Feedback on COMPILER_DESIGN_SPEC

### Strengths
- Well-structured, clear rationale for every design choice
- Correct phasing order (dialect → parser → lowering → polish)
- Custom `nn` dialect is the right choice (mirrors IREE, torch-mlir)
- Phase 1 is already implemented (`NNOps.td`, `NNDialect.cpp`, TableGen build)

### Issues and Gaps

**1. Weights/bias strategy is unspecified.**
`nn.dense` takes `%input`, `%weights`, `%bias` — but `.nn` files only say `dense 512 relu`. The spec needs to decide: are weights function arguments? Embedded constants? External references? This affects the entire IR structure.

**2. AST → MLIR IR translation is under-described.**
The spec says "walk AST, call `builder.create<nn::DenseOp>(...)`" but doesn't explain how the sequential layer chain connects. In Strix, `ToWAT()` tree-walks and the stack handles data flow. In MLIR, you must explicitly thread `mlir::Value` results from one op into the next op's inputs.

**3. `func.func` wrapping is unspecified.**
Strix wraps everything in `(module ... (func ...))`. The spec should describe what `func.func @forward(...)` looks like — its signature, how many functions one `.nn` file produces, and what the return type is.

**4. Phase 3 timeline is aggressive.**
Lowering `nn.dense` → `linalg.matmul + linalg.generic` (with bias broadcasting) is the hardest part. Budget 2-3 days, not 1.

**5. Makefile will struggle at Phase 3+.**
The current `-lMLIR -lLLVM` works for Phase 1, but Phase 3 needs `MLIRLinalgDialect`, `MLIRArithDialect`, `MLIRTensorDialect`, etc. Plan for CMake or `llvm-config` when the link list grows.

**6. `nn-opt.cpp` and `nn-compiler.cpp` are empty.**
Phase 1's testability (`nn-opt test.mlir` round-trips) can't be verified yet. `nn-opt.cpp` is ~20 lines — implement it first.

### Recommendations
1. Add a "weights strategy" section — decide if weights are function args or constants
2. Add `func.func` wrapping to the Architecture section
3. Budget Phase 3 as 2-3 days
4. Implement `nn-opt.cpp` to validate Phase 1 end-to-end
5. Plan for CMake migration after Phase 2

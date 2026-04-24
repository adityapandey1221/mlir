# NN-MLIR Compiler: First-Principles Roadmap

Every concept is explained from first principles, with analogies to the Strix compiler (CSE 450 Project 3) you already understand.

---

## How to Read This Document

Each section follows this pattern:
1. **What you already know** (Strix analogy)
2. **What's different in MLIR** (the conceptual shift)
3. **Why it's different** (the first-principles reason)
4. **What you'll write** (concrete code with line-by-line understanding)
5. **How to verify** (testable checkpoint)

---

## Day 1: Dialect Validation

### Goal: Prove that `nn.dense` and `nn.relu` exist as first-class MLIR operations

### 1.1 — What is a Dialect? (First Principles)

**What you already know from Strix:**

In Strix, you have a fixed set of AST node types hardcoded in C++:
```cpp
class ASTNode_NumLit : public ASTNode { ... };
class ASTNode_Var : public ASTNode { ... };
class ASTNode_Operator2 : public ASTNode { ... };
class ASTNode_Return : public ASTNode { ... };
```

Each node type knows its own semantics — what children it has, what it means, how to emit WAT. If you want to add a new node type (say `ASTNode_FunctionCall`), you write a new C++ class, add parsing logic, and add `ToWAT()`.

**What's different in MLIR:**

MLIR doesn't hardcode operation types. Instead, it has a plugin system called **dialects**. A dialect is a namespace of operations. For example:
- `arith` dialect → `arith.addf`, `arith.mulf`, `arith.constant`
- `linalg` dialect → `linalg.matmul`, `linalg.generic`
- `func` dialect → `func.func`, `func.return`
- `nn` dialect (yours!) → `nn.dense`, `nn.relu`

Each dialect is defined in **TableGen** (a declarative language), and `mlir-tblgen` generates the C++ classes automatically.

**Why it's different:**

In Strix, adding a new AST node means writing C++ by hand — the constructor, the children, `ToWAT()`, `Print()`, etc. This is fine for a small compiler, but doesn't scale. If you had 200 operation types (like LLVM does), hand-writing all that boilerplate would be unmaintainable.

MLIR's approach: you **declare** what an operation looks like (inputs, outputs, types, syntax) in TableGen, and the tool **generates** the C++ for you. You only hand-write the parts that are unique (like custom verification or lowering logic).

**Strix analogy:**
```
ASTNode_Operator2 (hand-written C++ class)  ←→  DenseOp (generated from NNOps.td)
```

### 1.2 — Understanding NNOps.td (What You Already Have)

Your `NNOps.td` is the **declaration** of your dialect. Let's map every line to Strix concepts:

```tablegen
def NN_Dialect : Dialect {
    let name = "nn";                    // Namespace prefix: "nn.dense", "nn.relu"
    let cppNamespace = "::mlir::nn";    // C++ namespace for generated code
}
```

**Strix analogy:** This is like declaring `namespace strix { }` — it just groups your operations together. In Strix, you didn't need this because there was only one "dialect" (your AST nodes).

```tablegen
class NN_Op<string mnemonic, list<Trait> traits = []>
    : Op<NN_Dialect, mnemonic, traits>;
```

**Strix analogy:** This is your `class ASTNode` base class. All Strix nodes inherit from `ASTNode`; all NN ops inherit from `NN_Op`.

```tablegen
def DenseOp : NN_Op<"dense"> {
    let arguments = (ins
        AnyRankedTensor:$input,     // Like ASTNode_Operator2's children[0]
        AnyRankedTensor:$weights,   // Like children[1]
        AnyRankedTensor:$bias       // Like a third child
    );
    let results = (outs
        AnyRankedTensor:$output     // Like the "return value" of ToWAT()
    );
    let assemblyFormat = [...];     // How to print/parse this op as text
}
```

**Strix analogy:** This is like defining `ASTNode_Operator2` — it has two children (left, right) and produces a value. But here:
- Children are called **arguments** (inputs to the operation)
- The "return value" is an explicit **result** (not implicit via the stack)
- The text format is declared, not hand-written in a `Print()` method

**Key difference from Strix:** In Strix, `ASTNode_Operator2` stores a `Token` to know which operator it is (`+`, `-`, `*`, etc.) — one class handles all binary ops. In MLIR, `DenseOp` and `ReluOp` are **separate classes**, each with their own specific inputs/outputs. This is more explicit but also more type-safe.

### 1.3 — Understanding the Glue Code

**`NNDialect.cpp`** — registers the dialect with MLIR:

```cpp
void NNDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "NN/NNOps.cpp.inc"  // TableGen generated: DenseOp, ReluOp
    >();
}
```

**Strix analogy:** In Strix, your AST nodes are "registered" implicitly — you just `#include "AST.hpp"` and use them. In MLIR, you must explicitly register your dialect so the framework knows about your operations. Think of it like `symbols.AddFunction(fun_name)` — you're telling the system "these operations exist."

**`NNOps.cpp`** — includes the generated implementations:

```cpp
#define GET_OP_CLASSES
#include "NN/NNOps.cpp.inc"  // TableGen generated: constructors, verifiers, printers, parsers
```

**Strix analogy:** This is all the boilerplate that TableGen writes for you. In Strix, you hand-wrote constructors like `ASTNode_Operator2(Token tok, ptr_t left, ptr_t right)`. In MLIR, TableGen generates equivalent constructors from your `.td` declaration.

### 1.4 — What You Need to Write: `nn-opt.cpp`

`nn-opt` is a tool that reads MLIR text, parses it, verifies it, and prints it back. It proves your dialect works.

**Strix analogy:** Imagine if you had a tool that reads a Strix `.strix` file, builds the AST, calls `Print()` on every node, and writes it back. If the round-trip produces the same output, you know your parser and AST are correct. `nn-opt` does exactly this for MLIR.

**What `nn-opt.cpp` does:**
1. Registers your `NNDialect` with MLIR's global registry
2. Calls `MlirOptMain()` — MLIR's built-in "read, verify, transform, print" driver
3. That's it. ~20 lines.

**Why so short?** Because MLIR already has a complete text parser and printer. Your `assemblyFormat` in `NNOps.td` told MLIR how to parse/print `nn.dense` and `nn.relu`. `nn-opt` just plugs your dialect into that existing infrastructure.

### 1.5 — Write a Test File: `test/test_dialect.mlir`

This is a hand-written MLIR file that uses your ops. It's what `nn-opt` will try to parse.

```mlir
// test/test_dialect.mlir
module {
  func.func @forward(
    %input: tensor<1x784xf32>,
    %w0: tensor<784x512xf32>, %b0: tensor<512xf32>
  ) -> tensor<1x512xf32> {
    %0 = nn.dense %input, %w0, %b0 : tensor<1x784xf32>, tensor<784x512xf32>, tensor<512xf32> -> tensor<1x512xf32>
    %1 = nn.relu %0 : tensor<1x512xf32> -> tensor<1x512xf32>
    return %1 : tensor<1x512xf32>
  }
}
```

**Strix analogy:** This is like writing a `.strix` test file by hand:
```
function forward(double x) : double {
  return x + 1.0;
}
```
You write it, feed it to your compiler, and check if it parses correctly.

**What each line means:**
- `module { }` — like `(module ...)` in your WAT output. A container for everything.
- `func.func @forward(...)` — like `(func $Fun0 (param ...) (result ...))`. Declares a function.
- `%input: tensor<1x784xf32>` — like `(param $var0 f64)`. A function parameter. `%input` is an SSA name (like your `$var0`), `tensor<1x784xf32>` is its type (1 row, 784 columns, 32-bit floats).
- `%0 = nn.dense ...` — like calling an operation that produces a result. `%0` is the SSA name for the output. In Strix, this value would be implicit on the WAT stack.
- `return %1` — like `(return)` in WAT, but explicitly names which value to return.

### 1.6 — Verification Checkpoint

```bash
make tablegen          # Generate .inc files from NNOps.td
make nn-opt            # Compile nn-opt
./nn-opt test/test_dialect.mlir    # Should print the same MLIR back
```

If the output matches the input (modulo whitespace), Phase 1 is complete.

---

## Day 2: Parser + IR Builder

### Goal: Parse `.nn` files and produce valid MLIR using the `OpBuilder` API

### 2.1 — The Conceptual Shift: Tree-Walking vs. SSA Building

**What you already know from Strix:**

In Strix, your parser builds a **tree**, then code gen **walks** that tree:

```
Parse_Expression() builds:     ToWAT() walks it:

  Operator2(-)                 children[0]->ToWAT()  → (f64.const 100)
   ├─ NumLit(100)              children[1]->ToWAT()  → (local.get $var0)
   └─ Var(val1)                AddCode("(f64.sub)")  → (f64.sub)
```

The tree structure defines data flow. Left child is evaluated first, result goes on the stack, then right child, then the operator consumes both from the stack.

**What's different in MLIR:**

There's no tree to walk. Your IR builder creates a **flat sequence of operations**, each producing a named value:

```
builder.create<nn::DenseOp>(...)   →  %0 = nn.dense %input, %w0, %b0 : ...
builder.create<nn::ReluOp>(...)    →  %1 = nn.relu %0 : ...
builder.create<nn::DenseOp>(...)   →  %2 = nn.dense %1, %w1, %b1 : ...
```

Data flow is explicit: `%1` uses `%0` because you passed the `mlir::Value` returned by the first `create<>()` call into the second one.

**Why it's different (first principles):**

A stack machine (WAT) handles data flow implicitly — values go on and come off the stack in order. This is simple but makes optimization hard. Which value is which? Are two operations independent? Can we reorder them?

SSA (Static Single Assignment) makes data flow **explicit and analyzable**:
- Every value has exactly one definition (`%0 = ...`)
- Every use names its source (`nn.relu %0`)
- You can instantly see which ops depend on which
- This is the foundation for every optimization pass

**Strix analogy for threading values:**

In Strix, your `Parse_Expression()` returns an `ast_node_t` (a `unique_ptr<ASTNode>`) and passes it to the parent node via `AddChild()`. In MLIR, `builder.create<>()` returns an `mlir::Value` and you pass it to the next `builder.create<>()` call. Same concept — passing results forward — but flat instead of nested.

```
Strix:
  ast_node_t left = Parse_Primary();           // produces a node
  ast_node_t right = Parse_Primary();          // produces another node
  MakeOpNode(op_token, left, right);           // consumes both

MLIR:
  Value dense_out = builder.create<DenseOp>(..., input, w0, b0);  // produces %0
  Value relu_out = builder.create<ReluOp>(..., dense_out);         // consumes %0, produces %1
```

### 2.2 — The `.nn` Parser

Your `.nn` format is **much simpler** than Strix's language. No expressions, no operator precedence, no nested scopes. It's line-oriented:

```
network deep_mlp       → NetworkDecl { name: "deep_mlp" }
input 784              → InputLayer { size: 784 }
dense 512 relu         → DenseLayer { size: 512, activation: "relu" }
dense 256 relu         → DenseLayer { size: 256, activation: "relu" }
dense 10               → DenseLayer { size: 10, activation: none }
```

**Strix analogy:** Your Strix `Parse_Function()` handles `function`, `(`, params, `)`, `:`, return type, `{`, body, `}` — 9 different token types in one function. Your `.nn` parser handles `network`, `input`, `dense`, numbers, and activation names — simpler grammar, simpler parser.

**Key difference:** No `Parse_Expression()` needed. No precedence climbing. No recursive descent for nested expressions. Your `.nn` AST is a **flat list of layers**, not a tree.

### 2.3 — The IR Builder

This is the new skill. Instead of `ToWAT()` printing strings, you call MLIR's `OpBuilder` API.

**Strix's approach** (string emission):
```cpp
// In ASTNode_NumLit::ToWAT():
AddCode(prefix, "(f64.const ", value, ")");

// In ASTNode_Operator2::ToWAT():
children[0]->ToWAT(symbols, prefix, false);  // emit left
children[1]->ToWAT(symbols, prefix, false);  // emit right
AddCode(prefix, "(f64.add)");                 // emit operator
```

**MLIR's approach** (typed API):
```cpp
// Create a constant (like ASTNode_NumLit):
Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0));

// Create a dense layer (like ASTNode_Operator2 but with 3 inputs):
Value dense_out = builder.create<nn::DenseOp>(loc, outputType, input, weights, bias);

// Create a relu (like a unary ASTNode_Operator1):
Value relu_out = builder.create<nn::ReluOp>(loc, outputType, dense_out);
```

**What each argument means:**
- `loc` — source location for error messages. Like `token.line_id` in Strix. MLIR tracks this so error messages say "line 5: type mismatch."
- `outputType` — the result tensor type (e.g., `tensor<1x512xf32>`). In Strix, everything was `f64`. In MLIR, types are explicit and verified.
- `input`, `weights`, `bias` — `mlir::Value` references to the inputs. Like passing `children[0]`, `children[1]` in Strix, but by value reference, not tree position.

### 2.4 — Building the Function Wrapper

**Strix's approach:**
```cpp
// In Strix::ToWAT():
std::cout << "(module\n";
std::cout << "  (func $Fun0 ";
for (size_t param_id : param_ids) {
    std::cout << "(param $var" << param_id << " f64) ";
}
std::cout << "(result f64)\n";
// ... body ...
std::cout << "  )\n";
std::cout << "  (export \"BinaryMath\" (func $Fun0))\n";
std::cout << ")\n";
```

**MLIR's approach:**
```cpp
// Create the module (like "(module ...)")
auto module = ModuleOp::create(loc);

// Build function type: (tensor, tensor, tensor, ...) -> tensor
auto funcType = builder.getFunctionType(inputTypes, outputTypes);

// Create function (like "(func $Fun0 (param...) (result...))")
auto func = builder.create<func::FuncOp>(loc, "forward", funcType);

// Create the function body block (like the "{...}" in Strix)
Block *entryBlock = func.addEntryBlock();
builder.setInsertionPointToStart(entryBlock);

// Now builder.create<>() calls will add ops inside this function
// ... create nn.dense, nn.relu, etc. ...

// Return the final value (like "(return)")
builder.create<func::ReturnOp>(loc, finalValue);
```

**Strix analogy:** The structure is identical:
1. Create outer container (`module` / `(module)`)
2. Create function with signature (`func::FuncOp` / `(func $Fun0 ...)`)
3. Fill in the body (create ops / emit WAT instructions)
4. Return the result (`func::ReturnOp` / `(return)`)

The difference is you're calling typed C++ APIs instead of printing strings.

### 2.5 — How Layer Dimensions Are Computed

Your parser needs to figure out tensor shapes from the `.nn` file:

```
input 784          → input shape: tensor<1x784xf32>
dense 512 relu     → weights: tensor<784x512xf32>, bias: tensor<512xf32>, output: tensor<1x512xf32>
dense 256 relu     → weights: tensor<512x256xf32>, bias: tensor<256xf32>, output: tensor<1x256xf32>
dense 10           → weights: tensor<256x10xf32>,  bias: tensor<10xf32>,  output: tensor<1x10xf32>
```

Pattern: each layer's input dimension = previous layer's output dimension. The `1` is the batch dimension.

**Strix analogy:** In Strix, all variables are `f64` — no dimension tracking needed. Here, you're threading dimension information through the layer list, similar to how Strix threads scope information through `IncScope()`/`DecScope()`.

### 2.6 — Verification Checkpoint

```bash
make nn-compiler
./nn-compiler test/simple_mlp.nn              # Should output valid MLIR
./nn-compiler test/simple_mlp.nn | ./nn-opt   # Round-trip: nn-opt can parse it
```

If `nn-opt` can read the output of `nn-compiler`, the IR is structurally valid.

---

## Days 3–4: Lowering Pass

### Goal: Transform `nn.dense` → `linalg.matmul` + bias add, `nn.relu` → `linalg.generic`

### 3.1 — What is Lowering? (First Principles)

**What you already know from Strix:**

In Strix, `ASTNode_Operator2::ToWAT()` is a form of "lowering" — it translates a high-level concept (`+` operator on two values) into a low-level instruction (`(f64.add)`). But it's a **one-step, all-or-nothing** translation baked into each AST node.

**What's different in MLIR:**

Lowering is a **separate pass** that runs after IR construction. It's not part of the op definition — it's a standalone transformation:

```
BEFORE lowering:   %0 = nn.dense %input, %w0, %b0 : ...
                   %1 = nn.relu %0 : ...

AFTER lowering:    %init = tensor.empty() : tensor<1x512xf32>
                   %zero = linalg.fill ins(%cst) outs(%init) : ...
                   %matmul = linalg.matmul ins(%input, %w0) outs(%zero) : ...
                   %biased = linalg.generic {add bias} ins(%matmul, %b0) outs(...) : ...
                   %relu = linalg.generic {max(0, x)} ins(%biased) outs(...) : ...
```

One `nn.dense` op **expands** into 3-4 lower-level ops. One `nn.relu` op becomes 1 `linalg.generic` op.

**Why it's separate (first principles):**

In Strix, if you wanted to add an optimization (say, constant folding `100.0 - 50.0` into `50.0`), where would you put it? There's no good place — `ToWAT()` is the only transformation, and it's baked into each node.

In MLIR, you can insert passes **between** levels:
```
nn dialect → [nn-level optimizations] → linalg dialect → [linalg-level optimizations] → ...
```

Because lowering is separate from op definition, you can:
- Optimize at the `nn` level (fuse consecutive layers) before lowering
- Optimize at the `linalg` level (tile for cache) after lowering
- Add new lowering targets (e.g., GPU) without changing `nn.dense`'s definition

### 3.2 — The Conversion Pattern Framework

**Strix analogy:**

In Strix, `ASTNode_Operator2::ToWAT()` is a big `if/else` chain:
```cpp
if (token.lexeme == "+") AddCode(prefix, "(f64.add)");
else if (token.lexeme == "-") AddCode(prefix, "(f64.sub)");
else if (token.lexeme == "*") AddCode(prefix, "(f64.mul)");
else if (token.lexeme == "/") AddCode(prefix, "(f64.div)");
else if (token.lexeme == "**") AddCode(prefix, "(call $pow)");
```

This says: "when I see `+`, emit `f64.add`; when I see `-`, emit `f64.sub`; etc."

MLIR's lowering patterns say the same thing, but more formally:

```cpp
struct DenseLowering : public OpConversionPattern<nn::DenseOp> {
    // "When I see nn.dense, replace it with linalg.matmul + bias add"
    LogicalResult matchAndRewrite(nn::DenseOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // 1. Get the input values (like accessing children[0], children[1] in Strix)
        Value input = adaptor.getInput();
        Value weights = adaptor.getWeights();
        Value bias = adaptor.getBias();

        // 2. Create the replacement ops (like AddCode("(f64.add)") but creating IR objects)
        Value init = rewriter.create<tensor::EmptyOp>(...);
        Value matmul = rewriter.create<linalg::MatmulOp>(..., input, weights, init);
        Value biased = rewriter.create<linalg::GenericOp>(...);  // add bias

        // 3. Replace the original op with the new result
        rewriter.replaceOp(op, biased);
        return success();
    }
};
```

**The pattern:**
1. **Match** — find an `nn.dense` op in the IR (like `if (token.lexeme == "+")`)
2. **Get inputs** — extract the operands (like `children[0]`, `children[1]`)
3. **Create replacements** — build new lower-level ops (like `AddCode(...)`)
4. **Replace** — swap the old op for the new ops (no Strix equivalent — Strix doesn't rewrite IR)

### 3.3 — What `nn.dense` Lowers To

`nn.dense` computes `output = input @ weights + bias`. This becomes:

```
Step 1: Create an empty output tensor
        %init = tensor.empty() : tensor<1x512xf32>

Step 2: Fill it with zeros (linalg.matmul accumulates into the output)
        %zero = linalg.fill ins(%cst_zero) outs(%init) : tensor<1x512xf32>

Step 3: Matrix multiply (input @ weights, accumulated into %zero)
        %matmul = linalg.matmul ins(%input, %weights) outs(%zero) : tensor<1x512xf32>

Step 4: Add bias (broadcast 1D bias across rows)
        %biased = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // matmul result
                             affine_map<(d0, d1) -> (d1)>,        // bias (broadcast)
                             affine_map<(d0, d1) -> (d0, d1)>],   // output
            iterator_types = ["parallel", "parallel"]
        } ins(%matmul, %bias) outs(%init2) {
            ^bb0(%a: f32, %b: f32, %c: f32):
                %sum = arith.addf %a, %b : f32
                linalg.yield %sum : f32
        } -> tensor<1x512xf32>
```

**Strix analogy for `linalg.generic`:** Think of it as a **generalized loop nest**. The `affine_map`s describe which indices to use for each operand (like array indexing), and the body (`^bb0`) describes the per-element computation. It's like a declarative version of:
```python
for d0 in range(batch):
    for d1 in range(out_features):
        output[d0][d1] = matmul_result[d0][d1] + bias[d1]
```

The `affine_map<(d0, d1) -> (d1)>` for bias is the **broadcasting** — it says "for each (d0, d1) pair, read bias at index [d1] only." This is how a 1D bias gets added to every row of a 2D matrix.

### 3.4 — What `nn.relu` Lowers To

`nn.relu` computes `output = max(0, input)`. This becomes:

```
%cst_zero = arith.constant 0.0 : f32
%relu = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // input
                     affine_map<(d0, d1) -> (d0, d1)>],   // output
    iterator_types = ["parallel", "parallel"]
} ins(%input) outs(%init) {
    ^bb0(%a: f32, %b: f32):
        %result = arith.maximumf %a, %cst_zero : f32
        linalg.yield %result : f32
} -> tensor<1x512xf32>
```

**Strix analogy:** This is simpler than `nn.dense`. It's like how `ASTNode_Operator1` (unary) is simpler than `ASTNode_Operator2` (binary) — one input, one output, element-wise operation.

### 3.5 — The ConversionTarget: "What's Legal?"

After lowering, you want to verify that **all `nn.*` ops are gone**. MLIR uses a `ConversionTarget` for this:

```cpp
ConversionTarget target(getContext());
target.addIllegalDialect<nn::NNDialect>();       // "nn.* ops are NOT allowed after this pass"
target.addLegalDialect<linalg::LinalgDialect>(); // "linalg.* ops ARE allowed"
target.addLegalDialect<arith::ArithDialect>();
target.addLegalDialect<tensor::TensorDialect>();
target.addLegalDialect<func::FuncDialect>();
```

**Strix analogy:** There's no equivalent in Strix — you never transform the IR, so there's nothing to validate. But imagine if Strix had a pass that converted `ASTNode_Operator2` into `ASTNode_WAT_Add` nodes, and you wanted to assert "no `ASTNode_Operator2` should remain after this pass." That's what `ConversionTarget` does.

### 3.6 — Verification Checkpoint

```bash
./nn-compiler test/simple_mlp.nn --lower
# Output should contain:
#   linalg.matmul, linalg.generic, arith.constant, tensor.empty
# Output should NOT contain:
#   nn.dense, nn.relu
```

You can also pipe through `nn-opt` to verify structural validity:
```bash
./nn-compiler test/simple_mlp.nn --lower | ./nn-opt
```

---

## Day 5: End-to-End Pipeline + Polish

### Goal: Wire everything together, add `--dump-pipeline`, write tests

### 5.1 — The `--dump-pipeline` Flag

**Strix analogy:** Imagine if Strix had a `--print-ast` flag that showed the AST before code gen, and then showed the WAT output. That's `--dump-pipeline` — it shows the IR at each stage.

```bash
./nn-compiler input.nn --dump-pipeline
```

Output:
```
=== Stage 1: NN Dialect (high-level) ===
module {
  func.func @forward(%input: tensor<1x784xf32>, ...) -> tensor<1x10xf32> {
    %0 = nn.dense %input, %w0, %b0 : ...
    %1 = nn.relu %0 : ...
    ...
  }
}

=== Stage 2: Linalg/Arith (lowered) ===
module {
  func.func @forward(%input: tensor<1x784xf32>, ...) -> tensor<1x10xf32> {
    %init = tensor.empty() : ...
    %matmul = linalg.matmul ins(%input, %w0) outs(%init) : ...
    ...
  }
}
```

This is the "progressive lowering made visual" — the killer demo for interviews.

### 5.2 — Test Suite

Create test files that validate each phase:

```
test/
├── test_dialect.mlir        # Phase 1: hand-written MLIR, round-trip with nn-opt
├── simple_mlp.nn            # Phase 2: basic network (input → dense → output)
├── deep_mlp.nn              # Phase 2: multi-layer network with relu
├── single_layer.nn          # Phase 2: minimal case (one dense layer, no activation)
└── run_tests.sh             # Runs all tests, reports pass/fail
```

**Strix analogy:** Exactly like your `tests/` directory with `test-00.strix` through `test-35.strix` and `run_tests.sh`.

### 5.3 — Verification Checkpoint (Final)

```bash
./test/run_tests.sh
# Expected:
# test_dialect.mlir    ... PASS (nn-opt round-trip)
# simple_mlp.nn        ... PASS (nn-compiler produces valid MLIR)
# simple_mlp.nn --lower ... PASS (no nn.* ops remain)
# deep_mlp.nn          ... PASS
# deep_mlp.nn --lower  ... PASS
```

---

## Summary: The Full Pipeline Comparison

```
STRIX COMPILER (what you know):
═══════════════════════════════
.strix file
  → Lexer (DFA)              emplex::Lexer, hand-written DFA table
  → Parser                   Parse_Function(), Parse_Expression(), precedence climbing
  → AST                      ASTNode tree with children pointers
  → Code Gen                 ToWAT() on each node, prints strings
  → .wat file                Stack-based WebAssembly Text

NN-MLIR COMPILER (what you're building):
════════════════════════════════════════
.nn file
  → Lexer                    Simple line-oriented tokenizer
  → Parser                   Flat layer list (no precedence needed)
  → AST                      NetworkAST (flat list, not a tree)
  → IR Builder               OpBuilder API, creates SSA operations
  → nn dialect IR             nn.dense, nn.relu (inspectable, transformable)
  → Lowering Pass             ConversionPatterns replace nn.* with linalg.*
  → linalg/arith/tensor IR   Standard MLIR ops (inspectable, optimizable)
  → [Future: more passes]    Bufferization, tiling, LLVM lowering
  → [Future: machine code]   Via LLVM backend

KEY DIFFERENCES:
  Strix: one big jump (AST → target code), no optimization, representation = transformation
  MLIR:  many small jumps (dialect → dialect), optimization at each level, representation ≠ transformation
```

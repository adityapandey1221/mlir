# NN-MLIR Compiler: Design Specification & First-Principles Justification

## Executive Summary

This document explains **why** we're building a neural network compiler using MLIR, and **how** the architecture choices follow from fundamental compiler design principles.

**TL;DR**: We're building a compiler that transforms neural network specifications into optimized machine code through progressive lowering. MLIR is the right tool because it handles the complexity of multiple intermediate representations and their transformations.

---

## Part 1: The Problem We're Solving

### What is a Neural Network Compiler?

A neural network compiler takes a high-level description of a network (weights, layers, activations) and produces executable code optimized for a target machine.

**Input**: A neural network specification
```
network deep_mlp
input 784
dense 512 relu
dense 256 relu
dense 10
```

**Output**: Optimized machine code that:
- Computes the forward pass correctly
- Runs as fast as possible on the target CPU/GPU
- Uses memory efficiently

### Why Not Just Use PyTorch/TensorFlow?

PyTorch and TensorFlow are **execution frameworks**. They handle training, backprop, and dynamic computation graphs. Our compiler is **static** — we compile once, run many times. That's fundamentally different and requires different tools.

**Framework approach** (PyTorch):
```python
model = MyModel()
model(input)  # Dynamic: graph built at runtime
output = model(input)  # Calls C++/CUDA backend
```

**Compiler approach** (ours):
```
network.nn → [compile once] → machine code → [link] → executable
```

**Why compiler is better**:
- ✅ No runtime overhead parsing/building graphs
- ✅ All optimization decisions made upfront
- ✅ Code generator can make aggressive optimizations
- ✅ Predictable performance (no surprises at runtime)

---

## Part 2: Why MLIR?

### Alternative Compiler Frameworks

| Framework | Pros | Cons | Use case |
|-----------|------|------|----------|
| **LLVM IR** | Proven, mature, excellent codegen | No high-level abstractions | Compiling languages (C++, Rust) |
| **Polly** (LLVM) | Good loop optimization | Hard to extend, limited IR | Static affine programs |
| **TVM** | ML-focused, good for hardware | Lower-level, less modular | Hardware acceleration |
| **MLIR** | Progressive lowering, multiple IRs, extensible | Steeper learning curve | Our choice ✅ |

### What is MLIR?

**MLIR = Multi-Level Intermediate Representation**

Instead of one fixed IR (like LLVM IR), MLIR allows you to define **multiple IRs at different abstraction levels** and transformations between them.

```
Your custom "nn" dialect (high-level, semantic)
    ↓ [lowering pass]
Standard "linalg" dialect (mid-level, algorithmic)
    ↓ [lowering pass]
"tensor" → "memref" (mid-level, memory-oriented)
    ↓ [lowering pass]
"scf" + "arith" (low-level, loops + arithmetic)
    ↓ [lowering pass]
"llvm" dialect (very low-level, close to machine)
    ↓ [code generation]
Machine code
```

### Why MLIR is Perfect for Neural Networks

1. **Semantic IR** — `nn.dense` is semantically clear (matrix multiply + bias), not buried in 50 linalg operations

2. **Progressive Lowering** — Each level of abstraction can be optimized independently
   - At the `nn` level: fuse multiple dense layers
   - At the `linalg` level: tile loops for cache
   - At the `scf` level: vectorize inner loops
   - At the `llvm` level: register allocation

3. **Composability** — Passes are independent. You can swap them, reorder them, or skip them
   - Maybe you don't want vectorization for a specific target
   - Maybe you want to insert debugging operations at a specific level
   - Easy to experiment

4. **Extensibility** — Adding a new operation or dialect is straightforward
   - Define it in TableGen
   - Write lowering patterns
   - No changes to the rest of the system

5. **Rich Op Ecosystem** — MLIR already has standard dialects (linalg, arith, tensor, scf, llvm, omp, etc.)
   - Don't reinvent matrix multiplication
   - Reuse community-maintained ops

---

## Part 3: The Architecture

### The Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NN-MLIR Compiler                             │
└─────────────────────────────────────────────────────────────────────┘

FRONTEND (Your Code)
├─ Input: simple_mlp.nn (text file)
├─ Lexer: Tokenize input
├─ Parser: Build AST (NetworkAST)
└─ Type checking: Validate connectivity

IR GENERATION (Your Code + MLIR Library)
├─ IR Builder: Walk AST, create MLIR operations
├─ Custom Dialect: Use nn.dense, nn.relu ops
└─ Output: mlir::ModuleOp with nn dialect ops

OPTIMIZATION PASSES (MLIR Library)
├─ Level 1: nn dialect optimizations (Phase 4+ future)
├─ Level 2: Lowering (nn → linalg/arith)
├─ Level 3: Bufferization (tensor → memref)
├─ Level 4: Loop optimization (tiling, vectorization)
├─ Level 5: Parallelization (OpenMP, async)
└─ Level 6: LLVM lowering

BACKEND (LLVM)
├─ LLVM IR generation
├─ LLVM optimization passes
├─ Code generation (native assembly)
└─ Linking

OUTPUT
└─ Executable machine code
```

### Resume Impact Analysis

For compiler engineer roles, what matters most on a resume is **depth at the right layers**:

| Skill | Resume Weight | Where You Demonstrate It |
|---|---|---|
| Writing lowering/conversion patterns | **Very High** — this is 80% of real MLIR work | Phase 3 |
| Building IR with OpBuilder API | **High** — shows you understand compiler infrastructure | Phase 2 |
| Defining a custom MLIR dialect | **High** — shows you can extend MLIR | Phase 1 |
| Writing a frontend parser | **Medium** — every compiler engineer has done this | Phase 2 |
| Bufferization, tiling, vectorization | **Nice to have** — mention as "future work" | Phases 5-7 (later) |

**Key insight**: A working `DenseOp → linalg.matmul` lowering pattern is worth more on your resume than Phases 5-8 combined, because it proves you can do the core daily work of an MLIR engineer.

### What to Skip This Week

- **Bufferization, tiling, vectorization (Phases 5-7)** — mention in README as "future work"
- **LLVM code gen (Phase 8)** — say "pipeline designed to extend to LLVM lowering"
- **CMake migration** — stick with Makefile, add link flags manually as needed
- **Fancy `.nn` syntax** — keep it dead simple, no error recovery

### The Five Phases (Days 1–5)

#### Phase 1: Dialect Validation (Day 1)

**What**: Validate the `nn` dialect (already defined) by building `nn-opt` and round-tripping a test `.mlir` file.

**Status**: `NNOps.td`, `NNDialect.cpp`, `NNOps.cpp`, and the Makefile are written. `nn-opt.cpp` needs implementation (~20 lines).

**Why this first**:
- Nothing else works without a validated dialect
- `nn-opt` gives you a feedback loop for all future work
- Forces you to verify the TableGen output actually compiles and links

**How**:
1. Implement `nn-opt.cpp` — register `NNDialect`, call `MlirOptMain`
2. Write `test/test_dialect.mlir` — hand-written MLIR using `nn.dense` and `nn.relu`
3. Run `./nn-opt test/test_dialect.mlir` — if it parses and prints back, Phase 1 is done

**Testable**: `nn-opt test.mlir` round-trips (parse → print → identical output)

---

#### Phase 2: Parser + IR Builder (Day 2)

**What**: Parse `.nn` files and generate MLIR using the real `OpBuilder` API.

**Why this order**:
- Dialect exists and is validated (Phase 1)
- Now we populate it with ops from parsed input
- Forces you to understand `OpBuilder`, `ModuleOp`, `FuncOp`, SSA value threading

**How**:
1. Write lexer — tokenize `.nn` input (simple, line-oriented — no precedence climbing needed)
2. Write parser — build a simple `NetworkAST` (flat list of layers, not a tree)
3. Write IR builder — walk the AST, compute tensor dimensions, call `builder.create<nn::DenseOp>(...)`
4. Thread `mlir::Value` results: each layer's output becomes the next layer's input
5. Call `module->print(llvm::outs())` to output MLIR text

**Key insight**: We use MLIR's **official API** (`OpBuilder`), not string concatenation. The critical new skill vs. Strix is **explicit SSA value threading** — instead of relying on a stack, you pass `mlir::Value` from one op to the next.

**Testable**: `nn-compiler input.nn | nn-opt` round-trips. If nn-opt can read it, it's valid MLIR.

---

#### Phase 3: Lowering Pass (Days 3–4)

**What**: Convert `nn.dense` → `linalg.matmul + linalg.generic` (bias add) and `nn.relu` → `linalg.generic` with `arith.maximumf(0, x)`.

**Why this is two days**:
- This is the **hardest and most valuable** phase
- Lowering patterns are the core skill for MLIR compiler engineers
- Getting `linalg.generic` affine maps right for bias broadcasting takes iteration

**Key concepts**:
- **Conversion patterns** — match high-level ops, replace with low-level ops
- **Type converters** — transform types during lowering (we don't need this; tensors stay tensors)
- **Partial lowering** — `nn.*` ops disappear, replaced with `linalg.*` and `arith.*` ops
- **`ConversionTarget`** — tells the framework "all `nn.*` ops are illegal after this pass"

**Why lower?**:
- `nn.dense` is semantic but not executable
- `linalg.matmul` is well-known, optimizable, has years of downstream passes
- Standard dialects unlock the entire MLIR optimization ecosystem

**Testable**: `nn-compiler input.nn --lower` produces zero `nn.*` ops in output. Only `linalg`, `arith`, `tensor`.

---

#### Phase 4: End-to-End Pipeline + `--dump-pipeline` (Day 5)

**What**: Wire phases together, add `--dump-pipeline` flag, comprehensive tests, documentation.

**`--dump-pipeline` flag**: Show the IR at each stage:
```bash
nn-compiler input.nn                  # outputs nn dialect IR
nn-compiler input.nn --lower          # outputs linalg/arith/tensor IR
nn-compiler input.nn --dump-pipeline  # shows both stages side-by-side
```
This makes progressive lowering **visual** — extremely compelling in interviews and demos.

**Why last**:
- All functionality working, now make it robust and presentable
- Test suite validates all phases work together
- Documentation makes this resume-ready

**Testable**: `test/run_tests.sh` passes all tests.

---

### Design Principles

#### Principle 1: Each Phase is Independent and Testable

```
Phase 1 ✓ → nn-opt round-trips → Phase 2 ✓ → compiler produces IR → Phase 3 ✓ → compiler --lower works
```

You can test each phase independently. If something breaks, you know exactly where.

#### Principle 2: Use MLIR APIs, Not String Manipulation

**Wrong way** (old project):
```cpp
output << "    %h0 = nn.dense %input, %w0, %b0 : ...";
```

**Right way** (this project):
```cpp
current = builder.create<nn::DenseOp>(loc, outType, input, weights, bias);
```

Why? The MLIR API:
- Ensures correctness (type checking, SSA form)
- Enables further transformations (passes can rewrite the ops)
- Shows you understand compiler infrastructure

#### Principle 3: Progressive Lowering (Not Big-Bang Translation)

**Wrong approach**: Parse `.nn` → directly generate LLVM IR (big-bang).

**Right approach** (ours): Parse `.nn` → `nn` dialect → linalg → memref → loops → llvm (progressive).

Why progressive is better:
- Each level is easier to understand and verify
- Optimizations can target the right level
- Bugs are easier to localize
- Extensions are easier (add a new pass between existing ones)

#### Principle 4: Reuse MLIR Standard Ops, Don't Reinvent

We use:
- `linalg.matmul` — well-tested, optimized across all targets
- `arith.*` — arithmetic (add, mul, cmp, select)
- `tensor.*` — tensor operations
- `scf.*` — structured control flow (loops)

We don't invent our own. This is crucial: every custom op you add is maintenance burden.

---

## Part 4: Why This Approach is Better for Learning

### vs. "Just Call TensorFlow's Compiler"

TensorFlow's `tf.function` or XLA compiler is a black box. You don't see the IR, the passes, or how lowering works.

**Our approach**: You write every component. You see every transformation. You understand the full pipeline.

### vs. "Just Parse to AST and Emit LLVM IR"

Some compilers parse → AST → LLVM IR directly. Simpler, but:
- You miss the value of intermediate representations
- Hard to add optimizations later (where do they go?)
- All optimization decisions must happen at once
- Not extensible

**Our approach**: Multiple intermediate levels. Easy to add passes anywhere.

### vs. "Use an Existing ML Compiler Framework"

XLA, TVM, Glow, etc. are production tools. You're not learning how they work, just using them.

**Our approach**: You build it from first principles, so you deeply understand it.

---

## Part 5: Key Design Decisions

### Decision 1: Weights as Function Arguments (Resolved)

The `.nn` format says `dense 512 relu` but `nn.dense` takes three operands: `%input`, `%weights`, `%bias`. Where do weights come from?

**Decision**: Weights and biases are **function arguments**, not embedded constants.

```mlir
// The compiled function takes weights as parameters at call time
func.func @forward(%input: tensor<1x784xf32>,
                   %w0: tensor<784x512xf32>, %b0: tensor<512xf32>,
                   %w1: tensor<512x256xf32>, %b1: tensor<256xf32>,
                   %w2: tensor<256x10xf32>,  %b2: tensor<10xf32>)
                   -> tensor<1x10xf32> {
  %0 = nn.dense %input, %w0, %b0 : ... -> tensor<1x512xf32>
  %1 = nn.relu %0 : tensor<1x512xf32> -> tensor<1x512xf32>
  %2 = nn.dense %1, %w1, %b1 : ... -> tensor<1x256xf32>
  %3 = nn.relu %2 : tensor<1x256xf32> -> tensor<1x256xf32>
  %4 = nn.dense %3, %w2, %b2 : ... -> tensor<1x10xf32>
  return %4 : tensor<1x10xf32>
}
```

**Why this approach**:
- Industry standard — IREE and torch-mlir do this for inference
- Simpler — no file I/O or constant embedding in the IR
- Flexible — the caller can provide any weights (different trained models, same compiled code)
- Separates concerns — the compiler handles computation, the runtime handles data loading

**What the alternatives were**:
- Embedded constants (`arith.constant dense<[...]>`) — bloats the IR, not how inference works in practice
- External file references — adds I/O complexity, not worth it for this project

### Decision 2: One `.nn` File = One `func.func`

Each `.nn` file produces a single MLIR module with a single `func.func @forward(...)`. The function:
- Takes `%input` as the first argument
- Takes `%weights_i, %bias_i` pairs for each dense layer
- Returns the output of the final layer

This mirrors how frameworks export models: one forward pass function.

### Decision 3: Function Argument Ordering Convention

Arguments follow this convention:
```
@forward(%input, %w0, %b0, %w1, %b1, ..., %wN, %bN) -> %output
```

The parser computes tensor dimensions from the `.nn` layer sizes and generates the full `func.func` signature automatically.

---

## Part 5b: The Design Choices

### Choice 1: Custom Dialect vs. Direct Lowering to Linalg

**Option A: Direct linalg** (simpler)
```
.nn file → parser → [generates linalg ops directly]
```

**Option B: Custom nn dialect** (what we chose)
```
.nn file → parser → [generates nn.dense/nn.relu] → [lowering pass] → [linalg ops]
```

Why Option B:
- ✅ `nn.dense` is **semantic** — it says "this is a dense layer" not "this is matmul + broadcast + add"
- ✅ Enables **nn-level optimizations** (fuse consecutive dense layers, common subexpression elimination, etc.)
- ✅ Easier to understand — someone reading the IR knows it's a neural network
- ✅ Future-proof — if you add conv2d, it's a new op at the nn level; lowering paths can reuse linalg

### Choice 2: Makefile vs. CMake

**Option A: CMake with `add_mlir_dialect()`** (simpler to write)
```cmake
add_mlir_dialect(NN NNOps)
```

**Option B: Manual Makefile** (what we chose)
```makefile
mlir-tblgen --gen-dialect-decls ... NNOps.td -o NNDialect.h.inc
mlir-tblgen --gen-op-decls ... NNOps.td -o NNOps.h.inc
...
```

Why Option B:
- ✅ You see every step (no magic)
- ✅ Easier to debug (run individual commands)
- ✅ Better for learning (you understand the build process)
- ✅ You can explain it to your professor ("I ran TableGen to generate the C++ from the TableGen spec")

### Choice 3: Explicit Parser (Lexer + Parser) vs. Text Pattern Matching

**Option A: Regex-based parsing** (simple)
```cpp
if (line.find("dense") != std::string::npos) { /* parse dense */ }
```

**Option B: Proper lexer + parser** (what we chose)
```cpp
Lexer lexer(source);
auto tokens = lexer.tokenize();
Parser parser(tokens);
auto ast = parser.parse();
```

Why Option B:
- ✅ Scales to richer syntax (future: convolutions, activation selection, etc.)
- ✅ Better error messages ("line 5: unexpected token")
- ✅ Standard compiler architecture (good to learn)
- ✅ Extensible (add new keywords or syntax easily)

### Choice 4: Real MLIR OpBuilder vs. String Concatenation

**Option A: String concat** (faster to write, wrong)
```cpp
llvm::outs() << "    %h0 = nn.dense %input, %w0, %b0 : ...";
```

**Option B: Real MLIR API** (what we chose)
```cpp
current = builder.create<nn::DenseOp>(loc, outType, input, weights, bias);
module->print(llvm::outs());
```

Why Option B:
- ✅ Type-safe (compiler checks operand types)
- ✅ Enables transformations (other passes can rewrite your ops)
- ✅ SSA form guaranteed (no manual value tracking)
- ✅ Shows you understand MLIR (crucial for job interviews, research)

---

## Part 6: Trade-Offs and Why We Accept Them

### Trade-off 1: More Code vs. More Learning

**Makefile approach requires writing more build rules than CMake.**

Consequence: ~30 more lines of Makefile, but you understand every line.

### Trade-off 2: Custom Dialect Overhead vs. Semantic Clarity

**Adding a custom dialect requires TableGen + glue code.**

Consequence: Extra files (`.td`, `.inc`), but the IR is more readable and optimizable.

### Trade-off 3: Proper Parser vs. Quick Regex

**Writing a lexer and parser is more code than regex matching.**

Consequence: ~200 extra lines of C++, but the parser is extensible and has proper error handling.

### Trade-off 4: Real MLIR IR vs. String Output

**Using `OpBuilder::create<>()` is more verbose than printf.**

Consequence: More API calls, but the IR is verifiable and transformable.

**We accept all these trade-offs because the goal is learning, not speed.**

---

## Part 7: Future Extensions and Why They Matter

### Why Phases 5–8 Are Worth Doing (After Days 1–4)

#### Phase 5: Bufferization (Tensor → MemRef)

**Problem**: Tensors are immutable (functional). Real code needs mutable buffers.

**Solution**: Convert `tensor<...>` to `memref<...>` (pointers + shape info).

**Lesson**: Understanding memory layout and allocation strategies.

#### Phase 6: Tiling + Vectorization

**Problem**: Naive matrix multiply on large tensors is slow (cache misses, no SIMD).

**Solution**: Tile loops to fit in L1 cache, vectorize inner loops.

**Lesson**: Performance optimization at the IR level.

#### Phase 7: Parallelization

**Problem**: Single-threaded code doesn't use multi-core CPUs.

**Solution**: Insert parallel loop markers, lower to OpenMP or async.

**Lesson**: Understanding parallelism and thread scheduling.

#### Phase 8: LLVM Code Generation

**Problem**: MLIR IR is not executable. Need machine code.

**Solution**: Lower through LLVM IR, use LLVM's code generator.

**Lesson**: Full end-to-end compilation to executable.

---

## Part 8: How This Relates to Real-World Compilers

### Similarities to IREE (Google's ML Compiler)

IREE (Intermediate Representation Execution Environment) follows the exact same pattern:

```
Input code (TensorFlow, PyTorch)
  → [Import]
  → IREE custom ops
  → [Lowering]
  → linalg/arith/tensor
  → [Bufferization]
  → memref/scf
  → [Code generation]
  → LLVM IR
  → Machine code (CPU, GPU, etc.)
```

### Similarities to torch-mlir (PyTorch's MLIR Integration)

torch-mlir lowers PyTorch operations to a custom `torch` dialect, then progressively lowers to standard MLIR dialects.

### Differences (Why Ours is Simpler)

- We support 1 operation (`nn.dense`) and 1 activation (`relu`)
- We only lower to CPU (not GPU, TPU, etc.)
- We don't handle dynamic shapes
- We don't have a training/gradients story

But the **architecture is identical**. You're learning the same patterns used in production compilers.

---

## Part 9: Verification Strategy

### How We Know Each Phase Works

#### Phase 1: Dialect Definition ✓
```bash
./nn-opt test.mlir  # Can it parse and print nn.dense/nn.relu?
```

#### Phase 2: Parser + IR Generation ✓
```bash
./nn-compiler input.nn | ./nn-opt  # Can nn-opt read the output?
```

#### Phase 3: Lowering ✓
```bash
./nn-compiler input.nn --lower  # Any nn.* ops remaining? (Should be 0)
```

#### Phase 4: End-to-End ✓
```bash
test/run_tests.sh  # All tests pass?
```

Each checkpoint is **concrete and testable**. No abstract "it should work."

---

## Part 10: What You'll Understand After Completing This

After Days 1–4, you'll understand:

1. **How MLIR works**: Multiple IRs, progressive lowering, op definitions via TableGen
2. **How to define a custom dialect**: What TableGen specs mean, how C++ code is generated
3. **How to build a compiler frontend**: Lexing, parsing, type checking, AST → IR
4. **How to write lowering passes**: Pattern matching, op replacement, type conversion
5. **How to use MLIR's C++ API**: `OpBuilder`, `ModuleOp`, `FunctionOp`, operations
6. **Compiler architecture**: Frontend → IR → passes → backend
7. **Build systems for MLIR**: How TableGen integrates with compilation

After Phases 5–8 (future), you'll additionally understand:

8. **Memory layout and bufferization**: Tensors vs. buffers, allocation strategies
9. **Loop optimization**: Tiling, vectorization, parallelization
10. **Full end-to-end code generation**: MLIR → LLVM IR → machine code

---

## Conclusion

This design is **deliberately educational**. Every choice prioritizes understanding over speed.

The result is a neural network compiler that:
- ✅ Is small enough to understand completely (not 100k LOC like PyTorch)
- ✅ Follows real compiler principles (like IREE, torch-mlir)
- ✅ Teaches you MLIR and compiler infrastructure
- ✅ Is extensible (add phases 5–8 when ready)
- ✅ Is presentable (clear, well-documented, architected properly)

**Ready to start?** Day 1, Ticket 1.1 awaits.
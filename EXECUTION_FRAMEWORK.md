# NN-MLIR Compiler: Framework Execution Plan

**Last Updated**: 2026-04-15  
**Status**: Phase 0 (Setup & Learning) — Ready to begin  
**Total Timeline**: ~5.5 days to interview-ready project

---

## Quick Reference: The Framework

### Core Principle: Mode-Based Delegation

| Mode | Definition | When to Use | Example |
|------|-----------|------------|---------|
| **Mode 1** | You drive decisions, Claude implements | Design choices where you own the reasoning | "Should DenseOp have the `Pure` trait? Why?" |
| **Mode 2** | You understand concepts, Claude implements procedures | Code with clear interfaces and loud failures | Lexer/Parser implementation |
| **Mode 3** | Claude fully handles | Boilerplate with no hidden decisions | CMake, lit tests, documentation |

### Why This Matters for Interviews
Mode 1 decisions are interview gold. You must be able to explain:
- Why you chose a custom dialect over using existing ops
- How you decided to lower nn.dense to linalg.matmul
- Why destination-passing style exists in linalg
- Tradeoffs between linalg.generic vs. arith.maxf for relu

Mode 3 tasks don't get asked about in interviews.

---

## Project Status (2026-04-15)

### What's Done
- ✅ **NNOps.td**: TableGen file with DenseOp and ReluOp fully defined
- ✅ **NNDialect.h/cpp**: Dialect registration code in place
- ✅ **nn-opt binary**: Builds and registers nn dialect correctly
- ✅ **Build system**: Makefile with proper TableGen invocation

### What Needs to Start
- ❌ **Phase 0**: Build validation, MLIR API learning, pass infrastructure proof
- ❌ **Phase 2**: Lexer/Parser for .nn files
- ❌ **Phase 2**: IR Builder (AST → MLIR IR)
- ❌ **Phase 3**: Lowering pass (nn → linalg)
- ❌ **Phase 4**: Testing and polish

### Current Code Structure
```
nn-mlir/
├── include/NN/
│   ├── NNOps.td              ✅ DONE
│   ├── NNDialect.h           ✅ DONE (minimal)
│   └── NNOps.h               ❌ GENERATED from TableGen
├── lib/NN/
│   ├── NNDialect.cpp         ✅ DONE
│   └── NNOps.cpp             ❌ Empty stub (auto-generated via TableGen)
├── tools/
│   ├── nn-opt.cpp            ✅ DONE
│   └── nn-compiler.cpp       ❌ Not started (Frontend + IR Builder)
├── tests/                    ❌ Empty (lit tests needed Phase 4)
├── Makefile                  ✅ WORKING
└── build/
    └── [generated files from TableGen]
```

---

## Execution Timeline

| Phase | Duration | Mode | Deliverable | Interview Value |
|-------|----------|------|-------------|-----------------|
| **0: Setup & Learning** | 1 day | — | Working build + API understanding | Foundation |
| **1: Dialect Finalization** | 1 day | Mode 1 | nn.dense/nn.relu complete + working roundtrip | Why custom dialect? Why these traits? |
| **2a: Lexer/Parser** | 0.5 day | Mode 2 | .nn file → AST | Standard knowledge |
| **2b: IR Builder** | 1 day | Mode 1 | AST → MLIR IR with SSA threading | How do you build IR? Where do weights go? |
| **3: Lowering Pass** | 1.5 days | Mode 1 | nn → linalg conversion patterns | Core compiler engineering (the main event) |
| **4: Testing/Polish** | 0.5 day | Mode 3 | Full test suite, clean repo | Professionalism |
| **TOTAL** | **~5.5 days** | | | **Interview-ready** |

---

## Phase 0: Setup & Learning (START HERE)

### Sub-Tasks (Do in Order)

#### 1. CMake Validation (45 min)
**Goal**: Confirm build system works correctly with MLIR

- Run: `cd nn-mlir && make clean && make`
- Expected output: `nn-opt` binary built without errors
- Checkpoint: `./nn-opt --help` shows the nn dialect in available dialects
- Current status: Build already works (from earlier Makefile run)

#### 2. MLIR API Hello-World (30 min)
**Goal**: Validate you can construct MLIR IR programmatically

- Create a small C++ program that:
  1. Creates an `mlir::MLIRContext`
  2. Creates an empty `mlir::ModuleOp`
  3. Dumps it to stdout using `module.dump()`
- Expected output: Valid MLIR text representation of an empty module
- Why this matters: Confirms MLIR linkage is correct and you can use the API

#### 3. Learning Protocol (3-4 hrs) — CRITICAL
**Goal**: Build conceptual understanding before Mode 1 decisions in Phase 2-3

Do all four sub-steps:

**3a. Analogy Anchoring**
- Map MLIR concepts to what you know from CSE 450 (compilers course)
- Example analogies:
  - Dialects ↔ ISA definitions in Strix
  - OpBuilder ↔ IR builder pattern from LLVM
  - TableGen ↔ declarative instruction set specs
  - Progressive lowering ↔ pass-based compilation (lexing → parsing → IR → lowering → codegen)
- **Question to answer**: Where do the analogies break? (MLIR's region-based IR? SSA threading? Destination-passing?)

**3b. Whiteboard Test**
- Draw the complete pipeline from input `.nn` file to final `linalg` IR
- Include:
  - What types flow between phases? (text → AST → MLIR IR → lowered IR)
  - What MLIR infrastructure do you use vs. write yourself? (OpBuilder is provided, lowering patterns you write)
  - Where do weights get created? (module-level constants? function arguments?)
  - How do SSA values chain between operations?
- Why: Forces you to see the whole system before implementing pieces

**3c. Failure Scenarios**
- What breaks silently (compiles but wrong) vs. loudly (crash/error)?
  - Silent: Type mismatches in tensor ranks, SSA values created in wrong order, pattern matching bugs
  - Loud: Syntax errors in .nn file, wrong builder API calls
- For each phase, identify what could go wrong:
  - Phase 1 (Dialect): Wrong type constraints on operands
  - Phase 2 (Frontend): Weights not created properly, SSA threading incorrect
  - Phase 3 (Lowering): Operands extracted wrong, linalg.matmul inputs/outputs swapped

**3d. Decision Pressure-Test**
Answer these questions (don't look them up yet):
- "Should `DenseOp` have the `Pure` trait? Why or why not?"
- "If relu lowers to `linalg.generic`, what are the index maps? If to `arith.maxf`, how do you iterate?"
- "Weights/bias in IR Builder: module constants or function arguments? What are the tradeoffs?"
- "How does the pass manager know when to stop lowering?"

#### 4. Lit + FileCheck Setup (20 min)
**Goal**: Validate testing infrastructure works

- Create `tests/simple.mlir`:
  ```mlir
  // RUN: nn-opt %s | FileCheck %s
  // CHECK: module
  module {
    func.func @test() {
      return
    }
  }
  ```
- Run: `lit tests/simple.mlir`
- Expected: Test passes
- Checkpoint: Confirms lit and FileCheck are configured

#### 5. Trivial Lowering Project (3-4 hrs)
**Goal**: Learn pass infrastructure before Phase 3 complexity

**What to build**: A single lowering pass that converts `toy.add %a, %b` → `arith.addi %a, %b`

**Why this project?**
- Forces all four components of pass infrastructure: OpConversionPattern, ConversionTarget, TypeConverter, dialect conversion driver
- One operation means no confusion from multiple patterns
- If this works, Phase 3 (nn → linalg) is just scaling to 2 patterns

**Sub-tasks**:
1. Define a toy dialect with a single `toy.add` operation (in toy.td)
2. Implement one `OpConversionPattern<toy::AddOp>` that creates `arith.addi`
3. Set up `ConversionTarget` (mark toy illegal, arith legal)
4. Register the pass in nn-opt
5. Write a lit test: `toy.add %a, %b` → `arith.addi %a, %b`

**Decisions you'll understand**:
- How `OpConversionPattern::matchAndRewrite` works
- Why ConversionTarget legality matters (controls what gets lowered)
- How `ConversionPatternRewriter::replaceOp` updates SSA uses
- How the pass manager invokes the conversion driver

**What this does NOT cover**:
- Linalg destination-passing style (you'll learn in Phase 3)
- Tensor shape calculations (Phase 3)
- Creating new intermediate values (Phase 3)

**Completion criteria**: `nn-opt --lower-toy-to-arith toy_add.mlir` produces valid arith IR; lit test passes

### Phase 0 Summary
- **Parallel work**: Items 1-2 and 4 (setup) can overlap. Item 3 (learning) is independent. Item 5 depends on 1-2.
- **Total time**: ~1 calendar day if focused
- **Checkpoint**: You have working build, understand MLIR concepts deeply, and have proven pass infrastructure works
- **Ready for**: Phase 1 dialect finalization

---

## Decision Point: How to Approach Phase 0?

### Three Options

**Option A: Lightweight Track** (skip learning protocol, learn by doing)
- Do items 1, 2, 4, 5 only
- Skip the full learning protocol (3a-3d)
- **Time**: 5-6 hours
- **Tradeoff**: You'll pick up concepts during Phase 2-3, but Mode 1 decisions will be slower initially
- **Best if**: You're confident learning MLIR APIs quickly and want to move fast

**Option B: Thorough Track** (full learning protocol first)
- Do all five items 1-5 completely
- **Time**: 1 full day
- **Tradeoff**: Slower start, but Phase 2-3 Mode 1 decisions will be confident and deep
- **Best if**: You want to understand the system thoroughly and nail interviews

**Option C: Balanced Track** (recommended)
- Do items 1, 2, 4 (basic validation: 1.5 hours)
- Do learning protocol items 3a, 3b (analogy + whiteboard: 1.5 hours)
- Skip 3c, 3d for now; answer them during Phase 2-3 as you implement
- Do trivial lowering project (5) fully (3-4 hours)
- **Time**: ~6-7 hours
- **Tradeoff**: Early productivity while building conceptual depth
- **Best if**: You want to start implementation soon but stay grounded in concepts

### Recommendation
**Go with Option C (Balanced)** if this is your first major MLIR project. It gets you to real code quickly while validating the infrastructure.

---

## Phase 1: Dialect Definition (READY TO START AFTER PHASE 0)

### Current State
- ✅ TableGen file (NNOps.td) is written with proper traits
- ✅ Dialect registration works
- ⚠️ Need to verify traits/constraints match intended semantics

### Mode 1 Decisions You Must Own
1. **Should DenseOp have `Pure` trait?**
   - Pure = no side effects, result depends only on inputs
   - Your choice: Yes, if weights/bias don't change during execution
   - Implication: Compiler can hoist/common dense ops if inputs are identical

2. **Should DenseOp have `SameOperandsAndResultType`?**
   - This trait says all operands and results have the same type
   - Your case: NO — input is (batch, in_features), output is (batch, out_features)
   - Implication: You must specify type constraints explicitly

3. **Type constraints on operands — are they too permissive?**
   - Current: `AnyRankedTensor` (any ranked tensor, any element type)
   - Better: Specify rank constraints (expect 2D tensors for weights, etc.)
   - Your choice: Stay permissive for flexibility or constrain for type safety?

### Checkpoint for Phase 1
- `make clean && make` succeeds
- `./nn-opt --help` lists nn dialect
- Create a file `test_ops.mlir`:
  ```mlir
  module {
    func.func @simple(%input: tensor<2x3xf32>) -> tensor<2x4xf32> {
      %weights = arith.constant dense<1.0> : tensor<3x4xf32>
      %bias = arith.constant dense<0.0> : tensor<4xf32>
      %output = nn.dense %input, %weights, %bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
      return %output : tensor<2x4xf32>
    }
  }
  ```
- Run: `./nn-opt test_ops.mlir`
- Expected: Valid MLIR output (roundtrips without error)
- If this works, Phase 1 is complete

---

## Phase 2: Frontend (AFTER PHASE 1)

### 2a: Lexer/Parser (Mode 2, 0.5 day)

**Pre-delegation brief** (you write this before delegating):
```
MODULE: Frontend parser
PURPOSE: Parse .nn files into AST
INPUT: Raw text from .nn file (example below)
OUTPUT: NetworkAST with layer specs

KEY DECISIONS:
  - Single-pass lexer, no lookahead (grammar is trivial)
  - AST is flat list of layers, no nesting
  - Error handling: line numbers + clear messages

EDGE CASES:
  - Missing activation: default to none
  - Invalid layer type: clear error with line number
  - Extra whitespace: ignore

DO NOT: 
  - Support layer types beyond dense/relu yet
  - Handle batch size specification (assume fixed)
```

**Example .nn file format** (you decide):
```
# Simple MLP
dense 784 128
relu
dense 128 10
```

**What Claude implements**:
- Lexer that tokenizes the .nn file
- Parser that builds an AST (NetworkAST struct with vector<Layer>)
- Error reporting with line numbers

**Your role (Mode 2)**:
- Understand the interface: what does the parser return?
- Review error messages: are they clear?
- Test with a sample .nn file

**Checkpoint**: Can parse sample_mlp.nn without errors

### 2b: IR Builder (Mode 1, 1 day)

**Goal**: Convert AST → MLIR IR (Module with func.func and nn operations)

**Mode 1 Decisions** (you drive these):

1. **Weights/Bias Storage**: How do you create and store them?
   - Option A: `arith.constant` inside the function (simple, but same weights for every invocation)
   - Option B: Function arguments (flexible, but need to pass weights at call time)
   - Option C: Module-level memrefs (more complex, allows weight updates)
   - Your choice: **Recommend Option A** for Phase 2 simplicity

2. **SSA Value Threading**: How do you chain operations?
   - First dense: input → (weights, bias) → output
   - Second dense: previous output → (new weights, bias) → output
   - Your understanding: How does OpBuilder know to thread SSA values?

3. **Tensor Types**: How do you construct `tensor<2x3xf32>`?
   - Use `RankedTensorType::get()` with static shapes
   - Question: How do you know the output shape after dense? (batch preserved, feature dims change)

4. **Return Op**: What type should func.func return?
   - Last layer output type (e.g., `tensor<batch x 10xf32>` for final dense)
   - How does the return op consume SSA values?

**Implementation Flow**:
```cpp
void IRBuilder::buildModule(const NetworkAST& ast) {
  // 1. Create ModuleOp
  auto module = mlir::ModuleOp::create(loc);
  
  // 2. Create func.func with signature (input tensor -> output tensor)
  auto funcOp = mlir::func::FuncOp::create(loc, "network", funcType);
  
  // 3. Insert operations into function body
  OpBuilder builder(funcOp.getBody());
  
  // 4. Create constants for weights/bias
  auto weightsConst = builder.create<arith::ConstantOp>(...);
  auto biasConst = builder.create<arith::ConstantOp>(...);
  
  // 5. Create nn.dense op
  auto dense = builder.create<nn::DenseOp>(input, weights, bias);
  
  // 6. Chain relu (input = dense.output)
  auto relu = builder.create<nn::ReluOp>(dense.output);
  
  // 7. Return final output
  builder.create<func::ReturnOp>(relu.output);
}
```

**Checkpoint**: `nn-compiler sample_mlp.nn | nn-opt` produces valid MLIR IR with correct SSA threading

---

## Phase 3: Lowering Pass (AFTER PHASE 2)

**Timeline**: 1.5 days  
**Mode**: Mode 1 (all decisions are yours)  
**Importance**: Core compiler engineering — interviews will focus here

### What You're Doing
Converting nn.dense/nn.relu to linalg/arith ops via pattern matching and rewriting

### Mode 1 Decisions (Must Articulate in Interviews)

#### 1. **Lowering nn.dense → ?**

**Decision**: How do you convert `nn.dense %input, %weights, %bias`?

**Option A**: Single `linalg.generic` operation
```mlir
linalg.generic {indexing_maps = [...], iterator_types = [...]}
  ins(%input, %weights, %bias) outs(%output)
  (%i, %w, %b) {
    %matmul = ... (custom computation)
    linalg.yield %matmul
  }
```

**Option B**: Separate ops (linalg.matmul + linalg.add)
```mlir
%matmul_out = linalg.matmul ins(%input, %weights) outs(%empty)
%final = linalg.add ins(%matmul_out, %bias) outs(%empty)
```

**Tradeoffs**:
- Option A: More control, harder to optimize further
- Option B: Clearer intent, easier for downstream passes to optimize matmul separately

**Your choice explanation**: "I chose Option B because linalg.matmul is a named op that backend compilers recognize and optimize specifically. Fusing with bias into a generic would lose that specialization."

#### 2. **Lowering nn.relu → ?**

**Decision**: How do you convert `nn.relu %input`?

**Option A**: `linalg.generic` with index maps
```mlir
linalg.generic {indexing_maps = [affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]}
  ins(%input) outs(%output)
  (%in) {
    %zero = arith.constant 0.0
    %max = arith.maxf %in, %zero
    linalg.yield %max
  }
```

**Option B**: `arith.maxf` with explicit loops (SCF)
```mlir
scf.for %i = %c0 to %dim0 {
  scf.for %j = %c0 to %dim1 {
    %val = memref.load ...
    %max = arith.maxf %val, %zero
    memref.store %max ...
  }
}
```

**Option C**: Custom linalg operation or vector dialect

**Tradeoffs**:
- Option A: Most composable, fits MLIR's linalg ecosystem
- Option B: Simple, explicit, but less amenable to vector optimizations
- Option C: Overkill for relu

**Your choice explanation**: "I chose Option A (linalg.generic) because it integrates with the linalg dialect family and allows downstream passes (like vectorization) to optimize element-wise ops across the whole program."

#### 3. **Destination-Passing Style**

**Concept**: Linalg ops don't allocate memory — you provide an output tensor/memref

```mlir
%empty = tensor.empty : tensor<2x4xf32>
%result = linalg.matmul ins(%a, %b) outs(%empty) -> tensor<2x4xf32>
```

**Why?**
- Allows buffer reuse in compiled code
- Makes memory management explicit (compiler controls allocation)
- Enables in-place updates with memref ops

**Interview question**: "Why doesn't linalg.matmul allocate its own output?"  
**Your answer**: "Because MLIR separates computation (ops) from memory management (bufferization). Linalg uses destination-passing to remain agnostic about where memory comes from — you could provide a pre-allocated buffer, a temporary, or a function argument. The bufferization pass decides allocation strategy later."

#### 4. **Type Converter**

**Decision**: Do tensor types change during lowering?

**Likely answer**: No — input/output types stay `tensor<...>` throughout. But nn.dense/nn.relu might have generic types that you need to validate.

**Type converter role**: Maps source types to target types. If no change needed, it's minimal.

#### 5. **Pass Registration and Management**

**Decision**: How does the pass get invoked?

```cpp
void registerNNToLinalgPass() {
  PassRegistration<LowerNNToLinalgPass>();
}
```

Then: `nn-opt --lower-nn-to-linalg input.mlir`

**Interview question**: "How does the pass manager know which dialects are legal after lowering?"  
**Your answer**: "The ConversionTarget specifies what's legal in the target — I mark nn dialect illegal and linalg/arith legal. The pass manager runs patterns until no illegal ops remain, or reports an error if it can't lower everything."

### Implementation Structure

```cpp
// 1. Define RewritePattern for nn.dense
struct DenseOpLowering : public OpConversionPattern<nn::DenseOp> {
  DenseOpLowering(MLIRContext *ctx) : OpConversionPattern(ctx) {}
  
  LogicalResult matchAndRewrite(
      nn::DenseOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Extract operands from adaptor
    auto input = adaptor.getInput();
    auto weights = adaptor.getWeights();
    auto bias = adaptor.getBias();
    
    // Create init tensor for output
    auto outputType = op.getResult().getType();
    auto empty = rewriter.create<tensor::EmptyOp>(loc, outputType);
    
    // Create linalg.matmul
    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, input, weights, empty);
    
    // Create linalg.add for bias
    auto emptyAdd = rewriter.create<tensor::EmptyOp>(loc, outputType);
    auto add = rewriter.create<linalg::AddOp>(
        loc, matmul.result(), bias, emptyAdd);
    
    // Replace nn.dense with the linalg result
    rewriter.replaceOp(op, add.result());
    return success();
  }
};

// 2. Define RewritePattern for nn.relu
struct ReluOpLowering : public OpConversionPattern<nn::ReluOp> {
  // Similar structure...
};

// 3. Set up ConversionTarget
ConversionTarget target(getContext());
target.addIllegalDialect<nn::NNDialect>();
target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, func::FuncDialect>();

// 4. Add patterns
RewritePatternSet patterns(ctx);
patterns.add<DenseOpLowering, ReluOpLowering>(ctx);

// 5. Run conversion
if (failed(applyPartialConversion(module, target, std::move(patterns))))
  return signalPassFailure();
```

### Checkpoint for Phase 3
- `nn-opt --lower-nn-to-linalg sample_mlp.mlir` produces linalg IR
- Verify output with `nn-opt -print-op-generic` to check SSA threading
- Lit test passes: input.mlir → output.mlir matches FileCheck patterns

---

## Phase 4: Testing & Polish (Mode 3)

**Duration**: 0.5 day  
**Fully delegate to Claude**

### Tasks
1. Comprehensive lit test suite
   - Parse-only tests (frontend validation)
   - Roundtrip tests (parse → IR → valid IR)
   - Lowering tests (nn IR → linalg IR)
2. FileCheck pattern library (common patterns for checking MLIR text)
3. CMake test target (`make test`)
4. Clean up repository
   - Move docs into `/docs`
   - Remove old files
   - Update README with usage, examples, pipeline diagram
5. Code comments (only where logic isn't obvious)

### Checkpoint
- `make test` passes all tests
- `README.md` explains the project and how to use it
- Code is clean and ready for interview review

---

## Interview Preparation Checklist

After completing all phases, you should be able to answer these without looking anything up:

### Dialect & TableGen (Phase 1)
- [ ] "What is a dialect in MLIR?"
- [ ] "Why define a custom nn dialect instead of using existing ops?"
- [ ] "What does TableGen generate? Why is it used?"
- [ ] "Explain the ODS traits you added to DenseOp. Why those specific ones?"
- [ ] "How does the dialect registration work (registry.insert<>())?"

### Frontend & IR Construction (Phase 2)
- [ ] "Walk me through how you construct a single nn.dense operation in IR."
- [ ] "How do you thread SSA values between operations? Show me the code."
- [ ] "Where do weights and biases come from in your IR? Why?"
- [ ] "How does OpBuilder work? What happens when you call builder.create<>()?"
- [ ] "What is a tensor type in MLIR? How do you construct RankedTensorType?"

### Lowering & Pattern Matching (Phase 3) — **Most Important**
- [ ] "Explain destination-passing style in linalg. Why does it exist?"
- [ ] "Walk me through the pattern matching for lowering nn.dense. What operands do you extract?"
- [ ] "What's the difference between a RewritePattern and a ConversionPattern?"
- [ ] "How does the ConversionTarget work? Why mark nn illegal and linalg legal?"
- [ ] "What's a type converter? Do you need one for your project? Why or why not?"
- [ ] "Why is progressive lowering better than direct source-to-LLVM compilation?"
- [ ] "How would you add shape inference to this compiler?"
- [ ] "What would break if tensors had dynamic dimensions?"
- [ ] "How would you add reshape or transpose operations? What would change?"

### Architecture & Design (Cross-cutting)
- [ ] "Compare your architecture to torch-mlir and IREE. What are the key differences?"
- [ ] "How would you extend this compiler to support other activation functions (tanh, sigmoid, etc.)?"
- [ ] "What optimizations could you add at the nn dialect level vs. the linalg level?"
- [ ] "How would you generate LLVM IR from your linalg IR?"
- [ ] "What would a test suite look like for a real compiler? How is yours different?"

---

## Current Action Items

### Immediate (Next Session)
1. **Mark Task #1 (Phase 0) as `in_progress`**
2. **Choose your track**: Lightweight (A), Thorough (B), or Balanced (C)?
3. **Start with Phase 0 item 1**: Validate build system

### After Phase 0
1. Move to Task #2 (Phase 1): Finalize dialect
2. Then Task #3-4 (Phase 2): Frontend
3. Then Task #5 (Phase 3): Lowering
4. Then Task #6 (Phase 4): Polish

---

## References & Resources

### Key Concepts to Understand
- **Dialects**: MLIR's plugin system for operation namespaces
- **TableGen**: Declarative language for code generation
- **OpBuilder**: C++ API for constructing MLIR operations
- **SSA (Static Single Assignment)**: Each value defined once, threaded through uses
- **Linalg**: Structured linear algebra dialect with destination-passing semantics
- **Progressive lowering**: Compilation as a sequence of dialect-to-dialect transformations
- **Pattern matching**: Recognizing operation patterns and rewriting them

### Files to Reference When Stuck
- `include/NN/NNOps.td` — TableGen operation definitions
- `lib/NN/NNDialect.cpp` — Dialect registration (watch how GET_OP_LIST works)
- `tools/nn-opt.cpp` — Entry point; shows dialect registration pattern

### Interview Resources
- Explain **why** lowering strategies matter (not just **how** to implement them)
- Be ready to **compare** your choices to industry compilers (torch-mlir, IREE, TVM, XLA)
- Articulate **tradeoffs**: Is linalg.generic better than arith.maxf? Under what conditions?

---

## Notes for Claude Code

**When the user asks to continue this project**:
1. Check this file first for context and current phase
2. Look at the memory files in `.claude/projects/.../memory/` for framework strategy
3. Check the task list (6 tasks in order)
4. Reference the current phase status section to know what's completed
5. Use Mode 1/2/3 discipline: don't skip Mode 1 decision-making, even if tempted to move fast

**Key constraints**:
- Phase 1, 2b, 3 are Mode 1 → user owns decisions, you implement
- Phase 2a, 4 are Mode 2/3 → you implement fully
- Don't abstract or over-engineer: stick to what the phase needs
- Interview value is in Phases 1-3 Mode 1 decisions; don't skimp there

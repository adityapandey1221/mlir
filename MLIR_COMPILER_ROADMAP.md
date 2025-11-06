# MLIR Compiler Roadmap: Building a Minimal Hardware Accelerator Compiler

**Goal**: Create a minimal, custom MLIR-based compiler that demonstrates core middle-end optimization principles used by hardware accelerators, built from first principles.

**Prerequisites**: C++ knowledge, basic understanding of neural network architectures
**Timeline**: 8-12 weeks (depending on time commitment)

---

## Table of Contents
1. [Foundation: Understanding the "Why"](#phase-1-foundation)
2. [Core Concepts: MLIR Building Blocks](#phase-2-core-concepts)
3. [Your First Dialect: TensorFlow-like Operations](#phase-3-first-dialect)
4. [Pattern Matching & Rewriting](#phase-4-pattern-matching)
5. [Hardware-Oriented Optimizations](#phase-5-hardware-optimizations)
6. [Memory Hierarchy & Data Movement](#phase-6-memory-hierarchy)
7. [Building the Complete Pipeline](#phase-7-complete-pipeline)
8. [Testing & Validation](#phase-8-testing)

---

## Phase 1: Foundation - Understanding the "Why" (Week 1)

### 1.1 Why Compilers Matter for Hardware Accelerators

**First Principles**: Modern neural networks are just mathematical computations. Hardware accelerators (GPUs, TPUs, custom ASICs) can execute these operations 100-1000x faster than CPUs, but only if the operations are expressed in a way that matches the hardware's capabilities.

**The Problem**:
- PyTorch/TensorFlow express computations in high-level operations (`matmul`, `conv2d`, `relu`)
- Hardware needs low-level instructions (specific memory layouts, tiled operations, parallel execution)
- The **middle-end compiler** bridges this gap through optimization

**Why MLIR?**
- Traditional compilers (LLVM) work at too low a level for ML operations
- MLIR lets you work at multiple abstraction levels simultaneously
- You can represent a matrix multiply as both a high-level operation AND its low-level implementation

### 1.2 The Three-Stage Compiler Mental Model

```
HIGH LEVEL (Python/Framework)
    ↓ Frontend
MIDDLE LEVEL (MLIR - THIS IS OUR FOCUS)
    ↓ Backend
LOW LEVEL (Assembly/Hardware Instructions)
```

**Our Focus**: The middle stage where we:
1. Represent operations abstractly
2. Optimize data flow and computation patterns
3. Transform for specific hardware characteristics

### 1.3 Practical Exercise: Set Up MLIR

**Objective**: Get MLIR building and understand the codebase structure

```bash
# Clone LLVM (includes MLIR)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure (minimal build for MLIR only)
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

# Build (this takes 30-60 minutes)
ninja check-mlir
```

**Key Directories to Explore**:
- `mlir/include/mlir/IR/` - Core IR definitions
- `mlir/include/mlir/Dialect/` - Built-in dialects (Arith, Tensor, Linalg)
- `mlir/lib/Transforms/` - Optimization passes
- `mlir/examples/toy/` - The canonical tutorial

**Deliverable**: Successfully build MLIR and run `mlir-opt --help`

---

## Phase 2: Core Concepts - MLIR Building Blocks (Week 2-3)

### 2.1 Understanding Operations (The Atom of MLIR)

**First Principles**: Everything in a compiler is an operation. An operation is a node in a computational graph.

```mlir
// Example: A simple addition
%result = arith.addi %a, %b : i32

// Anatomy:
// %result      - SSA (Static Single Assignment) value name
// arith.addi   - Operation name (dialect.operation)
// %a, %b       - Operands (inputs)
// : i32        - Type annotation
```

**Key Insight**: Operations are defined in **dialects** (namespaces for related operations)

### 2.2 Understanding Dialects (The Language You Invent)

**First Principles**: A dialect is a collection of operations that represent computations at a specific abstraction level.

**Examples**:
- `arith` dialect: Integer/float arithmetic (`addi`, `mulf`, etc.)
- `tensor` dialect: High-level tensor operations
- `linalg` dialect: Linear algebra operations (matmul, conv)
- **Your custom dialect**: Hardware-specific operations

**Why Multiple Dialects?**
Different hardware and optimization stages need different abstractions. You'll progressively lower from high-level to low-level dialects.

### 2.3 Understanding Types

MLIR is **strongly typed**. Types describe the shape and nature of data:

```mlir
// Scalar types
%x : i32           // 32-bit integer
%y : f32           // 32-bit float

// Tensor types (most important for ML)
%tensor : tensor<4x256x256xf32>  // 4 x 256 x 256 float tensor

// Memory types (for hardware)
%mem : memref<1024xf32>          // Memory reference (like a pointer)
```

**Key Distinction**:
- `tensor` = immutable, mathematical abstraction
- `memref` = mutable, physical memory representation

### 2.4 Understanding Regions and Blocks

**First Principles**: Some operations contain other operations. This creates hierarchy.

```mlir
// A function is an operation with a region
func.func @my_function(%arg0: f32) -> f32 {
  // This is a block (contains a sequence of operations)
  %result = arith.mulf %arg0, %arg0 : f32
  return %result : f32
}
```

**Hardware Relevance**: Loops and conditionals use regions to represent structured control flow.

### 2.5 Practical Exercise: Read and Understand MLIR IR

**Objective**: Build intuition by reading MLIR code

Create a file `example.mlir`:
```mlir
func.func @matrix_multiply(%A: tensor<128x256xf32>,
                           %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %C_empty = tensor.empty() : tensor<128x512xf32>
  %cst = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%cst : f32) outs(%C_empty : tensor<128x512xf32>) -> tensor<128x512xf32>

  %C = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                     outs(%C_init : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %C : tensor<128x512xf32>
}
```

Run: `mlir-opt example.mlir`

**Exercise**:
1. Identify each operation
2. Trace the data flow (%C_empty → %C_init → %C)
3. Understand why initialization is needed

**Deliverable**: Explain in your own words what each line does

---

## Phase 3: Your First Dialect - TensorFlow-like Operations (Week 3-4)

### 3.1 Design Your Dialect (Paper Design First!)

**First Principles**: Before writing code, decide what abstraction level you need.

**Our Target**: A minimal dialect for neural network layers

```mlir
// What operations do we need?
%y = nn.dense %x, %weights, %bias : tensor<NxD>, tensor<DxM>, tensor<M> -> tensor<NxM>
%y = nn.relu %x : tensor<NxM> -> tensor<NxM>
%y = nn.conv2d %input, %kernel : tensor<NxHxWxC>, tensor<KxKxCxF> -> tensor<NxH'xW'xF>
%y = nn.softmax %x : tensor<NxC> -> tensor<NxC>
```

**Design Questions**:
1. What are the minimum operations for a simple neural network?
2. What information does each operation need? (dimensions, types, attributes)
3. How will these eventually map to hardware?

### 3.2 Implement the Dialect in C++

**File Structure**:
```
my-mlir-project/
├── include/
│   └── NNDialect/
│       ├── NNDialect.h      // Dialect definition
│       ├── NNOps.h          // Operation definitions
│       └── NNOps.td         // TableGen operation specs
├── lib/
│   └── NNDialect/
│       ├── NNDialect.cpp
│       └── NNOps.cpp
└── CMakeLists.txt
```

**Step 3.2.1: Define Dialect** (`NNDialect.td`)

```tablegen
// TableGen is MLIR's code generation tool
def NN_Dialect : Dialect {
  let name = "nn";
  let summary = "Neural Network operations dialect";
  let description = [{
    This dialect contains high-level neural network operations
    suitable for hardware accelerator optimization.
  }];
  let cppNamespace = "::mlir::nn";
}
```

**Step 3.2.2: Define Operations** (`NNOps.td`)

```tablegen
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

def DenseOp : NN_Op<"dense", [Pure]> {
  let summary = "Fully connected (dense) layer";
  let description = [{
    Performs matrix multiplication followed by bias addition:
    output = input @ weights + bias
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$bias
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weights `,` $bias attr-dict `:` type($input) `,` type($weights) `,` type($bias) `->` type($output)
  }];
}

def ReluOp : NN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified Linear Unit activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

**Step 3.2.3: Build System** (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.20)
project(my-nn-compiler)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# TableGen our operations
set(LLVM_TARGET_DEFINITIONS NNOps.td)
mlir_tablegen(NNOps.h.inc -gen-op-decls)
mlir_tablegen(NNOps.cpp.inc -gen-op-defs)
mlir_tablegen(NNDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NNDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(NNOpsIncGen)

# Compile the dialect
add_mlir_library(NNDialect
  NNDialect.cpp
  NNOps.cpp
  DEPENDS NNOpsIncGen
)
```

### 3.3 Practical Exercise: Write Your First Operation

**Objective**: Implement a complete operation from scratch

1. Add a `nn.batch_norm` operation to your dialect
2. Define its semantics (mean subtraction, variance division, scale and shift)
3. Specify inputs (input tensor, scale, shift, mean, variance)
4. Generate code using TableGen
5. Write a test MLIR file that uses it

**Deliverable**: A working `nn.batch_norm` operation that parses and verifies

---

## Phase 4: Pattern Matching & Rewriting (Week 5-6)

### 4.1 Why Pattern Rewriting?

**First Principles**: Optimization is pattern matching + rewriting.

**Example Pattern**:
```mlir
// Before: Inefficient
%1 = nn.dense %x, %W1, %b1
%2 = nn.relu %1
%3 = nn.dense %2, %W2, %b2

// After: Fused (better for hardware)
%3 = nn.fused_dense_relu_dense %x, %W1, %b1, %W2, %b2
```

**Why This Matters for Hardware**:
- Fewer memory roundtrips
- Better cache utilization
- Opportunity for specialized hardware units

### 4.2 Declarative Pattern Rewriting (DRR)

MLIR provides a pattern matching DSL using TableGen:

```tablegen
// Fusion pattern: relu(dense(x)) -> fused_dense_relu(x)
def FuseDenseReluPattern : Pat<
  // Match this pattern
  (ReluOp (DenseOp $input, $weights, $bias)),

  // Replace with this
  (FusedDenseReluOp $input, $weights, $bias)
>;
```

### 4.3 Imperative Pattern Rewriting (C++)

For complex patterns, write C++ code:

```cpp
struct FuseDenseReluPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    // Check if input comes from a DenseOp
    auto denseOp = reluOp.getOperand().getDefiningOp<DenseOp>();
    if (!denseOp) return failure();

    // Check if DenseOp has only one use (safe to fuse)
    if (!denseOp->hasOneUse()) return failure();

    // Create fused operation
    rewriter.replaceOpWithNewOp<FusedDenseReluOp>(
      reluOp,
      denseOp.getInput(),
      denseOp.getWeights(),
      denseOp.getBias()
    );

    return success();
  }
};
```

### 4.4 Key Optimization Patterns for Hardware Accelerators

**Pattern 1: Operation Fusion**
- Fuse multiple operations to reduce memory traffic
- Example: `conv + batch_norm + relu` → `fused_conv_bn_relu`

**Pattern 2: Constant Folding**
- Compute known values at compile time
- Example: `%x = arith.addi %c1, %c2` → `%x = arith.constant 3`

**Pattern 3: Algebraic Simplification**
- Use mathematical identities
- Example: `%y = arith.muli %x, 1` → `%y = %x`

**Pattern 4: Data Layout Transformation**
- Change tensor layout for hardware efficiency
- Example: NCHW → NHWC for certain accelerators

### 4.5 Practical Exercise: Implement Fusion Patterns

**Objective**: Write patterns that reduce memory traffic

Implement these patterns:
1. Fuse `dense + relu`
2. Fuse `dense + relu + dense` (two-layer fusion)
3. Eliminate identity operations (`relu(relu(x))` → `relu(x)`)
4. Constant propagation through operations

Test with:
```bash
mlir-opt input.mlir -nn-fusion-pass -o optimized.mlir
```

**Deliverable**: Show before/after MLIR IR demonstrating your optimizations

---

## Phase 5: Hardware-Oriented Optimizations (Week 7-8)

### 5.1 Understanding Hardware Constraints

**First Principles**: Hardware accelerators have specific characteristics:

1. **Parallel Execution Units**: Can compute multiple operations simultaneously
2. **Memory Hierarchy**: Fast small cache, slow large DRAM
3. **Data Movement Cost**: Moving data is more expensive than computing
4. **Specialized Units**: Matrix multiply units, vector processors

**The Optimization Goal**: Maximize compute utilization while minimizing data movement

### 5.2 Tiling: Breaking Large Operations into Chunks

**First Principles**: Large matrices don't fit in fast memory. Solution: process in tiles.

```mlir
// Before: One large matmul
%C = linalg.matmul ins(%A, %B) outs(%C_init)
     : tensor<1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1024x1024xf32>

// After: Tiled into 64x64 blocks (fits in cache)
scf.for %i = 0 to 1024 step 64 {
  scf.for %j = 0 to 1024 step 64 {
    scf.for %k = 0 to 1024 step 64 {
      %A_tile = tensor.extract_slice %A[%i, %k][64, 64][1, 1]
      %B_tile = tensor.extract_slice %B[%k, %j][64, 64][1, 1]
      %C_tile = linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile_init)
      %C = tensor.insert_slice %C_tile into %C[%i, %j][64, 64][1, 1]
    }
  }
}
```

**Tile Size Selection**:
- Too small: Overhead from loop management
- Too large: Cache misses, no parallelism
- Optimal: Fits in L1/L2 cache, balances parallelism

### 5.3 Implementing Tiling Transformation

```cpp
struct TileMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
  TileMatmulPattern(MLIRContext *context, int tileSize)
      : OpRewritePattern<linalg::MatmulOp>(context), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    // Get dimensions
    auto aType = matmul.getInputs()[0].getType().cast<RankedTensorType>();
    auto shape = aType.getShape();

    // Create tiling loops
    SmallVector<Value> lowerBounds(3, rewriter.create<arith::ConstantIndexOp>(loc, 0));
    SmallVector<Value> upperBounds = {
      rewriter.create<arith::ConstantIndexOp>(loc, shape[0]),
      rewriter.create<arith::ConstantIndexOp>(loc, shape[1]),
      rewriter.create<arith::ConstantIndexOp>(loc, shape[2])
    };
    SmallVector<Value> steps(3, rewriter.create<arith::ConstantIndexOp>(loc, tileSize));

    // Build nested loops with tiled matmul inside
    // (Implementation details depend on your IR structure)

    return success();
  }

private:
  int tileSize;
};
```

### 5.4 Vectorization: SIMD Operations

**First Principles**: Modern processors can operate on vectors (multiple data elements) in one instruction.

```mlir
// Before: Scalar operations
scf.for %i = 0 to 1024 {
  %val = tensor.extract %input[%i]
  %result = arith.mulf %val, %scale
  tensor.insert %result into %output[%i]
}

// After: Vector operations (process 8 elements at once)
scf.for %i = 0 to 1024 step 8 {
  %vec = vector.transfer_read %input[%i] : tensor<1024xf32>, vector<8xf32>
  %result_vec = arith.mulf %vec, %scale_vec : vector<8xf32>
  vector.transfer_write %result_vec, %output[%i]
}
```

### 5.5 Practical Exercise: Implement Hardware Optimizations

**Objective**: Transform operations for efficient hardware execution

Tasks:
1. Implement loop tiling for matrix multiplication
2. Add a command-line parameter for tile size
3. Implement basic vectorization for element-wise operations
4. Measure theoretical performance (FLOPS) before and after

**Deliverable**:
- Optimized MLIR code with configurable tiling
- Performance analysis showing reduction in memory accesses

---

## Phase 6: Memory Hierarchy & Data Movement (Week 9-10)

### 6.1 Understanding the Memory Wall

**First Principles**: The Performance Equation

```
Time = (Compute_ops / FLOPS) + (Data_movement / Bandwidth)
```

For modern accelerators:
- Compute: ~100 TFLOPS (very fast)
- DRAM bandwidth: ~1 TB/s
- To saturate compute: need 100 FLOP per byte moved!

**Reality**: Most operations are memory-bound, not compute-bound.

### 6.2 The Memory Hierarchy

```
Register File:      ~100 KB,    ~10 TB/s,    1 cycle
L1 Cache:           ~100 KB,    ~5 TB/s,     4 cycles
L2 Cache:           ~10 MB,     ~2 TB/s,     20 cycles
DRAM:               ~40 GB,     ~1 TB/s,     200 cycles
```

**Optimization Strategy**: Keep data in fast memory as long as possible

### 6.3 From Tensors to Memory References

**Key Transition**: Moving from mathematical abstractions to physical memory

```mlir
// High-level: tensor (immutable, abstract)
func.func @compute(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = nn.relu %input : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// Low-level: memref (mutable, physical memory)
func.func @compute(%input: memref<1024xf32>, %output: memref<1024xf32>) {
  scf.for %i = 0 to 1024 {
    %val = memref.load %input[%i] : memref<1024xf32>
    %zero = arith.constant 0.0 : f32
    %result = arith.maxf %val, %zero : f32  // ReLU
    memref.store %result, %output[%i] : memref<1024xf32>
  }
  return
}
```

### 6.4 Buffer Allocation and Management

**Problem**: Where do we allocate memory?

```mlir
// Allocate in fast scratchpad memory (for accelerators)
%buffer = memref.alloc() : memref<64x64xf32, #scratchpad>

// Allocate in main memory
%buffer = memref.alloc() : memref<1024x1024xf32, #dram>
```

**Pattern**: Allocate temporaries in fast memory, inputs/outputs in DRAM

### 6.5 Implementing Buffer Optimization

**Lowering Pass**: tensor → memref

```cpp
struct TensorToMemrefLowering : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::ExtractOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Replace tensor.extract with memref.load
    rewriter.replaceOpWithNewOp<memref::LoadOp>(
      op,
      adaptor.getTensor(),
      adaptor.getIndices()
    );

    return success();
  }
};
```

### 6.6 Practical Exercise: Memory-Aware Compilation

**Objective**: Implement full tensor-to-memref lowering

Tasks:
1. Write a pass that converts tensor operations to memref operations
2. Implement buffer allocation for intermediate results
3. Add buffer reuse (same buffer for multiple ops when safe)
4. Implement copy operations for data movement between memory levels

**Deliverable**:
- A complete lowering pipeline: `tensor → memref → explicit loads/stores`
- Memory usage analysis showing buffer reuse

---

## Phase 7: Building the Complete Pipeline (Week 11)

### 7.1 The Compilation Pipeline Architecture

```
Input MLIR (nn dialect)
    ↓
[Optimization Passes]
    ↓ Pattern fusion
    ↓ Algebraic simplification
    ↓
[Tiling Pass]
    ↓ Insert loops with tile sizes
    ↓
[Lowering to linalg]
    ↓ Convert nn ops to linalg ops
    ↓
[Memory Planning]
    ↓ tensor → memref
    ↓ Buffer allocation
    ↓
[Vectorization]
    ↓ Scalar → vector operations
    ↓
[Lowering to LLVM]
    ↓ Convert to LLVM IR
    ↓
[LLVM Backend]
    ↓
Machine Code
```

### 7.2 Pass Manager Implementation

```cpp
void buildNNCompilerPipeline(OpPassManager &pm) {
  // High-level optimizations
  pm.addPass(createFusionPass());
  pm.addPass(createConstantFoldingPass());
  pm.addPass(createCSEPass());  // Common subexpression elimination

  // Hardware-specific optimizations
  pm.addPass(createTilingPass(/*tileSize=*/64));
  pm.addPass(createLayoutOptimizationPass());

  // Lowering
  pm.addPass(createLowerToLinalgPass());
  pm.addPass(createBufferDeallocationPass());
  pm.addPass(createConvertTensorToMemrefPass());

  // Low-level optimizations
  pm.addPass(createVectorizationPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createConvertToLLVMPass());
}
```

### 7.3 Creating Your Compiler Tool

**File**: `nn-opt.cpp` (similar to mlir-opt)

```cpp
int main(int argc, char **argv) {
  // Register dialects
  DialectRegistry registry;
  registry.insert<nn::NNDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<memref::MemRefDialect>();

  // Register passes
  registerNNPasses();

  // Parse command line and run passes
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "NN Compiler\n", registry)
  );
}
```

### 7.4 Practical Exercise: End-to-End Compilation

**Objective**: Compile a simple neural network to executable code

**Input**: A 2-layer MLP in your nn dialect
```mlir
func.func @mlp_inference(%input: tensor<1x784xf32>,
                         %w1: tensor<784x256xf32>,
                         %b1: tensor<256xf32>,
                         %w2: tensor<256x10xf32>,
                         %b2: tensor<10xf32>) -> tensor<1x10xf32> {
  %hidden = nn.dense %input, %w1, %b1
  %hidden_act = nn.relu %hidden
  %output = nn.dense %hidden_act, %w2, %b2
  return %output : tensor<1x10xf32>
}
```

**Pipeline**:
```bash
# Your compiler
nn-opt input.mlir \
  -nn-fusion \
  -nn-tiling \
  -convert-nn-to-linalg \
  -convert-linalg-to-loops \
  -convert-tensor-to-memref \
  -vectorize \
  -convert-to-llvm \
  -o output.ll

# LLVM backend
llc output.ll -o output.o
clang output.o -o mlp_inference
```

**Deliverable**:
- A working executable that runs your neural network
- Compilation pipeline that applies all your optimizations

---

## Phase 8: Testing & Validation (Week 12)

### 8.1 Testing Strategies

**Three Levels of Testing**:

1. **Unit Tests**: Individual passes
```cpp
TEST(FusionTest, FuseDenseRelu) {
  MLIRContext context;
  auto module = parseSourceString<ModuleOp>(R"(
    func.func @test(%x: tensor<10xf32>) -> tensor<10xf32> {
      %w = arith.constant dense<1.0> : tensor<10x10xf32>
      %b = arith.constant dense<0.0> : tensor<10xf32>
      %dense = nn.dense %x, %w, %b
      %relu = nn.relu %dense
      return %relu : tensor<10xf32>
    }
  )", &context);

  PassManager pm(&context);
  pm.addPass(createFusionPass());
  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Check that fusion happened
  auto func = module->lookupSymbol<func::FuncOp>("test");
  EXPECT_FALSE(func.walk([](ReluOp op) { return WalkResult::interrupt(); }).wasInterrupted());
}
```

2. **Integration Tests**: Full pipeline
```bash
# FileCheck tests (MLIR standard)
# RUN: nn-opt %s -nn-fusion | FileCheck %s

# CHECK-LABEL: func @test
func.func @test(%x: tensor<10xf32>) -> tensor<10xf32> {
  # CHECK-NOT: nn.relu
  # CHECK: nn.fused_dense_relu
  %w = arith.constant dense<1.0> : tensor<10x10xf32>
  %b = arith.constant dense<0.0> : tensor<10xf32>
  %dense = nn.dense %x, %w, %b
  %relu = nn.relu %dense
  return %relu : tensor<10xf32>
}
```

3. **End-to-End Tests**: Numerical correctness
```python
# Compare against reference implementation
import numpy as np

def test_mlp_correctness():
    # Generate random inputs
    x = np.random.randn(1, 784).astype(np.float32)
    w1 = np.random.randn(784, 256).astype(np.float32)
    b1 = np.random.randn(256).astype(np.float32)
    w2 = np.random.randn(256, 10).astype(np.float32)
    b2 = np.random.randn(10).astype(np.float32)

    # Reference implementation
    hidden = np.maximum(0, x @ w1 + b1)  # ReLU
    output_ref = hidden @ w2 + b2

    # Run compiled version
    output_compiled = run_compiled_mlp(x, w1, b1, w2, b2)

    # Check numerical accuracy
    np.testing.assert_allclose(output_ref, output_compiled, rtol=1e-5)
```

### 8.2 Performance Benchmarking

**What to Measure**:
1. **Compile Time**: How long does optimization take?
2. **Code Quality**: How fast is the generated code?
3. **Memory Usage**: How much memory is allocated?

```cpp
// Simple benchmarking harness
void benchmark_matmul() {
  auto start = std::chrono::high_resolution_clock::now();

  // Run compiled function
  for (int i = 0; i < 1000; i++) {
    compiled_matmul(A, B, C);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Calculate GFLOPS
  double flops = 2.0 * M * N * K * 1000;  // 1000 iterations
  double gflops = flops / duration.count() / 1000.0;

  std::cout << "Performance: " << gflops << " GFLOPS\n";
}
```

### 8.3 Debugging Tools

**Essential Debugging Flags**:
```bash
# Print IR after each pass
nn-opt input.mlir -nn-fusion -print-ir-after-all

# Print IR only for specific pass
nn-opt input.mlir -print-ir-after=nn-fusion

# Check IR validity
nn-opt input.mlir -verify-each

# Debug pattern matching
nn-opt input.mlir -debug-only=pattern-matcher
```

### 8.4 Practical Exercise: Build Test Suite

**Objective**: Comprehensive testing infrastructure

Tasks:
1. Write unit tests for each optimization pass
2. Create FileCheck integration tests
3. Implement numerical correctness tests
4. Build performance regression tests
5. Set up continuous integration (GitHub Actions/Jenkins)

**Deliverable**:
- Test suite with >80% code coverage
- CI pipeline that runs tests on every commit
- Performance dashboard tracking GFLOPS over time

---

## Reference Projects & Further Learning

### Minimal Working Examples

1. **MLIR Toy Tutorial**: `llvm-project/mlir/examples/toy/`
   - Canonical MLIR tutorial
   - Builds a complete compiler for a toy language
   - ~2000 lines of code

2. **Standalone MLIR Example**: `llvm-project/mlir/examples/standalone/`
   - Template for out-of-tree MLIR projects
   - Shows proper CMake setup

3. **IREE (Intermediate Representation Execution Environment)**
   - Production compiler for ML models
   - Excellent reference for hardware targeting
   - https://github.com/openxla/iree

### Key Papers & Resources

1. **MLIR: A Compiler Infrastructure for the End of Moore's Law**
   - Original MLIR paper
   - Explains the motivation and design

2. **Tensor Comprehensions**
   - Polyhedral optimization for ML
   - Background on loop transformations

3. **Halide**: Image processing DSL
   - Separation of algorithm and schedule
   - Inspiration for many MLIR concepts

### Hardware Accelerator Background

1. **NVIDIA CUDA Programming Guide**
   - Understanding GPU memory hierarchy
   - Thread/block execution model

2. **Google TPU Architecture**
   - Systolic arrays for matrix multiplication
   - Memory bandwidth optimization

3. **Cerebras Wafer-Scale Engine**
   - Extreme parallelism
   - On-chip SRAM optimization

---

## Common Pitfalls & How to Avoid Them

### Pitfall 1: Starting Too Low-Level
**Problem**: Jumping into LLVM IR directly
**Solution**: Stay at the tensor/linalg level as long as possible

### Pitfall 2: Ignoring Verification
**Problem**: Invalid IR that crashes later
**Solution**: Always run `-verify-each`, implement operation verifiers

### Pitfall 3: Over-Optimization Too Early
**Problem**: Trying to implement every optimization
**Solution**: Start with one optimization, get it working end-to-end

### Pitfall 4: Not Understanding TableGen
**Problem**: Fighting with TableGen errors
**Solution**: Study existing dialect definitions, use `-debug` flags

### Pitfall 5: Forgetting About Types
**Problem**: Type mismatches causing subtle bugs
**Solution**: Strong typing discipline, use typed APIs

---

## Success Criteria

By the end of this roadmap, you should be able to:

✅ **Understand** the role of middle-end compilers in hardware acceleration
✅ **Read** and comprehend MLIR code in multiple dialects
✅ **Design** a custom dialect for domain-specific operations
✅ **Implement** pattern-based optimizations (fusion, tiling, vectorization)
✅ **Lower** from high-level operations to memory operations
✅ **Compile** a simple neural network to executable code
✅ **Measure** and analyze the performance impact of optimizations
✅ **Test** compiler transformations for correctness and performance

---

## Final Project: Build a Minimal Accelerator Compiler

**Objective**: Synthesize everything into one complete project

**Requirements**:
1. Custom dialect with at least 5 neural network operations
2. Three optimization passes (fusion, tiling, vectorization)
3. Complete lowering pipeline to LLVM IR
4. Test suite with unit and integration tests
5. Performance benchmarks showing optimization impact
6. Documentation explaining design decisions

**Example Project**: "TinyNN Compiler"
- Input: Simple neural network in custom dialect
- Output: Optimized executable
- Target: CPU with AVX2 instructions
- Optimizations: Operation fusion, loop tiling, SIMD vectorization

**Timeline**: 2-3 weeks

**Deliverable**: GitHub repository with:
- Source code
- Build instructions
- Test suite
- Performance analysis
- Design document

---

## Next Steps Beyond This Roadmap

Once you complete this roadmap, you can:

1. **Add More Dialects**: GPU (NVVM, ROCDL), Custom ASICs
2. **Advanced Optimizations**: Automatic differentiation, memory planning, distributed execution
3. **Real Hardware**: Target actual accelerators (NVIDIA GPU, Google TPU, etc.)
4. **Integration**: Connect to PyTorch/TensorFlow frontends
5. **Production**: Contribute to IREE, XLA, or other production compilers

---

**Remember**: Compilers are complex, but they're built from simple concepts. Master each phase before moving to the next. Build small, test often, and gradually increase complexity.

Good luck on your MLIR compiler journey! 🚀

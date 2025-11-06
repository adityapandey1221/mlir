# MLIR Compiler Roadmap: 4-Week Intensive

**Goal**: Build a minimal MLIR compiler that optimizes a 2-layer neural network for CPU hardware accelerators.

**Prerequisites**: C++ knowledge, basic understanding of neural networks
**Timeline**: 4 weeks (intensive, ~20 hours/week = 80 hours total)
**Scope**: Matrix multiplication optimization with loop tiling

---

## What You'll Build

A complete compiler pipeline that:
1. Takes a 2-layer MLP (Multi-Layer Perceptron) as input
2. Fuses operations (dense + relu)
3. Tiles matrix multiplications for cache efficiency
4. Generates optimized executable code

**Example Input**:
```mlir
func.func @mlp(%input: tensor<1x784xf32>, ...) -> tensor<1x10xf32> {
  %h1 = nn.dense %input, %w1, %b1 : ...
  %h1_act = nn.relu %h1 : ...
  %output = nn.dense %h1_act, %w2, %b2 : ...
  return %output
}
```

**Example Output**: Optimized executable with 2-5x speedup from tiling.

---

## Table of Contents

### Week 1: Foundation & Dialect (Days 1-7)
0. [Hello World: Your First MLIR Program](#week-1-day-1-2-hello-world) - Days 1-2
1. [Core MLIR Concepts](#week-1-day-3-4-core-concepts) - Days 3-4
2. [Minimal Neural Network Dialect](#week-1-day-5-7-minimal-dialect) - Days 5-7

### Week 2: Optimization (Days 8-14)
3. [Pattern Rewriting & Fusion](#week-2-day-8-10-pattern-rewriting) - Days 8-10
4. [Loop Tiling for Hardware](#week-2-day-11-14-loop-tiling) - Days 11-14

### Week 3: Lowering & Pipeline (Days 15-21)
5. [Tensor to Memory Lowering](#week-3-day-15-17-lowering) - Days 15-17
6. [End-to-End Compilation](#week-3-day-18-21-pipeline) - Days 18-21

### Week 4: Integration (Days 22-28)
7. [Testing & Validation](#week-4-day-22-24-testing) - Days 22-24
8. [Final Project](#week-4-day-25-28-final) - Days 25-28

---

## Week 1, Day 1-2: Hello World

### Goal
Build and run your first MLIR program that generates a simple add function.

### What to Do

See the complete project in `mlir-hello-world/` directory with:
- `main.cpp` - Complete working program
- `CMakeLists.txt` - Build configuration
- `README.md` - Instructions and experiments

**Time Investment**: 4-6 hours
- 1-2 hours: Build MLIR (if needed)
- 1 hour: Build and run hello world
- 2-3 hours: Experiments and understanding

### Key Concepts
- MLIRContext: Manages all IR objects
- OpBuilder: Creates operations
- Operations: func.func, arith.addi, func.return
- SSA form: Immutable values

### Deliverable
✅ Working program that prints MLIR IR for an add function

---

## Week 1, Day 3-4: Core MLIR Concepts

### Goal
Understand the 5 core concepts needed to build a dialect.

### 1. Operations - The Building Blocks

```mlir
%result = arith.addi %a, %b : i32
//        ^operation  ^operands ^type
```

Every operation has:
- **Name**: `dialect.operation` (e.g., `arith.addi`)
- **Operands**: Inputs (SSA values)
- **Results**: Outputs (SSA values)
- **Attributes**: Compile-time constants
- **Regions**: Nested operations (for control flow)

### 2. Types - What Data Looks Like

```mlir
%x : i32                          // 32-bit integer
%y : f32                          // 32-bit float
%tensor : tensor<128x256xf32>     // 2D tensor
%mem : memref<1024xf32>           // Memory buffer
```

**Key Distinction**:
- `tensor` = immutable, value semantics (high-level)
- `memref` = mutable, pointer semantics (low-level hardware)

### 3. Dialects - Namespaces for Operations

Built-in dialects you'll use:
- `arith` - Arithmetic (addi, mulf, etc.)
- `func` - Functions
- `tensor` - Tensor operations
- `linalg` - Linear algebra (matmul)
- `scf` - Control flow (for, if)
- `memref` - Memory operations

### 4. Regions and Blocks

```mlir
func.func @example() {          // Region starts
  // This is a block
  %x = arith.constant 42 : i32
  return
}                                // Region ends
```

### 5. TableGen - Code Generation

Instead of writing boilerplate C++, use TableGen:

```tablegen
def AddOp : Op<"add"> {
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
}
```

Generates C++ code for you automatically.

### Practical Exercise (2-3 hours)

**Read existing MLIR code:**

```bash
# Create a sample
cat > matmul.mlir << 'EOF'
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %C_init = tensor.empty() : tensor<128x512xf32>
  %C = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                     outs(%C_init : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %C : tensor<128x512xf32>
}
EOF

# Validate it
mlir-opt matmul.mlir

# Try a transformation
mlir-opt matmul.mlir --linalg-tile="tile-sizes=64,64,64"
```

**Understand what each line does**:
1. What operations are used?
2. What are their operands?
3. What types do they have?
4. Trace the data flow

### Deliverable
✅ Can read and explain MLIR code
✅ Understand operations, types, and dialects

---

## Week 1, Day 5-7: Minimal Neural Network Dialect

### Goal
Create a minimal "nn" dialect with 3 operations: dense, relu, add.

### Project Structure

```
nn-dialect/
├── include/
│   └── NN/
│       ├── NNDialect.h
│       ├── NNOps.h
│       └── NNOps.td          ← Define operations here
├── lib/
│   └── NN/
│       ├── NNDialect.cpp
│       └── NNOps.cpp
├── tools/
│   └── nn-opt.cpp            ← Your compiler tool
└── CMakeLists.txt
```

### Step 1: Define the Dialect (1 hour)

**File: `include/NN/NNOps.td`**

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NN_Dialect : Dialect {
  let name = "nn";
  let cppNamespace = "::mlir::nn";
  let summary = "Neural network operations dialect";
}

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;
```

### Step 2: Define Operations (3-4 hours)

**Add to `NNOps.td`:**

```tablegen
// Dense layer: output = input @ weights + bias
def DenseOp : NN_Op<"dense", [Pure]> {
  let summary = "Fully connected layer";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$bias
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input `,` $weights `,` $bias attr-dict `:`
    type($input) `,` type($weights) `,` type($bias) `->` type($output)
  }];
}

// ReLU activation: output = max(0, input)
def ReluOp : NN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// Element-wise add
def AddOp : NN_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}
```

### Step 3: Build System (1 hour)

**File: `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(nn-dialect)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(TableGen)

# Generate code from TableGen
set(LLVM_TARGET_DEFINITIONS include/NN/NNOps.td)
mlir_tablegen(NNOps.h.inc -gen-op-decls)
mlir_tablegen(NNOps.cpp.inc -gen-op-defs)
mlir_tablegen(NNDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NNDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(NNOpsIncGen)

# Build the dialect library
add_mlir_library(NNDialect
  lib/NN/NNDialect.cpp
  lib/NN/NNOps.cpp
  DEPENDS NNOpsIncGen
)

# Build the compiler tool
add_llvm_executable(nn-opt tools/nn-opt.cpp)
target_link_libraries(nn-opt PRIVATE
  NNDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)
```

### Step 4: Create Compiler Tool (1 hour)

**File: `tools/nn-opt.cpp`**

```cpp
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "NN/NNDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nn::NNDialect>();
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "NN Dialect Optimizer\n", registry));
}
```

### Step 5: Test It (1 hour)

```bash
# Build
mkdir build && cd build
cmake .. -G Ninja -DMLIR_DIR=/path/to/mlir
ninja

# Create test file
cat > test.mlir << 'EOF'
func.func @test(%input: tensor<10xf32>,
                %w: tensor<10x5xf32>,
                %b: tensor<5xf32>) -> tensor<5xf32> {
  %dense = nn.dense %input, %w, %b :
    tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %relu : tensor<5xf32>
}
EOF

# Test parsing
./nn-opt test.mlir
```

### Deliverable
✅ Working nn-opt tool that can parse your dialect
✅ Can define operations in TableGen
✅ Understand the dialect compilation workflow

**Files to Complete**:
- `include/NN/NNOps.td` - Operation definitions
- `lib/NN/NNDialect.cpp` - Dialect implementation
- `lib/NN/NNOps.cpp` - Operation implementation
- `tools/nn-opt.cpp` - Compiler tool
- `CMakeLists.txt` - Build configuration

---

## Week 2, Day 8-10: Pattern Rewriting & Fusion

### Goal
Implement operation fusion to reduce memory traffic: `dense + relu → fused_dense_relu`

### Why Fusion Matters

```mlir
// Before: 2 memory roundtrips
%dense = nn.dense %input, %w, %b    // Write dense result to memory
%relu = nn.relu %dense               // Read from memory, write relu result

// After: 1 memory roundtrip
%fused = nn.fused_dense_relu %input, %w, %b  // Compute both, write once
```

**Hardware Benefit**: 2x reduction in memory bandwidth usage.

### Step 1: Add Fused Operation (1 hour)

**Add to `NNOps.td`:**

```tablegen
def FusedDenseReluOp : NN_Op<"fused_dense_relu", [Pure]> {
  let summary = "Fused dense + ReLU layer";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$bias
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input `,` $weights `,` $bias attr-dict `:`
    type($input) `,` type($weights) `,` type($bias) `->` type($output)
  }];
}
```

### Step 2: Implement Fusion Pattern (3-4 hours)

**File: `lib/NN/NNPatterns.cpp`**

```cpp
#include "NN/NNOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace nn {

// Pattern: relu(dense(x)) -> fused_dense_relu(x)
struct FuseDenseReluPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    // Check if input comes from a DenseOp
    auto denseOp = reluOp.getInput().getDefiningOp<DenseOp>();
    if (!denseOp)
      return failure();

    // Check if DenseOp has only one use (safe to fuse)
    if (!denseOp->hasOneUse())
      return failure();

    // Create fused operation
    rewriter.replaceOpWithNewOp<FusedDenseReluOp>(
      reluOp,
      reluOp.getType(),
      denseOp.getInput(),
      denseOp.getWeights(),
      denseOp.getBias()
    );

    return success();
  }
};

// Register pattern with the system
void populateFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseDenseReluPattern>(patterns.getContext());
}

} // namespace nn
} // namespace mlir
```

### Step 3: Create Fusion Pass (2-3 hours)

**File: `include/NN/NNPasses.td`**

```tablegen
include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"nn-fusion", "func::FuncOp"> {
  let summary = "Fuse neural network operations";
  let constructor = "mlir::nn::createFusionPass()";
}
```

**File: `lib/NN/FusionPass.cpp`**

```cpp
#include "NN/NNPasses.h"
#include "NN/NNOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace nn {

struct FusionPass : public impl::FusionPassBase<FusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFusionPatterns(patterns);

    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

} // namespace nn
} // namespace mlir
```

### Step 4: Test Fusion (1 hour)

```bash
# Create test
cat > fusion_test.mlir << 'EOF'
func.func @test(%input: tensor<1x784xf32>,
                %w: tensor<784x256xf32>,
                %b: tensor<256xf32>) -> tensor<1x256xf32> {
  %dense = nn.dense %input, %w, %b :
    tensor<1x784xf32>, tensor<784x256xf32>, tensor<256xf32> -> tensor<1x256xf32>
  %relu = nn.relu %dense : tensor<1x256xf32>
  return %relu : tensor<1x256xf32>
}
EOF

# Run fusion pass
./nn-opt fusion_test.mlir --nn-fusion

# Expected output: should see nn.fused_dense_relu instead of separate ops
```

### Deliverable
✅ Fusion pass that combines dense + relu
✅ Understand pattern matching and rewriting
✅ Can verify optimization worked

---

## Week 2, Day 11-14: Loop Tiling for Hardware

### Goal
Implement loop tiling for matrix multiplication to fit in CPU cache.

### The Problem

```
Large matmul (1024x1024): Doesn't fit in L1 cache (32KB)
→ Cache misses → Slow execution
```

### The Solution: Tiling

```
Break into small tiles (64x64): Fits in L1 cache
→ Reuse data → Fast execution
```

### Performance Impact

- **Without tiling**: ~10 GFLOPS (memory-bound)
- **With tiling**: ~50 GFLOPS (compute-bound)
- **Theoretical max**: ~100 GFLOPS (on modern CPU)

### Step 1: Lower to Linalg (2-3 hours)

First, convert your high-level operations to linalg operations that can be tiled.

**File: `lib/NN/LowerToLinalg.cpp`**

```cpp
// Pattern to lower nn.dense to linalg.matmul + linalg.add
struct DenseToLinalgPattern : public OpRewritePattern<DenseOp> {
  using OpRewritePattern<DenseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DenseOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Step 1: Matmul (input @ weights)
    auto matmul = rewriter.create<linalg::MatmulOp>(
      loc,
      ValueRange{op.getInput(), op.getWeights()},
      ValueRange{/* output buffer */}
    );

    // Step 2: Add bias
    auto add = rewriter.create<linalg::AddOp>(
      loc,
      ValueRange{matmul.getResult(0), op.getBias()},
      ValueRange{/* output buffer */}
    );

    rewriter.replaceOp(op, add.getResult(0));
    return success();
  }
};
```

### Step 2: Apply Tiling Transformation (4-5 hours)

Use MLIR's built-in tiling utilities:

**File: `lib/NN/TilingPass.cpp`**

```cpp
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

struct TilingPass : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();

    // Tile size (typically 32, 64, or 128)
    int64_t tileSize = 64;

    func.walk([&](linalg::MatmulOp matmulOp) {
      OpBuilder builder(matmulOp);

      // Tile the matmul into smaller blocks
      SmallVector<int64_t> tileSizes = {tileSize, tileSize, tileSize};

      auto tilingInterface = cast<TilingInterface>(matmulOp.getOperation());
      auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);

      FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCFForOp(builder, tilingInterface, options);

      if (failed(tilingResult))
        return;

      // Replace original op with tiled version
      matmulOp->replaceAllUsesWith(tilingResult->replacements);
      matmulOp->erase();
    });
  }
};
```

### Step 3: Understand the Transformation (1-2 hours)

**Before tiling:**
```mlir
%C = linalg.matmul ins(%A, %B) outs(%C_init) :
  tensor<1024x1024xf32>, tensor<1024x1024xf32> -> tensor<1024x1024xf32>
```

**After tiling (tile size = 64):**
```mlir
scf.for %i = 0 to 1024 step 64 {
  scf.for %j = 0 to 1024 step 64 {
    scf.for %k = 0 to 1024 step 64 {
      // Extract 64x64 tiles
      %A_tile = tensor.extract_slice %A[%i, %k][64, 64][1, 1]
      %B_tile = tensor.extract_slice %B[%k, %j][64, 64][1, 1]
      %C_tile = tensor.extract_slice %C[%i, %j][64, 64][1, 1]

      // Compute on small tile (fits in cache)
      %C_tile_result = linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile)

      // Insert result back
      %C_new = tensor.insert_slice %C_tile_result into %C[%i, %j][64, 64][1, 1]
    }
  }
}
```

### Step 4: Test and Benchmark (2-3 hours)

```bash
# Create large matmul test
cat > large_matmul.mlir << 'EOF'
func.func @matmul(%A: tensor<1024x1024xf32>,
                  %B: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %C_init = tensor.empty() : tensor<1024x1024xf32>
  %C = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%C_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %C : tensor<1024x1024xf32>
}
EOF

# Apply tiling
./nn-opt large_matmul.mlir --linalg-tile="tile-sizes=64,64,64" -o tiled.mlir

# Compare
mlir-opt large_matmul.mlir | wc -l  # Before
mlir-opt tiled.mlir | wc -l         # After (should be much longer)
```

**Tuning tile size:**
```bash
# Try different tile sizes
for size in 32 64 128 256; do
  echo "Testing tile size: $size"
  ./nn-opt large_matmul.mlir --linalg-tile="tile-sizes=$size,$size,$size"
  # Compile and benchmark
done
```

### Deliverable
✅ Tiling pass that breaks large matmuls into cache-friendly blocks
✅ Understand cache hierarchy and why tiling helps
✅ Can tune tile size for target hardware

---

## Week 3, Day 15-17: Tensor to Memory Lowering

### Goal
Lower from high-level tensors to low-level memory operations (memref).

### Why This Matters

- **Tensors**: Mathematical abstraction, immutable
- **Memrefs**: Physical memory, mutable, maps to hardware

Hardware doesn't understand tensors - it needs explicit memory operations.

### The Transformation

```mlir
// Before: High-level
func.func @add(%t1: tensor<1024xf32>, %t2: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = nn.add %t1, %t2 : tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// After: Low-level
func.func @add(%m1: memref<1024xf32>, %m2: memref<1024xf32>, %m_out: memref<1024xf32>) {
  scf.for %i = 0 to 1024 {
    %v1 = memref.load %m1[%i] : memref<1024xf32>
    %v2 = memref.load %m2[%i] : memref<1024xf32>
    %sum = arith.addf %v1, %v2 : f32
    memref.store %sum, %m_out[%i] : memref<1024xf32>
  }
  return
}
```

### Step 1: Use Bufferization (3-4 hours)

MLIR provides the "bufferization" framework to convert tensors to memrefs.

**File: `lib/NN/Bufferize.cpp`**

```cpp
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

void createBufferizationPipeline(OpPassManager &pm) {
  // Step 1: Convert linalg ops to use buffers
  pm.addPass(createLinalgBufferizePass());

  // Step 2: Convert tensor operations to memref
  pm.addPass(bufferization::createBufferizationPass());

  // Step 3: Finalize buffer deallocation
  pm.addPass(bufferization::createBufferDeallocationPass());
}
```

### Step 2: Lower Operations to Loops (2-3 hours)

Convert operations to explicit loops.

```cpp
// Lower linalg operations to loops
pm.addPass(createConvertLinalgToLoopsPass());

// Lower scf.for to control flow
pm.addPass(createConvertSCFToCFPass());
```

### Step 3: Test the Lowering (1-2 hours)

```bash
# Full lowering pipeline
./nn-opt test.mlir \
  --linalg-bufferize \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --buffer-deallocation \
  -o lowered.mlir

# Verify memref operations
cat lowered.mlir
# Should see: memref.alloc, memref.load, memref.store
```

### Deliverable
✅ Lowering pipeline from tensors to memrefs
✅ Understand memory allocation and deallocation
✅ Can see explicit loads/stores in generated code

---

## Week 3, Day 18-21: End-to-End Compilation

### Goal
Build a complete pipeline from your NN dialect to executable code.

### The Full Pipeline

```
NN Dialect (nn.dense, nn.relu)
    ↓ [Fusion Pass]
NN Dialect (nn.fused_dense_relu)
    ↓ [Lower to Linalg]
Linalg Dialect (linalg.matmul, linalg.generic)
    ↓ [Tiling Pass]
Tiled Linalg (nested scf.for loops)
    ↓ [Bufferization]
Memref Dialect (memref.load, memref.store)
    ↓ [Lower to LLVM]
LLVM IR
    ↓ [LLVM Backend]
Machine Code (executable)
```

### Step 1: Create Pipeline Pass (2-3 hours)

**File: `lib/NN/Pipeline.cpp`**

```cpp
void buildNNCompilerPipeline(OpPassManager &pm) {
  // ===== High-level optimizations =====
  pm.addPass(nn::createFusionPass());
  pm.addPass(createCSEPass());  // Common subexpression elimination

  // ===== Lower to Linalg =====
  pm.addPass(nn::createLowerToLinalgPass());

  // ===== Hardware optimizations =====
  pm.addPass(createLinalgTilingPass(/*tileSize=*/64));

  // ===== Lower to memory operations =====
  pm.addPass(createLinalgBufferizePass());
  pm.addPass(bufferization::createBufferizationPass());
  pm.addPass(createConvertLinalgToLoopsPass());

  // ===== Lower to LLVM IR =====
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}
```

### Step 2: Extend nn-opt Tool (1 hour)

**Update `tools/nn-opt.cpp`:**

```cpp
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all needed dialects
  registry.insert<mlir::nn::NNDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Register passes
  nn::registerNNPasses();
  mlir::registerTransformsPasses();
  mlir::linalg::registerPasses();

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "NN Compiler\n", registry));
}
```

### Step 3: Create Compilation Script (1 hour)

**File: `compile.sh`**

```bash
#!/bin/bash
set -e

INPUT=$1
OUTPUT=${2:-a.out}

echo "=== Stage 1: NN optimizations ==="
./nn-opt $INPUT \
  --nn-fusion \
  --cse \
  -o stage1.mlir

echo "=== Stage 2: Lower to Linalg ==="
./nn-opt stage1.mlir \
  --convert-nn-to-linalg \
  -o stage2.mlir

echo "=== Stage 3: Hardware optimizations ==="
./nn-opt stage2.mlir \
  --linalg-tile="tile-sizes=64,64,64" \
  -o stage3.mlir

echo "=== Stage 4: Bufferization ==="
./nn-opt stage3.mlir \
  --linalg-bufferize \
  --convert-linalg-to-loops \
  --buffer-deallocation \
  -o stage4.mlir

echo "=== Stage 5: Lower to LLVM ==="
./nn-opt stage4.mlir \
  --convert-scf-to-cf \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o stage5.mlir

echo "=== Stage 6: Translate to LLVM IR ==="
mlir-translate --mlir-to-llvmir stage5.mlir -o stage6.ll

echo "=== Stage 7: Compile to object file ==="
llc stage6.ll -o stage7.o -filetype=obj

echo "=== Stage 8: Link to executable ==="
clang stage7.o -o $OUTPUT

echo "Done! Executable: $OUTPUT"
```

### Step 4: Test End-to-End (2-3 hours)

**Create full test:**

```mlir
// mlp.mlir
func.func @mlp_inference(
  %input: tensor<1x784xf32>,
  %w1: tensor<784x256xf32>,
  %b1: tensor<256xf32>,
  %w2: tensor<256x10xf32>,
  %b2: tensor<10xf32>
) -> tensor<1x10xf32> {
  // Layer 1
  %h1 = nn.dense %input, %w1, %b1 :
    tensor<1x784xf32>, tensor<784x256xf32>, tensor<256xf32> -> tensor<1x256xf32>
  %h1_act = nn.relu %h1 : tensor<1x256xf32>

  // Layer 2
  %output = nn.dense %h1_act, %w2, %b2 :
    tensor<1x256xf32>, tensor<256x10xf32>, tensor<10xf32> -> tensor<1x10xf32>

  return %output : tensor<1x10xf32>
}
```

**Compile and run:**

```bash
chmod +x compile.sh
./compile.sh mlp.mlir mlp_executable

# Inspect each stage
echo "=== Original ==="
cat mlp.mlir

echo "=== After fusion ==="
cat stage1.mlir | grep "nn\."

echo "=== After tiling ==="
cat stage3.mlir | grep "scf.for"

echo "=== Final LLVM IR ==="
cat stage6.ll | head -50
```

### Step 5: Add Runtime Wrapper (2-3 hours)

Create C++ wrapper to call the compiled function:

**File: `runtime.cpp`**

```cpp
#include <iostream>
#include <vector>
#include <chrono>

// Forward declare the compiled MLIR function
extern "C" void mlp_inference(float* input, float* w1, float* b1,
                               float* w2, float* b2, float* output);

int main() {
  // Allocate memory
  std::vector<float> input(784);
  std::vector<float> w1(784 * 256);
  std::vector<float> b1(256);
  std::vector<float> w2(256 * 10);
  std::vector<float> b2(10);
  std::vector<float> output(10);

  // Initialize with random data
  for (auto& v : input) v = rand() / (float)RAND_MAX;
  for (auto& v : w1) v = rand() / (float)RAND_MAX - 0.5f;
  for (auto& v : b1) v = 0.0f;
  for (auto& v : w2) v = rand() / (float)RAND_MAX - 0.5f;
  for (auto& v : b2) v = 0.0f;

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1000; i++) {
    mlp_inference(input.data(), w1.data(), b1.data(),
                  w2.data(), b2.data(), output.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Average time: " << duration.count() / 1000.0 << " μs\n";
  std::cout << "Output sample: ";
  for (int i = 0; i < 10; i++) std::cout << output[i] << " ";
  std::cout << "\n";

  return 0;
}
```

**Compile with runtime:**

```bash
clang++ runtime.cpp stage7.o -o mlp_benchmark
./mlp_benchmark
```

### Deliverable
✅ Complete compilation pipeline
✅ Can compile NN dialect to executable
✅ Benchmark showing performance

---

## Week 4, Day 22-24: Testing & Validation

### Goal
Ensure your compiler produces correct and performant code.

### Three Types of Tests

1. **Unit Tests**: Individual passes
2. **Integration Tests**: Full pipeline
3. **Performance Tests**: Benchmark against reference

### Step 1: Unit Tests with FileCheck (4-6 hours)

MLIR uses FileCheck for testing transformations.

**File: `test/fusion.mlir`**

```mlir
// RUN: nn-opt %s --nn-fusion | FileCheck %s

// CHECK-LABEL: func @test_fusion
func.func @test_fusion(%input: tensor<10xf32>,
                       %w: tensor<10x5xf32>,
                       %b: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK-NOT: nn.dense
  // CHECK-NOT: nn.relu
  // CHECK: nn.fused_dense_relu
  %dense = nn.dense %input, %w, %b :
    tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %relu : tensor<5xf32>
}
```

**Run tests:**

```bash
# Using LLVM's lit testing tool
lit test/
```

**Create more tests:**

- `test/tiling.mlir` - Verify tiling creates loops
- `test/lowering.mlir` - Verify memref operations
- `test/pipeline.mlir` - Full pipeline test

### Step 2: Numerical Correctness (4-6 hours)

**File: `test/correctness.cpp`**

```cpp
#include <cassert>
#include <cmath>
#include <iostream>

extern "C" void mlp_inference(float* input, float* w1, float* b1,
                               float* w2, float* b2, float* output);

// Reference implementation
void mlp_reference(const float* input, const float* w1, const float* b1,
                   const float* w2, const float* b2, float* output) {
  // Layer 1: dense + relu
  float h1[256] = {0};
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 784; j++) {
      h1[i] += input[j] * w1[j * 256 + i];
    }
    h1[i] += b1[i];
    h1[i] = std::max(0.0f, h1[i]);  // ReLU
  }

  // Layer 2: dense
  for (int i = 0; i < 10; i++) {
    output[i] = 0;
    for (int j = 0; j < 256; j++) {
      output[i] += h1[j] * w2[j * 10 + i];
    }
    output[i] += b2[i];
  }
}

int main() {
  // Allocate and initialize
  float input[784], w1[784*256], b1[256], w2[256*10], b2[10];
  float output_compiled[10], output_reference[10];

  for (int i = 0; i < 784; i++) input[i] = (i % 100) / 100.0f;
  for (int i = 0; i < 784*256; i++) w1[i] = (i % 200 - 100) / 100.0f;
  for (int i = 0; i < 256; i++) b1[i] = 0.0f;
  for (int i = 0; i < 256*10; i++) w2[i] = (i % 200 - 100) / 100.0f;
  for (int i = 0; i < 10; i++) b2[i] = 0.0f;

  // Run both versions
  mlp_inference(input, w1, b1, w2, b2, output_compiled);
  mlp_reference(input, w1, b1, w2, b2, output_reference);

  // Compare results
  float max_error = 0.0f;
  for (int i = 0; i < 10; i++) {
    float error = std::abs(output_compiled[i] - output_reference[i]);
    max_error = std::max(max_error, error);
    std::cout << "Output[" << i << "]: compiled=" << output_compiled[i]
              << ", reference=" << output_reference[i]
              << ", error=" << error << "\n";
  }

  std::cout << "\nMax error: " << max_error << "\n";

  // Assert correctness (allow small floating point errors)
  assert(max_error < 1e-4 && "Numerical error too large!");

  std::cout << "✅ Correctness test PASSED\n";
  return 0;
}
```

### Step 3: Performance Benchmark (2-3 hours)

**File: `benchmark/benchmark.cpp`**

```cpp
#include <iostream>
#include <chrono>
#include <vector>

extern "C" void mlp_inference(float* input, float* w1, float* b1,
                               float* w2, float* b2, float* output);

void benchmark(int iterations) {
  std::vector<float> input(784), w1(784*256), b1(256), w2(256*10), b2(10), output(10);

  // Warmup
  for (int i = 0; i < 100; i++) {
    mlp_inference(input.data(), w1.data(), b1.data(), w2.data(), b2.data(), output.data());
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    mlp_inference(input.data(), w1.data(), b1.data(), w2.data(), b2.data(), output.data());
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avg_time = duration / (double)iterations;

  // Calculate GFLOPS
  // Layer 1: 784 * 256 * 2 (matmul) = 401,408 ops
  // Layer 2: 256 * 10 * 2 (matmul) = 5,120 ops
  double total_ops = 401408 + 5120;
  double gflops = (total_ops / avg_time) / 1000.0;

  std::cout << "Average time: " << avg_time << " μs\n";
  std::cout << "Throughput: " << gflops << " GFLOPS\n";
}

int main() {
  std::cout << "=== Benchmark with tiling ===\n";
  benchmark(10000);
  return 0;
}
```

**Compare with/without optimization:**

```bash
# Without tiling
./nn-opt mlp.mlir --convert-nn-to-linalg --linalg-bufferize ... -o no_tiling.mlir
# Compile and benchmark
./benchmark_no_tiling

# With tiling
./nn-opt mlp.mlir --convert-nn-to-linalg --linalg-tile="tile-sizes=64,64,64" --linalg-bufferize ... -o with_tiling.mlir
# Compile and benchmark
./benchmark_with_tiling

# Compare speedup
```

### Deliverable
✅ Unit tests with FileCheck
✅ Numerical correctness validation
✅ Performance benchmarks showing optimization impact

---

## Week 4, Day 25-28: Final Project

### Goal
Polish everything into a complete, working compiler.

### Checklist

#### 1. Complete Dialect (2-3 hours)
- [ ] All operations defined in TableGen
- [ ] Documentation comments
- [ ] Verification methods
- [ ] Assembly format

#### 2. All Passes Implemented (2-3 hours)
- [ ] Fusion pass
- [ ] Lowering to Linalg
- [ ] Tiling pass
- [ ] Bufferization
- [ ] LLVM lowering

#### 3. Build System (1 hour)
- [ ] CMakeLists.txt working
- [ ] All dependencies specified
- [ ] Builds without warnings

#### 4. Testing (2-3 hours)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Correctness tests pass
- [ ] Performance benchmarks run

#### 5. Documentation (2-3 hours)
- [ ] README with build instructions
- [ ] Example usage
- [ ] Architecture diagram
- [ ] Performance results

#### 6. Demo (2-3 hours)

Create a compelling demo:

**File: `demo.sh`**

```bash
#!/bin/bash

echo "========================================="
echo "NN Compiler Demo"
echo "========================================="

echo -e "\n📝 Input: 2-layer MLP"
cat examples/mlp.mlir

echo -e "\n🔧 Compiling with optimizations..."
./compile.sh examples/mlp.mlir mlp_optimized

echo -e "\n🔧 Compiling without optimizations..."
./compile.sh examples/mlp.mlir mlp_baseline --no-tiling

echo -e "\n✅ Running correctness test..."
./correctness_test

echo -e "\n📊 Benchmarking..."
echo "Baseline (no tiling):"
./mlp_baseline --benchmark
echo ""
echo "Optimized (with tiling):"
./mlp_optimized --benchmark

echo -e "\n🎉 Demo complete!"
```

### Final Project Structure

```
nn-compiler/
├── README.md                    # Project overview
├── docs/
│   └── architecture.md          # Compiler architecture
├── include/
│   └── NN/
│       ├── NNDialect.h
│       ├── NNOps.h
│       ├── NNOps.td
│       └── NNPasses.h
├── lib/
│   └── NN/
│       ├── NNDialect.cpp
│       ├── NNOps.cpp
│       ├── FusionPass.cpp
│       ├── LowerToLinalg.cpp
│       ├── TilingPass.cpp
│       └── Pipeline.cpp
├── tools/
│   └── nn-opt.cpp
├── test/
│   ├── fusion.mlir
│   ├── tiling.mlir
│   ├── lowering.mlir
│   └── pipeline.mlir
├── benchmark/
│   ├── benchmark.cpp
│   └── correctness.cpp
├── examples/
│   ├── mlp.mlir
│   └── simple.mlir
├── scripts/
│   ├── compile.sh
│   └── demo.sh
└── CMakeLists.txt
```

### Deliverable

✅ Complete compiler project
✅ Comprehensive documentation
✅ Working demo
✅ Performance analysis

**Expected Results**:
- 2-5x speedup from tiling (depending on hardware)
- Numerical accuracy within 1e-4
- Clean, well-documented code
- Understanding of end-to-end compiler flow

---

## Success Criteria

After 4 weeks, you should be able to:

✅ **Understand MLIR fundamentals**
- Operations, types, dialects, regions
- SSA form and IR structure

✅ **Build a custom dialect**
- Define operations in TableGen
- Implement dialect in C++
- Create custom compiler tools

✅ **Implement optimizations**
- Pattern rewriting for fusion
- Loop tiling for cache efficiency
- Understand hardware constraints

✅ **Build compilation pipelines**
- High-level to low-level lowering
- Tensor to memref conversion
- LLVM code generation

✅ **Validate compiler correctness**
- Write tests with FileCheck
- Numerical correctness validation
- Performance benchmarking

✅ **Complete end-to-end project**
- Working compiler for 2-layer MLP
- Measurable performance improvements
- Professional-quality code

---

## What's Next?

After completing this 4-week intensive, you can:

### Extend Your Compiler
- Add more operations (conv2d, pooling, batch_norm)
- Implement more optimizations (operator scheduling, memory planning)
- Target different hardware (GPU, custom accelerators)

### Contribute to Real Projects
- **IREE**: Production ML compiler
- **torch-mlir**: PyTorch to MLIR
- **LLVM/MLIR**: Core infrastructure

### Advanced Topics
- Auto-tuning tile sizes
- Automatic differentiation
- Distributed compilation
- Hardware-specific backends (CUDA, Vulkan)

---

## Resources

### Documentation
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Dialect Documentation](https://mlir.llvm.org/docs/Dialects/)
- [TableGen Documentation](https://llvm.org/docs/TableGen/)

### Tutorials
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR Examples](https://github.com/llvm/llvm-project/tree/main/mlir/examples)

### Papers
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)

### Community
- [MLIR Discourse](https://discourse.llvm.org/c/mlir/31)
- [LLVM Discord](https://discord.gg/xS7Z362)

---

## Tips for Success

1. **Start Simple**: Get each phase working before moving to the next
2. **Test Early**: Write tests as you build, not after
3. **Read Examples**: Study existing dialects in MLIR
4. **Ask Questions**: MLIR community is very helpful
5. **Iterate**: First make it work, then make it fast
6. **Track Progress**: Keep notes on what you learn each day
7. **Stay Focused**: It's only 4 weeks - stay on track!

**Remember**: The goal is a minimal but complete compiler. Don't get distracted by extra features. Finish the core pipeline first!

Good luck! 🚀

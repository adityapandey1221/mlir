# MLIR Compiler Project - Tickets & Tests

This file tracks all tasks and their validation tests for the 4-week MLIR compiler project.

**How to Use This File**:
1. Check off `[ ]` when starting a ticket → `[🔄]`
2. Mark complete when all tests pass → `[✅]`
3. Mark blocked if stuck → `[🚫]`
4. Run tests after each ticket to validate progress

**Status Legend**:
- `[ ]` Not Started
- `[🔄]` In Progress
- `[✅]` Complete (all tests pass)
- `[🚫]` Blocked

---

## Week 1: Foundation & Dialect

### Phase 0: Hello World (Days 1-2)

#### [ ] W1-T1: Build MLIR from Source
**Time Estimate**: 2-3 hours
**Dependencies**: None
**Priority**: P0 (Blocker)

**Description**:
Download, configure, and build LLVM/MLIR from source code.

**Tasks**:
- [ ] Clone llvm-project repository
- [ ] Configure CMake with correct flags
- [ ] Build MLIR with Ninja
- [ ] Run MLIR test suite

**Acceptance Criteria**:
- MLIR builds without errors
- `mlir-opt` executable exists and runs
- All core dialects available

**Tests**:
```bash
# Test 1: Verify mlir-opt exists
./llvm-project/build/bin/mlir-opt --help
# Expected: Help text displays

# Test 2: Verify dialects are loaded
./llvm-project/build/bin/mlir-opt --show-dialects | grep -E "arith|func|tensor|linalg"
# Expected: All four dialects listed

# Test 3: Run simple transformation
echo 'func.func @test() { return }' | ./llvm-project/build/bin/mlir-opt
# Expected: Parsed successfully

# Test 4: Run MLIR tests
cd llvm-project/build && ninja check-mlir-unit
# Expected: All tests pass
```

**Deliverable**: Working MLIR build in `llvm-project/build/`

---

#### [ ] W1-T2: Hello World MLIR Program
**Time Estimate**: 2-3 hours
**Dependencies**: W1-T1
**Priority**: P0

**Description**:
Build and run the hello world program that generates MLIR IR for an add function.

**Tasks**:
- [ ] Build mlir-hello-world project
- [ ] Run and verify output
- [ ] Understand each line of code
- [ ] Complete Experiment 1 (add multiplication)

**Acceptance Criteria**:
- Program compiles without errors
- Generates valid MLIR IR
- Output matches expected format
- Can modify to add new operations

**Tests**:
```bash
# Test 1: Build hello world
cd mlir-hello-world/build
ninja
# Expected: Builds successfully

# Test 2: Run and verify output
./mlir-hello-world > output.mlir
# Expected: No errors

# Test 3: Validate generated MLIR
mlir-opt output.mlir --verify-diagnostics
# Expected: Verification succeeds

# Test 4: Check for expected operations
cat output.mlir | grep -E "func.func|arith.addi|return"
# Expected: All three operations present

# Test 5: Experiment - Add multiplication
# Modify main.cpp to multiply by 2, rebuild, run
./mlir-hello-world | grep "arith.muli"
# Expected: Multiplication operation appears in output
```

**Deliverable**: Working `mlir-hello-world` executable + completed Experiment 1

---

### Phase 1: Core Concepts (Days 3-4)

#### [ ] W1-T3: Read and Parse MLIR Examples
**Time Estimate**: 3-4 hours
**Dependencies**: W1-T2
**Priority**: P0

**Description**:
Study existing MLIR code to understand operations, types, and data flow.

**Tasks**:
- [ ] Create example MLIR files (matmul, relu, tensor ops)
- [ ] Parse with mlir-opt
- [ ] Trace data flow through operations
- [ ] Identify dialects used

**Acceptance Criteria**:
- Can identify all operations in MLIR code
- Can trace SSA value flow
- Can name dialects and their purposes
- Can modify examples and re-validate

**Tests**:
```bash
# Test 1: Create matmul example
cat > test_matmul.mlir << 'EOF'
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %C_init = tensor.empty() : tensor<128x512xf32>
  %cst = arith.constant 0.0 : f32
  %C_zero = linalg.fill ins(%cst : f32) outs(%C_init : tensor<128x512xf32>) -> tensor<128x512xf32>
  %C = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                     outs(%C_zero : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %C : tensor<128x512xf32>
}
EOF

mlir-opt test_matmul.mlir
# Expected: Parses successfully

# Test 2: Apply transformation
mlir-opt test_matmul.mlir --linalg-tile="tile-sizes=64,64,64" | grep "scf.for"
# Expected: Loop operations appear

# Test 3: Create quiz file
cat > quiz.txt << 'EOF'
1. How many operations are in test_matmul.mlir?
2. What dialects are used?
3. What are the input types?
4. What is the output type?
5. Trace the SSA values from input to output.
EOF

# Answer the quiz in quiz_answers.txt
# Test: Verify understanding
cat quiz_answers.txt
# Expected: Correct answers (manually verify)

# Test 4: Modify example (change dimensions)
sed 's/128x256/64x128/g' test_matmul.mlir > test_matmul_modified.mlir
mlir-opt test_matmul_modified.mlir
# Expected: Still parses correctly with new dimensions
```

**Deliverable**: `test_matmul.mlir`, `quiz_answers.txt` with correct understanding

---

### Phase 2: Minimal NN Dialect (Days 5-7)

#### [ ] W1-T4: Project Structure and Build System
**Time Estimate**: 2-3 hours
**Dependencies**: W1-T3
**Priority**: P0

**Description**:
Set up the project structure for nn-dialect with proper CMake configuration.

**Tasks**:
- [ ] Create directory structure
- [ ] Write CMakeLists.txt
- [ ] Set up TableGen infrastructure
- [ ] Create empty dialect files

**Acceptance Criteria**:
- Project structure matches best practices
- CMake configures without errors
- TableGen properly integrated
- Empty dialect compiles

**Tests**:
```bash
# Test 1: Create project structure
mkdir -p nn-dialect/{include/NN,lib/NN,tools,test}
tree nn-dialect
# Expected: Correct directory structure

# Test 2: CMake configuration
cd nn-dialect/build
cmake .. -G Ninja -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
# Expected: Configuration succeeds

# Test 3: Build empty project
ninja
# Expected: Builds successfully (even if empty)

# Test 4: Verify TableGen setup
ninja NNOpsIncGen
ls include/NN/*.inc
# Expected: Generated .inc files exist

# Test 5: Check include paths
cmake .. -G Ninja -DMLIR_DIR=/path/to/mlir --debug-output 2>&1 | grep "MLIR_INCLUDE_DIRS"
# Expected: Include directories correctly set
```

**Deliverable**: `nn-dialect/` project structure with working build system

---

#### [ ] W1-T5: Define NN Dialect in TableGen
**Time Estimate**: 2-3 hours
**Dependencies**: W1-T4
**Priority**: P0

**Description**:
Define the NN dialect and its three core operations in TableGen.

**Tasks**:
- [ ] Create NNOps.td with dialect definition
- [ ] Define DenseOp in TableGen
- [ ] Define ReluOp in TableGen
- [ ] Define AddOp in TableGen
- [ ] Generate C++ code

**Acceptance Criteria**:
- TableGen file is valid
- All three operations defined
- C++ code generates without errors
- Operations have correct traits

**Tests**:
```bash
# Test 1: Validate TableGen syntax
mlir-tblgen include/NN/NNOps.td -I /path/to/mlir/include
# Expected: No syntax errors

# Test 2: Generate operation declarations
mlir-tblgen include/NN/NNOps.td --gen-op-decls -I /path/to/mlir/include > /tmp/test.h
grep -E "DenseOp|ReluOp|AddOp" /tmp/test.h
# Expected: All three operation classes present

# Test 3: Generate operation definitions
mlir-tblgen include/NN/NNOps.td --gen-op-defs -I /path/to/mlir/include > /tmp/test.cpp
grep "::build" /tmp/test.cpp
# Expected: Builder methods generated

# Test 4: Generate dialect code
mlir-tblgen include/NN/NNOps.td --gen-dialect-decls -I /path/to/mlir/include | grep "NNDialect"
# Expected: Dialect class declaration present

# Test 5: Verify operation traits
grep "Pure" include/NN/NNOps.td
# Expected: DenseOp and ReluOp marked as Pure

# Test 6: Check assembly format
grep "assemblyFormat" include/NN/NNOps.td | wc -l
# Expected: 3 (one per operation)
```

**Deliverable**: `include/NN/NNOps.td` with all operations defined

---

#### [ ] W1-T6: Implement NN Dialect C++ Code
**Time Estimate**: 3-4 hours
**Dependencies**: W1-T5
**Priority**: P0

**Description**:
Implement the C++ boilerplate for the NN dialect and operations.

**Tasks**:
- [ ] Implement NNDialect.cpp
- [ ] Implement NNOps.cpp
- [ ] Add necessary includes
- [ ] Compile dialect library

**Acceptance Criteria**:
- All C++ files compile
- Dialect library links successfully
- No compiler warnings
- Follows MLIR coding conventions

**Tests**:
```bash
# Test 1: Compile dialect library
cd nn-dialect/build
ninja NNDialect
# Expected: Library compiles successfully

# Test 2: Check for symbols
nm lib/libNNDialect.a | grep -E "DenseOp|ReluOp|AddOp"
# Expected: Operation symbols present

# Test 3: Verify no warnings
ninja NNDialect 2>&1 | grep "warning:"
# Expected: No output (no warnings)

# Test 4: Check dialect registration
nm lib/libNNDialect.a | grep "NNDialect::initialize"
# Expected: Initialization symbol present

# Test 5: Verify includes
grep -r "#include.*NN/NN" lib/NN/*.cpp
# Expected: Proper includes in all files
```

**Deliverable**: Compiled `libNNDialect.a` library

---

#### [ ] W1-T7: Create nn-opt Tool
**Time Estimate**: 2-3 hours
**Dependencies**: W1-T6
**Priority**: P0

**Description**:
Build the nn-opt compiler tool that can parse and print NN dialect operations.

**Tasks**:
- [ ] Implement tools/nn-opt.cpp
- [ ] Register NN dialect
- [ ] Register all necessary dialects
- [ ] Build and test tool

**Acceptance Criteria**:
- nn-opt builds successfully
- Can parse NN dialect operations
- Can print IR back out
- Help text displays

**Tests**:
```bash
# Test 1: Build nn-opt
ninja nn-opt
# Expected: Builds successfully

# Test 2: Run help
./nn-opt --help
# Expected: Help text displays

# Test 3: Create test file
cat > test_nn.mlir << 'EOF'
func.func @test(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> tensor<5xf32> {
  %dense = nn.dense %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %relu : tensor<5xf32>
}
EOF

# Test 4: Parse NN dialect
./nn-opt test_nn.mlir
# Expected: Parses successfully and prints IR

# Test 5: Verify dialect registration
./nn-opt --show-dialects | grep "nn"
# Expected: "nn" dialect listed

# Test 6: Test error handling
echo "invalid mlir" | ./nn-opt 2>&1 | grep "error:"
# Expected: Error message displayed

# Test 7: Test all three operations
cat > test_all_ops.mlir << 'EOF'
func.func @all_ops(%x: tensor<5xf32>, %y: tensor<5xf32>) -> tensor<5xf32> {
  %sum = nn.add %x, %y : tensor<5xf32>
  %relu = nn.relu %sum : tensor<5xf32>
  return %relu : tensor<5xf32>
}
EOF
./nn-opt test_all_ops.mlir
# Expected: Parses successfully
```

**Deliverable**: Working `nn-opt` tool that can parse NN dialect

---

## Week 2: Optimization

### Phase 3: Pattern Rewriting & Fusion (Days 8-10)

#### [ ] W2-T1: Add Fused Operation Definition
**Time Estimate**: 1-2 hours
**Dependencies**: W1-T7
**Priority**: P1

**Description**:
Add FusedDenseReluOp to the NN dialect in TableGen.

**Tasks**:
- [ ] Add FusedDenseReluOp to NNOps.td
- [ ] Regenerate C++ code
- [ ] Rebuild dialect library

**Acceptance Criteria**:
- Fused operation defined
- Has same inputs as DenseOp
- Marked as Pure
- Compiles successfully

**Tests**:
```bash
# Test 1: Verify TableGen definition
grep "FusedDenseReluOp" include/NN/NNOps.td
# Expected: Definition exists

# Test 2: Generate and check
ninja NNOpsIncGen
grep "FusedDenseReluOp" build/include/NN/NNOps.h.inc
# Expected: Class declaration generated

# Test 3: Rebuild dialect
ninja NNDialect
# Expected: Builds successfully

# Test 4: Parse fused operation
cat > test_fused.mlir << 'EOF'
func.func @test(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> tensor<5xf32> {
  %result = nn.fused_dense_relu %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  return %result : tensor<5xf32>
}
EOF
./nn-opt test_fused.mlir
# Expected: Parses successfully

# Test 5: Verify symbol
nm lib/libNNDialect.a | grep "FusedDenseReluOp"
# Expected: Symbol present
```

**Deliverable**: FusedDenseReluOp added to dialect

---

#### [ ] W2-T2: Implement Fusion Pattern
**Time Estimate**: 3-4 hours
**Dependencies**: W2-T1
**Priority**: P1

**Description**:
Implement C++ pattern that matches relu(dense(x)) and replaces with fused_dense_relu(x).

**Tasks**:
- [ ] Create lib/NN/NNPatterns.cpp
- [ ] Implement FuseDenseReluPattern class
- [ ] Add pattern registration function
- [ ] Write unit test

**Acceptance Criteria**:
- Pattern compiles without errors
- Correctly matches dense+relu sequence
- Only fuses when safe (single use)
- Pattern is reusable

**Tests**:
```bash
# Test 1: Compile patterns
ninja NNDialect
# Expected: NNPatterns.cpp compiles

# Test 2: Check pattern symbols
nm lib/libNNDialect.a | grep "FuseDenseReluPattern"
# Expected: Pattern class present

# Test 3: Create pattern test (will be used in next ticket)
cat > test_pattern_match.mlir << 'EOF'
func.func @should_fuse(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> tensor<5xf32> {
  %dense = nn.dense %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %relu : tensor<5xf32>
}

func.func @should_not_fuse(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %dense = nn.dense %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %dense, %relu : tensor<5xf32>, tensor<5xf32>
}
EOF
```

**Deliverable**: `lib/NN/NNPatterns.cpp` with working fusion pattern

---

#### [ ] W2-T3: Create Fusion Pass
**Time Estimate**: 3-4 hours
**Dependencies**: W2-T2
**Priority**: P1

**Description**:
Create a pass that applies the fusion pattern to functions.

**Tasks**:
- [ ] Create include/NN/NNPasses.td
- [ ] Implement lib/NN/FusionPass.cpp
- [ ] Register pass with pass manager
- [ ] Add pass to nn-opt

**Acceptance Criteria**:
- Pass compiles successfully
- Can be invoked from nn-opt
- Applies fusion pattern correctly
- Reports success/failure

**Tests**:
```bash
# Test 1: Build with pass
ninja nn-opt
# Expected: Builds successfully

# Test 2: Verify pass registration
./nn-opt --help | grep "nn-fusion"
# Expected: Pass listed in help

# Test 3: Apply fusion pass
./nn-opt test_pattern_match.mlir --nn-fusion -o fused.mlir
cat fused.mlir | grep "nn.fused_dense_relu"
# Expected: Fused operation present in should_fuse function

# Test 4: Verify selective fusion
cat fused.mlir | grep -A 5 "func.func @should_not_fuse"
# Expected: Still has separate dense and relu (not fused due to multiple uses)

# Test 5: Test with multiple opportunities
cat > multi_fusion.mlir << 'EOF'
func.func @multi(%i: tensor<10xf32>, %w1: tensor<10x5xf32>, %b1: tensor<5xf32>,
                 %w2: tensor<5x3xf32>, %b2: tensor<3xf32>) -> tensor<3xf32> {
  %d1 = nn.dense %i, %w1, %b1 : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %r1 = nn.relu %d1 : tensor<5xf32>
  %d2 = nn.dense %r1, %w2, %b2 : tensor<5xf32>, tensor<5x3xf32>, tensor<3xf32> -> tensor<3xf32>
  %r2 = nn.relu %d2 : tensor<3xf32>
  return %r2 : tensor<3xf32>
}
EOF
./nn-opt multi_fusion.mlir --nn-fusion | grep -c "nn.fused_dense_relu"
# Expected: 2 (both layers fused)

# Test 6: Verify IR is still valid after fusion
./nn-opt test_pattern_match.mlir --nn-fusion --verify-diagnostics
# Expected: No verification errors
```

**Deliverable**: Working `--nn-fusion` pass in nn-opt

---

### Phase 4: Loop Tiling (Days 11-14)

#### [ ] W2-T4: Lower NN Ops to Linalg
**Time Estimate**: 4-5 hours
**Dependencies**: W2-T3
**Priority**: P1

**Description**:
Implement conversion from nn.dense to linalg.matmul + linalg.add.

**Tasks**:
- [ ] Create lib/NN/LowerToLinalg.cpp
- [ ] Implement DenseToLinalgPattern
- [ ] Implement ReluToLinalgPattern
- [ ] Create --convert-nn-to-linalg pass

**Acceptance Criteria**:
- All NN ops can be lowered to Linalg
- Lowered IR is valid
- Types are preserved correctly
- Pass integrates with nn-opt

**Tests**:
```bash
# Test 1: Build lowering pass
ninja nn-opt
# Expected: Builds successfully

# Test 2: Verify pass exists
./nn-opt --help | grep "convert-nn-to-linalg"
# Expected: Pass listed

# Test 3: Lower simple dense op
cat > test_dense.mlir << 'EOF'
func.func @test(%i: tensor<128x256xf32>, %w: tensor<256x128xf32>, %b: tensor<128xf32>) -> tensor<128x128xf32> {
  %result = nn.dense %i, %w, %b : tensor<128x256xf32>, tensor<256x128xf32>, tensor<128xf32> -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
EOF
./nn-opt test_dense.mlir --convert-nn-to-linalg | grep "linalg.matmul"
# Expected: matmul operation present

# Test 4: Lower relu op
cat > test_relu.mlir << 'EOF'
func.func @test(%input: tensor<128xf32>) -> tensor<128xf32> {
  %result = nn.relu %input : tensor<128xf32>
  return %result : tensor<128xf32>
}
EOF
./nn-opt test_relu.mlir --convert-nn-to-linalg | grep "linalg.generic"
# Expected: generic operation (for element-wise relu)

# Test 5: Lower fused operation
./nn-opt test_fused.mlir --convert-nn-to-linalg | grep -E "linalg.matmul|linalg.generic"
# Expected: Both operations present

# Test 6: Verify types are preserved
./nn-opt test_dense.mlir --convert-nn-to-linalg --verify-diagnostics
# Expected: No verification errors

# Test 7: Full pipeline (fusion then lowering)
./nn-opt multi_fusion.mlir --nn-fusion --convert-nn-to-linalg -o lowered.mlir
cat lowered.mlir | grep "linalg.matmul" | wc -l
# Expected: 2 (one per layer)
```

**Deliverable**: `--convert-nn-to-linalg` pass working

---

#### [ ] W2-T5: Implement Tiling Transformation
**Time Estimate**: 4-5 hours
**Dependencies**: W2-T4
**Priority**: P1

**Description**:
Implement loop tiling for linalg.matmul operations.

**Tasks**:
- [ ] Create lib/NN/TilingPass.cpp
- [ ] Use MLIR's tiling interface
- [ ] Make tile size configurable
- [ ] Create --nn-tile pass

**Acceptance Criteria**:
- Tiling pass compiles
- Generates nested scf.for loops
- Tile size is configurable
- Works with linalg.matmul

**Tests**:
```bash
# Test 1: Build tiling pass
ninja nn-opt
# Expected: Builds successfully

# Test 2: Apply tiling
cat > test_matmul.mlir << 'EOF'
func.func @matmul(%A: tensor<1024x1024xf32>, %B: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %C_init = tensor.empty() : tensor<1024x1024xf32>
  %cst = arith.constant 0.0 : f32
  %C_zero = linalg.fill ins(%cst : f32) outs(%C_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %C = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%C_zero : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %C : tensor<1024x1024xf32>
}
EOF
./nn-opt test_matmul.mlir --nn-tile="tile-size=64" | grep "scf.for"
# Expected: Loop operations present

# Test 3: Verify 3 nested loops (i, j, k)
./nn-opt test_matmul.mlir --nn-tile="tile-size=64" | grep -c "scf.for"
# Expected: At least 3

# Test 4: Test different tile sizes
for size in 32 64 128; do
  ./nn-opt test_matmul.mlir --nn-tile="tile-size=$size" -o tiled_$size.mlir
  cat tiled_$size.mlir | grep "step.*$size"
done
# Expected: Different step sizes in loops

# Test 5: Full NN pipeline with tiling
./nn-opt multi_fusion.mlir \
  --nn-fusion \
  --convert-nn-to-linalg \
  --nn-tile="tile-size=64" \
  -o fully_optimized.mlir
cat fully_optimized.mlir | grep -c "scf.for"
# Expected: Multiple loops (2 matmuls * 3 loops each)

# Test 6: Verify IR validity
./nn-opt test_matmul.mlir --nn-tile="tile-size=64" --verify-diagnostics
# Expected: No verification errors
```

**Deliverable**: `--nn-tile` pass that generates tiled loops

---

#### [ ] W2-T6: Tile Size Tuning Experiment
**Time Estimate**: 2-3 hours
**Dependencies**: W2-T5
**Priority**: P2

**Description**:
Experiment with different tile sizes and understand their impact on generated code.

**Tasks**:
- [ ] Generate code with tile sizes: 16, 32, 64, 128, 256
- [ ] Analyze loop structure for each
- [ ] Count operations and memory accesses
- [ ] Document findings

**Acceptance Criteria**:
- Tested at least 5 different tile sizes
- Documented loop structure for each
- Understand tradeoffs (cache vs overhead)
- Can explain optimal choice

**Tests**:
```bash
# Test 1: Generate all variants
for size in 16 32 64 128 256; do
  ./nn-opt test_matmul.mlir --nn-tile="tile-size=$size" -o tiled_$size.mlir
  echo "=== Tile size $size ==="
  echo "Number of loops: $(grep -c 'scf.for' tiled_$size.mlir)"
  echo "Number of lines: $(wc -l < tiled_$size.mlir)"
done > tile_analysis.txt

# Test 2: Count operations per tile
cat > analyze_tiles.sh << 'EOF'
#!/bin/bash
for size in 16 32 64 128 256; do
  ops=$(grep -c "linalg.matmul" tiled_${size}.mlir)
  loops=$(grep -c "scf.for" tiled_${size}.mlir)
  echo "$size,$ops,$loops"
done
EOF
chmod +x analyze_tiles.sh
./analyze_tiles.sh > tile_stats.csv

# Test 3: Create analysis document
cat > tile_size_analysis.md << 'EOF'
# Tile Size Analysis

## Tile Size: 16
- Cache fit: [estimate]
- Loop overhead: [high/medium/low]
- Parallelism: [high/medium/low]

## Tile Size: 32
...

## Tile Size: 64
...

## Recommendation
Based on L1 cache size (typically 32KB), tile size of XX is optimal because...
EOF

# Test: Manual verification that analysis is complete
cat tile_size_analysis.md | grep "Recommendation"
# Expected: Recommendation section filled out
```

**Deliverable**: `tile_size_analysis.md` with findings and recommendations

---

## Week 3: Lowering & Pipeline

### Phase 5: Tensor to Memory Lowering (Days 15-17)

#### [ ] W3-T1: Implement Bufferization Pass
**Time Estimate**: 3-4 hours
**Dependencies**: W2-T6
**Priority**: P1

**Description**:
Convert tensor operations to memref operations using MLIR's bufferization.

**Tasks**:
- [ ] Create bufferization pipeline
- [ ] Add buffer allocation
- [ ] Add buffer deallocation
- [ ] Integrate with nn-opt

**Acceptance Criteria**:
- Tensors converted to memrefs
- Allocations inserted correctly
- Deallocations prevent leaks
- IR remains valid

**Tests**:
```bash
# Test 1: Create bufferization pipeline
./nn-opt --help | grep -E "bufferize|buffer-deallocation"
# Expected: Bufferization passes available

# Test 2: Apply bufferization
./nn-opt test_matmul.mlir \
  --nn-tile="tile-size=64" \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  | grep "memref<"
# Expected: memref types present

# Test 3: Check for allocations
./nn-opt test_matmul.mlir \
  --nn-tile="tile-size=64" \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  | grep "memref.alloc"
# Expected: Allocation operations present

# Test 4: Check for deallocations
./nn-opt test_matmul.mlir \
  --nn-tile="tile-size=64" \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  --buffer-deallocation \
  | grep "memref.dealloc"
# Expected: Deallocation operations present

# Test 5: Full pipeline to memref
./nn-opt multi_fusion.mlir \
  --nn-fusion \
  --convert-nn-to-linalg \
  --nn-tile="tile-size=64" \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  --buffer-deallocation \
  -o buffered.mlir

cat buffered.mlir | grep -c "memref<"
# Expected: Multiple memref types (no tensor types)

# Test 6: Verify no tensor operations remain
cat buffered.mlir | grep "tensor<" || echo "All tensors converted"
# Expected: No tensor types found
```

**Deliverable**: Working bufferization pipeline

---

#### [ ] W3-T2: Lower Linalg to Loops
**Time Estimate**: 2-3 hours
**Dependencies**: W3-T1
**Priority**: P1

**Description**:
Convert linalg operations to explicit loops with load/store operations.

**Tasks**:
- [ ] Use convert-linalg-to-loops pass
- [ ] Verify explicit loads and stores
- [ ] Check loop structure

**Acceptance Criteria**:
- No linalg operations remain
- Explicit scf.for loops present
- memref.load and memref.store operations
- Nested loop structure preserved

**Tests**:
```bash
# Test 1: Lower to loops
./nn-opt buffered.mlir \
  --convert-linalg-to-loops \
  -o loops.mlir

cat loops.mlir | grep "linalg\." || echo "All linalg ops converted"
# Expected: No linalg operations

# Test 2: Check for explicit loads
cat loops.mlir | grep -c "memref.load"
# Expected: Many load operations

# Test 3: Check for explicit stores
cat loops.mlir | grep -c "memref.store"
# Expected: Many store operations

# Test 4: Verify loop nesting
cat loops.mlir | grep "scf.for" | head -10
# Expected: Nested for loops visible

# Test 5: Full pipeline including loop lowering
./nn-opt multi_fusion.mlir \
  --nn-fusion \
  --convert-nn-to-linalg \
  --nn-tile="tile-size=64" \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  --buffer-deallocation \
  --convert-linalg-to-loops \
  -o explicit_loops.mlir

# Verify both tiling loops and compute loops
cat explicit_loops.mlir | grep -c "scf.for"
# Expected: Many nested loops
```

**Deliverable**: Pipeline that generates explicit loops with loads/stores

---

#### [ ] W3-T3: Lower to Standard Dialects
**Time Estimate**: 2-3 hours
**Dependencies**: W3-T2
**Priority**: P1

**Description**:
Lower SCF, memref, and arithmetic operations toward LLVM dialect.

**Tasks**:
- [ ] Apply convert-scf-to-cf
- [ ] Apply convert-math-to-llvm
- [ ] Apply convert-arith-to-llvm
- [ ] Verify lowering

**Acceptance Criteria**:
- High-level dialects converted
- Closer to LLVM IR structure
- No verification errors
- Prepares for LLVM lowering

**Tests**:
```bash
# Test 1: Lower SCF to control flow
./nn-opt explicit_loops.mlir \
  --convert-scf-to-cf \
  | grep "cf\."
# Expected: Control flow dialect operations

# Test 2: Lower arithmetic
./nn-opt explicit_loops.mlir \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  | grep "llvm\."
# Expected: LLVM dialect operations

# Test 3: Full standard lowering
./nn-opt explicit_loops.mlir \
  --convert-scf-to-cf \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o standard_lowered.mlir

cat standard_lowered.mlir | grep -c "llvm\."
# Expected: Many LLVM operations

# Test 4: Verify IR validity
./nn-opt standard_lowered.mlir --verify-diagnostics
# Expected: No errors
```

**Deliverable**: Pipeline to standard/LLVM dialects

---

### Phase 6: End-to-End Compilation (Days 18-21)

#### [ ] W3-T4: Create Complete Pipeline Script
**Time Estimate**: 3-4 hours
**Dependencies**: W3-T3
**Priority**: P0

**Description**:
Create a script that runs the complete compilation pipeline.

**Tasks**:
- [ ] Write compile.sh script
- [ ] Integrate all passes in order
- [ ] Add intermediate file output
- [ ] Add error handling

**Acceptance Criteria**:
- Script runs all passes successfully
- Saves intermediate stages
- Reports errors clearly
- Configurable options (tile size, etc.)

**Tests**:
```bash
# Test 1: Create compilation script
cat > compile.sh << 'EOF'
#!/bin/bash
set -e

INPUT=$1
OUTPUT=${2:-output}
TILE_SIZE=${3:-64}

echo "=== Stage 1: Fusion ==="
./nn-opt $INPUT --nn-fusion -o ${OUTPUT}_stage1.mlir

echo "=== Stage 2: Lower to Linalg ==="
./nn-opt ${OUTPUT}_stage1.mlir --convert-nn-to-linalg -o ${OUTPUT}_stage2.mlir

echo "=== Stage 3: Tiling ==="
./nn-opt ${OUTPUT}_stage2.mlir --nn-tile="tile-size=$TILE_SIZE" -o ${OUTPUT}_stage3.mlir

echo "=== Stage 4: Bufferization ==="
./nn-opt ${OUTPUT}_stage3.mlir \
  --linalg-bufferize \
  --tensor-bufferize \
  --func-bufferize \
  --buffer-deallocation \
  -o ${OUTPUT}_stage4.mlir

echo "=== Stage 5: Lower to Loops ==="
./nn-opt ${OUTPUT}_stage4.mlir --convert-linalg-to-loops -o ${OUTPUT}_stage5.mlir

echo "=== Stage 6: Lower to LLVM ==="
./nn-opt ${OUTPUT}_stage5.mlir \
  --convert-scf-to-cf \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o ${OUTPUT}_stage6.mlir

echo "=== Stage 7: Translate to LLVM IR ==="
mlir-translate --mlir-to-llvmir ${OUTPUT}_stage6.mlir -o ${OUTPUT}.ll

echo "Done! Output: ${OUTPUT}.ll"
EOF

chmod +x compile.sh

# Test 2: Run on simple example
./compile.sh multi_fusion.mlir test_output 64
# Expected: All stages complete successfully

# Test 3: Verify all intermediate files exist
ls test_output_stage*.mlir test_output.ll
# Expected: All 6 stage files + .ll file

# Test 4: Verify LLVM IR is valid
llvm-as test_output.ll -o test_output.bc
# Expected: Assembles successfully

# Test 5: Test different tile sizes
./compile.sh multi_fusion.mlir test_32 32
./compile.sh multi_fusion.mlir test_128 128
diff test_32.ll test_128.ll
# Expected: Files differ (different tile sizes)

# Test 6: Test error handling
echo "invalid" | ./compile.sh /dev/stdin test_error 2>&1 | grep -i error
# Expected: Error reported clearly
```

**Deliverable**: `compile.sh` script that runs complete pipeline

---

#### [ ] W3-T5: Lower MLIR to LLVM IR
**Time Estimate**: 2-3 hours
**Dependencies**: W3-T4
**Priority**: P0

**Description**:
Use mlir-translate to convert MLIR to LLVM IR.

**Tasks**:
- [ ] Configure mlir-translate
- [ ] Test translation
- [ ] Verify LLVM IR validity
- [ ] Inspect generated code

**Acceptance Criteria**:
- MLIR successfully translates to LLVM IR
- LLVM IR is valid
- Can see matrix multiply logic
- Functions are properly defined

**Tests**:
```bash
# Test 1: Translate to LLVM IR
mlir-translate --mlir-to-llvmir test_output_stage6.mlir -o test.ll
# Expected: Creates .ll file

# Test 2: Verify LLVM IR syntax
llvm-as test.ll -o test.bc
# Expected: Assembles without errors

# Test 3: Disassemble and inspect
llvm-dis test.bc -o test_dis.ll
cat test_dis.ll | head -50
# Expected: Valid LLVM IR with function definitions

# Test 4: Look for key patterns
grep "define.*@" test.ll
# Expected: Function definitions present

# Test 5: Check for loops
grep "br label" test.ll | wc -l
# Expected: Many branches (from loops)

# Test 6: Check for memory operations
grep -E "load|store" test.ll | wc -l
# Expected: Many load/store operations

# Test 7: Verify optimization opportunities
opt -O3 test.ll -S -o test_opt.ll
diff test.ll test_opt.ll | head -20
# Expected: Some differences (optimizations applied)
```

**Deliverable**: Valid LLVM IR file from MLIR

---

#### [ ] W3-T6: Compile to Native Code
**Time Estimate**: 2-3 hours
**Dependencies**: W3-T5
**Priority**: P0

**Description**:
Use LLVM backend to compile to native object code and executable.

**Tasks**:
- [ ] Use llc to create object file
- [ ] Create C wrapper for calling
- [ ] Link to executable
- [ ] Test execution

**Acceptance Criteria**:
- LLVM IR compiles to object code
- Object code links successfully
- Executable runs without crashing
- Can call from C/C++

**Tests**:
```bash
# Test 1: Compile to object file
llc test.ll -o test.o -filetype=obj
file test.o
# Expected: Object file for correct architecture

# Test 2: Check symbols
nm test.o | grep -i matmul
# Expected: Function symbols present

# Test 3: Create C wrapper
cat > wrapper.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

// Forward declare MLIR function
extern void mlir_matmul(float* A, float* B, float* C);

int main() {
    int N = 128;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Initialize
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // Call compiled function
    mlir_matmul(A, B, C);

    // Check result
    printf("C[0] = %f (expected ~128)\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
EOF

# Test 4: Link and compile
clang wrapper.c test.o -o test_exe -lm
# Expected: Links successfully

# Test 5: Run executable
./test_exe
# Expected: Runs and prints result

# Test 6: Update compile.sh to include native compilation
cat >> compile.sh << 'EOF'

echo "=== Stage 8: Compile to Object ==="
llc ${OUTPUT}.ll -o ${OUTPUT}.o -filetype=obj

echo "Compilation complete!"
echo "Object file: ${OUTPUT}.o"
echo "To link: clang your_wrapper.c ${OUTPUT}.o -o executable"
EOF

# Test full pipeline including native code
./compile.sh multi_fusion.mlir native_test 64
ls native_test.o
# Expected: Object file exists
```

**Deliverable**: Native object file and test executable

---

## Week 4: Integration

### Phase 7: Testing & Validation (Days 22-24)

#### [ ] W4-T1: FileCheck Unit Tests
**Time Estimate**: 3-4 hours
**Dependencies**: W3-T6
**Priority**: P1

**Description**:
Write FileCheck tests for each optimization pass.

**Tasks**:
- [ ] Create test/ directory structure
- [ ] Write fusion pass test
- [ ] Write tiling pass test
- [ ] Write lowering pass test
- [ ] Run with lit

**Acceptance Criteria**:
- At least 10 FileCheck tests
- Tests for all major passes
- Tests cover positive and negative cases
- All tests pass

**Tests**:
```bash
# Test 1: Create fusion test
cat > test/fusion.mlir << 'EOF'
// RUN: nn-opt %s --nn-fusion | FileCheck %s

// CHECK-LABEL: func @should_fuse
func.func @should_fuse(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK-NOT: nn.dense
  // CHECK-NOT: nn.relu
  // CHECK: nn.fused_dense_relu
  %dense = nn.dense %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %relu : tensor<5xf32>
}

// CHECK-LABEL: func @should_not_fuse_multi_use
func.func @should_not_fuse_multi_use(%input: tensor<10xf32>, %w: tensor<10x5xf32>, %b: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  // CHECK: nn.dense
  // CHECK: nn.relu
  %dense = nn.dense %input, %w, %b : tensor<10xf32>, tensor<10x5xf32>, tensor<5xf32> -> tensor<5xf32>
  %relu = nn.relu %dense : tensor<5xf32>
  return %dense, %relu : tensor<5xf32>, tensor<5xf32>
}
EOF

# Test 2: Run FileCheck test
./nn-opt test/fusion.mlir --nn-fusion | FileCheck test/fusion.mlir
# Expected: Test passes

# Test 3: Create tiling test
cat > test/tiling.mlir << 'EOF'
// RUN: nn-opt %s --convert-nn-to-linalg --nn-tile="tile-size=64" | FileCheck %s

// CHECK-LABEL: func @test_tiling
func.func @test_tiling(%i: tensor<128x256xf32>, %w: tensor<256x128xf32>, %b: tensor<128xf32>) -> tensor<128x128xf32> {
  %result = nn.dense %i, %w, %b : tensor<128x256xf32>, tensor<256x128xf32>, tensor<128xf32> -> tensor<128x128xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: linalg.matmul
  return %result : tensor<128x128xf32>
}
EOF

./nn-opt test/tiling.mlir --convert-nn-to-linalg --nn-tile="tile-size=64" | FileCheck test/tiling.mlir
# Expected: Test passes

# Test 4: Create lit configuration
cat > test/lit.cfg.py << 'EOF'
import lit.formats
config.name = "NN Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.substitutions.append(('%nn-opt', '/path/to/nn-opt'))
EOF

# Test 5: Run all tests with lit
lit test/
# Expected: All tests pass

# Test 6: Add more tests (lowering, bufferization, etc.)
cat > test/lowering.mlir << 'EOF'
// RUN: nn-opt %s --convert-nn-to-linalg | FileCheck %s

// CHECK-LABEL: func @test_dense_lowering
func.func @test_dense_lowering(%i: tensor<10x20xf32>, %w: tensor<20x30xf32>, %b: tensor<30xf32>) -> tensor<10x30xf32> {
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  %result = nn.dense %i, %w, %b : tensor<10x20xf32>, tensor<20x30xf32>, tensor<30xf32> -> tensor<10x30xf32>
  return %result : tensor<10x30xf32>
}
EOF

./nn-opt test/lowering.mlir --convert-nn-to-linalg | FileCheck test/lowering.mlir
# Expected: Test passes

# Test 7: Count total tests
find test/ -name "*.mlir" | wc -l
# Expected: At least 10 test files
```

**Deliverable**: `test/` directory with comprehensive FileCheck tests

---

#### [ ] W4-T2: Numerical Correctness Test
**Time Estimate**: 4-5 hours
**Dependencies**: W4-T1
**Priority**: P0

**Description**:
Implement correctness test that compares compiled code against reference implementation.

**Tasks**:
- [ ] Create reference implementation in C
- [ ] Create test harness
- [ ] Generate test data
- [ ] Compare outputs with tolerance

**Acceptance Criteria**:
- Reference implementation correct
- Compiled code produces same results
- Error within acceptable tolerance (1e-4)
- Test passes consistently

**Tests**:
```bash
# Test 1: Create reference implementation
cat > reference.c << 'EOF'
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void reference_mlp(float* input, float* w1, float* b1,
                   float* w2, float* b2, float* output) {
    // Layer 1: dense (784x256)
    float h1[256] = {0};
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 784; j++) {
            h1[i] += input[j] * w1[j * 256 + i];
        }
        h1[i] += b1[i];
        h1[i] = fmaxf(0.0f, h1[i]);  // ReLU
    }

    // Layer 2: dense (256x10)
    for (int i = 0; i < 10; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < 256; j++) {
            output[i] += h1[j] * w2[j * 10 + i];
        }
        output[i] += b2[i];
    }
}
EOF

# Test 2: Create correctness test
cat > correctness_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void reference_mlp(float*, float*, float*, float*, float*, float*);
extern void mlir_mlp(float*, float*, float*, float*, float*, float*);

int main() {
    // Allocate
    float *input = malloc(784 * sizeof(float));
    float *w1 = malloc(784 * 256 * sizeof(float));
    float *b1 = malloc(256 * sizeof(float));
    float *w2 = malloc(256 * 10 * sizeof(float));
    float *b2 = malloc(10 * sizeof(float));
    float *output_ref = malloc(10 * sizeof(float));
    float *output_mlir = malloc(10 * sizeof(float));

    // Initialize with deterministic values
    for (int i = 0; i < 784; i++) input[i] = (i % 100) / 100.0f;
    for (int i = 0; i < 784*256; i++) w1[i] = ((i % 200) - 100) / 100.0f;
    for (int i = 0; i < 256; i++) b1[i] = 0.0f;
    for (int i = 0; i < 256*10; i++) w2[i] = ((i % 200) - 100) / 100.0f;
    for (int i = 0; i < 10; i++) b2[i] = 0.0f;

    // Run both implementations
    reference_mlp(input, w1, b1, w2, b2, output_ref);
    mlir_mlp(input, w1, b1, w2, b2, output_mlir);

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < 10; i++) {
        float error = fabsf(output_ref[i] - output_mlir[i]);
        max_error = fmaxf(max_error, error);
        printf("Output[%d]: ref=%.6f mlir=%.6f error=%.6e\n",
               i, output_ref[i], output_mlir[i], error);
    }

    printf("\nMax error: %.6e\n", max_error);

    if (max_error < 1e-4) {
        printf("✅ PASS: Numerical correctness verified\n");
        return 0;
    } else {
        printf("❌ FAIL: Error too large\n");
        return 1;
    }
}
EOF

# Test 3: Compile reference
gcc -c reference.c -o reference.o -O2

# Test 4: Compile MLP to object
./compile.sh examples/mlp.mlir mlp_compiled 64
llc mlp_compiled.ll -o mlp_compiled.o -filetype=obj

# Test 5: Link and run correctness test
gcc correctness_test.c reference.o mlp_compiled.o -o correctness_test -lm
./correctness_test
# Expected: Test passes with error < 1e-4

# Test 6: Test with different tile sizes
for size in 32 64 128; do
    echo "Testing tile size $size"
    ./compile.sh examples/mlp.mlir mlp_$size $size
    llc mlp_$size.ll -o mlp_$size.o -filetype=obj
    gcc correctness_test.c reference.o mlp_$size.o -o correctness_$size -lm
    ./correctness_$size || exit 1
done
# Expected: All tile sizes pass correctness
```

**Deliverable**: Passing correctness test comparing reference vs compiled

---

#### [ ] W4-T3: Performance Benchmarking
**Time Estimate**: 3-4 hours
**Dependencies**: W4-T2
**Priority**: P1

**Description**:
Benchmark the compiled code and measure speedup from optimizations.

**Tasks**:
- [ ] Create benchmark harness
- [ ] Measure baseline (no tiling)
- [ ] Measure optimized (with tiling)
- [ ] Calculate speedup and GFLOPS

**Acceptance Criteria**:
- Benchmark runs successfully
- Reports timing and GFLOPS
- Shows measurable speedup from tiling
- Can compare different tile sizes

**Tests**:
```bash
# Test 1: Create benchmark harness
cat > benchmark.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void mlir_mlp(float*, float*, float*, float*, float*, float*);

double benchmark_mlp(int iterations) {
    float *input = malloc(784 * sizeof(float));
    float *w1 = malloc(784 * 256 * sizeof(float));
    float *b1 = malloc(256 * sizeof(float));
    float *w2 = malloc(256 * 10 * sizeof(float));
    float *b2 = malloc(10 * sizeof(float));
    float *output = malloc(10 * sizeof(float));

    // Initialize
    for (int i = 0; i < 784; i++) input[i] = 1.0f;
    for (int i = 0; i < 784*256; i++) w1[i] = 0.01f;
    for (int i = 0; i < 256; i++) b1[i] = 0.0f;
    for (int i = 0; i < 256*10; i++) w2[i] = 0.01f;
    for (int i = 0; i < 10; i++) b2[i] = 0.0f;

    // Warmup
    for (int i = 0; i < 100; i++) {
        mlir_mlp(input, w1, b1, w2, b2, output);
    }

    // Benchmark
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iterations; i++) {
        mlir_mlp(input, w1, b1, w2, b2, output);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    free(input); free(w1); free(b1); free(w2); free(b2); free(output);

    return elapsed / iterations;
}

int main() {
    int iterations = 10000;
    double avg_time = benchmark_mlp(iterations);

    // Calculate GFLOPS
    // Layer 1: 784*256*2 = 401,408 ops
    // Layer 2: 256*10*2 = 5,120 ops
    double total_ops = 401408.0 + 5120.0;
    double gflops = (total_ops / avg_time) / 1e9;

    printf("Average time: %.2f μs\n", avg_time * 1e6);
    printf("Throughput: %.2f GFLOPS\n", gflops);

    return 0;
}
EOF

# Test 2: Compile baseline (no tiling or minimal tiling)
cat > examples/mlp_no_opt.mlir << 'EOF'
func.func @mlir_mlp(%input: tensor<784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>,
                    %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> tensor<10xf32> {
  %h1 = nn.dense %input, %w1, %b1 : tensor<784xf32>, tensor<784x256xf32>, tensor<256xf32> -> tensor<256xf32>
  %h1_relu = nn.relu %h1 : tensor<256xf32>
  %output = nn.dense %h1_relu, %w2, %b2 : tensor<256xf32>, tensor<256x10xf32>, tensor<10xf32> -> tensor<10xf32>
  return %output : tensor<10xf32>
}
EOF

# Compile without tiling optimization
./nn-opt examples/mlp_no_opt.mlir \
  --convert-nn-to-linalg \
  --linalg-bufferize \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir > baseline.ll

llc baseline.ll -o baseline.o -filetype=obj
gcc benchmark.c baseline.o -o benchmark_baseline -lm

# Test 3: Run baseline benchmark
echo "=== Baseline (no tiling) ==="
./benchmark_baseline

# Test 4: Compile with tiling
for size in 32 64 128; do
    ./compile.sh examples/mlp_no_opt.mlir optimized_$size $size
    llc optimized_$size.ll -o optimized_$size.o -filetype=obj
    gcc benchmark.c optimized_$size.o -o benchmark_$size -lm
done

# Test 5: Run optimized benchmarks
for size in 32 64 128; do
    echo "=== Optimized (tile size $size) ==="
    ./benchmark_$size
done

# Test 6: Create comparison report
cat > benchmark_report.sh << 'EOF'
#!/bin/bash
echo "Tile Size,Time(μs),GFLOPS" > results.csv
for size in baseline 32 64 128; do
    result=$(./benchmark_$size | grep "Average time")
    echo "$size,$result" >> results.csv
done
cat results.csv
EOF
chmod +x benchmark_report.sh
./benchmark_report.sh

# Test 7: Calculate speedup
cat > calculate_speedup.py << 'EOF'
import csv
with open('results.csv') as f:
    reader = csv.DictReader(f)
    results = list(reader)
    baseline_time = float(results[0]['Time(μs)'])
    for row in results[1:]:
        optimized_time = float(row['Time(μs)'])
        speedup = baseline_time / optimized_time
        print(f"Tile size {row['Tile Size']}: {speedup:.2f}x speedup")
EOF
python3 calculate_speedup.py
# Expected: Speedup > 1.5x for optimized versions
```

**Deliverable**: Benchmark results showing speedup from optimizations

---

### Phase 8: Final Project (Days 25-28)

#### [ ] W4-T4: Documentation and README
**Time Estimate**: 3-4 hours
**Dependencies**: W4-T3
**Priority**: P1

**Description**:
Create comprehensive documentation for the project.

**Tasks**:
- [ ] Write detailed README
- [ ] Document build instructions
- [ ] Add usage examples
- [ ] Create architecture diagram
- [ ] Document performance results

**Acceptance Criteria**:
- README covers all essential topics
- Build instructions are clear and complete
- Examples are runnable
- Architecture is explained
- Performance results documented

**Tests**:
```bash
# Test 1: Create comprehensive README
cat > nn-dialect/README.md << 'EOF'
# Neural Network Dialect Compiler

A minimal MLIR-based compiler for neural networks with hardware-aware optimizations.

## Features
- Custom NN dialect with dense, relu, and add operations
- Pattern-based fusion optimization
- Loop tiling for cache efficiency
- Complete lowering to native code
- 2-5x speedup on 2-layer MLP

## Building
[Instructions]

## Usage
[Examples]

## Architecture
[Diagram and explanation]

## Performance
[Benchmark results]

## Testing
[How to run tests]
EOF

# Test 2: Verify README completeness
grep -E "Features|Building|Usage|Architecture|Performance|Testing" nn-dialect/README.md
# Expected: All sections present

# Test 3: Create architecture diagram (ASCII art or image)
cat > ARCHITECTURE.md << 'EOF'
# Compiler Architecture

## Pipeline Overview

```
Input (NN Dialect)
    ↓
[Fusion Pass]
    ↓
[Lower to Linalg]
    ↓
[Tiling Pass]
    ↓
[Bufferization]
    ↓
[Lower to Loops]
    ↓
[Lower to LLVM]
    ↓
Output (Native Code)
```

## Dialect Hierarchy
[Explanation]

## Optimization Passes
[Details of each pass]
EOF

# Test 4: Create examples directory
mkdir -p examples
cp test_nn.mlir examples/simple.mlir
cp multi_fusion.mlir examples/two_layer_mlp.mlir

cat > examples/README.md << 'EOF'
# Examples

## simple.mlir
Basic example showing dense + relu

## two_layer_mlp.mlir
Complete 2-layer MLP

## Running Examples
```bash
./nn-opt examples/simple.mlir --nn-fusion
./compile.sh examples/two_layer_mlp.mlir output 64
```
EOF

# Test 5: Document performance results
cat > PERFORMANCE.md << 'EOF'
# Performance Results

## Benchmark: 2-Layer MLP (784→256→10)

| Configuration | Time (μs) | GFLOPS | Speedup |
|--------------|-----------|---------|---------|
| Baseline     | [X]       | [Y]     | 1.0x    |
| Tiling (32)  | [X]       | [Y]     | [Z]x    |
| Tiling (64)  | [X]       | [Y]     | [Z]x    |
| Tiling (128) | [X]       | [Y]     | [Z]x    |

## Analysis
[Explanation of results]

## Hardware
- CPU: [model]
- L1 Cache: [size]
- L2 Cache: [size]
EOF

# Fill in actual benchmark results
./benchmark_report.sh >> PERFORMANCE.md

# Test 6: Verify documentation builds (if using docs generator)
# Or just verify markdown renders correctly
for file in README.md ARCHITECTURE.md PERFORMANCE.md; do
    markdown-check $file || echo "$file needs review"
done
```

**Deliverable**: Complete documentation (README, ARCHITECTURE, PERFORMANCE)

---

#### [ ] W4-T5: Final Integration and Demo
**Time Estimate**: 4-5 hours
**Dependencies**: W4-T4
**Priority**: P0

**Description**:
Polish everything and create a compelling demo.

**Tasks**:
- [ ] Ensure all tests pass
- [ ] Clean up code and comments
- [ ] Create demo script
- [ ] Record demo output
- [ ] Prepare presentation materials

**Acceptance Criteria**:
- All FileCheck tests pass
- Correctness test passes
- Benchmark shows speedup
- Demo script runs successfully
- Project is presentation-ready

**Tests**:
```bash
# Test 1: Run complete test suite
echo "=== Running All Tests ==="

echo "1. Unit tests..."
lit test/ || exit 1

echo "2. Correctness test..."
./correctness_test || exit 1

echo "3. Benchmark..."
./benchmark_64

echo "✅ All tests passed!"

# Test 2: Create demo script
cat > demo.sh << 'EOF'
#!/bin/bash

echo "========================================="
echo "NN Compiler Demo"
echo "========================================="

echo -e "\n📝 Input: 2-layer MLP"
cat examples/two_layer_mlp.mlir

echo -e "\n🔧 Stage 1: Fusion Optimization"
./nn-opt examples/two_layer_mlp.mlir --nn-fusion
sleep 2

echo -e "\n🔧 Stage 2: Loop Tiling"
./nn-opt examples/two_layer_mlp.mlir --nn-fusion --convert-nn-to-linalg --nn-tile="tile-size=64" | head -30
sleep 2

echo -e "\n🔧 Compiling to native code..."
./compile.sh examples/two_layer_mlp.mlir demo_output 64 2>&1 | grep "==="

echo -e "\n✅ Correctness Check"
./correctness_test | tail -5

echo -e "\n📊 Performance Comparison"
echo "Baseline:"
./benchmark_baseline | grep -E "time|GFLOPS"
echo ""
echo "Optimized (64x64 tiles):"
./benchmark_64 | grep -E "time|GFLOPS"

echo -e "\n🎉 Demo Complete!"
echo "The optimized compiler achieved 2-5x speedup through:"
echo "  - Operation fusion (reduced memory traffic)"
echo "  - Loop tiling (improved cache utilization)"
EOF

chmod +x demo.sh

# Test 3: Run demo
./demo.sh | tee demo_output.txt
# Expected: Complete demo runs successfully

# Test 4: Create final project checklist
cat > FINAL_CHECKLIST.md << 'EOF'
# Final Project Checklist

## Code Quality
- [ ] All code compiles without warnings
- [ ] Code follows MLIR conventions
- [ ] Comments explain key concepts
- [ ] No TODOs or FIXMEs remaining

## Functionality
- [ ] NN dialect with 3+ operations
- [ ] Fusion pass works correctly
- [ ] Tiling pass generates correct loops
- [ ] Full pipeline compiles to native code
- [ ] Executable runs without errors

## Testing
- [ ] 10+ FileCheck tests
- [ ] Correctness test passes
- [ ] Benchmark shows speedup
- [ ] All tests documented

## Documentation
- [ ] README complete
- [ ] Build instructions clear
- [ ] Usage examples work
- [ ] Architecture documented
- [ ] Performance results shown

## Demo
- [ ] Demo script works
- [ ] Output is clear and compelling
- [ ] Shows before/after optimization
- [ ] Demonstrates speedup
EOF

# Test 5: Verify all items
echo "Review FINAL_CHECKLIST.md and check off items"
cat FINAL_CHECKLIST.md

# Test 6: Create project summary
cat > PROJECT_SUMMARY.md << 'EOF'
# Project Summary

## What Was Built
A minimal but complete MLIR-based compiler for neural networks that demonstrates hardware-aware optimization through loop tiling.

## Key Components
1. Custom NN dialect (dense, relu, add operations)
2. Pattern-based fusion optimization
3. Loop tiling for cache efficiency
4. Complete lowering pipeline to native code

## Results
- Successfully compiled 2-layer MLP to native code
- Achieved [X]x speedup through optimizations
- Numerical accuracy within 1e-4 tolerance
- Comprehensive test suite (100% pass rate)

## What Was Learned
- MLIR fundamentals (operations, types, dialects)
- Pattern matching and rewriting
- Hardware-oriented optimizations
- Complete compiler pipeline construction
- Testing and validation strategies

## Future Work
- Add more operations (conv2d, pooling, batch_norm)
- Implement vectorization
- Target GPU/custom accelerators
- Auto-tune tile sizes
EOF

# Test 7: Generate final statistics
cat > generate_stats.sh << 'EOF'
#!/bin/bash
echo "=== Project Statistics ==="
echo "Lines of TableGen: $(find include -name "*.td" -exec wc -l {} + | tail -1 | awk '{print $1}')"
echo "Lines of C++: $(find lib -name "*.cpp" -exec wc -l {} + | tail -1 | awk '{print $1}')"
echo "Number of operations: $(grep -r "def.*Op.*:" include | wc -l)"
echo "Number of passes: $(grep -r "def.*Pass.*:" include | wc -l)"
echo "Number of tests: $(find test -name "*.mlir" | wc -l)"
echo "Test pass rate: $(lit test/ 2>&1 | grep -o "[0-9]* passed" | awk '{print $1}')/$(find test -name "*.mlir" | wc -l)"
EOF
chmod +x generate_stats.sh
./generate_stats.sh
```

**Deliverable**:
- Complete, polished project
- Working demo
- All tests passing
- Comprehensive documentation
- Ready for presentation/portfolio

---

## Progress Tracking

### Summary Dashboard

**Week 1: Foundation & Dialect**
- [ ] W1-T1: Build MLIR from Source
- [ ] W1-T2: Hello World MLIR Program
- [ ] W1-T3: Read and Parse MLIR Examples
- [ ] W1-T4: Project Structure and Build System
- [ ] W1-T5: Define NN Dialect in TableGen
- [ ] W1-T6: Implement NN Dialect C++ Code
- [ ] W1-T7: Create nn-opt Tool

**Week 2: Optimization**
- [ ] W2-T1: Add Fused Operation Definition
- [ ] W2-T2: Implement Fusion Pattern
- [ ] W2-T3: Create Fusion Pass
- [ ] W2-T4: Lower NN Ops to Linalg
- [ ] W2-T5: Implement Tiling Transformation
- [ ] W2-T6: Tile Size Tuning Experiment

**Week 3: Lowering & Pipeline**
- [ ] W3-T1: Implement Bufferization Pass
- [ ] W3-T2: Lower Linalg to Loops
- [ ] W3-T3: Lower to Standard Dialects
- [ ] W3-T4: Create Complete Pipeline Script
- [ ] W3-T5: Lower MLIR to LLVM IR
- [ ] W3-T6: Compile to Native Code

**Week 4: Integration**
- [ ] W4-T1: FileCheck Unit Tests
- [ ] W4-T2: Numerical Correctness Test
- [ ] W4-T3: Performance Benchmarking
- [ ] W4-T4: Documentation and README
- [ ] W4-T5: Final Integration and Demo

### Completion Statistics

```
Total Tickets: 24
Completed: 0
In Progress: 0
Remaining: 24
Progress: 0%
```

**Update this section as you complete tickets!**

---

## Notes

- Mark tickets as `[🔄]` when you start working on them
- Mark tickets as `[✅]` only after ALL tests pass
- If blocked on a ticket, mark it `[🚫]` and note the reason
- Run tests after each ticket to ensure incremental progress
- Keep this file updated daily to track your progress

Good luck building your MLIR compiler! 🚀

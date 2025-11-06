# MLIR Hello World

Your first MLIR program! This project demonstrates the basics of the MLIR C++ API by programmatically constructing a simple function that adds two integers.

## What This Program Does

Generates the following MLIR code:

```mlir
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

## Prerequisites

You need MLIR installed. Choose one option:

### Option A: Install Pre-built LLVM/MLIR (Quick)

**Ubuntu/Debian:**
```bash
sudo apt-get install llvm-17 llvm-17-dev mlir-17-tools libmlir-17-dev
```

**macOS:**
```bash
brew install llvm@17
```

### Option B: Build from Source (Recommended for Learning)

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

ninja check-mlir  # Takes 30-60 minutes
```

## Building This Project

```bash
mkdir build
cd build

# If using pre-built MLIR
cmake .. -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

# If you built from source
cmake .. -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm

ninja
```

## Running

```bash
./mlir-hello-world
```

Expected output:
```mlir
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

## Key Concepts Demonstrated

1. **MLIRContext**: The universe for all MLIR objects
2. **OpBuilder**: Factory for creating operations
3. **ModuleOp**: Top-level container
4. **Operations**: Building blocks (func.func, arith.addi, func.return)
5. **SSA Form**: Values are immutable (%arg0, %arg1, %0)

## Experiments to Try

### 1. Add Multiplication

After the addition, multiply the result by 2:

```cpp
auto two = builder.create<mlir::arith::ConstantIntOp>(
  builder.getUnknownLoc(), 2, i32Type
);
mlir::Value multiplied = builder.create<mlir::arith::MulIOp>(
  builder.getUnknownLoc(), result, two
);
// Return multiplied instead of result
builder.create<mlir::func::ReturnOp>(
  builder.getUnknownLoc(),
  multiplied
);
```

### 2. Create Multiple Functions

Add a second function that subtracts:

```cpp
// After creating the add function
auto subFuncOp = builder.create<mlir::func::FuncOp>(
  builder.getUnknownLoc(),
  "subtract",
  funcType
);
mlir::Block *subBlock = subFuncOp.addEntryBlock();
builder.setInsertionPointToStart(subBlock);

mlir::Value diff = builder.create<mlir::arith::SubIOp>(
  builder.getUnknownLoc(),
  subBlock->getArgument(0),
  subBlock->getArgument(1)
);

builder.create<mlir::func::ReturnOp>(
  builder.getUnknownLoc(),
  diff
);
```

### 3. Use Floating Point

Change from `i32` to `f32` and use `arith.addf`:

```cpp
auto f32Type = builder.getF32Type();
auto funcType = builder.getFunctionType({f32Type, f32Type}, {f32Type});
// ...
mlir::Value result = builder.create<mlir::arith::AddFOp>(
  builder.getUnknownLoc(),
  arg0,
  arg1
);
```

## Common Errors

### "MLIR_DIR not found"

Make sure you specify the correct path:
```bash
cmake .. -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir
```

### "undefined reference to mlir::..."

Add missing libraries to `CMakeLists.txt`:
```cmake
target_link_libraries(mlir-hello-world
  PRIVATE
    MLIRIR
    MLIRFunc
    # Add any other needed libraries here
)
```

### Module verification failed

The module is invalid. Print it to see the issue:
```cpp
module.print(llvm::errs());
if (failed(mlir::verify(module))) {
  llvm::errs() << "Module verification failed\n";
}
```

## Next Steps

Once this works, you're ready to:
- ✅ Understand why MLIR exists (Phase 1 of roadmap)
- ✅ Learn about dialects and types (Phase 2)
- ✅ Build your own custom dialect (Phase 3)

See `../MLIR_COMPILER_ROADMAP.md` for the complete learning path!

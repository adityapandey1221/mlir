# NN-MLIR Compiler

This project is a small neural-network compiler built with MLIR.

It takes a simple `.nn` file such as:

```text
network mlp
input 784
dense 256 relu
dense 10
```

and compiles it in two stages:

1. `.nn` source -> MLIR with a custom `nn` dialect
2. `nn` dialect MLIR -> lowered MLIR using `linalg`, `tensor`, and `arith`

The implementation lives in `nn-mlir/`.

## What The Tools Do

- `nn-compiler` reads a `.nn` file and generates MLIR
- `nn-opt` reads MLIR and can run the lowering pass

## Build

```bash
cd nn-mlir
make
```

## Most Important Commands

### 1. Compile a `.nn` file to `nn` dialect MLIR

```bash
./nn-compiler tests/frontend_basic.nn
```

This prints MLIR containing operations like:

- `nn.dense`
- `nn.relu`

### 2. Check that the generated MLIR is valid

```bash
./nn-compiler tests/frontend_basic.nn | ./nn-opt
```

This does **not** lower the program. It only checks that the frontend produced valid MLIR that `nn-opt` can parse and print.

### 3. Compile a `.nn` file directly to lowered MLIR

```bash
./nn-compiler --lower tests/frontend_basic.nn
```

This runs the lowering stage and prints MLIR using standard dialects such as:

- `linalg`
- `tensor`
- `arith`

This is the simplest end-to-end command in the project.

## Extra Test Inputs

These example `.nn` files are included:

- `tests/frontend_basic.nn` - basic two-layer MLP
- `tests/frontend_single_layer.nn` - one dense layer
- `tests/frontend_deep.nn` - deeper multilayer network
- `tests/frontend_comments.nn` - comments and blank lines

Examples:

```bash
./nn-compiler tests/frontend_single_layer.nn
./nn-compiler tests/frontend_deep.nn | ./nn-opt
./nn-compiler --lower tests/frontend_deep.nn
./nn-compiler tests/frontend_comments.nn
```

## MLIR-Only Tests

There are also hand-written MLIR test files:

- `tests/dialect_roundtrip.mlir`
- `tests/lowering_basic.mlir`

Useful commands:

```bash
./nn-opt tests/dialect_roundtrip.mlir
./nn-opt --pass-pipeline='builtin.module(lower-nn-to-standard)' tests/lowering_basic.mlir
```

## Full Writeup

The full project explanation is in [PROJECT_WRITEUP.md](PROJECT_WRITEUP.md).

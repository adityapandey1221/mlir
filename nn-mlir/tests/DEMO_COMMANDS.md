# Demo Commands

Use these exact commands for the submission/demo flow:

```bash
cd nn-mlir
make
./nn-opt tests/dialect_roundtrip.mlir
./nn-compiler tests/frontend_basic.nn
./nn-compiler --lower tests/frontend_basic.nn
```

If you need a quick smoke check of the handwritten frontend path:

```bash
./nn-compiler tests/frontend_basic.nn | ./nn-opt
```

If you want to run the lowering pass explicitly through `nn-opt` instead of using `nn-compiler --lower`:

```bash
./nn-opt --pass-pipeline='builtin.module(lower-nn-to-standard)' tests/lowering_basic.mlir
```

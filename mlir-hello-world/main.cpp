#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  // 1. Create an MLIR context (manages all IR objects)
  mlir::MLIRContext context;

  // 2. Register the dialects we'll use
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();

  // 3. Create a builder (used to construct IR)
  mlir::OpBuilder builder(&context);

  // 4. Create a module (top-level container)
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // 5. Set insertion point to the module body
  builder.setInsertionPointToEnd(module.getBody());

  // 6. Create the function type: (i32, i32) -> i32
  auto i32Type = builder.getI32Type();
  auto funcType = builder.getFunctionType(
    {i32Type, i32Type},  // Input types
    {i32Type}             // Output types
  );

  // 7. Create the function operation
  auto funcOp = builder.create<mlir::func::FuncOp>(
    builder.getUnknownLoc(),
    "add",      // Function name
    funcType
  );

  // 8. Create the function body (a block with two arguments)
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // 9. Get the function arguments
  mlir::Value arg0 = entryBlock->getArgument(0);
  mlir::Value arg1 = entryBlock->getArgument(1);

  // 10. Create the addition operation
  mlir::Value result = builder.create<mlir::arith::AddIOp>(
    builder.getUnknownLoc(),
    arg0,
    arg1
  );

  // 11. Create the return operation
  builder.create<mlir::func::ReturnOp>(
    builder.getUnknownLoc(),
    result
  );

  // 12. Verify the module is well-formed
  if (failed(mlir::verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  // 13. Print the generated MLIR
  module.print(llvm::outs());

  return 0;
}

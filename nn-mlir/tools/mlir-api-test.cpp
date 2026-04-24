#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"

int main() {
  // Step 1: Create an MLIR context
  mlir::MLIRContext context;

  // Step 2: Register all standard dialects
  mlir::registerAllDialects(context);

  // Step 3: Parse an empty module from a string
  std::string code = R"(module {})";
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddMemoryBuffer(
      llvm::MemoryBuffer::getMemBuffer(code),
      llvm::SMLoc());

  auto module = mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  if (!module) {
    llvm::errs() << "Failed to parse module\n";
    return 1;
  }

  // Step 4: Dump it to stdout
  module->dump();
  llvm::outs() << "\n✓ MLIR API test passed: empty module created and dumped\n";

  return 0;
}

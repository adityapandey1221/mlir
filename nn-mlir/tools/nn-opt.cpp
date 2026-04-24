#include "NN/NNDialect.h"
#include "../lib/Conversion/NNToStandard.cpp"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard MLIR dialects
  mlir::registerAllDialects(registry);

  // Register our NN dialect
  registry.insert<mlir::nn::NNDialect>();

  return asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NN MLIR Optimizer", registry));
}

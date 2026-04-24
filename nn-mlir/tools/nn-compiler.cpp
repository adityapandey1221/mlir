#include "../parser/NNParser.h" // DSL parser
#include "../lib/Conversion/NNToStandard.cpp"

#include "NN/NNDialect.h"
#include "NN/NNOps.h"

// Core MLIR APIs
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nn;

namespace {

// Data structure to store the network layer
struct LayerIRInfo {
  unsigned outputSize = 0;
  bool applyRelu = false;
};

LogicalResult buildModule(const nn_frontend::Network &network,
                          MLIRContext &context, ModuleOp &module) {
  OpBuilder builder(&context); // Create an MLIR builder tied to the context
  Location loc = builder.getUnknownLoc(); // use an "unknown" source location for all emitted ops

  module = ModuleOp::create(loc); // create a top-level MLIR module
  builder.setInsertionPointToEnd(module.getBody());

  auto f32 = builder.getF32Type();
  SmallVector<Type> argumentTypes;
  argumentTypes.push_back(
      RankedTensorType::get({1, network.inputSize}, f32)); //network input type is tensor<1 x inputSize x f32>

  SmallVector<LayerIRInfo> layers;
  layers.reserve(network.layers.size());

  unsigned currentSize = network.inputSize;
  for (const auto &layer : network.layers) {
    argumentTypes.push_back(
        RankedTensorType::get({currentSize, layer.units}, f32)); // Add a weight tensor type
    argumentTypes.push_back(RankedTensorType::get({layer.units}, f32)); //Add a bias tensor type
    layers.push_back({layer.units, layer.applyRelu});
    currentSize = layer.units;
  }

  auto resultType = RankedTensorType::get({1, currentSize}, f32); // Create result type tensor<1 x outputSize x f32>
  SmallVector<Type> resultTypes{resultType};
  auto functionType = builder.getFunctionType(argumentTypes, resultTypes);
  auto func = builder.create<func::FuncOp>(loc, network.name, functionType);

  Block *entryBlock = func.addEntryBlock(); // Create block holding the function arguments
  builder.setInsertionPointToStart(entryBlock);

  // Initialize SSA threading
  Value current = entryBlock->getArgument(0);
  unsigned argIndex = 1;

  /*
  For each layer:

  fetch its weight argument
  fetch its bias argument
  create nn.dense
  if needed, create nn.relu
  update current
  */ 

  for (const auto &layer : layers) {
    Value weights = entryBlock->getArgument(argIndex++);
    Value bias = entryBlock->getArgument(argIndex++);

    auto denseType = RankedTensorType::get({1, layer.outputSize}, f32);
    current = builder.create<DenseOp>(loc, denseType, current, weights, bias)
                  .getOutput();

    if (layer.applyRelu) {
      current = builder.create<ReluOp>(loc, denseType, current).getOutput();
    }
  }

  builder.create<func::ReturnOp>(loc, current); // Crete final current value
  return success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                       llvm::cl::desc("<input file>"),
                                       llvm::cl::Required);
  llvm::cl::opt<bool> lowerToStandard(
      "lower",
      llvm::cl::desc("Lower nn dialect IR to standard linalg/tensor/arith IR"),
      llvm::cl::init(false));
  llvm::cl::ParseCommandLineOptions(argc, argv, "NN frontend\n");

  // Get input file
  auto fileOrErr = llvm::MemoryBuffer::getFile(inputFile);
  if (!fileOrErr) {
    llvm::errs() << "error: unable to read input file `"
                 << inputFile << "`\n";
    return 1;
  }

  // Call parseNetworkText
  nn_frontend::Network network;
  std::string errorMessage;
  if (!nn_frontend::parseNetworkText(fileOrErr.get()->getBuffer().str(),
                                     network, errorMessage)) {
    llvm::errs() << "parse error: " << errorMessage << "\n";
    return 1;
  }

  // Register MLIR dialects and load context
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<NNDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Call buildModule; prints MLIR text containing nn.dense and nn.relu to stdout
  ModuleOp module;
  if (failed(buildModule(network, context, module))) {
    llvm::errs() << "error: failed to build MLIR module\n";
    return 1;
  }

  if (lowerToStandard) {
    PassManager pm(&context);
    pm.addPass(mlir::createLowerNNToStandardPass());
    if (failed(pm.run(module))) {
      llvm::errs() << "error: lowering pass failed\n";
      return 1;
    }
  }

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}

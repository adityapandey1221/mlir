#include "NN/NNDialect.h"
#include "NN/NNOps.h"

using namespace mlir;
using namespace mlir::nn;

// The #define flag tells TableGen which section of the .cpp.inc file to include
#define GET_DIALECT_DEFS
#include "NN/NNDialect.cpp.inc"

// Register all operations in this dialect
void NNDialect::initialize() {
    addOperations<
    // GET_OP_LIST tells TableGen to include the operation list (DenseOp, ReluOp)
    #define GET_OP_LIST
    #include "NN/NNOps.cpp.inc"
    >();
}
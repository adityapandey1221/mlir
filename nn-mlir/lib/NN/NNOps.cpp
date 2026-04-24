#include "NN/NNOps.h"                                                                                                                              
#include "NN/NNDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::nn;

// The #define flag tells TableGen which section of the .cpp.inc file to include.
// GET_OP_CLASSES means "include the operation class implementations"
#define GET_OP_CLASSES
#include "NN/NNOps.cpp.inc"
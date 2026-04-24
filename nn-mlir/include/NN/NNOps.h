#ifndef NN_OPS_H                                                                                                                                     
#define NN_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES // Tell generator: emit operation class declarations from NNOps.h.inc
#include "NN/NNOps.h.inc"

#endif // NN_OPS_H
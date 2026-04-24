// This file is intentionally included from tools/nn-opt.cpp so the pass can ship without a Makefile change.
// This file handles the loweing pass from custom nn dialect to standard MLIR dialects
#include "NN/NNDialect.h"
#include "NN/NNOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// Helper function to retun a vector of parallel iterator types for linalg.generic
static SmallVector<utils::IteratorType>
makeParallelIterators(unsigned rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

// Returns identity indexing maps for linalg.generic
static AffineMap makeIdentityMap(unsigned rank, MLIRContext *ctx) {
  return AffineMap::getMultiDimIdentityMap(rank, ctx);
}

// Creates indexing map for broadcasting a 1D bias across last dimension of a 2D tensor
static AffineMap makeBroadcastLastDimMap(unsigned rank, MLIRContext *ctx) {
  if (rank == 0)
    return AffineMap::get(0, 0, {}, ctx);
  return AffineMap::get(rank, 0, getAffineDimExpr(rank - 1, ctx), ctx);
}

// Returns a zero attribute for float and integer types
static FailureOr<Attribute> makeZeroAttr(Type type, OpBuilder &builder) {
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  return failure();
}

// Class responsible for lowering nn.relu
class ReluLowering final : public OpConversionPattern<nn::ReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(nn::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || inputType != resultType)
      return failure();

    Type elementType = resultType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();

    auto zeroAttr = makeZeroAttr(elementType, rewriter);
    if (failed(zeroAttr))
      return failure();

    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto empty = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{}); // Initialize output tensor

    SmallVector<AffineMap> maps = {
        makeIdentityMap(resultType.getRank(), ctx),
        makeIdentityMap(resultType.getRank(), ctx),
    };
    auto iterators = makeParallelIterators(resultType.getRank());

    auto relu = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{empty.getResult()}, maps, iterators, "nn.relu", "",
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value value = args[0];
          auto zero = nestedBuilder.create<arith::ConstantOp>(
              nestedLoc, cast<TypedAttr>(*zeroAttr));
          Value result;
          if (elementType.isa<FloatType>()) {
            result = nestedBuilder
                         .create<arith::MaxFOp>(nestedLoc, value,
                                                zero.getResult())
                         .getResult();
          } else {
            result = nestedBuilder
                         .create<arith::MaxSIOp>(nestedLoc, value,
                                                zero.getResult())
                         .getResult();
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(op, relu.getResultTensors().front());
    return success();
  }
};

// This class is responsible for lowering nn.dense
class DenseLowering final : public OpConversionPattern<nn::DenseOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Extract and validate types
  LogicalResult
  matchAndRewrite(nn::DenseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto weightsType = dyn_cast<RankedTensorType>(adaptor.getWeights().getType());
    auto biasType = dyn_cast<RankedTensorType>(adaptor.getBias().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !weightsType || !biasType || !resultType)
      return failure();

    if (inputType.getRank() != 2 || weightsType.getRank() != 2 ||
        biasType.getRank() != 1 || resultType.getRank() != 2)
      return failure();

    if (inputType.getElementType() != weightsType.getElementType() ||
        inputType.getElementType() != biasType.getElementType() ||
        inputType.getElementType() != resultType.getElementType())
      return failure();

    Type elementType = resultType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();

    auto zeroAttr = makeZeroAttr(elementType, rewriter);
    if (failed(zeroAttr))
      return failure();
    
    // Initialize output tensor
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(*zeroAttr));
    auto empty = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});

    auto filled = rewriter.create<linalg::FillOp>(
        loc, TypeRange{resultType}, ValueRange{zero.getResult()},
        ValueRange{empty.getResult()});
    
    // Computes the matrix multiplication part of nn.dense
    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getWeights()},
        ValueRange{filled.getResultTensors().front()});
    
    // Setup for bias broadcasting to add a 1D bias vector across all rows
    auto biasDest = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    SmallVector<AffineMap> maps = {
        makeIdentityMap(2, ctx),
        makeBroadcastLastDimMap(2, ctx),
        makeIdentityMap(2, ctx),
    };
    auto iterators = makeParallelIterators(2);
    
    // emit linalg::GenericOp for bias add
    auto biased = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType},
        ValueRange{matmul.getResultTensors().front(), adaptor.getBias()},
        ValueRange{biasDest.getResult()}, maps, iterators, "nn.dense.bias",
        "", [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = args[0];
          Value rhs = args[1];
          Value result;
          if (elementType.isa<FloatType>()) {
            result = nestedBuilder
                         .create<arith::AddFOp>(nestedLoc, lhs, rhs)
                         .getResult();
          } else {
            result = nestedBuilder
                         .create<arith::AddIOp>(nestedLoc, lhs, rhs)
                         .getResult();
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(op, biased.getResultTensors().front()); // Replace original op
    return success();
  }
};

// Wraps the rewrite patterns into an actual MLIR pass
struct LowerNNToStandardPass final
    : public PassWrapper<LowerNNToStandardPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-nn-to-standard"; }
  StringRef getDescription() const final {
    return "Lower nn dialect ops to linalg, tensor, and arith";
  }

  // Registers the dialects that the lowered IR will use
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  // Entry point for the lowering pass
  void runOnOperation() final {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    patterns.add<ReluLowering, DenseLowering>(typeConverter, &context);

    ConversionTarget target(context);
    target.addIllegalDialect<nn::NNDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, tensor::TensorDialect>();

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, frozen)))
      signalPassFailure();
  }
};

static PassRegistration<LowerNNToStandardPass> registerNNLoweringPass;

} // namespace

namespace mlir {

std::unique_ptr<Pass> createLowerNNToStandardPass() {
  return std::make_unique<LowerNNToStandardPass>();
}

} // namespace mlir

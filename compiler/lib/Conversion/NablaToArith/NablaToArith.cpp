#include "Conversion/NablaToArith/NablaToArith.hpp"

#include "../Utils.hpp"
#include "Dialect/Nabla/IR/Nabla.hpp"
#include "Dialect/Nabla/IR/NablaInterface.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace nabla {

class AccumulateToArithAddF;
class AccumulateToArithAddI;
class AdjointToArith;

class NablaToArith : public impl::NablaToArithBase<NablaToArith> {
  void runOnOperation() override {
    RewritePatternSet patterns{&getContext()};
    patterns
        .insert<AccumulateToArithAddF, AccumulateToArithAddI, AdjointToArith>(
            &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createNablaToArith() {
  return std::make_unique<NablaToArith>();
}

class AccumulateToArithAddF : public OpRewritePattern<AccumulateOp> {
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getType().isa<FloatType>()) {
      return failure();
    }

    rewriter.replaceOp(op,
                       accumulate<arith::AddFOp>(op.getOperands(), rewriter));
    return success();
  }
};

class AccumulateToArithAddI : public OpRewritePattern<AccumulateOp> {
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getType().isa<IntegerType>()) {
      return failure();
    }

    rewriter.replaceOp(op,
                       accumulate<arith::AddIOp>(op.getOperands(), rewriter));
    return success();
  }
};

class AdjointToArith : public OpRewritePattern<AdjointOp> {
  using OpRewritePattern<AdjointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AdjointOp op,
                                PatternRewriter& rewriter) const override {
    auto target = op.getTarget();
    auto owner = target.getDefiningOp();

    // 判断是否实现了 AdjointInterface
    auto adj = dyn_cast<AdjointInterface>(owner);
    if (!adj) {
      return failure();
    }

    // 如果 dtarget 为空，生成默认 dtarget
    auto dtarget = op.getDtarget();
    if (!dtarget) {
      auto type = target.getType();
      if (isa<FloatType>(type)) {
        auto attr = rewriter.getFloatAttr(type, 1.0);
        dtarget = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
      } else {
        auto attr = rewriter.getIntegerAttr(type, 1);
        dtarget = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
      }
    }

    rewriter.replaceOp(op, adj.adjoint(dtarget, rewriter));
    return success();
  }
};

}  // namespace nabla
}  // namespace mlir

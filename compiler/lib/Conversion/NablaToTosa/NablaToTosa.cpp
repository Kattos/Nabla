#include "Conversion/NablaToTosa/NablaToTosa.hpp"

#include "../Utils.hpp"
#include "Dialect/Nabla/IR/Nabla.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace nabla {

class AccumulateToTosaAdd;

class NablaToTosa : public impl::NablaToTosaBase<NablaToTosa> {
  void runOnOperation() override {
    RewritePatternSet patterns{&getContext()};
    patterns.insert<AccumulateToTosaAdd>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createNablaToTosa() {
  return std::make_unique<NablaToTosa>();
}

class AccumulateToTosaAdd : public OpRewritePattern<AccumulateOp> {
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getType().isa<TensorType>()) {
      return failure();
    }

    rewriter.replaceOp(op, accumulate<tosa::AddOp>(op.getOperands(), rewriter));
    return success();
  }
};

}  // namespace nabla
}  // namespace mlir

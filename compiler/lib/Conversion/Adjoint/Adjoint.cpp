#include "Conversion/Adjoint/Adjoint.hpp"

#include "Dialect/Nabla/IR/Nabla.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace nabla {

class AdjointConversion;

class AdjointPass : public impl::AdjointPassBase<AdjointPass> {
  void runOnOperation() override {
    RewritePatternSet patterns{&getContext()};
    patterns.insert<AdjointConversion>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createAdjointPass() {
  return std::make_unique<AdjointPass>();
}

class AdjointConversion : public OpRewritePattern<AdjointOp> {
  using OpRewritePattern<AdjointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AdjointOp op,
                                PatternRewriter& rewriter) const override {
    // TODO: 调用 target 的 owner 实现的 adjoint 接口
    return failure();
  }
};

}  // namespace nabla
}  // namespace mlir

#include "Math.hpp"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir {
namespace nabla {

using namespace math;

class ExpAdjoint;

void registerMathAdjointInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, MathDialect*) {
    ExpOp::attachInterface<ExpAdjoint>(*context);
  });
}

class ExpAdjoint : public AdjointInterface::ExternalModel<ExpAdjoint, ExpOp> {
 public:
  SmallVector<Value> adjoint(Operation* op, Value dtarget,
                             OpBuilder& builder) const {
    auto exp = cast<ExpOp>(op);
    auto helper = ArithBuilder{builder, op->getLoc()};
    return SmallVector<Value, 1>{helper.mul(dtarget, exp)};
  }
};

}  // namespace nabla
}  // namespace mlir

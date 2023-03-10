#include "Arith.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"

namespace mlir {
namespace nabla {

using namespace arith;

class AddFAdjoint;
class MulFAdjoint;
class SubFAdjoint;

void registerArithAdjointInterface(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* context, ArithDialect*) {
    AddFOp::attachInterface<AddFAdjoint>(*context);
    MulFOp::attachInterface<MulFAdjoint>(*context);
    SubFOp::attachInterface<SubFAdjoint>(*context);
  });
}

class AddFAdjoint
    : public AdjointInterface::ExternalModel<AddFAdjoint, AddFOp> {
 public:
  SmallVector<Value> adjoint(Operation* op, Value dtarget,
                             OpBuilder& builder) const {
    return SmallVector<Value, 2>{dtarget, dtarget};
  }
};

class MulFAdjoint
    : public AdjointInterface::ExternalModel<MulFAdjoint, MulFOp> {
 public:
  SmallVector<Value> adjoint(Operation* op, Value dtarget,
                             OpBuilder& builder) const {
    auto mul = cast<MulFOp>(op);
    auto helper = ArithBuilder{builder, op->getLoc()};
    return SmallVector<Value, 2>{helper.mul(dtarget, mul.getLhs()),
                                 helper.mul(dtarget, mul.getRhs())};
  }
};

class SubFAdjoint
    : public AdjointInterface::ExternalModel<SubFAdjoint, SubFOp> {
 public:
  SmallVector<Value> adjoint(Operation* op, Value dtarget,
                             OpBuilder& builder) const {
    auto neg = builder.create<NegFOp>(op->getLoc(), dtarget);
    return SmallVector<Value, 2>{dtarget, neg};
  }
};

}  // namespace nabla
}  // namespace mlir

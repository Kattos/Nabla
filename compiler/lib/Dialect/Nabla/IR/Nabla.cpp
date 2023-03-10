#include "Dialect/Nabla/IR/Nabla.hpp"

#define GET_OP_CLASSES
#include "Dialect/Nabla/IR/Nabla.cpp.inc"
#include "Dialect/Nabla/IR/NablaDialect.cpp.inc"

namespace mlir {
namespace nabla {

void NablaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Nabla/IR/Nabla.cpp.inc"
      >();
}

LogicalResult AdjointOp::verify() {
  auto target = getTarget();
  if (!target.isa<OpResult>()) {
    return emitOpError("target must be an op result");
  }

  auto dtarget = getDtarget();
  if (dtarget && dtarget.getType() != target.getType()) {
    return emitOpError("dtarget must have the same type as target");
  }

  auto definingOp = target.getDefiningOp();
  auto sources = definingOp->getOperands();
  auto dsources = getDsources();
  if (sources.size() != dsources.size()) {
    return emitOpError("dsources must have the same size as sources");
  }

  auto sourcesTy = definingOp->getOperandTypes();
  auto dsourcesTy = dsources.getTypes();
  if (sourcesTy != dsourcesTy) {
    return emitOpError("dsources must have the same type as sources");
  }

  return success();
}

LogicalResult AccumulateOp::verify() {
  if (getNumOperands() == 0) {
    return emitOpError("must have at least one operand");
  }

  return success();
}

}  // namespace nabla
}  // namespace mlir

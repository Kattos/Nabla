#include "Dialect/Nabla/IR/NablaInterface.hpp"

#include "AdjointInterface/Arith.hpp"
#include "AdjointInterface/Math.hpp"
#include "Dialect/Nabla/IR/Nabla.hpp"
#include "Dialect/Nabla/IR/NablaInterface.cpp.inc"

namespace mlir {
namespace nabla {

void registerAdjointInterface(DialectRegistry& registry) {
  registerArithAdjointInterface(registry);
  registerMathAdjointInterface(registry);
}

}  // namespace nabla
}  // namespace mlir

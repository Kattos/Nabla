#ifndef NABLA_INTERFACE_HPP
#define NABLA_INTERFACE_HPP

// clang-format off
#include "mlir/IR/OpDefinition.h"
#include "Dialect/Nabla/IR/NablaInterface.h.inc"
// clang-format on

namespace mlir {
namespace nabla {
void registerAdjointInterface(DialectRegistry& registry);
}  // namespace nabla
}  // namespace mlir

#endif  // NABLA_INTERFACE_HPP

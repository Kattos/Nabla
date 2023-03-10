#ifndef NABLA_ADJOINT_CONVERSION_HPP
#define NABLA_ADJOINT_CONVERSION_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nabla {

#define GEN_PASS_DEF_ADJOINTPASS
#define GEN_PASS_DECL_ADJOINTPASS
#include "Conversion/Conversion.hpp.inc"

std::unique_ptr<Pass> createAdjointPass();

}  // namespace nabla
}  // namespace mlir

#endif  // NABLA_ADJOINT_CONVERSION_HPP

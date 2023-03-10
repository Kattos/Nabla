#ifndef NABLA_CONVERSION_TO_ARITH_HPP
#define NABLA_CONVERSION_TO_ARITH_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nabla {

#define GEN_PASS_DEF_NABLATOARITH
#define GEN_PASS_DECL_NABLATOARITH
#include "Conversion/Conversion.hpp.inc"

std::unique_ptr<Pass> createNablaToArith();

}  // namespace nabla
}  // namespace mlir

#endif  // NABLA_CONVERSION_TO_ARITH_HPP
